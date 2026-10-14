"""Original physical-scale soil-image regression with nested sample validation.

Real feature extraction and all fitted regression use CUDA GPU 0 explicitly.
Pure tensor helpers also accept tiny CPU tensors for synthetic unit checks.
No competition images, labels, fitted features, or model arrays are bundled here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import unicodedata

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch

ID = "sample_id"
DIAMETERS = np.array([0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2., 6.3, 20., 63., 200.])
COLUMNS = ["0.002", "0.0063", "0.02", "0.063", "0.2", "0.63", "2", "6.3", "20", "63", "200"]
INTERVAL_WEIGHTS = np.diff(np.log10(DIAMETERS))
WEIGHT_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
DEVICE = "cuda:0"
SEED = 2026
PHYSICAL_SIDES_MM = (16.0, 64.0)
CENTERS = ((0.35, 0.35), (0.65, 0.35), (0.5, 0.5), (0.35, 0.65), (0.65, 0.65))
IMAGE_SIZE = 224
BATCH_SIZE = 32


@dataclass(frozen=True)
class Candidate:
    method: str
    alpha: float = 0.0

    @property
    def name(self):
        return "median" if self.method == "median" else f"{self.method}-ridge-{self.alpha:g}"


# Fixed before looking at any competition score. Nested LOOCV chooses among
# these on fitting samples only; the outer held-out sample never chooses a model.
CANDIDATES = (Candidate("median"),) + tuple(
    Candidate(kernel, alpha) for kernel in ("linear", "rbf") for alpha in (0.1, 1.0, 10.0)
)


def canonical(value: str) -> str:
    """Exact deterministic orthographic normalization, never fuzzy matching."""
    value = str(value).casefold().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
    value = unicodedata.normalize("NFKD", value)
    return "".join(character for character in value if character.isalnum() and not unicodedata.combining(character))


def validate_ids(frame: pd.DataFrame) -> None:
    if ID not in frame or frame[ID].isna().any() or frame[ID].duplicated().any():
        raise ValueError("Sample IDs must be present, non-null, and unique.")
    normalized = [canonical(value) for value in frame[ID]]
    if any(not value for value in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("Empty or ambiguous normalized sample IDs.")


def validate_cdfs(values: np.ndarray) -> None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != 11:
        raise ValueError("Expected eleven cumulative percentages per sample.")
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise ValueError("CDF values must be finite percentages in [0,100].")
    if np.any(np.diff(values, axis=1) < -1e-7) or not np.allclose(values[:, -1], 100, atol=1e-7, rtol=0):
        raise ValueError("CDFs must be nondecreasing and end at exactly 100.")


def emd_rows(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Official image formula: LEFT endpoints, not a trapezoidal integral."""
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.shape != prediction.shape or truth.ndim != 2 or truth.shape[1] != 11:
        raise ValueError("Metric arrays must have equal (samples,11) shapes.")
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Metric input must be finite.")
    return (np.abs(truth[:, :-1] - prediction[:, :-1]) * INTERVAL_WEIGHTS).sum(axis=1)


def pava(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Weighted least-squares isotonic projection by pool-adjacent-violators."""
    values = np.asarray(values, dtype=float)
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape or not np.isfinite(values).all():
        raise ValueError("PAVA expects finite one-dimensional values and matching weights.")
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("PAVA weights must be finite and positive.")
    levels, masses, lengths = [], [], []
    for value, weight in zip(values, weights):
        levels.append(float(value)); masses.append(float(weight)); lengths.append(1)
        while len(levels) > 1 and levels[-2] > levels[-1]:
            total = masses[-2] + masses[-1]
            average = (levels[-2] * masses[-2] + levels[-1] * masses[-1]) / total
            length = lengths[-2] + lengths[-1]
            levels[-2:] = [average]; masses[-2:] = [total]; lengths[-2:] = [length]
    return np.repeat(levels, lengths)


def project_cdfs(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim == 1:
        values = values[None]
    if values.shape[1] != 11 or not np.isfinite(values).all():
        raise ValueError("CDF projection requires finite eleven-column predictions.")
    result = np.asarray([np.r_[np.clip(pava(row[:10], INTERVAL_WEIGHTS), 0, 100), 100.] for row in values])
    validate_cdfs(result)
    return result


def load_metadata(data: Path):
    labels = pd.read_csv(data / "Training_labels_updated.csv", dtype={ID: str})
    sample = pd.read_csv(data / "sample_submission.csv", dtype={ID: str})
    ppm = pd.read_csv(data / "ppm_updated.csv")
    validate_ids(labels); validate_ids(sample)
    if list(labels.columns) != [ID] + COLUMNS or list(sample.columns) != [ID] + COLUMNS:
        raise ValueError("Updated labels or sample submission have an unexpected exact column schema.")
    validate_cdfs(labels[COLUMNS].to_numpy(float))
    if set(labels[ID]) & set(sample[ID]):
        raise ValueError("Training and test sample IDs overlap.")
    required = {"phone", "camera", "width", "height", "ppm"}
    if not required <= set(ppm):
        raise ValueError("Updated PPM camera metadata is missing columns.")
    for column in ("width", "height", "ppm"):
        ppm[column] = pd.to_numeric(ppm[column], errors="raise")
        if not np.isfinite(ppm[column]).all() or not (ppm[column] > 0).all():
            raise ValueError(f"Camera {column} must be finite and positive.")
    return labels, sample, ppm


def match_photo(path: Path, sample_ids: list[str], ppm: pd.DataFrame) -> tuple[str, dict]:
    """Remove only documented photo suffixes and exact camera prefixes.

    In particular, iPhone_16 and iPhone16 agree; Muenster and Münster agree;
    commas, underscores and spaces are orthographic separators. Longer camera
    prefixes take precedence over Motorola Edge for the Edge 60 Fusion model.
    """
    stem = re.sub(r"\s*\(\d+\)$", "", path.stem)
    stem = re.sub(r"_\d{1,3}$", "", stem)
    normalized = canonical(stem)
    aliases = []
    for row in ppm.to_dict("records"):
        for field in ("phone", "camera"):
            prefix = canonical(row[field])
            if prefix and normalized.startswith(prefix):
                aliases.append((len(prefix), prefix, row))
    if not aliases:
        raise ValueError(f"Unknown camera prefix: {path.name}")
    aliases.sort(key=lambda item: (-item[0], item[1]))
    longest = aliases[0][0]
    cameras = {canonical(alias[2]["phone"]) for alias in aliases if alias[0] == longest}
    if len(cameras) != 1:
        raise ValueError(f"Ambiguous camera prefix: {path.name}")
    _, prefix, camera = aliases[0]
    remainder = normalized[len(prefix):]
    identities = {canonical(sample_id): sample_id for sample_id in sample_ids}
    if len(identities) != len(sample_ids):
        raise ValueError("Sample identity normalization collided.")
    if remainder not in identities:
        raise ValueError(f"Photo does not match an exact known sample ID: {path.name}")
    return identities[remainder], camera


def photo_ledger(data: Path, sample_ids: list[str], ppm: pd.DataFrame, split: str) -> list[dict]:
    directory_tag = "training-all_photos_updated" if split == "train" else "test_all_photos"
    paths = sorted(path for path in data.rglob("*") if path.is_file()
                   and path.suffix.casefold() in {".jpg", ".jpeg", ".png"}
                   and any(part.casefold() == directory_tag for part in path.parts))
    if not paths:
        raise FileNotFoundError(f"No {split} images under the official {directory_tag} directory.")
    ledger = []
    for path in paths:
        sample_id, camera = match_photo(path, sample_ids, ppm)
        ledger.append({"path": path, "sample_id": sample_id, "camera": camera})
    observed = {row[ID] for row in ledger}
    if observed != set(sample_ids):
        raise ValueError(f"Missing {split} photographs for: {sorted(set(sample_ids)-observed)}")
    return ledger


def physical_crop(image: Image.Image, ppm: float, side_mm: float | None,
                  center: tuple[float, float]) -> tuple[Image.Image, float]:
    if ppm <= 0 or not math.isfinite(ppm):
        raise ValueError("Crop PPM must be finite and positive.")
    width, height = image.size
    maximum = max(1, int(0.90 * min(width, height)))
    side = maximum if side_mm is None else min(maximum, max(1, round(ppm * side_mm)))
    left = int(np.clip(round(center[0] * width - side / 2), 0, width - side))
    top = int(np.clip(round(center[1] * height - side / 2), 0, height - side))
    crop = image.crop((left, top, left + side, top + side)).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return crop, side / ppm


def effective_ppm(camera: dict, image_size: tuple[int, int]) -> tuple[float, dict]:
    """Correct uniformly resized photos using calibrated camera dimensions.

    EXIF orientation can exchange width/height, so choose the matching aspect
    orientation first. A cropped, differently shaped image does not establish a
    resize ratio; report the unresolved calibration rather than silently call it
    a verified physical scale.
    """
    width, height = image_size
    native_width, native_height = float(camera["width"]), float(camera["height"])
    orientations = ((native_width, native_height), (native_height, native_width))
    nominal_w, nominal_h = min(orientations, key=lambda pair: abs(math.log((width / pair[0]) / (height / pair[1]))))
    scale_x, scale_y = width / nominal_w, height / nominal_h
    consistent = abs(math.log(scale_x / scale_y)) <= 0.03
    # For matching aspect, width establishes the calibrated resizing factor.
    # For an unmatched crop the supplied calibration remains unverified; this
    # flag is retained in the report for inspection of the actual Kaggle images.
    scaling = scale_x if consistent else 1.0
    value = float(camera["ppm"]) * scaling
    return value, {"resize_scale_x": scale_x, "resize_scale_y": scale_y,
                   "aspect_consistent_resize": consistent, "effective_ppm": value,
                   "scale_status": "resolution_corrected" if consistent else "aspect_mismatch_calibration_unverified"}


def texture_features(rgb: np.ndarray, actual_side_mm: float) -> np.ndarray:
    """Fixed color/contrast and Fourier bands in cycles/mm; no fitted bins."""
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = rgb @ np.array([0.299, 0.587, 0.114], np.float32)
    values = [*rgb.mean(axis=(0, 1)), *rgb.std(axis=(0, 1)),
              *np.quantile(gray, [0.1, 0.5, 0.9]), float(gray.std())]
    dx, dy = np.diff(gray, axis=1), np.diff(gray, axis=0)
    values += [float(np.abs(dx).mean()), float(np.abs(dy).mean()),
               float((dx**2).mean()), float((dy**2).mean())]
    window = np.outer(np.hanning(len(gray)), np.hanning(gray.shape[1]))
    spectrum = np.abs(np.fft.fft2((gray - gray.mean()) * window)) ** 2
    fy = np.fft.fftfreq(gray.shape[0], d=actual_side_mm / gray.shape[0])
    fx = np.fft.fftfreq(gray.shape[1], d=actual_side_mm / gray.shape[1])
    radial = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    bands = (0., 0.03125, 0.125, 0.5, 2., 8., np.inf)
    total = max(float(spectrum.sum()), 1e-12)
    values += [float(spectrum[(radial >= low) & (radial < high)].sum() / total)
               for low, high in zip(bands[:-1], bands[1:])]
    # Missing physical coverage is represented explicitly if a crop is clamped.
    values.append(float(np.log1p(actual_side_mm)))
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite image texture features.")
    return result


def cuda_setup() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Enable Kaggle CUDA GPU 0. This real experiment has no CPU fallback.")
    torch.cuda.set_device(0)
    torch.set_num_threads(4)
    torch.manual_seed(SEED); np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats(0)


def load_encoder(weights: Path):
    from torchvision.models import resnet18
    actual = hashlib.sha256(weights.read_bytes()).hexdigest()
    if actual != WEIGHT_SHA256:
        raise ValueError("Encoder file differs from the pinned official ImageNet ResNet18 weights.")
    encoder = resnet18(weights=None)
    encoder.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    encoder.fc = torch.nn.Identity()
    encoder = encoder.to(DEVICE).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


@torch.inference_mode()
def extract_features(encoder, ledger: list[dict]) -> tuple[np.ndarray, list[dict], list[tuple[str, int]]]:
    mean = torch.tensor([0.485, 0.456, 0.406], device=DEVICE)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=DEVICE)[None, :, None, None]
    rows, photo_report, blocks = [], [], None
    for index, photo in enumerate(ledger):
        with Image.open(photo["path"]) as original:
            image = ImageOps.exif_transpose(original).convert("RGB")
        ppm_value, scaling_report = effective_ppm(photo["camera"], image.size)
        if not scaling_report["aspect_consistent_resize"]:
            print(json.dumps({"calibration_warning": photo["path"].name, **scaling_report}), flush=True)
        views, textures, policies = [], [], []
        for side_mm in (*PHYSICAL_SIDES_MM, None):
            centers = CENTERS if side_mm is not None else ((0.5, 0.5),)
            for center in centers:
                crop, actual_mm = physical_crop(image, ppm_value, side_mm, center)
                rgb = np.asarray(crop)
                views.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.)
                textures.append(texture_features(rgb, actual_mm))
                policies.append("context" if side_mm is None else f"{side_mm:g}mm")
        embeddings = []
        for start in range(0, len(views), BATCH_SIZE):
            batch = torch.stack(views[start:start+BATCH_SIZE]).to(DEVICE)
            embeddings.append(encoder((batch - mean) / std).cpu().numpy().astype(np.float64))
        embeddings = np.concatenate(embeddings)
        textures = np.asarray(textures)
        feature, current_blocks = [], []
        for policy in ("16mm", "64mm", "context"):
            keep = np.asarray(policies) == policy
            feature.append(embeddings[keep].mean(axis=0)); current_blocks.append((f"cnn_{policy}", embeddings.shape[1]))
            feature.append(textures[keep].mean(axis=0)); current_blocks.append((f"texture_{policy}", textures.shape[1]))
        rows.append(np.concatenate(feature))
        if blocks is None:
            blocks = current_blocks
        elif blocks != current_blocks:
            raise ValueError("Feature layout changed between images.")
        photo_report.append({"sample_id": photo[ID], "filename": photo["path"].name,
                             "camera": photo["camera"]["phone"], "ppm": float(photo["camera"]["ppm"]),
                             "image_width": image.width, "image_height": image.height,
                             "nominal_width": int(photo["camera"]["width"]), "nominal_height": int(photo["camera"]["height"]),
                             "views": len(views), **scaling_report})
        if (index + 1) % 10 == 0 or index + 1 == len(ledger):
            print(json.dumps({"features_completed_photos": index + 1, "total": len(ledger)}), flush=True)
    return np.asarray(rows), photo_report, blocks


def aggregate_samples(features: np.ndarray, ledger: list[dict], sample_ids: list[str]) -> np.ndarray:
    """Equal camera weights per sample; photos never become validation units."""
    if len(features) != len(ledger) or not np.isfinite(features).all():
        raise ValueError("Feature rows must match the finite photo ledger.")
    samples = []
    for sample_id in sample_ids:
        cameras = sorted({canonical(row["camera"]["phone"]) for row in ledger if row[ID] == sample_id})
        if not cameras:
            raise ValueError(f"No feature rows for sample {sample_id}")
        by_camera = []
        for camera in cameras:
            rows = [i for i, row in enumerate(ledger) if row[ID] == sample_id and canonical(row["camera"]["phone"]) == camera]
            by_camera.append(features[rows].mean(axis=0))
        samples.append(np.asarray(by_camera).mean(axis=0))
    return np.asarray(samples)


def fit_scaler(x: torch.Tensor, blocks: list[tuple[str, int]]) -> dict:
    if x.ndim != 2 or len(x) < 2 or sum(length for _, length in blocks) != x.shape[1]:
        raise ValueError("Scaler needs >=2 fitting samples and a complete feature layout.")
    center = x.mean(dim=0)
    scale = x.std(dim=0, unbiased=False)
    # Constant fitting dimensions carry no estimated signal. Do not activate
    # them because a held-out/test camera happens to differ in that dimension.
    active = scale > 1e-8
    scale = scale.clamp_min(1e-8)
    weights, offset = torch.zeros_like(scale), 0
    for _, length in blocks:
        count = int(active[offset:offset+length].sum())
        weights[offset:offset+length] = 1.0 / math.sqrt(max(count, 1))
        offset += length
    return {"center": center, "scale": scale, "active": active, "weights": weights}


def apply_scaler(x: torch.Tensor, scaler: dict) -> torch.Tensor:
    z = ((x - scaler["center"]) / scaler["scale"]).clamp(-5., 5.)
    return z * scaler["active"] * scaler["weights"]


def kernel(a: torch.Tensor, b: torch.Tensor, method: str) -> torch.Tensor:
    if method == "linear":
        return a @ b.T / 6.0
    if method == "rbf":
        distance = (a.square().sum(1)[:, None] + b.square().sum(1)[None] - 2 * (a @ b.T)).clamp_min(0)
        return torch.exp(-distance / 12.0)
    raise ValueError("Unknown kernel.")


def fit_predict(x: torch.Tensor, y: torch.Tensor, query: torch.Tensor,
                candidate: Candidate, blocks: list[tuple[str, int]]) -> tuple[np.ndarray, dict]:
    if x.ndim != 2 or y.shape != (len(x), 11) or query.ndim != 2 or query.shape[1] != x.shape[1]:
        raise ValueError("Regression fitting/query shapes differ.")
    if candidate.method == "median":
        # Quantile implements the usual midpoint median for even sample counts.
        median = torch.quantile(y, 0.5, dim=0)
        result = median[None].repeat(len(query), 1)
        return project_cdfs(result.cpu().numpy()), {"method": candidate.name, "median_cdf": median}
    if candidate.alpha <= 0:
        raise ValueError("Ridge regularization must be positive.")
    scaler = fit_scaler(x, blocks)
    a, b = apply_scaler(x, scaler), apply_scaler(query, scaler)
    center_y = y.mean(dim=0)
    gram = kernel(a, a, candidate.method)
    coefficients = torch.linalg.solve(gram + candidate.alpha * torch.eye(len(a), device=x.device, dtype=x.dtype), y - center_y)
    result = kernel(b, a, candidate.method) @ coefficients + center_y
    state = {"candidate": candidate.name, "center": scaler["center"], "scale": scaler["scale"],
             "active": scaler["active"], "weights": scaler["weights"], "fitting_scaled_features": a,
             "target_center": center_y, "dual_coefficients": coefficients}
    return project_cdfs(result.cpu().numpy()), state


def loocv_candidates(x: torch.Tensor, y: torch.Tensor, blocks, candidates=CANDIDATES) -> tuple[list[dict], dict[str, np.ndarray]]:
    if len(x) < 4:
        raise ValueError("Inner sample LOOCV requires at least four fitting samples.")
    predictions = {candidate.name: np.zeros((len(x), 11), float) for candidate in candidates}
    for held in range(len(x)):
        keep = torch.arange(len(x), device=x.device) != held
        for candidate in candidates:
            value, _ = fit_predict(x[keep], y[keep], x[held:held+1], candidate, blocks)
            predictions[candidate.name][held] = value[0]
    truth = y.cpu().numpy()
    scores = [{"candidate": candidate.name, "mean_emd": float(emd_rows(truth, predictions[candidate.name]).mean())}
              for candidate in candidates]
    return scores, predictions


def select_candidate(scores: list[dict], candidates=CANDIDATES) -> Candidate:
    indices = {candidate.name: i for i, candidate in enumerate(candidates)}
    best = min(scores, key=lambda row: (row["mean_emd"], indices[row["candidate"]]))
    return next(candidate for candidate in candidates if candidate.name == best["candidate"])


def nested_loocv(x: torch.Tensor, y: torch.Tensor, sample_ids: list[str], blocks, candidates=CANDIDATES) -> dict:
    if len(x) != len(sample_ids) or len(set(sample_ids)) != len(sample_ids) or len(x) < 5:
        raise ValueError("Outer LOOCV needs >=5 distinct soil samples.")
    outer = np.zeros((len(x), 11)); median = np.zeros_like(outer); choices = []
    for held, sample_id in enumerate(sample_ids):
        keep = torch.arange(len(x), device=x.device) != held
        fitting_ids = [value for i, value in enumerate(sample_ids) if i != held]
        inner_scores, _ = loocv_candidates(x[keep], y[keep], blocks, candidates)
        chosen = select_candidate(inner_scores, candidates)
        outer[held] = fit_predict(x[keep], y[keep], x[held:held+1], chosen, blocks)[0][0]
        median[held] = fit_predict(x[keep], y[keep], x[held:held+1], Candidate("median"), blocks)[0][0]
        choices.append({"held_out_sample_id": sample_id, "fitting_sample_ids": fitting_ids,
                        "selected_candidate": chosen.name, "inner_loocv_scores": inner_scores})
        print(json.dumps({"outer_sample": sample_id, "completed": held + 1, "total": len(x), "choice": chosen.name}), flush=True)
    truth = y.cpu().numpy()
    return {"nested_predictions": outer, "median_predictions": median,
            "nested_emd_rows": emd_rows(truth, outer), "median_emd_rows": emd_rows(truth, median),
            "outer_choices": choices}


def make_submission(sample: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    validate_ids(sample)
    if list(sample.columns) != [ID] + COLUMNS or predictions.shape != (len(sample), 11):
        raise ValueError("Submission schema or row count mismatch.")
    validate_cdfs(predictions)
    result = pd.DataFrame(predictions, columns=COLUMNS)
    result.insert(0, ID, sample[ID].astype(str).to_numpy())
    if list(result.columns) != list(sample.columns) or not result[ID].equals(sample[ID].reset_index(drop=True)):
        raise ValueError("Submission failed exact sample ID/column ordering.")
    return result


def run_experiment(data: Path, weights: Path, output: Path) -> dict:
    started = time.perf_counter()
    cuda_setup(); output.mkdir(parents=True, exist_ok=True)
    labels, sample, ppm = load_metadata(data)
    train_ledger = photo_ledger(data, labels[ID].tolist(), ppm, "train")
    test_ledger = photo_ledger(data, sample[ID].tolist(), ppm, "test")
    encoder = load_encoder(weights)
    train_photos, train_report, blocks = extract_features(encoder, train_ledger)
    test_photos, test_report, test_blocks = extract_features(encoder, test_ledger)
    if blocks != test_blocks:
        raise ValueError("Fitting and test feature layouts differ.")
    train_features = aggregate_samples(train_photos, train_ledger, labels[ID].tolist())
    test_features = aggregate_samples(test_photos, test_ledger, sample[ID].tolist())
    del encoder; torch.cuda.empty_cache()
    x = torch.tensor(train_features, dtype=torch.float64, device=DEVICE)
    y = torch.tensor(labels[COLUMNS].to_numpy(float), dtype=torch.float64, device=DEVICE)
    query = torch.tensor(test_features, dtype=torch.float64, device=DEVICE)
    nested = nested_loocv(x, y, labels[ID].tolist(), blocks)
    # Full-fitting LOOCV picks the final model. Outer aggregate scores never
    # decide the candidate family/alpha or select a different submitted policy.
    selection_scores, _ = loocv_candidates(x, y, blocks)
    final_candidate = select_candidate(selection_scores)
    predictions, model_state = fit_predict(x, y, query, final_candidate, blocks)
    torch.save({"state": model_state, "candidate": asdict(final_candidate), "blocks": blocks,
                "fitting_ids": labels[ID].tolist(), "encoder_sha256": WEIGHT_SHA256}, output / "regressor.pt")
    make_submission(sample, predictions).to_csv(output / "submission.csv", index=False, float_format="%.8f")
    for name, values in (("nested_oof", nested["nested_predictions"]), ("median_oof", nested["median_predictions"])):
        pd.DataFrame({ID: labels[ID], **{column: values[:, j] for j, column in enumerate(COLUMNS)}}).to_csv(output / f"{name}.csv", index=False)
    torch.cuda.synchronize(0)
    report = {"method": "frozen physical-scale ImageNet image features and nested sample-LOOCV GPU kernel ridge",
              "device": DEVICE, "gpu_name": torch.cuda.get_device_name(0), "seed": SEED,
              "train_samples": len(labels), "test_samples": len(sample), "train_photos": len(train_ledger), "test_photos": len(test_ledger),
              "feature_dimension": x.shape[1], "feature_blocks": blocks, "physical_sides_mm": PHYSICAL_SIDES_MM,
              "views_per_photo": 11, "camera_ppm_policy": "supplied PPM corrected by actual/nominal width when oriented aspect ratios agree; unmatched crops flagged unverified",
              "encoder_weights_sha256": WEIGHT_SHA256, "encoder_finetuned": False,
              "nested_sample_loocv_mean_emd": float(nested["nested_emd_rows"].mean()),
              "sample_loocv_training_median_mean_emd": float(nested["median_emd_rows"].mean()),
              "final_candidate": final_candidate.name, "final_selection_loocv_scores": selection_scores,
              "outer_choices": nested["outer_choices"], "elapsed_seconds": time.perf_counter() - started,
              "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(0),
              "metric": "mean over samples of sum_i=0..9 abs(CDF_i-pred_i)*log10(diameter_i+1/diameter_i)",
              "validation_note": "Outer soil samples do not enter inner fitting, scaling, or candidate choice. Candidate policy fixed before this run. These are local validation scores, not leaderboard scores.",
              "limitations": "Only 24 labeled samples; test iPhone collection differs from Motorola/Samsung training. Sample LOOCV does not estimate that camera/collection shift. Cropped images with mismatching calibration aspect are flagged; their physical scale is unverified."}
    (output / "experiment_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "photo_metadata.json").write_text(json.dumps({"train": train_report, "test": test_report}, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "outer_choices"}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args()
    run_experiment(args.data, args.weights, args.output)


if __name__ == "__main__":
    main()
