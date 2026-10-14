"""Synthetic checks only; no competition image extraction or model evaluation."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

import soil_model as soil


def cameras():
    # Invented calibration values and sample identities, not competition rows.
    return pd.DataFrame([
        {"phone": "Motorola Edge", "camera": "motorola edge 20", "width": 4000, "height": 1800, "ppm": 10.0},
        {"phone": "Motorola Edge 60 Fusion", "camera": "Motorola Edge 60 fusion", "width": 4096, "height": 2304, "ppm": 12.0},
        {"phone": "iPhone 16", "camera": "iPhone 16", "width": 5712, "height": 4284, "ppm": 20.0},
    ])


class MetricAndCDFTests(unittest.TestCase):
    def test_metric_uses_left_endpoints_and_ignores_last_error(self):
        truth = np.zeros((1, 11))
        for j in range(10):
            prediction = truth.copy(); prediction[0, j] = 17
            self.assertAlmostEqual(soil.emd_rows(truth, prediction)[0], 17 * soil.INTERVAL_WEIGHTS[j])
        last = truth.copy(); last[0, -1] = 100
        self.assertEqual(soil.emd_rows(truth, last)[0], 0)

    def test_metric_maximum_is_500_for_valid_cdfs(self):
        low = np.zeros((1, 11)); low[:, -1] = 100
        high = np.full((1, 11), 100.)
        soil.validate_cdfs(low); soil.validate_cdfs(high)
        self.assertAlmostEqual(soil.emd_rows(low, high)[0], 500.)
        self.assertAlmostEqual(soil.emd_rows(high, low)[0], 500.)

    def test_weighted_pava_known_solution(self):
        np.testing.assert_allclose(soil.pava([10, 0, 20], [1, 3, 1]), [2.5, 2.5, 20])
        np.testing.assert_allclose(soil.pava([3, 2, 1]), [2, 2, 2])

    def test_projection_enforces_bounds_order_and_endpoint(self):
        raw = np.array([[-5, 8, 4, 120, 10, 99, 103, 105, 106, 110, -100]], float)
        projected = soil.project_cdfs(raw)
        soil.validate_cdfs(projected)
        self.assertEqual(projected[0, -1], 100.)
        self.assertTrue((np.diff(projected) >= 0).all())
        with self.assertRaises(ValueError):
            soil.project_cdfs(np.full((1, 11), np.nan))

    def test_exact_submission_schema_and_id_order(self):
        sample = pd.DataFrame({soil.ID: ["with space", "Soil_Muenster_2_5_depth"],
                               **{c: [0., 0.] for c in soil.COLUMNS}})
        predictions = np.tile(np.linspace(0, 100, 11), (2, 1))
        frame = soil.make_submission(sample, predictions)
        self.assertEqual(list(frame.columns), [soil.ID] + soil.COLUMNS)
        self.assertEqual(frame[soil.ID].tolist(), sample[soil.ID].tolist())
        bad = predictions.copy(); bad[:, -1] = 99.
        with self.assertRaises(ValueError):
            soil.make_submission(sample, bad)
        with self.assertRaises(ValueError):
            soil.make_submission(sample.rename(columns={"2": "2.0"}), predictions)


class PhotoIdentityAndScaleTests(unittest.TestCase):
    def test_longer_camera_prefix_and_training_photo_suffix(self):
        uid, camera = soil.match_photo(Path("Motorola_Edge_60_fusion_SAMPLE42_01.jpg"), ["SAMPLE42"], cameras())
        self.assertEqual(uid, "SAMPLE42")
        self.assertEqual(camera["phone"], "Motorola Edge 60 Fusion")
        uid, camera = soil.match_photo(Path("Motorola_Edge_SAMPLE42_01.jpg"), ["SAMPLE42"], cameras())
        self.assertEqual(camera["phone"], "Motorola Edge")

    def test_german_decimal_and_iphone_underscore_aliases(self):
        ids = ["Soil_Muenster_2_5_depth", "Soil_DemoField_B"]
        uid, _ = soil.match_photo(Path("iPhone_16_Soil_Münster_2,5_depth (5).JPG"), ids, cameras())
        self.assertEqual(uid, ids[0])
        uid, _ = soil.match_photo(Path("iPhone16_Soil_DemoField_B.JPG"), ids, cameras())
        self.assertEqual(uid, ids[1])
        with self.assertRaises(ValueError):
            soil.match_photo(Path("iPhone16_Soil_unknown (1).JPG"), ids, cameras())

    def test_ambiguous_sample_normalization_is_rejected(self):
        with self.assertRaises(ValueError):
            soil.validate_ids(pd.DataFrame({soil.ID: ["A-B", "AB"]}))

    def test_camera_balanced_sample_pooling_not_photo_level(self):
        ledger = [{soil.ID: "sample", "camera": {"phone": "a"}},
                  {soil.ID: "sample", "camera": {"phone": "a"}},
                  {soil.ID: "sample", "camera": {"phone": "b"}}]
        features = np.array([[0., 2.], [2., 4.], [9., 11.]])
        first = soil.aggregate_samples(features, ledger, ["sample"])
        np.testing.assert_array_equal(first, [[5., 7.]])
        enlarged = ledger + [{soil.ID: "sample", "camera": {"phone": "a"}}] * 4
        repeated = np.r_[features, np.tile([[1., 3.]], (4, 1))]
        np.testing.assert_array_equal(soil.aggregate_samples(repeated, enlarged, ["sample"]), first)

    def test_resize_scale_corrects_ppm_and_exif_dimensions(self):
        camera = {"width": 4000, "height": 1800, "ppm": 12.}
        value, report = soil.effective_ppm(camera, (1000, 450))
        self.assertEqual(value, 3.)
        self.assertTrue(report["aspect_consistent_resize"])
        rotated, report = soil.effective_ppm(camera, (450, 1000))
        self.assertEqual(rotated, 3.)
        self.assertTrue(report["aspect_consistent_resize"])
        _, report = soil.effective_ppm(camera, (1000, 1000))
        self.assertFalse(report["aspect_consistent_resize"])

    def test_physical_crop_retains_explicit_scale_when_clamped(self):
        image = Image.new("RGB", (100, 80), "white")
        crop, actual_mm = soil.physical_crop(image, ppm=10., side_mm=64., center=(.5, .5))
        self.assertEqual(crop.size, (224, 224))
        self.assertAlmostEqual(actual_mm, 7.2)
        texture = soil.texture_features(np.asarray(crop), actual_mm)
        self.assertTrue(np.isfinite(texture).all())


class FoldIsolationTests(unittest.TestCase):
    def test_scaler_fits_only_fitting_rows_and_zeroes_constant_dimensions(self):
        fit = torch.tensor([[1., 7.], [2., 7.], [3., 7.]], dtype=torch.float64)
        state = soil.fit_scaler(fit, [("features", 2)])
        before = {key: value.clone() for key, value in state.items()}
        transformed = soil.apply_scaler(torch.tensor([[1e9, -1e9]], dtype=torch.float64), state)
        for key in state:
            torch.testing.assert_close(before[key], state[key], rtol=0, atol=0)
        self.assertEqual(transformed[0, 1], 0.)

    def test_outer_prediction_cannot_depend_on_its_held_out_target(self):
        x = torch.tensor([[i, i*i, i+1, (i % 2), i*.5, i*.1] for i in range(6)], dtype=torch.float64)
        y = torch.tensor([np.r_[np.linspace(2, 80, 10) + i, 100.] for i in range(6)], dtype=torch.float64)
        blocks = [("cnn", 3), ("texture", 3)]
        candidates = (soil.Candidate("median"), soil.Candidate("linear", 1.))
        first = soil.nested_loocv(x, y, [f"sample-{i}" for i in range(6)], blocks, candidates)
        altered = y.clone(); altered[0, :10] = torch.clamp(altered[0, :10] + 19, max=99.)
        second = soil.nested_loocv(x, altered, [f"sample-{i}" for i in range(6)], blocks, candidates)
        np.testing.assert_array_equal(first["nested_predictions"][0], second["nested_predictions"][0])
        self.assertEqual(first["outer_choices"][0], second["outer_choices"][0])
        for choice in first["outer_choices"]:
            self.assertNotIn(choice["held_out_sample_id"], choice["fitting_sample_ids"])

    def test_even_count_baseline_uses_midpoint_median(self):
        x = torch.zeros(2, 1, dtype=torch.float64)
        y = torch.tensor([np.r_[np.zeros(10), 100.], np.full(11, 100.)], dtype=torch.float64)
        result, _ = soil.fit_predict(x, y, x[:1], soil.Candidate("median"), [("features", 1)])
        np.testing.assert_array_equal(result, [np.r_[np.full(10, 50.), 100.]])

    def test_real_experiment_has_no_cpu_fallback(self):
        with patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "no CPU fallback"):
                soil.run_experiment(Path("unused"), Path("unused"), Path("unused"))


if __name__ == "__main__":
    unittest.main()
