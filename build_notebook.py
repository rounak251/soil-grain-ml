"""Build an original self-contained private/offline Kaggle GPU notebook.

Supply your own notebook and private pretrained-weight input IDs. This helper
writes local files only; it performs no uploads, account access, or downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def cell(kind: str, text: str) -> dict:
    value = {"cell_type": kind, "id": hashlib.sha256(text.encode()).hexdigest()[:12],
             "metadata": {}, "source": text.splitlines(keepends=True)}
    if kind == "code":
        value.update({"execution_count": None, "outputs": []})
    return value


def build(kernel_id: str, weights_dataset: str, destination: Path) -> dict:
    pattern = r"[A-Za-z0-9_-]+/[A-Za-z0-9_-]+"
    for name, identifier in (("kernel ID", kernel_id), ("weight dataset ID", weights_dataset)):
        if re.fullmatch(pattern, identifier) is None:
            raise ValueError(f"Use an owner/slug for the {name}.")
    root = Path(__file__).resolve().parent
    source = (root / "soil_model.py").read_text()
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "source_sha256": source_hash,
        },
        "cells": [
            cell("markdown", "# Soil grain size: physical views and nested sample validation\n\n"
                 "Original AI-assisted ResNet18 baseline. Frozen official ImageNet features and fitted "
                 "kernel regression use CUDA GPU 0. All photos and cameras of a sample stay together. "
                 "Inner fitting-sample LOOCV chooses the fixed candidate; learned preprocessing uses "
                 "fitting samples only. Outputs are bounded monotonic CDFs ending at 100.\n\n"
                 "Run privately with supplied competition data and official pinned weights. Internet "
                 "is disabled and no downloads occur. Keep generated labels, per-sample predictions, "
                 "calibration rows, and fitted arrays private under their applicable terms.\n"),
            cell("code", "from pathlib import Path\n" +
                 f"Path('/kaggle/working/soil_model.py').write_text({source!r})\n"),
            cell("code", "from pathlib import Path\nimport hashlib\nimport sys\n"
                 "sys.path.insert(0, '/kaggle/working')\n"
                 "from soil_model import run_experiment, WEIGHT_SHA256\n"
                 "matches = sorted(Path('/kaggle/input').rglob('Training_labels_updated.csv'))\n"
                 "if len(matches) != 1:\n"
                 "    raise ValueError('Attach exactly one updated soil competition input')\n"
                 "weights = [p for p in sorted(Path('/kaggle/input').rglob('*.pth'))\n"
                 "           if 'resnet18' in p.name.lower()\n"
                 "           and hashlib.sha256(p.read_bytes()).hexdigest() == WEIGHT_SHA256]\n"
                 "if len(weights) != 1:\n"
                 "    raise ValueError('Attach exactly one pinned official ResNet18 weight file')\n"
                 "report = run_experiment(matches[0].parent, weights[0], Path('/kaggle/working'))\n"
                 "print({'submission_validated': True, 'device': report['device'],\n"
                 "       'nested_mean_emd': report['nested_sample_loocv_mean_emd']})\n"),
        ],
    }
    destination.mkdir(parents=True, exist_ok=True)
    notebook_path = destination / "soil-grain-ml.ipynb"
    notebook_path.write_text(json.dumps(notebook, indent=2) + "\n")
    metadata = {
        "id": kernel_id, "title": "Soil grain ML: nested sample validation",
        "code_file": notebook_path.name, "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "competition_sources": ["soil-grain-size-from-photos"],
        "dataset_sources": [weights_dataset], "kernel_sources": [],
    }
    (destination / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return {"source_sha256": source_hash, "notebook": str(notebook_path),
            "kernel_id": kernel_id, "private_gpu_offline": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-id", required=True)
    parser.add_argument("--weights-dataset", required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "notebook")
    args = parser.parse_args()
    print(json.dumps(build(args.kernel_id, args.weights_dataset, args.output)))


if __name__ == "__main__":
    main()
