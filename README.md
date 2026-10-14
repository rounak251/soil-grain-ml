# Soil grain size distributions from images

A small-data image-regression project using physical-scale image views, frozen ResNet18 features, nested whole-sample validation, and valid cumulative grain-size predictions. The original project code was developed with AI assistance for [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos).

The updated training input has **24 labeled soil samples and 127 photographs**; the test input has **10 samples and 35 photographs**. All feature extraction, fitted regression, and real competition evaluation ran in a **private Kaggle GPU notebook on Tesla T4**. Synthetic checks run locally; aggregate validation metrics were also independently recomputed from private saved predictions without refitting a model.

## Verified baseline

Results recorded on **14 October 2026**. EMD is lower-is-better.

| Evaluation | Mean EMD | Meaning |
| --- | ---: | --- |
| Nested leave-one-soil-sample-out validation | **44.13481663** | Outer held-out samples are excluded from inner model selection and fitted preprocessing. |
| Training-median baseline on the same outer samples | **97.24831519** | A constant CDF fitted separately to each outer fitting set. |
| Final candidate's ordinary LOOCV selection score | 42.50983538 | Used to select the final model; this is not independent evaluation. |
| Official Kaggle public leaderboard | **66.01225** | Completed submission; the final private score is not available. |

The final selection was **linear kernel ridge, regularization 1**, with the ResNet18 encoder frozen. Script runtime was **57.97548 seconds**, excluding notebook startup and scoring overhead. [Aggregate execution record](evidence/baseline.json) preserves device, policy, source hashes, and submission reference without sample rows, targets, or model arrays.

Twenty-four labels make validation uncertain. Training images use Motorola/Samsung cameras, whereas test images use iPhone 14/16 from a different collection. The public score exceeds nested validation by **21.87743 EMD**; the available evidence does not establish the cause. Sample-level validation does not measure the full camera/site shift. This is a measured baseline and engineering portfolio, with no winning or commercial-performance claim.

## Engineering contribution

- **Physical calibration:** correct supplied pixels-per-millimeter for uniform resizing after accounting for image orientation; flag incompatible aspect ratios. Keep clamped crops' actual physical coverage as a feature.
- **Multiple image views:** five 16 mm crops, five 64 mm crops, and one context view per photo, each resized to 224 pixels. Combine frozen CNN embeddings with fixed color, contrast, and Fourier features in physical frequency units.
- **Camera-balanced pooling:** average views, then photos within each camera, then cameras within each soil sample. A camera with more photos does not receive more sample weight.
- **Nested sample validation:** keep every photo and camera of a held-out soil sample together. Inner sample LOOCV chooses the median baseline or linear/RBF ridge with alpha 0.1, 1, or 10. Scaling, constant-feature detection, target centering, and fitting use only that fold's fitting samples.
- **Valid CDF output:** weighted pool-adjacent-violators projection, clipping to [0, 100], and a fixed endpoint of 100. Check exact sample IDs, row order, columns, finite values, and monotonicity.

The feature extraction, matching, calibration, regression policy, projection, and packaging are original project code. The ResNet architecture and pretrained representation come from torchvision and its upstream training resources; the encoder was not trained from scratch here.

## Run synthetic checks

Use Python 3.11 or newer with a compatible PyTorch/torchvision installation. Synthetic tests run on CPU and need no competition input or pretrained file.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Eighteen tests exercise the official left-endpoint metric, its 500-point maximum, weighted monotonic projection, identity matching, camera pooling, calibrated resizing, strict submission schema, fold-only scaling, held-out-target isolation, exact notebook source embedding, and private/offline GPU metadata. The GPU experiment fails explicitly if CUDA is unavailable. Dependency ranges support setup; results across different library versions are not promised to be bitwise identical.

## Reproduce on a private Kaggle GPU notebook

1. Obtain the competition inputs through your own Kaggle account under the [official rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules). Attach the updated competition dataset.
2. Obtain the official [ResNet18 ImageNet V1 weight file](https://download.pytorch.org/models/resnet18-f37072fd.pth). Attach it as a private Kaggle input that you control. The code verifies its complete SHA-256 before loading it; the weight file is not distributed in this repository.
3. Build a self-contained notebook and private GPU metadata using your notebook ID and weight-input ID:

```bash
python build_notebook.py \
  --kernel-id YOUR_USERNAME/soil-grain-ml-private \
  --weights-dataset YOUR_USERNAME/YOUR_PRIVATE_WEIGHTS_INPUT
```

4. Inspect the generated files in `notebook/`. They keep **GPU enabled, internet disabled, and private visibility**. Run the notebook in Kaggle after attaching both inputs. It requires CUDA GPU 0 and performs no dependency installations or weight downloads.
5. Inspect the output validation and aggregate report. Keep generated per-sample CSVs, calibration records, and fitted arrays in private storage. Submit through your own Kaggle account only after checking the CSV.

You can upload the generated notebook through the Kaggle interface, or use an authenticated local Kaggle CLI with `kaggle kernels push -p notebook`. Authentication credentials stay in your local account setup; they are not notebook inputs.

The expected input files are `Training_labels_updated.csv`, `ppm_updated.csv`, and `sample_submission.csv`, plus the `Training-All_Photos_updated` and `Test_All_Photos` image directories. Nested image folders are supported. The CLI on a Kaggle GPU environment is:

```bash
python soil_model.py \
  --data /path/to/attached/competition \
  --weights /path/to/attached/resnet18-f37072fd.pth \
  --output /kaggle/working
```

## Metric and interpretation

For each sample, sum the absolute CDF errors at the first ten diameter **left endpoints**, weighted by the difference between consecutive base-10 log diameters. Then average over samples. Values are percentages; diameter endpoints are 0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2, 6.3, 20, 63, and 200 mm. The last endpoint must equal 100 for output validity, although it is outside the ten error terms. [Official evaluation](https://www.kaggle.com/competitions/soil-grain-size-from-photos/overview/evaluation).

PAVA is a least-squares monotonic projection; it does not exactly optimize this absolute-error metric. The outer score evaluates the fixed inner-selection procedure. The final full-training LOOCV score chooses the submission model and has a different role.

## Code, data, and pretrained terms

Original project code is under the [MIT license](LICENSE). Dependencies retain their own licenses. torchvision's [BSD-3-Clause software license](https://github.com/pytorch/vision/blob/main/LICENSE) is distinct from its [pretrained-model terms notice](https://github.com/pytorch/vision#pre-trained-model-license). No unrestricted commercial license for ImageNet weights is asserted. The soil adapter keeps physical crops and uses the encoder's RGB mean/std; this differs from the generic classification resize/crop preset. [Official ResNet18 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

Competition data is supplied separately under its rules, which permit competition, noncommercial research, and educational use. This source release does not redistribute photographs, target labels, sample-level predictions, fitted weights, or private run logs. Any use with client data requires separately authorized inputs and an appropriate pretrained-model license. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Project map

```text
soil_model.py                 GPU feature extraction, fitting, evaluation, output
build_notebook.py             Configurable private/offline Kaggle packaging
requirements.txt              Dependency ranges
tests/                        Synthetic model and notebook contract checks
evidence/baseline.json         Aggregate verified result and provenance
SOURCE_MANIFEST.json           Release file hashes
THIRD_PARTY.md                 Data/software/pretrained provenance
```
