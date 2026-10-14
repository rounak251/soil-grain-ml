# Sources and license boundaries

## Original project code

The soil adapter and selection/evaluation workflow are original, AI-assisted project work. `soil_model.py` preserves the source of the verified ResNet18 baseline. Synthetic test identities and calibration values in this release are invented fixtures. The [MIT license](LICENSE) covers this original code and documentation.

## torchvision and upstream model

The code imports torchvision's ResNet18 implementation rather than copying that implementation. torchvision software is distributed under its [BSD-3-Clause license](https://github.com/pytorch/vision/blob/main/LICENSE).

The baseline consumes the official `ResNet18_Weights.IMAGENET1K_V1` state dictionary, supplied separately by the user:

- Documentation: [ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).
- Original download: [resnet18-f37072fd.pth](https://download.pytorch.org/models/resnet18-f37072fd.pth).
- SHA-256: `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`.
- Architecture paper: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).

The pretrained file and ImageNet data are not bundled. Software licensing does not establish the rights for every pretrained-model use. Consult torchvision's [pretrained-model terms notice](https://github.com/pytorch/vision#pre-trained-model-license) and the relevant upstream data terms for the intended use.

## Competition inputs and results

Competition: [Predicting Soil Grain Size Distributions from Images](https://www.kaggle.com/competitions/soil-grain-size-from-photos), sponsored by BOKU's Institute of Geotechnical Engineering. Obtain inputs under the [competition rules](https://www.kaggle.com/competitions/soil-grain-size-from-photos/rules); their permitted uses include competition and noncommercial academic/educational work.

The public release includes original processing code and aggregate measurements. Competition photos, label rows, sample-level predictions, fitted feature/model arrays, and account credentials are not included. `evidence/baseline.json` summarizes private Kaggle execution evidence and a scored submission. It is not a public mirror of the private notebook's data or logs, and it does not make all underlying artifacts publicly inspectable.

Other imported libraries—PyTorch, NumPy, pandas, and Pillow—retain their own licenses. Install them from their official distributions; they are not vendored in this repository.
