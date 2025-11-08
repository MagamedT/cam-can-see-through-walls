# Figure Generation Code

This folder contains the notebooks and helper scripts used to reproduce the figures and qualitative analyses from the paper.

## Contents

- `heatmap.py` — common utilities for loading datasets, models, and computing CAM heatmaps.
- `camMaps_on_stackmix_and_stackgen.ipynb` — builds several figures and Table 1 using STACK-MIX and STACK-GEN images.
- `mnist_experiment.ipynb` — sanity-checks the masking mechanism on MNIST and illustration of the Theorem 1.
- `data/` — expected location for processed datasets.
- `model/` — expected location for pre-trained VGG checkpoints.

## Prerequisites

1. Follow the root README to create a Python environment and install dependencies (`torch`, `torchvision`, `pytorch-grad-cam`, etc.).
2. Unzip `datasets.zip` to `figure_code/data/`.
3. Ensure `params_vgg16_baseline.pt` and `params_vgg16_masked.pt` also reside in `figure_code/model/`.

Resulting structure:
```
figure_code/
├── cam_on_masked_cnn.py
├── camMaps_on_cutemixAndGen.ipynb
├── mnist_experiment.ipynb
└── data/
    ├── stackgen/
    ├── stackmix/
└── model/
    ├── params_vgg16_baseline.pt
    └── params_vgg16_masked.pt
```
