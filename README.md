# CAM-Based Methods Can See Through Walls

This repository contains the ompanion code for
[CAM-Based Methods Can See Through Walls](https://arxiv.org/abs/2404.01964).
It includes:

- the two proposed image datasets STACK-MIX and STACK-GEN;
- two VGG16 networks, one of them unable to see the bottom of the input image;
- the code to reproduce our experiments.

The notebook reproducing Figure 1 of the paper is [here](https://github.com/MagamedT/cam-can-see-through-walls/blob/main/figure_code/camMaps_on_stackmix_and_stackgen.ipynb).

## Quick Start

1. **Clone and enter the repo**

   ```bash
   git clone https://github.com/MagamedT/cam-can-see-through-walls
   cd cam-can-see-through-walls
   ```

2. **Install the minimal requirements**

   - PyTorch 2.1 (CUDA if available)
   - `pytorch-grad-cam`
   - `torchvision`
   - `matplotlib`, `numpy`, `Pillow`
   Optional: NVIDIA DALI 1.35.0 (only required for re-training with Imagenet-1k).
   If you are on a different CUDA toolkit, swap the nvidia-dali-cuda120 line for the build that matches your driver (e.g., nvidia-dali-cuda110).
Alternativaly: create a Python environment (Python ≥ 3.9):

   ```bash
   conda create -n cam-walls python=3.11 -y
   conda activate cam-walls
   ```
   then 
   ```bash
   pip install -r requirements.txt
   ```

3. **Get the datasets and the models**

   The datasets and model weights can be found in [this drive](https://drive.google.com/drive/folders/19LbSZcABEv3E0KB4KzlYQebjo8qDxr1S?usp=sharing).
   Unzip the data into `figure_code/data/`
   ```bash
   unzip datasets.zip -d figure_code/data
   ```
   Ensure the following files are in `figure_code/data/` and `figure_code/model/`:
   ```
   figure_code/data/
   ├── stackmix/                      # STACK-MIX images (imagenet data)
   ├── stackgen/                  # STACK-GEN images (generative model)
   figure_code/model/
   ├── params_vgg16_baseline.pt      # pre-trained baseline VGG checkpoint
   └── params_vgg16_masked.pt        # pre-trained masked VGG checkpoint
   ```

4. **Reproduce the results**

   - `figure_code/mnist_experiment.ipynb` replicates the MNIST experiment
   - `figure_code/camMaps_on_stackmix_and_stackgen.ipynb` generates Figure 1 and Table 1

Note that GPU use is optional; the scripts automatically map checkpoints to CPU when CUDA is unavailable.

## Re-training the Network

To reproduce training:
1. Download Imagenet-1k (train/validation splits) from [Hugging Face](https://huggingface.co/datasets/imagenet-1k).
2. Update `data_root_dir` in `masked_vgg16_training/preprocess_imagenet1k_huggingface.py` and run it to create the curated subset.
3. Configure `vgg16_training.py`:
   - Set `data_root_dir` to the preprocessed Imagenet directory.

## Citation

If you use this code, please use the following to cite our work:

```bibtex
@inproceedings{taimeskhanov2024cam,
  title={CAM-Based Methods Can See Through Walls},
  author={Taimeskhanov, Magamed and Sicre, Ronan and Garreau, Damien},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={332--348},
  year={2024},
  organization={Springer}
}
```
