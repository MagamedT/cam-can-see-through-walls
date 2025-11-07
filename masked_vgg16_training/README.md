# Masked VGG16 Training

This directory holds the preprocessing and training scripts for the masked and baseline VGG16-style models.

## Files

- `vgg16_training.py` — trains the custom VGG16 with a optional mask.
- `preprocess_imagenet1k_huggingface.py` — prepares Imagenet-1k downloaded from Hugging Face.
- `LOC_synset_mapping.txt` — mapping from synset IDs to human-readable labels.

## Requirements

- Python ≥ 3.9
- PyTorch 2.1
- NVIDIA CUDA (optional but recommended)
- (Optional) NVIDIA DALI 1.35.0 for accelerated data loading
- Imagenet-1k dataset from Hugging Face (`datasets/imagenet-1k`)

Install dependencies via the environment created in the root README or with:

```bash
pip install torch torchvision pytorch-grad-cam python-dali
```

## Prepare Imagenet-1k

1. Download and extract the Hugging Face Imagenet-1k dataset.
2. Place `LOC_synset_mapping.txt` inside the extracted dataset directory (or update paths accordingly).
3. Edit `preprocess_imagenet1k_huggingface.py`:
   ```python
   data_root_dir = "/absolute/path/to/imagenet-1k"
   output_dir = "/absolute/path/to/preprocessed-imagenet"
   ```
4. Run the preprocessing:
   ```bash
   python preprocess_imagenet1k_huggingface.py
   ```

The script will create a curated subset tailored for training the masked model.

## Train the Model

1. Open `vgg16_training.py` and configure:
   ```python
   data_root_dir = "/absolute/path/to/preprocessed-imagenet"
   save_dir = "/absolute/path/to/checkpoints"
   masked = True  # set False for baseline
   ```
   Additional hyperparameters (learning rate, epochs, optimizer settings) can be tweaked inside the script.

2. Launch training:
   ```bash
   python vgg16_training.py
   ```

3. After training completes, copy the resulting checkpoints to `figure_code/data/` so the figure notebook can pick them up.

## Troubleshooting
- **Memory errors**: reduce batch size, and if using DALI, ensure adequate GPU memory.

