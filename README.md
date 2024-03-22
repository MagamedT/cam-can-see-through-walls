# Overview

This is the Python code for the experiments of the submission **CAM-Based Methods Can See through Walls**

# Installation

Python version should be >= 3.9
The two main dependencies are: PyTorch 2.1 and DALI (1.35.0 -c420f32).
The imagenet1k dataset can be downloaded from [Huggingface](https://huggingface.co/datasets/imagenet-1k).

# Script Usage

mieux detailler et dire comment use les script, genre dl imagenet dans le data_root_dir

In the training script (vgg16_training.py) and preprocessing script (preprocess_imagenet1k_huggingface), there is a global path variable named **data_root_dir** that you need to fill with your Imagenet1-k dataset location. 
