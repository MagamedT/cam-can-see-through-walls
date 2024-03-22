# Overview

This is the official repository for the reproduction of experiments in **CAM-Based Methods Can See through Walls**

# Installations

The two main dependencies are: PyTorch 2.1 and DALI (1.35.0 -c420f32).
The imagenet1k dataset is downloaded from [Huggingface](https://huggingface.co/datasets/imagenet-1k).

# Script Usage

In the training script (vgg16_training.py) and preprocessing script (preprocess_imagenet1k_huggingface), there is a global path variable named **data_root_dir** that you need to fill with your Imagenet1-k dataset location. 
