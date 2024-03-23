# Overview

This is the Python code for the experiments of the submission **CAM-Based Methods Can See through Walls**

# Installation

Python version should be >= 3.9.
The three main dependencies are: PyTorch 2.1, DALI (1.35.0 -c420f32) and the last version of [grad-cam](https://github.com/jacobgil/pytorch-grad-cam).
The imagenet1k dataset can be downloaded from [Hugging Face](https://huggingface.co/datasets/imagenet-1k).

# Folder hierarchy overview

- *vgg16_experiment* contains the preprocessing code of Imagenet-1k and the training code for our VGG16-like model. Also, we give in this folder the pre-trained weights for our two models (masked VGG16 and unmasked VGG16)
- *figure_code* contains two google colab notebooks (we higly advice to import them to Google Colab) to generate Figure 1, 3, 7, 8, 9 and Table 1 of our paper.




