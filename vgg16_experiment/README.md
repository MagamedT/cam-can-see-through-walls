# Folder content

- this folder contains the two pre-train weights of our models, namely masked VGG and unmasked (baseline) VGG.
- It contains the training code of this models and the preprocessing code for Imagenet-1k downloaded from Hugging Face.
- Lastly, it contains *LOC_synset_mapping.txt*, a .txt with the mapping between the human-readable labels and the corresponding synsets.

# Script Usage.

- To prepare the Imagenet-1k dataset for use, start by downloading it from [Hugging Face](https://huggingface.co/datasets/imagenet-1k). After downloading, ensure the file *LOC_synset_mapping.txt* is placed in the same directory as the downloaded dataset. This file is crucial for mapping the synset IDs to human-readable labels.
Next, use the script *preprocess_imagenet1k_huggingface.py* to preprocess the dataset. Before running the script, you must specify the location of the unzipped Imagenet-1k dataset by setting the **data_root_dir** variable within the script. This variable should be the path to the root directory of the unzipped Imagenet-1k dataset.
- To train a VGG16-like model, either with or without a masking mechanism, you can utilize the *vgg16_training.py* script. It's important to ensure that the **data_root_dir** variable within the script is correctly set. This variable should contain the path to where the Imagenet-1k dataset has been extracted and is accessible. Make sure to unzip the Imagenet-1k dataset and update the **data_root_dir** with the appropriate path to the dataset's root directory before initiating the training process.
