# Script Usage
We highly advice to use the following jupyter notebooks in Google Colab.

- The usage of *mnist_experiment.ipynb* is straightforward.
- To utilize the notebook *camMaps_on_cutemixAndGen.ipynb*, it's required to first obtain our two datasets, namely STACK-MIX and STACK-GEN (found in supplementary material). These datasets should be downloaded and stored on Google Drive. Once stored, at the beginning of the notebook, you need to define and set two variables: **weights_path** and **folder_path_dataset**.
Indeed, **folder_path_dataset** should point to the locations where the datasets are stored on your Google Drive, facilitating access to the data within the notebook. Additionally, the **weights_path** variable needs to be set with the path to the pre-trained VGG models (both unmasked and masked versions) that are available within the *vgg16_experiment* directory. You do this by uploading this weights to your Google Drive.
