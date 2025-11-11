import os
from pathlib import Path
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
from PIL import Image

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, LayerCAM

import matplotlib.pyplot as plt

import numpy as np

import random

import ast




# PATH SETUP TO THE FOLDER CONTAINING THE VGG16 WEIGHTS AND THE UNZIPPED DATA.
DATA_PATH = Path(__file__).resolve().parent / "data"
MODEL_PATH = Path(__file__).resolve().parent / "model"
# PATH SETUP FOR IMAGENET LABELS
LABELS_FILE = Path(__file__).resolve().parent / "imagenet1000_clsidx_to_labels.txt"
with open(LABELS_FILE) as f:
    IMAGENET_IDX_TO_LABEL = {int(k): v for k, v in ast.literal_eval(f.read()).items()}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 3
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Custom dataset class to load stack-mix and stack-gen.
class CustomImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.image_paths = [os.path.join(directory, file)
                            for file in sorted(os.listdir(directory))
                            if file.lower().endswith((".png", ".jpg", ".jpeg"))
                           ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    


def load_dataset(dataset_name = 'stackmix', images_idx = []):
    """
    Load the dataset, either stackgen or stackmix.
    `images_idx` is a list of the images to load. If the list is empty then the whole 100 images from the choosen dataset are loaded.
    """
    # Define a transform to convert the image to a PyTorch tensor
    if dataset_name == 'stackmix':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    # Load all images from the specified folder
    path_to_data = DATA_PATH / dataset_name
    dataset = CustomImageDataset(directory=path_to_data, transform=transform)

    # Create a DataLoader to batch the images
    size_dataset = len([name for name in os.listdir(path_to_data) if os.path.isfile(os.path.join(path_to_data, name))])
    dataloader = DataLoader(dataset, batch_size=int(size_dataset), shuffle=False)

    # get the whole 100 images of one of our dataset.
    if images_idx == []:
        dataset_batch = next(iter(dataloader)).to(DEVICE)
    else:
        dataset_batch = next(iter(dataloader))[images_idx].to(DEVICE)
    
    return dataset_batch

def load_model(masked = False):
    # get vgg16 with: None ("IMAGENET1K_V1" to get the pretrained model)
    model = torchvision.models.vgg16(weights = None).to(DEVICE)

    # remove last max-pooling
    index_to_remove = 30
    new_features = list(model.features.children())[:index_to_remove] + list(model.features.children())[index_to_remove+1:]

    # remove global average pooling which is not in the VGG paper of Simonyan.
    custom_model = nn.Sequential(*new_features, nn.Flatten(start_dim=1),model.classifier)

    # update size of first linear layer and of the last convolutional layer
    custom_model[28] = nn.Conv2d(512,256,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    custom_model[-1][0] = nn.Linear(256*14*14, custom_model[-1][0].out_features)

    custom_model = custom_model.to(DEVICE)

    if masked:
        checkpoint = torch.load(MODEL_PATH / "params_vgg16_masked.pt",  map_location="cpu")
    else:
        checkpoint = torch.load(MODEL_PATH / "params_vgg16_baseline.pt",  map_location="cpu")

    custom_model.load_state_dict(checkpoint)

    if masked:
        # For safety: re-apply weights masking in the first dense layer to create a "dead zone" in the precedent feature maps. This is already the case but just in case.
        with torch.no_grad():
            mask = torch.ones((14,14)).to(DEVICE)
            mask[-9:, :] = 0
            mask = mask.unsqueeze(0).repeat(256,1,1)
            mask = mask.unsqueeze(0).repeat(4096,1,1,1)
            mask = mask.reshape(4096,-1)

            custom_model[-1][0].weight *= mask
    

    custom_model = custom_model.to(DEVICE)
    # If model not put into eval() mode, the output will not be deterministic
    custom_model.eval()
    
    return custom_model

def get_heatmaps(model, dataset_batch, class_target = None, layers_idx = None, method= "GradCAM"):
    """ 

    method can be any of ["GradCAM", "HiResCAM", "ScoreCAM", "GradCAMPlusPlus", "AblationCAM", "XGradCAM", "EigenCAM", "LayerCAM"] 
    target_layers is the list of layers to compute CAM saliency maps on.
    class_target is the class for which we compute CAM, if None, then computed for the highest scoring class.
    """
    # Mapping of method names to pytorch_grad_cam CAM classes
    cam_methods = {
        "GradCAM": GradCAM,
        "HiResCAM": HiResCAM,
        "ScoreCAM": ScoreCAM,
        "GradCAMPlusPlus": GradCAMPlusPlus,
        "AblationCAM": AblationCAM,
        "XGradCAM": XGradCAM,
        "EigenCAM": EigenCAM,
        "LayerCAM": LayerCAM,
    }

    # set model to eval mode and give the last rectified activations maps as reference for computations of the saliency maps.
    model.eval()
    assert layers_idx != None
    target_layers = [model[idx] for idx in layers_idx]

    # Instantiate the chosen CAM method
    if method in cam_methods:
        cam_class = cam_methods[method]
        cam = cam_class(model=model, target_layers=target_layers)
    else:
        raise ValueError(f"Unsupported CAM method: {method}")

    # target = None means that the saliency maps will be computed for the highest scoring class of each images.
    grayscale_cam = cam(input_tensor=dataset_batch, targets=class_target)

    return grayscale_cam

def plot_heatmap_with_wall(heatmaps, dataset, idx = 0, save = False):

    name_to_save = f"../figures/img_{idx}.png"

    image = np.transpose((dataset[idx]/ (1000/225) + 0.5).squeeze().detach().cpu(), (1, 2, 0))
    gradCamMaps_tensor = heatmaps[idx]

    fig, ax = plt.subplots(1, frameon=False)

    # plot img and saliency map
    ax.imshow(image, alpha = 1.)
    ax.imshow(gradCamMaps_tensor, cmap = "jet", alpha = 0.7)

    # plot red hatched line
    xline_position = image.shape[0] / 1.32
    ax.axhline(y=xline_position, color='red', linestyle='--', linewidth=4)

    # Add a hatched zone to the left of the vertical line
    ax.fill_betweenx(y=[xline_position-1/2, image.shape[1]-1/2], x1=-1, x2=image.shape[1], color='red', alpha=0.2, hatch='/')

    # fig setup and save
    ax.axis('off')

    # Before saving, adjust the figure's layout
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save==True:
        # Use 'bbox_inches' and 'pad_inches' to remove the white borders
        fig.savefig(name_to_save, bbox_inches='tight', pad_inches=0)

    plt.show()

def plot_10_random_heatmaps(heatmaps, dataset):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))  # Creates a 2x5 grid of subplots

    for i in range(10):  # Looping through the first 10 images
        ax = axes[i // 5, i % 5]  # Determines the position of the subplot

        # First image layer
        index_img = random.randint(0, 99)

        ax.imshow(np.transpose((dataset[index_img] / (1000 / 225) + 0.5).squeeze().detach().cpu(), (1, 2, 0)), alpha=1)

        # Second image layer (CAM overlay)
        ax.imshow(heatmaps[index_img], alpha=0.6, cmap="jet")

        ax.axis('off')  # Hides the axis

    plt.show()  # Displays the subplot

def plot_image(model, dataset, idx = 0):
    image = dataset[idx]
    logits = model(image.unsqueeze(0))
    predicted_class = int(logits.argmax())
    label = IMAGENET_IDX_TO_LABEL.get(predicted_class, "Unknown class")

    plt.imshow(np.transpose((image / (1000 / 225) + 0.5).squeeze().detach().cpu(), (1, 2, 0)), alpha=1)
    plt.title(f"top-1 prediction: {label} (class_id={predicted_class})")
    plt.axis("off")
    plt.show()



