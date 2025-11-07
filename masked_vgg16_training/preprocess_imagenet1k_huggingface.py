import os
import csv 
import shutil
import random


# ROOT path containing the mapping between synset and index (LOC_synset_mapping.txt) and imagenet data.
data_root_dir = "YOUR_DATA_DIR"



# Load the mapping between synset and index
synset_file = open(data_root_dir + "/LOC_synset_mapping.txt", "r")
synset_folder_index = dict() # dict containing the pairs (synset, imagenet index)

for i, line in enumerate(synset_file.readlines()):
    line_elements = line.split(" ", 1)

    synset = line_elements[0]
    classes_name = line_elements[1]

    synset_folder_index[synset] = i



# get all folder name to create.
all_train_folder_list = list(synset_folder_index.keys())

# create all the folder in train also in val folder
val_folder_path = data_root_dir + "/data/val/"
train_folder_path = data_root_dir + "/data/train/"

def create_folder_classes(root_path):
    for folder_name in all_train_folder_list:
        folder_path = os.path.join(root_path, folder_name)
        # print(folder_path)
        os.makedirs(folder_path)

create_folder_classes(val_folder_path)
create_folder_classes(train_folder_path)


# put all images in the corresponding class folder
def put_img_in_folders(root_path):
    # preprocessing root (train, val, test) directory: put every image into a class folder.
    for file in os.scandir(root_path):
        if file.is_file():
            # image file are in .JPEG format
            folder_name = file.name.replace(".JPEG", "").split("_")[-1]
            folder_path = root_path + folder_name

            shutil.move(file, folder_path)
        
put_img_in_folders(val_folder_path)
put_img_in_folders(train_folder_path)



# rename all 'synset' filename into 'classes index' filename
def rename_synsets_to_classesIndex(root_dir):
    subfolders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    for subfolder in subfolders:
        if subfolder in synset_folder_index:
            old_folder_path = os.path.join(root_dir, subfolder)

            new_folder_name = str(synset_folder_index[subfolder])
            new_folder_path = os.path.join(root_dir, new_folder_name)
            os.rename(old_folder_path, new_folder_path)

rename_synsets_to_classesIndex(train_folder_path)
rename_synsets_to_classesIndex(val_folder_path)
   

# generate files for DALI dataloader that contrains file name and associated labels

def create_image_label_association_txt(root_folder, output_file):
    """
    Create a text file listing all images in the root folder with their corresponding subfolder names and label in imagenet.
    
    :param root_folder: Path to the root folder containing subfolders for each class.
    :param output_file: Path to the output .txt file.
    """
    with open(output_file, 'w') as f:
        for subdir, _, files in os.walk(root_folder):
            if subdir == root_folder:
                # Skip the root directory itself
                continue

            # Get the subfolder (class) name
            class_name = os.path.basename(subdir)

            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Write the file name and subfolder name to the output file
                    f.write(f"{os.path.join(class_name, file)} {class_name}\n")

# Usage for val folder
root_folder = val_folder_path  # Replace with the path to your val folder if needed
output_file = data_root_dir + "/val_labels.txt"   # Replace with the path where you want to save the .txt file
create_image_label_association_txt(root_folder, output_file)

# Usage for train folder
root_folder = train_folder_path  # Replace with the path to your train folder if needed
output_file = data_root_dir + "/train_labels.txt"   # Replace with the path where you want to save the .txt file
create_image_label_association_txt(root_folder, output_file)
