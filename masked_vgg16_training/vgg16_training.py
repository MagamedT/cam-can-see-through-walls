############################################################################################################

from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import nvidia.dali.fn as fn
from nvidia.dali.plugin.pytorch import DALIGenericIterator, DALIClassificationIterator, LastBatchPolicy


import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F



from torch.utils.data import Dataset
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
from PIL import Image


import numpy as np
import math

import gc
import os


# Set rand
torch.manual_seed(3)
np.random.seed(3)

# empty cache
gc.collect()
torch.cuda.empty_cache()

# Set backend state
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.enabled =  True

############################################################################################################

# Set wandb offline
os.environ["WANDB_MODE"] = "offline"

# Set device to gpu 0 if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############# Create Imagenet Dataloaders ###################################################################


# AverageMeter of NVIDIA repository, found in https://github.com/NVIDIA/DALI/blob/c420f32d5f5d899215e11f5d1c4b8f437b32dd95/docs/examples/use_cases/pytorch/resnet50/main.py
# Define average meter object which will be used to store the average and current value of a metric.
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Define function to convert tensor to python float
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

# Define function to compute model accuracy
def model_accuracy(net, loader, topk=(5,)):
    maxk = max(topk)
    acc = AverageMeter()

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, target = data[0]['data'], data[0]['label'].to(device) # already on the gpu

            output = F.softmax(net(inputs), dim=1)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))

            acc.update(to_python_float(res), inputs.size(0)) # this also resets the accuracy computation
    return acc.avg.item()

################################ CREATE DALI DATALOADERS ############################################################################

# To run with different data, see documentation of nvidia.dali.fn.readers.file
# points to https://github.com/NVIDIA/DALI_extra

# CHANGE THIS TO YOUR imagenet root PATH as processed in preprocess_imagenet1k_huggingface.py
data_root_dir = ""
images_dir_val = os.path.join(data_root_dir, 'data/val')
images_dir_train = os.path.join(data_root_dir, 'data/train')


# Define the pipeline, present in the DALI documentation for ResNet50.
@pipeline_def(num_threads=4, device_id=0)
def create_dali_pipeline(data_dir, labels_file, crop=224, size=224, dali_cpu=False, is_training=True):
    images, labels = fn.readers.file(file_root=data_dir,
                                     random_shuffle=is_training,
                                     pad_last_batch=True,
                                     name="Reader", file_list = labels_file)
    dali_device = 'cpu' if dali_cpu else 'gpu'
    decoder_device = 'cpu' if dali_cpu else 'mixed'
    # ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
    device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
    host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
    # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
    preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
    preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
    if is_training:
        images = fn.decoders.image_random_crop(images,
                                               device=decoder_device, output_type=types.RGB,
                                               device_memory_padding=device_memory_padding,
                                               host_memory_padding=host_memory_padding,
                                               preallocate_width_hint=preallocate_width_hint,
                                               preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
        images = fn.resize(images,
                           device=dali_device,
                           resize_x=crop,
                           resize_y=crop,
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = fn.random.coin_flip(probability=0.5)
    else:
        images = fn.decoders.image(images,
                                   device=decoder_device,
                                   output_type=types.RGB)
        images = fn.resize(images,
                           device=dali_device,
                           size=size,
                           mode="not_smaller",
                           interp_type=types.INTERP_TRIANGULAR)
        mirror = False

    images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(crop, crop),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)
    labels = labels.gpu()
    return images, labels


BATCH_SIZE = 256

# Define valide loader

val_labels_file = data_root_dir + "/val_labels.txt"

val_pipe = create_dali_pipeline(batch_size=BATCH_SIZE,
                                    data_dir=images_dir_val,
                                    labels_file=val_labels_file, 
                                    is_training=False) 

val_pipe.build()

val_data = DALIClassificationIterator(
    val_pipe,
    last_batch_policy=LastBatchPolicy.PARTIAL,
    reader_name='Reader',
    auto_reset=True
)

# Define train loader

train_labels_file = data_root_dir + "/train_labels.txt"

train_pipe = create_dali_pipeline(batch_size=BATCH_SIZE,
                                    data_dir=images_dir_train,
                                    labels_file=train_labels_file, 
                                    is_training=True) 

train_pipe.build()

train_data = DALIClassificationIterator(
    train_pipe,
    last_batch_policy=LastBatchPolicy.PARTIAL,
    reader_name='Reader',
    auto_reset=True
)

##### Create custom model with vgg16 #####

# get vgg16 with: "IMAGENET1K_V1", None
model = torchvision.models.vgg16(weights = None).to(device)

# remove last max-pooling
index_to_remove = 30
new_features = list(model.features.children())[:index_to_remove] + list(model.features.children())[index_to_remove+1:]

# remove average pooling
custom_model = nn.Sequential(*new_features, nn.Flatten(start_dim=1),model.classifier)

# update size of first linear layer and last convolutional layer
custom_model[28] = nn.Conv2d(512,256,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
custom_model[-1][0] = nn.Linear(256*14*14, custom_model[-1][0].out_features) # here, custom_model[-1][0].out_features = 4096

custom_model = custom_model.to(device)


# Weight masking in layer connecting Features and Classifier
with torch.no_grad():
    m = torch.ones((14,14)).to(device)
    m[-9:, :] = 0
    m = m.unsqueeze(0).repeat(256,1,1)
    m = m.unsqueeze(0).repeat(4096,1,1,1)
    m = m.reshape(4096,-1)

    custom_model[-1][0].weight *= m


#### Define training process ####

use_amp = True
LR = 3.0 * 1e-2
EPOCHS = 90
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
LR_GAMMA = 0.1
LR_STEP_SIZE = 30
# path for model checkpoints.
PATH_MODEL = data_root_dir + f"/model.pt"
FP16_MODE = True
resume = False


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(custom_model.parameters(), lr=LR, momentum = MOMENTUM, weight_decay = WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
if FP16_MODE:
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, growth_factor=2, backoff_factor=0.5 ,growth_interval=100)

if resume:
    checkpoint = torch.load(data_root_dir + f"/model.pt")
    
    custom_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])


#### Setup Wandb ####

run = wandb.init(project='custom_vgg16_train',
                 
                 config={
                     "lr":LR,
                     "epochs":EPOCHS,
                     "momentum":MOMENTUM,
                     "weight_decay":WEIGHT_DECAY,
                     "lr_gamma":LR_GAMMA,
                     "lr_step_size":LR_STEP_SIZE,  
                 })

# Define training loop.
custom_model.train()
for epoch in range(EPOCHS):
  running_loss = 0
  for i, batch in enumerate(train_data):

    inputs, labels = batch[0]['data'], batch[0]['label'].type(torch.LongTensor).squeeze().to(device) # already on device

    optimizer.zero_grad(set_to_none=True)

    
    with torch.cuda.amp.autocast(enabled=FP16_MODE):
        out = custom_model(inputs)
        loss = criterion(out, labels)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    with torch.no_grad():
        custom_model[-1][0].weight *= m


    running_loss += loss.item()
    

    
  # log metrics at the end of each epoch.
  train_loader_len = int(math.ceil(train_data._size / BATCH_SIZE))
  loss_to_print = running_loss / train_loader_len # in our case train_loader_len = nearly 5000
  val_to_print = model_accuracy(custom_model, val_data, topk=(5,))
  custom_model.train() # reset model mode

  wandb.log({'epoch': epoch+1, 'loss': loss_to_print, 'val_accuracy': val_to_print})


  # reset record variables
  running_loss = 0.0

  # step scheduler after an epoch is ended.
  scheduler.step()

  # checkpoint every epochs
  torch.save({
            'epoch': epoch,
            'model_state_dict': custom_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            }, PATH_MODEL)


print('Finished Training')


# save model after training.
torch.save(custom_model.state_dict(), data_root_dir + f'/params_vgg16.pt')
