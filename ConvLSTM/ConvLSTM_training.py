import torch,torchvision
print(f"torch version：{torch.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"cuDNN version is: {torch.backends.cudnn.version()}")

from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision.transforms as T


import pytorchvideo
import av
print(av.__version__)

import torch

import sys

from torchvision.datasets import UCF101

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from Seq2Seq import Seq2Seq
from torch.utils.data import DataLoader

import io
import imageio
from ipywidgets import widgets, HBox


# These are some minimal variables used to configure and load the dataset:
ucf_data_dir = "../UCF101_sample/UCF-101"
ucf_label_dir = "../UCF101_sample/TrainTestList"
frames_per_clip =5
step_between_clips = 5
batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


tfs = T.Compose([
    # TODO: this should be done by a video-level transfrom when PyTorch provides transforms.ToTensor() for video
    # scale in [0, 1] of type float
    # T.Lambda(lambda x: x / 255.),
    # reshape into (C,T,H,W) from (T, H, W, C) for easier convolutions #### (to match ConvLSTM stuff)
    # might need C in index 1
    T.Lambda(lambda x: x.permute(3, 0, 1, 2)),
    # rescale to the most common size
    T.Lambda(lambda x: nn.functional.interpolate(x, (64, 64))),
])

def custom_collate(batch):

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    filtered_batch = []
    for video, _, _ in batch:
      filtered_batch.append(video)
    
    # new_batch = torch.utils.data.dataloader.default_collate(filtered_batch)
    new_batch = torch.stack(filtered_batch)   
    new_batch = new_batch / 255.0                        
    new_batch = new_batch.to(device)                   

    # first 4 frames are input, 5th frame is target               
    return new_batch[:,:,0:4, :, :], new_batch[:,:,4, :, :]   


# create train loader (allowing batches and other extras)
train_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                       step_between_clips=step_between_clips, train=True, transform=tfs)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=custom_collate)
# create test loader (allowing batches and other extras)
test_dataset = UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip,
                      step_between_clips=step_between_clips, train=False, transform=tfs)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                          collate_fn=custom_collate)


# Get a batch
input, _ = next(iter(test_loader))

# Reverse process before displaying
input = input.cpu().numpy() * 255.0     


# The input video frames are grayscale, thus single channel
model = Seq2Seq(num_channels=3, num_kernels=64, 
kernel_size=(3, 3), padding=(1, 1), activation="relu", 
frame_size=(64, 64), num_layers=3).to(device)

## CHANGE THIS IF LOADING IN PREVIOUS MODEL
pretrained = False
pretrained_path = 'ConvLSTMmodel_epoch0.pth'

if pretrained == True: 
    model.load_state_dict(torch.load(pretrained_path))

optim = Adam(model.parameters(), lr=1e-6)

# Binary Cross Entropy, target pixel values either 0 or 1
criterion = nn.BCELoss(reduction='sum')

num_epochs = 20

for epoch in range(1, num_epochs+1):
    
    train_loss = 0                                                 
    model.train()                                                  
    for batch_num, (input, target) in enumerate(train_loader, 1):  
        # input = input[:, 0, :, :, :].unsqueeze(1)
        output = model(input)      
        # print(output.detach().numpy())                         
        loss = criterion(output.flatten(), target.flatten())       
        loss.backward()                                            
        optim.step()                                               
        optim.zero_grad()                                           
        train_loss += loss.item()                                 
    train_loss /= len(train_loader.dataset)                       

    val_loss = 0                                                 
    model.eval()                                                   
    with torch.no_grad():                                          
        for input, target in test_loader:                          
            output = model(input)                                   
            loss = criterion(output.flatten(), target.flatten())   
            val_loss += loss.item()                                
    val_loss /= len(test_loader.dataset)                            

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))
    
    # Save intermediate pytorch model at every epoch
    torch.save(model.state_dict(), 'ConvLSTMmodel_epoch' + str(epoch) + '.pth')

    # Save intermediate gif results at every epoch
    # Write input video as gif
    with io.BytesIO() as gif:
        new_input = np.array(input[0, :, :, :, :]) * 255.0
        new_input = new_input.transpose(1, 2, 3, 0)
        imageio.mimsave(gif, new_input, "GIF", fps = 5)
        imageio.mimwrite("input_gif_epoch" + str(epoch),video.astype(np.uint8),"GIF",fps=5)    
        #input_gif = gif.getvalue()
    # Write target video as gif
    with io.BytesIO() as gif:
        new_target = np.array(target[0, :, :, :].unsqueeze(0))* 255.0
        new_target = new_target.transpose(0, 2, 3, 1)
        imageio.mimsave(gif, new_target, "GIF", fps = 5)    
        imageio.mimwrite("target_gif_epoch" + str(epoch),video.astype(np.uint8),"GIF",fps=5) 
        #target_gif = gif.getvalue()

    # Write output video as gif
    with io.BytesIO() as gif:
        new_output = output[0, :, :, :].unsqueeze(0).detach().numpy()* 255.0
        new_output = new_output.transpose(0, 2, 3, 1)
        imageio.mimsave(gif, new_output, "GIF", fps = 5)  
        imageio.mimwrite("output_gif_epoch" + str(epoch),video.astype(np.uint8),"GIF",fps=5) 
        #output_gif = gif.getvalue()


