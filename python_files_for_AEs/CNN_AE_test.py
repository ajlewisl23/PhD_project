### Import libraries
import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torch.optim as optim
from torchinfo import summary

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import xarray as xr
import random as rand
from pathlib import Path
import scipy
import skimage as ski
import torcheval

### Load list of filenames of storm_data.npy files
storm_filenames = os.listdir("../storm_pixels_data/")[1:]

storm_labels_dict = {}

for idx, ID in enumerate(storm_filenames):
    storm_labels_dict[idx] = ID[:-4]

### Setup train-test partition of file labels
train_val_split = 1/3

val_size = int(np.floor(len(storm_labels_dict)*train_val_split))
train_size = int(len(storm_labels_dict) - val_size)
#print(test_size, train_size)

shuffled_labels = list(storm_labels_dict.values())
rand.shuffle(shuffled_labels)
train_labels = shuffled_labels[:val_size]
val_labels = shuffled_labels[val_size:]

partition = {"train":train_labels, "val":val_labels}

### Define Classes for data transforms
class RandomPad(object):
    """Pad the object up to a certain resolution placing the data pixels randomly within the box"""
    
    def __init__(self, output_res):
        assert isinstance(output_res, int)
        self.output_res = output_res
        self.threshold = 233
        
    def __call__(self, data):
        h, w = data.shape
        padded_data = np.ones((self.output_res, self.output_res))*self.threshold
        if (h<self.output_res) & (w<self.output_res):
            bl_corner_y, bl_corner_x = rand.randint(0, self.output_res-h), rand.randint(0, self.output_res-w)
            padded_data[bl_corner_y:bl_corner_y+h, bl_corner_x:bl_corner_x+w] = data
            padded_data = torch.from_numpy(padded_data)
            return padded_data
        
        else:
            padded_data = torch.from_numpy(padded_data)
            return padded_data

class CentrePadAndNormalise(object):
    """Pad the object up to a certain resolution placing the data pixels randomly within the box"""
    
    def __init__(self, output_res):
        assert isinstance(output_res, int)
        self.output_res = output_res
        self.upper_threshold = 233
        self.lower_bound = 180
        
    def __call__(self, data, idx):
        # Normalise to between 0 (upper threshold K) and 1 (lower bound K) 
        data = (self.upper_threshold - data)/(self.upper_threshold-self.lower_bound)
        try:
            h, w = data.shape
        except ValueError:
            print(idx)
        padded_data = np.zeros((self.output_res, self.output_res))
        img_centre = np.floor(self.output_res/2)
        if (h<self.output_res) & (w<self.output_res):
            padded_data[int(img_centre-np.floor(h/2)):int(img_centre+np.ceil(h/2)), int(img_centre-np.floor(w/2)):int(img_centre+np.ceil(w/2))] = data
            padded_data = torch.from_numpy(padded_data)
            return padded_data
        
        else:
            padded_data = torch.from_numpy(padded_data)
            return padded_data

### Define how to load the data from the storm data .npy files
class StormPixelsDataset(Dataset):
    """Storm Pixels dataset."""
    def __init__(self, list_IDs, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = "../storm_pixels_data/"
        #self.labels = labels
        self.list_IDs = list_IDs
        #self.files = np_file_paths
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, idx):
        ID = self.list_IDs[idx]
        X = np.float32(np.load(self.root_dir+ID+".npy"))
        if self.transform:
            X = self.transform(X, idx)
        X = X.float()
        X = X.unsqueeze(0)
        
        return X, ID
        """
        print("Loading"+self.root_dir+"/"+self.files[idx])
        x = np.load(self.root_dir+"/"+self.files[idx])
        x = torch.from_numpy(x).float()
        return x
        """

### Define model
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def forward(self, input, size=81920):
        return input.view(input.size(0), size, 64, 64)

### Define the CNN in PyTorch
class CNN_VAE(nn.Module):
    def __init__(self):
        super(CNN_VAE, self).__init__()
        
        # Build encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 20, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.Dropout(p=0.5)
            #torch.reshape()
            # nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(128, 128, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            #nn.AdaptiveAvgPool2d((1, 1))
            Flatten(),
            #nn.Linear(81920,2)
        )

        # Construct Latent layers
        self.mean_layer = nn.Linear(81920, 2)
        self.logvar_layer = nn.Linear(81920, 2)
        nn.Linear(2, 81920)
        #print(low.shape)
        
        # Build decoder
        self.decoder = nn.Sequential(
            UnFlatten(),
            # nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(20, 20, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(20, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        """self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        """
    #super(Encoder, self).__init__()

    def encode(self, x):
        x = self.encoder(x)
        mean, log_var = self.mean_layer(x), self.logvar_layer(x)
        return mean, log_var

    def reparameterisation(self, mean, var):
        epsilon = torch.randn_like(var).to(device)
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)
        
    def forward(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterisation(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var#, reduced_space

### Setup dataloaders and instantiate model
model = CNN_VAE()
print(model)

# Parameters
params = {'batch_size': 64,
          'shuffle': True,
          'num_workers': 6}

# Generators
train_set = StormPixelsDataset(partition['train'], transform=CentrePadAndNormalise(256))
train_loader = torch.utils.data.DataLoader(train_set, **params)

val_set = StormPixelsDataset(partition['val'], transform=CentrePadAndNormalise(256))
val_loader = torch.utils.data.DataLoader(val_set, **params)

### Setup model and move to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CNN_VAE().to(device)

### Define loss function
def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
  return reproduction_loss + KLD

### Setup optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

### Train autoencoder
max_epochs = 20
# Loop over epochs
loss_over_time = []
for epoch in range(max_epochs):
    overall_loss = 0
    # Training
    for local_batch, local_labels in train_loader:
        #print(local_batch, local_labels)
        # Transfer to GPU
        local_batch = local_batch[:-1].to(device)
        optimizer.zero_grad()
        x_hat, mean, log_var = model(local_batch)
        #loss = criterion(output, local_batch)
        loss = loss_function(x, x_hat, mean, log_var)

        overall_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        #print(local_batch)
        #print(output)
    for local_val_batch, local_val_labels in val_loader:
        local_val_vatch = local_val_batch = local_val_batch[:-1].to(device)
        val_output = model(local_val_batch)
        val_loss = criterion(val_output, local_val_batch)
    loss_over_time.append([epoch, loss, val_loss])
    if epoch % 5== 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, max_epochs, loss.item()))

### Store loss over time
loss_arr = np.empty((len(loss_over_time), 3))
for i in range(len(loss_over_time)):
    loss_arr[i] = loss_over_time[i][0], loss_over_time[i][1].detach().numpy(), loss_over_time[i][2].detach().numpy()

### Save loss over time
np.save("loss_arr.npy", loss_arr)

### Save model parameters
torch.save(model.state_dict(), "model_dropout=0.1")

### Load model
model_load = CNN_AE()
model_load.load_state_dict(torch.load("model_dropout=0.1"))
model_load.eval()
