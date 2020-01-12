
#########################################################
#					NN Testing Script					#
#########################################################
#
# IMPORTANT:
# This scripts needs to be in the same folder of 
# 'best_net_params_4.pth' and 'MNIST.dat' to work properly
#
# This script will return the MSE of the trained autoencoder
# over the input dataset and will build a ".mat" dataset with 
# the reconstructed images
#


# Import needed libraries
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from autoencoder_structure import Autoencoder


### If cuda is available set the device to GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    device = torch.device("cpu")



# DEFINE NAME AND DIRECTORY OF THE DATASET
filename = '../Lab_03/MNIST.mat'





# Define the dataset class for 'MNIST.mat'
class MNISTmat_dataset(Dataset):
    
    # Initialize the dataset
    def __init__(self, filename):
        
        # Load the MNIST dataset
        data = sio.loadmat(filename)
        self.images = torch.Tensor(data['input_images']).reshape(-1,1,28,28)
        self.labels = torch.Tensor(data['output_labels']).squeeze()

    # This function returns the noised sample, the original one and the label
    def __getitem__(self, idx):
        return (self.images[idx], self.labels[idx])
    
    # Function that defines the length of the dataset
    def __len__(self):
        return len(self.images)




# Load the trained network
net = Autoencoder(4, device)
net.load_state_dict(torch.load('best_net_params_4.pth', map_location=device))
net.to(device)
net.eval()

# Build the 'MNIST.dat' dataset and dataloader
MNIST_dataset = MNISTmat_dataset(filename)
MNIST_dataloader = DataLoader(MNIST_dataset, batch_size=512, shuffle=False)


# Define the loss function
loss_fn = torch.nn.MSELoss()

# Test net without using the implemented method
# because we also want all the reconstructed images
with torch.no_grad(): 
    conc_out = torch.Tensor().float().to(device)
    conc_label = torch.Tensor().float().to(device)
    conc_digits = torch.Tensor().float().to(device)

    for sample_batch in MNIST_dataloader:

        # Extract data and move tensors to the selected device
        image_batch = sample_batch[0].to(device)
        digit_batch = sample_batch[1].to(device)

        # Forward pass
        out = net(image_batch)
        
        # Concatenate with previous outputs
        conc_out = torch.cat([conc_out, out])
        conc_label = torch.cat([conc_label, image_batch])
        conc_digits = torch.cat([conc_digits, digit_batch])

    # Evaluate global loss
    test_loss = loss_fn(conc_out, conc_label)


# Print MSE 
print('MSE :', test_loss.item())

# Build matlab file of reconstructed images
MNIST_dict =  {'__header__': b'MATLAB 5.0 MAT-file, Reconstructed MNIST images',
               '__version__': '1.0',
               '__globals': [],
               'input_images': np.array(conc_out.reshape(len(conc_out), 784)),
               'output_labels': np.array(conc_digits) }
sio.savemat('reconstructed_images.mat', MNIST_dict)