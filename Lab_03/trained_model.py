
#########################################################
#					NN Testing Script					#
#########################################################
#
# IMPORTANT:
# This scripts needs to be in the same folder of 
# 'winning_net.dat' and 'MNIST.dat' to work properly
#


# Import needed libraries
import scipy.io as sio
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim



# Define Neural Network class
class NN_model(nn.Module):
    
    def __init__(self, Ni, Nh1, Nh2, No, dropout=0.5, act_func=nn.LeakyReLU()):
        super().__init__()
        
        self.fc1 = nn.Linear(Ni, Nh1)
        self.fc2 = nn.Linear(Nh1, Nh2)
        self.fc3 = nn.Linear(Nh2, No)
        
        self.drop = nn.Dropout(dropout)
        self.act  = act_func
        
    def forward(self, x, additional_out=False):
        
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.act(self.fc2(x)))
        out = self.fc3(x)    # since we will use the cross entropy as loss function, we don't
                             # need a softmax activation function for the last layer
        if additional_out:
            return out, x
        
        return out



# Load the trained winning network
winning_net = torch.load('winning_net.dat')
winning_net.eval()

# Load the MNIST dataset
data = sio.loadmat('MNIST.mat')
images = torch.Tensor(data['input_images'])
labels = torch.LongTensor(data['output_labels']).squeeze()

# Compute its accuracy over the MNIST dataset
with torch.no_grad():
    net_output = winning_net(images)
    correct_answers = [np.argmax(net_output[i])==labels[i] for i in range(len(labels))]
    print('Accuracy:', np.sum(correct_answers)/len(labels))