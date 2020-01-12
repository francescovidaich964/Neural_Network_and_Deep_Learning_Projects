

# This is the architecture of the autoencoder. 
# Inside the class we also define different methods 
# used for training and testing the model.

import torch
from torch import nn

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



class Autoencoder(nn.Module):
    
    def __init__(self, encoded_space_dim, device, dropout=0):
        super().__init__()
        
        # Store the device as a class member
        self.device = device
        
        ### Encoder
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.Dropout(dropout),
            nn.ReLU(True)
        )
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 64),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(64, encoded_space_dim)
        )
        
        ### Decoder
        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 64),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.Linear(64, 3 * 3 * 32),
            nn.Dropout(dropout),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.Dropout(dropout),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def encode(self, x):
        # Apply convolutions
        x = self.encoder_cnn(x)
        # Flatten
        x = x.view([x.size(0), -1])
        # Apply linear layers
        x = self.encoder_lin(x)
        return x
    
    def decode(self, x):
        # Apply linear layers
        x = self.decoder_lin(x)
        # Reshape
        x = x.view([-1, 32, 3, 3])
        # Apply transposed convolutions
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x
    
    
    
    # -------- New functions ---------
    
    ### Function that perform an epoch of the training
    def train_epoch(self, dataloader, loss_fn, optim, denoise_mode=False, log=False):
        
        # set training mode
        self.train()
        
        for sample_batch in dataloader:
            
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(self.device)
            
            # Forward pass (if denoise_mode, use original image as target)
            output = self.forward(image_batch)
            if denoise_mode == False:
                loss = loss_fn(output, image_batch)
            else:
                loss = loss_fn(output, sample_batch[1].to(self.device))
            
            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # Print loss if requested
            if log:
                print('\t partial train loss: %f' % (loss.data))

         
        
    ### Testing function
    def test_epoch(self, dataloader, loss_fn, denoise_mode=False):
        
        # Validation
        self.eval() # Evaluation mode (e.g. disable dropout)
        
        with torch.no_grad(): # No need to track the gradients
            
            conc_out = torch.Tensor().float().to(self.device)
            conc_label = torch.Tensor().float().to(self.device)
            
            for sample_batch in dataloader:
                
                # Extract data and move tensors to the selected device
                image_batch = sample_batch[0].to(self.device)
               
                # Forward pass
                out = self.forward(image_batch)
                
                # Concatenate with previous outputs 
                # (if denoise_mode, use original image as target)
                conc_out = torch.cat([conc_out, out])#.cpu()])
                
                if denoise_mode == False:
                    conc_label = torch.cat([conc_label, image_batch]) #.cpu()])
                else:
                    conc_label = torch.cat([conc_label, sample_batch[1].to(self.device)]) #.cpu()])
            
            # Evaluate global loss
            val_loss = loss_fn(conc_out, conc_label)
        
        return val_loss.data
    
    
    
    ### Function to get the encoded representation of an input dataset
    def get_enc_representation(self, dataset, return_label=False):
        encoded_samples = np.array([])

        for sample in tqdm(dataset):
            img = sample[0].unsqueeze(0).to(self.device)
            label = sample[1]

            # Encode image
            self.eval()
            with torch.no_grad():
                encoded_img  = self.encode(img)

            # Append to list
            np_encoded_img = encoded_img.flatten().cpu().numpy() 
            if return_label:
                if encoded_samples.size == 0:
                    encoded_samples = np.array( [[np_encoded_img, label],] )
                else:
                    encoded_samples = np.append(encoded_samples, [[np_encoded_img, label]], axis=0)
            else:
                if encoded_samples.size == 0:
                    encoded_samples = np.array([np_encoded_img,])
                else:
                    encoded_samples = np.append(encoded_samples, [np_encoded_img], axis=0)

        return encoded_samples 
    
    
    
    ### Function that plots or store the generated image decoded from the given enc_sample
    def generate_from_encoded_sample(self, enc_sample, filename=None):

        # Decode the sample to produce the image
        self.eval()
        with torch.no_grad():
            encoded_value = torch.tensor(enc_sample).float().unsqueeze(0)
            new_img  = self.decode(encoded_value)

        # plot the generated image, store it if required
        plt.figure(figsize=(8,6))
        plt.imshow(new_img.squeeze().cpu().numpy(), cmap='gist_gray')
        if filename != None: 
            plt.savefig(filename)
        else:
            plt.show()
        plt.close()

        return
    