

# This is the architecture of the autoencoder. 
# Inside the class we also define different methods 
# used for training and testing the model.

class Autoencoder(nn.Module):
    
    def __init__(self, encoded_space_dim, dropout=0):
        super().__init__()
        
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
    def train_epoch(self, dataloader, loss_fn, optimizer, log=True):
        
        # set training mode
        self.train()
        
        for sample_batch in dataloader:
            # Extract data and move tensors to the selected device
            image_batch = sample_batch[0].to(device)
            # Forward pass
            output = self.forward(image_batch)
            loss = loss_fn(output, image_batch)
            # Backward pass
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            # Print loss if requested
            if log:
                print('\t partial train loss: %f' % (loss.data))

         
        
    ### Testing function
    def test_epoch(self, dataloader, loss_fn, optimizer):
        
        # Validation
        self.eval() # Evaluation mode (e.g. disable dropout)
        
        with torch.no_grad(): # No need to track the gradients
            
            conc_out = torch.Tensor().float()
            conc_label = torch.Tensor().float()
            
            for sample_batch in dataloader:
                # Extract data and move tensors to the selected device
                image_batch = sample_batch[0].to(device)
                # Forward pass
                out = self.forward(image_batch)
                # Concatenate with previous outputs
                conc_out = torch.cat([conc_out, out.cpu()])
                conc_label = torch.cat([conc_label, image_batch.cpu()]) 
            
            # Evaluate global loss
            val_loss = loss_fn(conc_out, conc_label)
        
        return val_loss.data
    
    
    
    ### Function to get the encoded representation of an input dataset
    def get_enc_representation(dataset, net):
        encoded_samples = []

        for sample in tqdm(dataset):
            img = sample[0].unsqueeze(0)
            label = sample[1]

            # Encode image
            net.eval()
            with torch.no_grad():
                encoded_img  = self.encode(img)

            # Append to list
            encoded_samples.append((encoded_img.flatten().numpy(), label))

        return encoded_samples 
    
    
    
    ### Function that plots the generated image decoded from the given enc_sample
    def generate_from_encoded_sample(enc_sample, net):

        # Decode the sample to produce the image
        net.eval()
        with torch.no_grad():
            encoded_value = torch.tensor(enc_sample).float().unsqueeze(0)
            new_img  = net.decode(encoded_value)

        # plot the generated image
        plt.figure(figsize=(12,10))
        plt.imshow(new_img.squeeze().numpy(), cmap='gist_gray')
        plt.show()

        return