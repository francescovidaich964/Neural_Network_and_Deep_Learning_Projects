

# This file contains the functions that corrupts the dataset
#  - Gauss_noised_dataset: add a gaussian noise with selected std
#  - Obscured_dataset: Hide (set to zero) a portion of the images (it will be a square 
#                      of the desired size with random centre)


import torch
import numpy as np



def Gauss_noised_dataset(dataset, std_noise = 0.2):
    
    # Generate all Gaussian noises
    noises = torch.Tensor(np.random.normal(0,std_noise,(len(dataset),28,28))).cpu()

    # Add them to the images and clip results to be contained in [0,1]
    # Store everything with the format of the MNIST dataset
    noisy_data = []
    for i in range(len(dataset)):
        noisy_image = torch.clamp(dataset[i][0][0] + noises[i], min=0, max=1)
        noisy_sample = ( noisy_image.reshape([1,28,28]), dataset[i][1])
        noisy_data.append(noisy_sample)

    return noisy_data




def Obscured_dataset(dataset, square_size = 5):

    # Generate random squares with starting position defined by 
    # their low edge(the square still has to be fully contained in 
    # the image bounds, so there are some constraints)

    edges = np.random.randint(low = 0, 
                              high = 28-square_size,
                              size = (len(dataset), 2) )

    # Obscure each image of the dataset with the correspondent square
    # Store everything with the format of the MNIST dataset
    obscured_data = []
    for i in range(len(dataset)):

        # define square indexes
        x_min = edges[i,0]
        x_max = edges[i,0]+square_size
        y_min = edges[i,1]
        y_max = edges[i,1]+square_size

        # Set to zero elements of the image covered by the square
        obscured_image = dataset[i][0][0]
        obscured_image[ x_min:x_max, y_min:y_max ] = torch.zeros((square_size,square_size))

        # build sample for the MNIST format
        obscured_sample = ( obscured_image.reshape([1,28,28]), dataset[i][1])
        obscured_data.append(obscured_sample)

    return obscured_data
