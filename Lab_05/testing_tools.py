

# Here the already provided tools for testing the network are
# reorganized into funtions to be used in the notebook when needed

import torch
import numpy as np
import matplotlib.pyplot as plt
import random


# Plot and store the comparison image between an original 
# sample and the reconstructed one (already provided)
def plot_comparison(net, sample, store_fig=False):
    
    # build comparison image
    #img = test_dataset[0][0].unsqueeze(0).to(device)
    img = sample[0].unsqueeze(0).to(net.device)
    net.eval()
    with torch.no_grad():
        rec_img  = net(img)
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
    axs[0].set_title('Original image')
    axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
    axs[1].set_title('Reconstructed image')
    plt.tight_layout()
    plt.pause(0.1)
    
    # Show figure and store them if desired
    if store_fig:
        plt.savefig('Comparison_plot.png')
    plt.show()
    plt.close()
    return





# Function to plot the first two variables of the encoded 
# representation of the encoded samples given as argument
def plot_encoded_space(encoded_samples, n_samples_to_plot = 1000):
    
    # define colors associated to each label
    color_map = { 0: '#1f77b4',
                  1: '#ff7f0e',
                  2: '#2ca02c',
                  3: '#d62728',
                  4: '#9467bd',
                  5: '#8c564b',
                  6: '#e377c2',
                  7: '#7f7f7f',
                  8: '#bcbd22',
                  9: '#17becf' }
    
    ### Visualize encoded space
    encoded_samples_reduced = random.sample(encoded_samples, n_samples_to_plot)
    plt.figure(figsize=(8,6))
    for enc_sample, label in tqdm(encoded_samples_reduced):
        plt.plot(enc_sample[0], enc_sample[1], marker='.', color=color_map[label])
    plt.grid(True)
    plt.legend([plt.Line2D([0], [0], ls='', marker='.', color=c, label=l) for l, c in color_map.items()], color_map.keys())
    plt.tight_layout()
    plt.show()
    
    return
