

# Here the already provided tools for testing the network are
# reorganized into funtions to be used in the notebook when needed

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

import random
import imageio


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









# Function that takes two digits, start from the corresponding 
# centroids of the encoded digits and build the gif of the 
# reconstructed images of the points between them (to see the evolution)
def morphing_digits_gif(net, encoded_samples, start_digit, end_digit, 
                        t_update=0.1, n_frames=200, gif_directory='results/'):
    
    # Find centroids of the two digits in the endoded space
    start_samples = encoded_samples[ encoded_samples[:,1] == start_digit ]
    end_samples = encoded_samples[ encoded_samples[:,1] == end_digit ]
    start_centroid = np.stack(start_samples[:,0]).mean(axis=0)
    end_centroid = np.stack(end_samples[:,0]).mean(axis=0)

    # Compute array with intermediate points
    middle_points = np.array([ np.linspace(start_centroid[0], end_centroid[0], n_frames), 
                               np.linspace(start_centroid[1], end_centroid[1], n_frames),    
                               np.linspace(start_centroid[2], end_centroid[2], n_frames),    
                               np.linspace(start_centroid[3], end_centroid[3], n_frames) ])

    # Decode every intermediate point and store the correspondent image in a temp folder
    os.makedirs('tmp_gif_dir')
    for i in range(n_frames):
        filename = 'tmp_gif_dir/image_'+str(i).zfill(3)+'.png'
        net.generate_from_encoded_sample(middle_points[:,i], filename)

    # Order all filenames
    all_filenames = os.listdir('tmp_gif_dir')
    all_filenames = sorted(all_filenames)

    # Build the GIF
    images = []
    for filename in all_filenames:
        images.append(imageio.imread('tmp_gif_dir/'+filename))
    
    # Store the GIF
    gif_filename = gif_directory+'from_'+str(start_digit)+'_to_'+str(end_digit)+'_smooth_evol.gif'
    imageio.mimsave(gif_filename, images, format='GIF', duration=t_update)
    print('GIF is stored as '+gif_filename)

    # Delete temp files and folder
    for filename in all_filenames:
        os.remove('tmp_gif_dir/'+filename)
    os.rmdir('tmp_gif_dir/')
    return
