

# Here the already provided tools for testing the network are
# reorganized into funtions to be used in the notebook when needed



# Plot and store the comparison image between an original 
# sample and the reconstructed one (already provided)
def plot_comparison(net, test_dataset):
    
    # build comparison image
    img = test_dataset[0][0].unsqueeze(0).to(device)
    net.eval()
    with torch.no_grad():
        rec_img  = net(img)
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
    axs[0].set_title('Original image')
    axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
    axs[1].set_title('Reconstructed image (EPOCH %d)' % (epoch + 1))
    plt.tight_layout()
    plt.pause(0.1)
    
    # Save figures
    os.makedirs('autoencoder_progress_%d_features' % encoded_space_dim, exist_ok=True)
    plt.savefig('autoencoder_progress_%d_features/epoch_%d.png' % (encoded_space_dim, epoch + 1))
    plt.show()
    plt.close()
    return





# function to get the encoded representation of some input dataset
def get_encoded_representation(dataset, net):
    encoded_samples = []
    
    for sample in tqdm(dataset):
        img = sample[0].unsqueeze(0)
        label = sample[1]
        
        # Encode image
        net.eval()
        with torch.no_grad():
            encoded_img  = net.encode(img)
            
        # Append to list
        encoded_samples.append((encoded_img.flatten().numpy(), label))
    
    return encoded_samples  




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




def decode_custom_sample(enc_sample, net):
    
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