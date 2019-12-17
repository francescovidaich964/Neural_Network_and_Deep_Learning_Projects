
#########################################################
#					RNN Testing Script					#
#########################################################
#
# This scripts needs to be in the same directory of 
# the folder "trained_models" and "piano-midi-de"
# to work, but other directories can be passed as arguments
#
# Choose the number of beats to generate and the initial
# conditions:  - 'sample'   : Start from the first 2 measures of a random song
#                               (You should first do the pre-processing to 
#                                convert every song to unmpy object)
#              - 'generate' : Start from scratch 
#   maybe no   - 'custom'   : Start from a custom midi file
#


# Import libraries
import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pypianoroll

# Import network structure
from RNN_structure import RNN_net


# Read parameters 
parser = argparse.ArgumentParser(description='Generate sample starting from a given text')

parser.add_argument('--n_beats', type=int, default=24, help='Number of beats to generate')
parser.add_argument('--mode', type=str, default='sample', help='Sample generation mode')
parser.add_argument('--model_dir', type=str, default='trained_models/first_net.pth', help='Network model directory')
parser.add_argument('--npy_songs_dir', type=str, default='numpy_piano-midi-de', help='Numpy songs directory (needed for mode "sample")')
parser.add_argument('--init_seq_length', type=int, default=32, help='Length of the sequence taken in input by the network')
parser.add_argument('--sample_notes', type=bool, default=False, help='Policy to choose notes')

# Parse input arguments
args = parser.parse_args()


# Check if CUDA cores are available
if torch.cuda.is_available():
    useCuda = True
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    useCuda = False
    device = torch.device('cpu')




# Load trained network (use same network parameters)
hidden_units = 128
layers_num = 2
dropout_prob = 0.4
net = RNN_net(hidden_units, layers_num, dropout_prob)
net.to(device)

net.load_state_dict(torch.load(args.model_dir, map_location=device))
net.eval()



###### GET ORIGINAL SAMPLE with the selected mode ######

# If mode is 'generate': Start from a null sequence
if (args.mode == 'generate'):
    full_sample = np.zeros((args.init_seq_length, 88))

# If mode is 'sample', Start from the beginning of a song in the dataset
elif (args.mode == 'sample'):

    # Pick one song from the dataset
    songs_list = np.array(os.listdir(args.npy_songs_dir+'/'))
    song_name = np.random.choice(songs_list)
    song = np.load(args.npy_songs_dir + '/' + song_name)
    print('Loaded starting sample:', song_name)

    # Take its beginning
    full_sample = song[ : args.init_seq_length ]

# If the mode is not allowed, print error
else:
    print('The typed mode is not allowed')


#######################################################



# Use the network to continue the original sample by
# producing one (1/16)th at each iteration 

out_state = None

for i in range(4*args.n_beats):

    # Predict num of notes and probs of each notes
    sample = torch.Tensor([full_sample[ -args.init_seq_length : ]])
    out_piano, out_num_notes, out_state = net(sample, out_state)
    #print(out_piano)

    next_piano_probs = nn.Softmax()(out_piano[:,-1,:]).detach().numpy().squeeze()
    print(np.sort(next_piano_probs))
    next_num_notes = out_num_notes[:,-1].detach().numpy().squeeze()
    num_notes = int(next_num_notes.round())


    # Take the most probable notes (sampling them from the out distribution)
    if args.sample_notes == False:
        notes = next_piano_probs.argsort()[-num_notes:]
    else:
        notes = np.empty(num_notes)
        for j in range(num_notes):
            # Sample notes until a new one is picked
            while True:
                notes[j] = np.random.choice(np.arange(88), p=next_piano_probs)
                if np.all(notes[j] != notes[:j]):
                    break


    # set the sampled notes to 1 and the others to 0
    print(notes)
    next_piano = np.zeros(88, dtype=int)
    next_piano[notes.astype(int)] = 1
    print(next_piano)

    # store the notes of the new timestep and
    full_sample = np.append(full_sample, [next_piano], axis=0)


# Now that we have the full sample, convert it to file midi
new_song = np.zeros((len(full_sample)*6, 128))
new_song[:,21:109] = np.repeat(full_sample, 6, axis=0)
new_midi = pypianoroll.Track(new_song)
new_midi = pypianoroll.Multitrack(tracks=[new_midi])
pypianoroll.write(new_midi, 'generated_tracks/gen_sample.mid')