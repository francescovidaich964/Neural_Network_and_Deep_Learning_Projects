
#########################################################
#					RNN Testing Script					#
#########################################################
#
# This scripts needs to be in the same directory of 
# the folder "trained_models" and "piano-midi-de"
# to work, but other directories can be passed as arguments
#
# Choose the number of beats to generate and the initial
# conditions:  - 'select'   : Start from the first 2 measures of a selected song
#              - 'sample'   : Start from the first 2 measures of a random song
#                               (You should first do the pre-processing to 
#                                convert every song to unmpy object)
#


# Import libraries
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import pypianoroll

# Import network structure
from RNN_structure import RNN_net


# Check if CUDA cores are available
if torch.cuda.is_available():
    useCuda = True
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    useCuda = False
    device = torch.device('cpu')




########## PARAMETERS (set what you want) ############


parser = argparse.ArgumentParser(description='Generate a sample starting from the beginnig of one song of the dataset')

parser.add_argument('--seed', type=str, default=None, help='Choose the starting song from the pre-processed dataset (random if not used)')
parser.add_argument('--model_dir', type=str, default='trained_models/first_net.pth', help='Network model directory')
parser.add_argument('--npy_dataset', type=str, default='numpy_piano-midi-de', help='Pre-processed dataset directory')

parser.add_argument('--seed_length', type=int, default=32, help='Number of timesteps of initial sequence')
parser.add_argument('--length', type=int, default=96, help='Number of timesteps to be generated (i.e. length/4 beats of music)')


# Don't change from here to obtain best results

sample_notes = True              # if True, sample notes (do not pick ones with higher probs)
apply_probs_corrections = True   # if True, apply penalization (if argmax) or bonus (if sample)
                                 # to notes that are played for the i-th consecutive time

######################################################




# Store all args in old variables
args = parser.parse_args()

model_dir = args.model_dir
npy_songs_dir = args.npy_dataset
init_seq_length = args.seed_length
timesteps_to_generate = args.length

if args.seed==None:
    mode = 'sample_song'
else:
    mode = 'select_song'
    init_seq_name = args.seed




################### FUNCTIONS ########################

# Function to get the initial sequence following the selected mode
def get_sample(mode):

    # If mode is 'select', Start from the beginning of a 
    # selected song of the dataset
    if (mode == 'select_song'):

        # Load the song from the dataset
        song = np.load(npy_songs_dir + '/' + init_seq_name)
        print('\nLoaded starting sample:', init_seq_name)   

        # Take its beginning
        sample = song[ : init_seq_length ]


    # If mode is 'sample', Start from the beginning of a 
    # random song of the dataset
    elif (mode == 'sample_song'):

        # Pick one song from the dataset
        songs_list = np.array(os.listdir(npy_songs_dir+'/'))
        song_name = np.random.choice(songs_list)
        song = np.load(npy_songs_dir + '/' + song_name)
        print('\nLoaded starting sample:', song_name)

        # Take its beginning
        sample = song[ : init_seq_length ]

    # If the mode is not allowed, print error
    else:
        print('\nThe typed mode is not allowed')

    return sample


# ---------------------------------------------------

# Penalty value for a note that is played for the i-th consecutive time
# (start from 1 and decreases over time, they will be multiplied to probs)
def penalty_val(i, a=50, b=0.5):
    if (i == 0):
        return 1.0
    else:
        #return (1-i/a) / ((1+b)-i/a) + b/(1+b)
        return np.exp(-b*i)


# ---------------------------------------------------

# Bonus value for a note that is sampled that will be multiplied to its prob
# (start from 2 and decrease exponentially to 1)
def bonus_val(i, a=0.25):
    if (i == 0):
        return 1.0
    else:
        return np.exp(-a*i)+1


# ---------------------------------------------------

# Select notes to play using the outputs of the network 
def select_notes_to_play(out_piano_probs, out_num_notes, play_times):

    # Convert net output to numpy arrays
    piano_probs = F.softmax(out_piano_probs[0,-1,:], dim=0).detach().numpy().squeeze()
    num_notes = out_num_notes[:,-1].detach().numpy().squeeze()
    num_notes = int(num_notes.round())


    # Take the most probable notes (or sample them from the distribution)
    if sample_notes == False:

        # If flag is True, apply penalty to notes that are played for the i-th time
        if apply_probs_corrections: 
            piano_probs = piano_probs * [penalty_val(play_times[i]) for i in range(88)]
            #print([penalty_val(times_for_penal[i]) for i in range(88)])

        # Take notes with highest (penalized) probability
        sorted_probs = piano_probs.argsort()[::-1]
        notes = sorted_probs[:num_notes]

    else:

        # if flag is True, apply bonus to notes that have been sampled in previous timesteps
        if apply_probs_corrections:
            piano_probs = piano_probs * [bonus_val(play_times[i]) for i in range(88)]
            piano_probs = piano_probs / np.sum(piano_probs)
            #print([bonus_val(play_times[i]) for i in range(88)])
       
        notes = np.empty(num_notes)

        # Sample 'num_notes' different notes from the piano distribution
        for j in range(num_notes):

            # Sample notes until a new one is picked
            while True:
                notes[j] = np.random.choice(np.arange(88), p=piano_probs)
                if np.all(notes[j] != notes[:j]):
                    break

    return notes


###################################################







################# MAIN PROGRAM ####################

# Load trained network (use same network parameters)
hidden_units = 128
layers_num = 2
dropout_prob = 0.4
net = RNN_net(hidden_units, layers_num, dropout_prob)
net.to(device)

net.load_state_dict(torch.load(model_dir, map_location=device))
net.eval()


# Use 'get_sample()' function to build the initial sequence
full_sample = get_sample(mode=mode)
sample = torch.Tensor([full_sample])

# initial conditions
rnn_state = None
play_times = np.zeros(88, dtype=int)


# Use the network to continue the original sample, producing one (1/16)th at each iteration

for i in range(timesteps_to_generate):

    # Predict num of notes and probs of each notes
    out_piano, out_num_notes, rnn_state = net(sample, rnn_state)

    # Use function to select the notes to play given the output of the net
    notes = select_notes_to_play(out_piano, out_num_notes, play_times)

    # Set the sampled notes to 1 and the others to 0
    next_piano = np.zeros(88, dtype=int)
    next_piano[notes.astype(int)] = 1

    # Prepare sample for next iteration and update play_times of each note (for penalty)
    sample = torch.Tensor([[next_piano]])
    play_times = np.where(next_piano==0, 0, play_times+1)

    # Store the notes of the new timestep
    full_sample = np.append(full_sample, [next_piano], axis=0)


# Now that we have the full sample, convert it to file midi
new_song = np.zeros((len(full_sample)*6, 128), dtype=bool)
new_song[:,21:109] = np.repeat(full_sample, 6, axis=0)
new_midi = pypianoroll.Track(new_song, name='Notes')
new_midi = pypianoroll.Multitrack(tracks=[new_midi], tempo=100)
pypianoroll.write(new_midi, 'generated_tracks/gen_orig_sample.mid')
print()

# Print the generted midi song
low_note, high_note = new_midi.get_active_pitch_range()
fig, axs = new_midi.plot(xtick='step')
axs[0].set_ylim(low_note-6,high_note+6)
plt.rcParams["figure.figsize"] = [9,5]
plt.savefig('generated_tracks/plot_gen_orig_sample.png')