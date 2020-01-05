
#########################################################
#                   RNN Testing Script                  #
#########################################################
#
# This scripts needs to be in the same directory of 
# the folder "trained_models" and "piano-midi-de"
# to work, but other directories can be passed as arguments
#
# Choose the number of beats to generate and the initial
# conditions:  - 'select'   : Start from the first 2 measures of a selected song
#              - 'sample'   : Start from the first 2 measures of a random song
#                               (For both you should first do the pre-processing 
#                                to convert every song to unmpy object)
#              - 'generate' : Start from scratch (does not work, produce empty track)
#


# Import libraries
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pypianoroll

# Import network structure
from RNN_structure import RNN_two_hands_net


# Check if CUDA cores are available
if torch.cuda.is_available():
    useCuda = True
    device = torch.device('cuda')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    useCuda = False
    device = torch.device('cpu')




########## PARAMETERS (set what you want) ############

mode = 'sample'

sample_R_notes = True            # if True, sample notes from prob distribution
sample_L_notes = True            # if False, take notes with highest probabilities (worse results)

apply_probs_corrections = True   # if True, apply penalization (if argmax) or bonus (if sample) to
                                 # notes that are played for the i-th consecutive time (reduce randomness)

model_dir = 'trained_models/two_hands_net.pth'   # path to trained model
npy_songs_dir = 'numpy_piano-midi-de_two_hands'  # directory with numpy songs
init_seq_name = 'beeth_appass_2.npy'             # if 'select', choose song here

n_beats = 24           # generate 4*n_beats timesteps
init_seq_length = 32   # number of timesteps of initial sequence

######################################################






################### FUNCTIONS ########################

# Function to get the initial sequence following the selected mode
def get_sample(mode):

    # If mode is 'generate': Start from a null sequence
    if (mode == 'generate'):
        sample = np.zeros((2, init_seq_length, 88))

    # If mode is 'select', Start from the beginning of a 
    # selected song of the dataset
    elif (mode == 'select'):

        # Load the song from the dataset
        song = np.load(npy_songs_dir + '/' + init_seq_name)
        print('\nLoaded starting sample:', init_seq_name)   

        # Take its beginning
        sample = song[:, : init_seq_length ]


    # If mode is 'sample', Start from the beginning of a 
    # random song of the dataset
    elif (mode == 'sample'):

        # Pick one song from the dataset
        songs_list = np.array(os.listdir(npy_songs_dir+'/'))
        song_name = np.random.choice(songs_list)
        song = np.load(npy_songs_dir + '/' + song_name)
        print('\nLoaded starting sample:', song_name)

        # Take its beginning
        sample = song[:, : init_seq_length ]

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
        return np.exp(-b*i)


# ---------------------------------------------------

# Bonus value for a note that is sampled that will be multiplied to its prob
# (start from 2 and decrease exponentially to 1)
def bonus_val(i, a=0.25, b=1):
    if (i == 0):
        return 1.0
    else:
        return b*np.exp(-a*i)+1


# ---------------------------------------------------

# Select notes to play using the outputs of the network 
def select_notes_to_play(out_piano_probs, out_num_notes, sample_notes, played_times, correction_weight=0.25):

    # Convert net output to numpy arrays
    piano_probs = F.softmax(out_piano_probs[0,-1,:], dim=0).detach().numpy().squeeze()
    num_notes = out_num_notes[0,-1].detach().numpy().squeeze()
    num_notes = int(num_notes.round())


    # Take the most probable notes (or sample them from the distribution)
    if sample_notes == False:

        # If flag is True, apply penalty to notes that are played for the i-th time
        if apply_probs_corrections: 
            piano_probs = piano_probs * [penalty_val(played_times[i], correction_weight) for i in range(88)]
            #print([penalty_val(played_times[i]) for i in range(88)])

        # Take notes with highest (penalized) probability
        sorted_probs = piano_probs.argsort()[::-1]
        notes = sorted_probs[:num_notes]

    else:

        # if flag is True, apply bonus to notes that have been sampled in previous timesteps
        if apply_probs_corrections:
            piano_probs = piano_probs * [bonus_val(played_times[i], correction_weight) for i in range(88)]
            piano_probs = piano_probs / np.sum(piano_probs)
            #print([bonus_val(played_times[i]) for i in range(88)])
       
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
dropout_prob = 0.4
net = RNN_two_hands_net(dropout_prob)
net.to(device)

net.load_state_dict(torch.load(model_dir, map_location=device))
net.eval()


# Use 'get_sample()' function to build the initial sequence
full_sample = get_sample(mode=mode)
sample = torch.Tensor([full_sample])

# initial conditions
rnn_state = None
R_played_times = np.zeros(88, dtype=int)    # number of times that a note is played consecutively
L_played_times = np.zeros(88, dtype=int)    #  by corresponding hand during sample generation


# Use the network to continue the original sample, producing one (1/16)th at each iteration

for i in range(4*n_beats):

    # Predict num of notes and probs of each notes
    out_R_piano, out_R_num_notes, out_L_piano, out_L_num_notes, rnn_state = net(sample, rnn_state)

    # Use function to select the notes to play given the output of the net (L should play longer notes)
    R_notes = select_notes_to_play(out_R_piano, out_R_num_notes, sample_R_notes, R_played_times)
    L_notes = select_notes_to_play(out_L_piano, out_L_num_notes, sample_L_notes, L_played_times, 0.1)

    print('\nRight hand notes', R_notes)
    print('Left  hand notes', L_notes)

    # Set the sampled notes to 1 and the others to 0
    next_R_piano = np.zeros(88, dtype=int)
    next_L_piano = np.zeros(88, dtype=int)
    next_R_piano[R_notes.astype(int)] = 1
    next_L_piano[L_notes.astype(int)] = 1

    # Prepare sample for next iteration and update played_times of each note (for penalty/bonus)
    sample = torch.Tensor([[[next_R_piano], [next_L_piano]]])
    R_played_times = np.where( next_R_piano, R_played_times+1, 0)
    L_played_times = np.where( next_L_piano, L_played_times+1, 0)

    # Store the notes of the new timestep 
    full_sample = np.append(full_sample, [[next_R_piano],[next_L_piano]] , axis=1)


# Now that we have the full sample, convert it to file midi
new_song = np.zeros((2,len(full_sample[0])*6, 128), dtype=bool)
new_song[:,:,21:109] = np.repeat(full_sample, 6, axis=1)
new_R_midi = pypianoroll.Track(new_song[0], name = 'Right hand')
new_L_midi = pypianoroll.Track(new_song[1], name = 'Left hand')
new_midi = pypianoroll.Multitrack(tracks=[new_R_midi, new_L_midi], tempo=100)
pypianoroll.write(new_midi, 'generated_tracks/gen_two_hands_sample.mid')

