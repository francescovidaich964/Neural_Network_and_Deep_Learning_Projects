
# THIS SCRIPT IS NEEDED TO SAVE THE PRE-PROCESSED
# MIDI SONGS AS NUMPY OBJECTS (to be sampled with Dataloader)

import os
import numpy as np
import pypianoroll



# For each song of each composer, preprocess the midi files and save them as np objects  
for composer in os.listdir('piano-midi-de'):
    for song_name in os.listdir('piano-midi-de/' + composer):    

        # Upload midi song
        song = pypianoroll.parse('piano-midi-de/' + composer + '/' + song_name)

        # Remove all tracks from midi files apart from 'right hand' and 'left_hand'
        while len(song.tracks)>2:
            song.remove_tracks(2)

        # Binarize the intensity of the notes
        song = pypianoroll.binarize(song)

        # Merge 'right hand' and 'left_hand' tracks (produce numpy obj)
        npy_song = song.get_merged_pianoroll(mode='any')

        # Reduce data size: - Reduce time resolution -> 1 step = (1/16)th of a measure
        # 					- Crop pitch range to grand piano keys
        npy_song = npy_song[::6,21:109]#.astype('int')

        # Store numpy object
        # (sparse matrix is stored instead of 'on' indexes because in this way
        #  the dataloader takes less time to build the batches)
        npy_filename = 'numpy_piano-midi-de/' + composer + '_' + song_name[:-4] + '.npy'
        np.save(npy_filename, npy_song)
