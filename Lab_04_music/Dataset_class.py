

import os
import numpy as np
from torch.utils import data



##############################################################################
############# STANDARD DATASET CLASS (no multitrack for hands) ###############
##############################################################################

class Piano_Dataset(data.Dataset):
    
    ### Initialize the dataset loading all the songs stored before as numpy objects
    def __init__(self, filepath, transpose=[-2,-1,0,1,2], sample_length=32):
       
        # initialize empty members of the dataset
        self.all_songs = {}
        self.IDs = np.array([])
        self.lengths = np.array([])
        self.indexes = np.array([])
        self.sample_length = sample_length
        
        ### Get songs, their IDs, lengths and possible sequences
        for song_name in os.listdir(filepath+'/'):
            
            # Load song from the stored numpy object
            song = np.load(filepath+'/'+song_name)
            
            song_length = len(song)

            # Fill the dataset with all the song transpositions
            for tune in transpose:
                
                # *** Check that we don't lose any notes with the transposition
                # If YES, do not save transposed song
                # If NO, compute the transposed song and store it
                
                transp_song = np.zeros((song_length,88), dtype=int)
                
                if (tune == 0):
                    transp_song = song
                    transp_name = song_name[:-4]+'_orig'
                
                elif (tune > 0):
                    if np.any(song[:,-tune:]): # ***
                        break
                    transp_song[:,tune:] = np.roll(song, tune, axis=1)[:,tune:]
                    transp_name = song_name[:-4]+'_+'+str(tune)
                
                else:
                    if np.any(song[:,:-tune]): # ***
                        break
                    transp_song[:,:tune] = np.roll(song, tune, axis=1)[:,:tune]
                    transp_name = song_name[:-4]+'_'+str(tune)
                    
                # update class members
                self.all_songs[transp_name] = transp_song
                self.IDs = np.append(self.IDs, song_name)
                self.lengths = np.append(self.lengths, song_length)

                # build list of indexes to identify each possible sample 
                # that can be extracted from the song (and will be used as input)
                possible_sequences = np.array([(transp_name, i) for i in range(song_length-(sample_length+1))] )
                if len(self.indexes)==0:
                    self.indexes = possible_sequences
                else:    
                    self.indexes = np.append(self.indexes, possible_sequences, axis=0)

                
    ### Function that returns the number of possible sequences
    def __len__(self):
            return len(self.indexes)
        

    ### Function that returns all the contained songs
    def get_songs(self):
        return self.all_songs
    
        
    ### Return the correct sample given the index
    def __getitem__(self, idx):
        
        # Get the corresponding sample, the following step and its active notes
        song_name, start = self.indexes[idx]
        start = int(start)
        song = self.all_songs[song_name]
        
        sample = song[ start : start+self.sample_length]
        following_step = song[start + (self.sample_length+1)]
        active_notes = sum(following_step)
        
        # Return samples as 'float32' variables (because it is the dtype of torch default tensor)
        return sample.astype('float32'), following_step.astype('float32'), active_notes.astype('float32')
    
    
    ### Compute weigths to pass to the BCEWithLogitsLoss() in order to give
    ### more importance to notes that are less frequently played in the samples
    def get_pos_weights(self):
        
        all_active_notes = np.ones(88, dtype=int) # Start from 1 to avoid having
        all_notes = 0                             # elements equal to 0

        #for song in os.listdir('numpy_piano-midi-de'): OLD
        for name, song in self.all_songs.items():

            # Read midi file and get its lowest and highest note
            all_active_notes += sum(song)
            all_notes += len(song)

        return (all_notes - all_active_notes) / all_active_notes





###########################################################################################
############# MULTITRACK DATASET CLASS (hands are stored in separate tracks) ##############
###########################################################################################

class Piano_Dataset_two_hands(data.Dataset):
    
    ### Initialize the dataset loading all the songs stored before as numpy objects
    def __init__(self, filepath, transpose=[-1,0,1], sample_length=32):
       
        # initialize empty members of the dataset
        self.all_songs = {}
        self.lengths = np.array([])
        self.indexes = np.array([])
        self.sample_length = sample_length
        
        ### Get songs, their IDs, lengths and possible sequences
        for song_name in os.listdir(filepath+'/'):
            
            # Load song from the stored numpy object
            song = np.load(filepath+'/'+song_name)
            song_length = len(song[0])


            # Fill the dataset with all the song transpositions
            for tune in transpose:
                
                # *** Program checks that we don't lose any notes with the transposition
                # If YES, do not save transposed song
                # If NO, compute the transposed song and store it
            
                transp_song = np.zeros((2,song_length,88), dtype=int)

                if (tune == 0):
                    transp_song = song
                    transp_name = song_name[:-4]+'_orig'
                
                elif (tune > 0):
                    if np.any(song[:,:,-tune:]): # ***
                        break
                    transp_song[0,:,tune:] = np.roll(song[0], tune, axis=1)[:,tune:]
                    transp_song[1,:,tune:] = np.roll(song[1], tune, axis=1)[:,tune:]
                    transp_name = song_name[:-4]+'_+'+str(tune)
                
                else:
                    if np.any(song[:,:,:-tune]): # ***
                        break
                    transp_song[0,:,:tune] = np.roll(song[0], tune, axis=1)[:,:tune]
                    transp_song[1,:,:tune] = np.roll(song[1], tune, axis=1)[:,:tune]
                    transp_name = song_name[:-4]+'_'+str(tune)
                    
                # Update class members
                self.all_songs[transp_name] = transp_song
                self.lengths = np.append(self.lengths, song_length)

                # Build list of indexes to identify each possible sample 
                # that can be extracted from the song (and will be used as input)
                possible_sequences = np.array([(transp_name, i) for i in range(song_length-(sample_length+1))] )
                if len(self.indexes)==0:
                    self.indexes = possible_sequences
                else:    
                    self.indexes = np.append(self.indexes, possible_sequences, axis=0)

                
    ### Function that returns the number of possible sequences
    def __len__(self):
            return len(self.indexes)
        
    
    ### Function that returns all the contained songs
    def get_songs(self):
        return self.all_songs
    

    ### Return the correct sample given the index
    def __getitem__(self, idx):
        
        # Get the corresponding sample, the following step and its active notes
        song_name, start = self.indexes[idx]
        start = int(start)
        song = self.all_songs[song_name]
        
        sample = song[:, start : start+self.sample_length]
        following_step = song[:, start + (self.sample_length+1)]
        active_notes = np.array([sum(following_step[0]), sum(following_step[1])])
        
        # Return samples as 'float32' variables (because it is the dtype of torch default tensor)
        return sample.astype('float32'), following_step.astype('float32'), active_notes.astype('float32')

    
    ### Compute weigths to pass to the BCEWithLogitsLoss() in order to give
    ### more importance to notes that are less frequently played in the samples
    def get_pos_weights(self):
        
        all_right_active_notes = np.ones(88, dtype=int) # Start from 1 to avoid having
        all_left_active_notes = np.ones(88, dtype=int) # elements equal to 0
        all_notes = 0

        #for song in os.listdir('numpy_piano-midi-de'): OLD
        for name, song in self.all_songs.items():

            # Read midi file and get its lowest and highest note
            all_right_active_notes += sum(song[0])
            all_left_active_notes  += sum(song[1])
            all_notes += len(song[0])  # equal to number of timesteps

        # Compute weights for each note as  '# of zeros' / '# of ones'
        right_weights = (all_notes - all_right_active_notes) / all_right_active_notes
        left_weights = (all_notes - all_left_active_notes) / all_left_active_notes

        return right_weights, left_weights

