
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data



####################################################################
############# ORIGINAL RNN (no multitrack for hands) ###############
####################################################################

class RNN_net(nn.Module):
    
    def __init__(self, lstm_hidden_units, lstm_layers_num, dropout_prob=0, 
                 dense_piano_units=100, dense_notes_units=10):
        super().__init__()
        
        # Define recurrent layer
        self.rnn = nn.LSTM(input_size = 88, 
                           hidden_size = lstm_hidden_units,
                           num_layers = lstm_layers_num,
                           dropout = dropout_prob,
                           batch_first = True)
       
        # Define dropout layer
        self.drop = nn.Dropout(dropout_prob)
        
        # Define Dense layers before outputs
        self.dense_piano = nn.Linear(lstm_hidden_units, dense_piano_units)
        self.dense_notes = nn.Linear(lstm_hidden_units, dense_notes_units)
        
        # Define output layers
        self.out_piano = nn.Linear(dense_piano_units, 88)
        self.out_num_notes = nn.Linear(dense_notes_units, 1)
        

    def forward(self, x, state=None):
        # LSTM
        x, rnn_state = self.rnn(x, state)
        # Dense layers
        x_piano = F.leaky_relu(self.dense_piano(x))
        x_num_notes = F.leaky_relu(self.dense_notes(x))
        # Output estimation
        x_piano = self.out_piano(x_piano)
        x_num_notes = self.out_num_notes(x_num_notes)
        
        return x_piano, x_num_notes, rnn_state
    

    def train_batch(self, sample_batch, next_step_batch, num_notes_batch, loss_fn_piano,
                    loss_fn_num_notes, optimizer, loss_coeff=1, log=False):
            
        ### Forward pass
        optimizer.zero_grad()
        net_out_piano, net_out_num_notes, _ = self.forward(sample_batch)
        
        ### Update network
        # Evaluate losses only for last output (final loss is given by the sum)
        loss_piano = loss_fn_piano(net_out_piano[:, -1, :], next_step_batch)
        loss_num_notes = loss_fn_num_notes(net_out_num_notes[:, -1].squeeze(), num_notes_batch)
        loss = loss_coeff*loss_piano + loss_num_notes
        
        if log==True:
            print('Loss for next step piano:         ', loss_piano)
            print('Loss for next step nuber of notes:', loss_num_notes)
            
        # Backward pass and update
        loss.backward()
        optimizer.step()
        
        # Return average batch loss
        return float(loss_piano.data), float(loss_num_notes.data)




################################################################################
############# MULTITRACK RNN (hands are elaborated independently) ##############
################################################################################


class RNN_two_hands_net(nn.Module):
    
    def __init__(self,  dropout_prob=0, lstm_hidden_units=150, lstm_layers_num=2,
                 dense_hand_units=120, dense_piano_units=100, dense_notes_units=10):
        super().__init__()
        
        # Define recurrent layer
        self.rnn = nn.LSTM(input_size = 176, 
                           hidden_size = lstm_hidden_units,
                           num_layers = lstm_layers_num,
                           dropout = dropout_prob,
                           batch_first = True)
       
        # Define dropout layer
        self.drop = nn.Dropout(dropout_prob)
        
        # Define Dense layers before outputs
        self.dense_right = nn.Linear(lstm_hidden_units, dense_hand_units)
        self.dense_left = nn.Linear(lstm_hidden_units, dense_hand_units)
        self.dense_R_piano = nn.Linear(dense_hand_units, dense_piano_units)
        self.dense_R_notes = nn.Linear(dense_hand_units, dense_notes_units)
        self.dense_L_piano = nn.Linear(dense_hand_units, dense_piano_units)
        self.dense_L_notes = nn.Linear(dense_hand_units, dense_notes_units)
        
        # Define output layers
        self.out_R_piano = nn.Linear(dense_piano_units, 88)
        self.out_R_num_notes = nn.Linear(dense_notes_units, 1)
        self.out_L_piano = nn.Linear(dense_piano_units, 88)
        self.out_L_num_notes = nn.Linear(dense_notes_units, 1)
    

    def forward(self, x, state=None):

        # Concatenate R and L input for LSTM input (concatenate last dimension)
        x = torch.cat( (x[:,0], x[:,1]), -1)
        
        # LSTM
        x, rnn_state = self.rnn(x, state)

        # --- Branch for right hand ---
        x_R = F.leaky_relu(self.dense_right(x))
        x_R_piano = F.leaky_relu(self.dense_R_piano(x_R))
        x_R_num_notes = F.leaky_relu(self.dense_R_notes(x_R))
        # Output for right hand
        x_R_piano = self.out_R_piano(x_R_piano)
        x_R_num_notes = self.out_R_num_notes(x_R_num_notes)
        
        # --- Branch for left hand ---
        x_L = F.leaky_relu(self.dense_left(x))
        x_L_piano = F.leaky_relu(self.dense_L_piano(x_L))
        x_L_num_notes = F.leaky_relu(self.dense_L_notes(x_L))
        # Output for left hand
        x_L_piano = self.out_L_piano(x_L_piano)
        x_L_num_notes = self.out_L_num_notes(x_L_num_notes)

        return x_R_piano, x_R_num_notes, x_L_piano, x_L_num_notes, rnn_state
    


    def train_batch(self, sample_batch, next_step_batch, num_notes_batch, 
                    loss_fn_R_piano, loss_fn_L_piano, loss_fn_num_notes, 
                    optimizer, loss_coeff=1, log=False):
            
        ### Forward pass
        optimizer.zero_grad()
        net_out_R_piano, net_out_R_num_notes, net_out_L_piano, net_out_L_num_notes, _ = self.forward(sample_batch)
        
        ### Update network
        # Evaluate losses only for last output (final loss is given by the sum of all losses)
        loss_R_piano = loss_fn_R_piano(net_out_R_piano[:, -1, :], next_step_batch[:,0])       
        loss_L_piano = loss_fn_L_piano(net_out_L_piano[:, -1, :], next_step_batch[:,1])
        loss_R_num_notes = loss_fn_num_notes(net_out_R_num_notes[:, -1].squeeze(), num_notes_batch[:,0])
        loss_L_num_notes = loss_fn_num_notes(net_out_L_num_notes[:, -1].squeeze(), num_notes_batch[:,1])
        loss = loss_coeff*(loss_R_piano+loss_L_piano) + (loss_R_num_notes+loss_L_num_notes)
        
        if log==True:
            print('Loss for next step right piano:          ', loss_R_piano)
            print('Loss for next step right number of notes:', loss_R_num_notes)
            print('Loss for next step left piano:           ', loss_L_piano)
            print('Loss for next step left number of notes: ', loss_L_num_notes)
            
        # Backward pass and update
        loss.backward()
        optimizer.step()
        
        # Return average batch losses
        loss_piano = [float(loss_R_piano.data), float(loss_L_piano.data)]
        loss_notes = [float(loss_R_num_notes.data), float(loss_L_num_notes.data)]
        return loss_piano, loss_notes

