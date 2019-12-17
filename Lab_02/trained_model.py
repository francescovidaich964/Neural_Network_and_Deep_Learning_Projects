
#########################################################
#					NN Testing Script					#
#########################################################
#
# IMPORTANT:
# This scripts needs to be in the same folder of 
# 'winning_net.dat' and 'test_set.py' to work properly
#
# The network class definition ends at line 200, the important
# part of the script (regarding the MSE) starts there
#


# Import needed libraries
import numpy as np
import dill



# -------------------- Define network class -------------------

class Network():
    
    def __init__(self, Ni, Nh1, Nh2, No, act, L1_lambda):
            
        ### WEIGHT INITIALIZATION (Xavier)
        # Initialize hidden weights and biases (layer 1)
        Wh1 = (np.random.rand(Nh1, Ni) - 0.5) * np.sqrt(12 / (Nh1 + Ni))
        Bh1 = np.zeros([Nh1, 1])
        self.WBh1 = np.concatenate([Wh1, Bh1], 1) # Weight matrix including biases
        # Initialize hidden weights and biases (layer 2)
        Wh2 = (np.random.rand(Nh2, Nh1) - 0.5) * np.sqrt(12 / (Nh2 + Nh1))
        Bh2 = np.zeros([Nh2, 1])
        self.WBh2 = np.concatenate([Wh2, Bh2], 1) # Weight matrix including biases
        # Initialize output weights and biases
        Wo = (np.random.rand(No, Nh2) - 0.5) * np.sqrt(12 / (No + Nh2))
        Bo = np.zeros([No, 1])
        self.WBo = np.concatenate([Wo, Bo], 1) # Weight matrix including biases
        
        ### ACTIVATION FUNCTION (given as argument)
        self.act = eval(act)
        self.act_der = eval(act + '_der')
        
        ### Set lambda for L1 REGULARIZATION
        self.L1_lambda = L1_lambda
        
        
    def forward(self, x, additional_out=False):
        
        # Convert to numpy array
        x = np.array(x)
        
        ### Hidden layer 1
        # Add bias term
        X = np.append(x, 1)
        # Forward pass (linear)
        H1 = np.matmul(self.WBh1, X)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        Z1 = np.append(Z1, 1)
        # Forward pass (linear)
        H2 = np.matmul(self.WBh2, Z1)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        Z2 = np.append(Z2, 1)
        # Forward pass (linear)
        Y = np.matmul(self.WBo, Z2)
        # NO activation function
        
        if additional_out:
            return Y.squeeze(), Z2
        
        return Y.squeeze()
        
                                    # If not specified, do not use Gradient Clipping
    def update(self, x, label, lr, clip_norm = np.inf):
        
        # Convert to numpy array
        X = np.array(x)
        
        ### Hidden layer 1
        # Add bias term
        X = np.append(X, 1)
        # Forward pass (linear)
        H1 = np.matmul(self.WBh1, X)
        # Activation function
        Z1 = self.act(H1)
        
        ### Hidden layer 2
        # Add bias term
        Z1 = np.append(Z1, 1)
        # Forward pass (linear)
        H2 = np.matmul(self.WBh2, Z1)
        # Activation function
        Z2 = self.act(H2)
        
        ### Output layer
        # Add bias term
        Z2 = np.append(Z2, 1)
        # Forward pass (linear)
        Y = np.matmul(self.WBo, Z2)
        # NO activation function
        
        # Evaluate the derivative terms
        D1 = Y - label    # derivate of the loss wrt Y
        D2 = Z2
        D3 = self.WBo[:,:-1]
        D4 = self.act_der(H2)
        D5 = Z1
        D6 = self.WBh2[:,:-1]
        D7 = self.act_der(H1)
        D8 = X
        
        # Layer Error
        Eo = D1
        Eh2 = np.matmul(Eo, D3) * D4
        Eh1 = np.matmul(Eh2, D6) * D7
        
        # Derivative for weight matrices (gradients)
        dWBo = np.matmul(Eo.reshape(-1,1), D2.reshape(1,-1))
        dWBh2 = np.matmul(Eh2.reshape(-1,1), D5.reshape(1,-1))
        dWBh1 = np.matmul(Eh1.reshape(-1,1), D8.reshape(1,-1))
        
# NEW   # GRADIENT CLIPPING
        # Compute grad norm; if norm > 'clip_norm', rescale grad to have norm = 'clip_norm'
        grad_norm = np.linalg.norm(np.concatenate([dWBo.flatten(),dWBh2.flatten(),dWBh1.flatten()]))
        if grad_norm >= clip_norm:
            dWBo  = dWBo/grad_norm * clip_norm
            dWBh2 = dWBh2/grad_norm * clip_norm
            dWBh1 = dWBh1/grad_norm * clip_norm
        
        # Update the weights (considering L1 regularization)
        self.WBh1 -= lr * (dWBh1 + self.L1_lambda*np.sign(self.WBh1))
        self.WBh2 -= lr * (dWBh2 + self.L1_lambda*np.sign(self.WBh2))
        self.WBo -= lr * (dWBo + self.L1_lambda*np.sign(self.WBo))
        
        # Evaluate loss function
        loss = (Y - label)**2/2 + self.L1_lambda * ( np.abs(self.WBh1).sum() + 
                                                     np.abs(self.WBh2).sum() + 
                                                     np.abs(self.WBo).sum() )
        
        return loss
    
    
    def plot_weights(self):
    
        fig, axs = plt.subplots(3, 1, figsize=(12,6))
        axs[0].hist(self.WBh1.flatten(), 20)
        axs[1].hist(self.WBh2.flatten(), 50)
        axs[2].hist(self.WBo.flatten(), 20)
        plt.legend()
        plt.grid()
        plt.show()

    
    def train(self, x_train, y_train, x_test, y_test, num_epochs, lr, en_decay=False, lr_final=0, return_log=False, out_log=False):

        # if we want to keep track of the errors, initialize arrays to contain them
        if out_log or return_log: 
            train_loss_log = []
            test_loss_log = []
        
        if en_decay:
            lr_decay = (lr_final / lr)**(1 / num_epochs)
        
        for num_ep in range(num_epochs):
            
            # IF 'en_decay' is true, compute new learning rate for each epoch
            if en_decay:
                lr *= lr_decay
                
            # Train single epoch (sample by sample, no batch for now)
            #train_loss_vec = [self.update(x, y, lr) for x, y in zip(x_train, y_train)]
            train_loss_vec = [self.update(x, y, lr, clip_norm=10) for x, y in zip(x_train, y_train)]
            avg_train_loss = np.mean(train_loss_vec)
            
            # Test network (The test score is given only by the MSE, we don't add the reg term)
            y_test_est = np.array([self.forward(x) for x in x_test])
            avg_test_loss = np.mean((y_test_est - y_test)**2/2) # + self.L1_lambda * ( np.abs(self.WBh1).sum() + 
                                                               #                     np.abs(self.WBh2).sum() + 
                                                               #                     np.abs(self.WBo).sum() ))
            
            # if we want to keep track of errors, add them the their array
            if return_log or out_log:
                train_loss_log.append(avg_train_loss)
                test_loss_log.append(avg_test_loss)
                # IF 'out_log' is true, save and print train/test errors for each epoch
                if out_log:
                    print('Epoch %d - lr: %.5f - Train loss: %.5f - Test loss: %.5f' % (num_ep + 1, lr, avg_train_loss, avg_test_loss))

        # IF 'return_log' is true, return arrays with train/test loss values
        # ELSE return only the final values of the trained network
        if return_log:
            return train_loss_log, test_loss_log
        else:
            return avg_train_loss, avg_test_loss
        




# ------------- Compute loss of the best net --------------------------------

# Import the best network
import dill
with open('winning_net.dat', 'rb') as input: 
    best_net = dill.load(input) 

# Import the test set and split X and Y columns
test_data = np.genfromtxt('test_set.txt', delimiter=',')
x_test = test_data[:,0]
y_test = test_data[:,1]

# Compute the estimated y and the Mean Squared Error
estimated_y =  np.array([best_net.forward(x) for x in x_test])
avg_test_loss = np.mean((estimated_y - y_test)**2)
print("MSE:", avg_test_loss)