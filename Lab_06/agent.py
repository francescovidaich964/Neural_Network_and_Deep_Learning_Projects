import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

class Agent:

    states = 1
    actions = 1
    discount = 0.9
    max_reward = 1
    qtable = np.matrix([1])
    softmax = False
    sarsa = False
    

    # initialize
    def __init__(self, states, actions, discount, max_reward, softmax, sarsa):
        self.states = states
        self.actions = actions
        self.discount = discount
        self.max_reward = max_reward
        self.softmax = softmax
        self.sarsa = sarsa
        # initialize Q table
        self.qtable = np.ones([states, actions], dtype = float) * max_reward / (1 - discount)
        

    # update function (Sarsa and Q-learning)
    def update(self, state, action, reward, next_state, alpha, epsilon):

        # find the next action (greedy for Q-learning, using the choosen policy for SARSA)
        if (self.sarsa):
            next_action = self.select_action(next_state, epsilon)  # eps-greedy or softmax (on-policy)
        else:
            next_action = self.select_action(next_state, 0)  # greedy (off-policy)

        # calculate long-term reward with bootstrap method
        observed = reward + self.discount * self.qtable[next_state, next_action]
        # bootstrap update
        self.qtable[state, action] = self.qtable[state, action] * (1 - alpha) + observed * alpha
        

    # action policy: implements epsilon greedy and softmax
    def select_action(self, state, epsilon):
        qval = self.qtable[state]
        prob = []

        if (self.softmax) and (epsilon!=0):
            # use Softmax distribution
            prob = sp.softmax(qval / epsilon)
        else:
            # assign equal value to all actions
            prob = np.ones(self.actions) * epsilon / (self.actions - 1)
            # the best action is taken with probability 1 - epsilon
            prob[np.argmax(qval)] = 1 - epsilon

        return np.random.choice(range(0, self.actions), p = prob)


    # Function that plots a visualization of the estimated values of each state
    def visualize_values(self, x, y, filename=None):

        #Plot values, computed as the max Q-value of every state
        plt.imshow(self.qtable.max(axis=1).reshape(x,y))
        plt.colorbar()
        plt.title("Value map")
        if filename==None:
            plt.show()
        else:
            plt.savefig(filename)

        