#########################################################
#					NN Testing Script					#
#########################################################
#
# IMPORTANT:
# This scripts needs to be in the same folder of the
# modules 'agent.py', 'environment.py' and 'training.py'
#
# This script will train an agent on a maze (agent can't
# pass through them), it will plot the learnt values of
# each state and it will build an animation of the
# trained agent playing one episode
#



import dill
import numpy as np
import matplotlib.pyplot as plt

import agent
import environment
import training


episodes = 2000         # number of training episodes
episode_length = 168    # maximum episode length
x = 10                  # horizontal size of the box
y = 10                  # vertical size of the box
goal = [0,0]            # objective point
initial = [9,9]         # Initial state (set None to have random ones)

discount = 0.9          # exponential discount factor
softmax = False         # set to true to use Softmax policy
sarsa = False           # set to true to use the Sarsa algorithm

ep_per_epoch = 100      # print results nd store model each 'ep_per_epoch' episodes



# Define the maze
walls = [      [0,1],                              [0,7],
                           [1,3],      [1,5],[1,6],[1,7],[1,8],
               [2,1],[2,2],            [2,5],
         [3,0],[3,1],            [3,4],            [3,7],[3,8],
                           [4,3],            [4,6],      [4,8],
               [5,1],[5,2],[5,3],[5,4],      [5,6],
               [6,1],                        [6,6],      [6,8],[6,9],
                           [7,3],[7,4],[7,5],[7,6],
               [8,1],[8,2],[8,3],      [8,5],            [8,8],
                                       [9,5],      [9,7],[9,8]
        ]



# Initialize the agent (grid with 'x*y' states and 5 possible actions)
learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=True, sarsa=False)

# Train the agent
episodes = 2000
mean_rewards_per_step = training.train_agent(learner, x, y, initial, goal, episodes, 
                                             eps_decay_exp=True, walls=walls, episode_length=168)

# Plot the results
learner.visualize_values(x,y)

# Build GIF of the episode
print('Start building GIF of the episode')
training.play_episode_gif('trained_model_episode.gif', learner, x, y,
                          initial, goal, walls, episode_length=168)