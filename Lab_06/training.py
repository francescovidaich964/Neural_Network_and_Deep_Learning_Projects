
import os
import dill
import numpy as np
import matplotlib.pyplot as plt
import imageio

import agent
import environment



# Function that trains the agent in input
def train_agent(learner, x, y, initial, goal, episodes, eps_decay_exp=False, 
                walls=None, wall_penal=None, log=True, discount=0.9, 
                ep_per_epoch=100, episode_length=50):
    
    # alpha profile
    alpha = np.ones(episodes) * 0.25

    # Define epsilon-decay process
    if eps_decay_exp:
        dec_factor = pow( 0.001/0.8, 1/(episodes-1) ) 
        epsilon = np.array([0.8 * (dec_factor)**i for i in range(episodes)]) 
    else: 
        epsilon = np.linspace( 0.8, 0.001,episodes )


    # perform the training
    mean_rewards = np.zeros(episodes//ep_per_epoch)
    for index in range(0, episodes):

        # start from a random state if it was set to None
        if initial == None:
            initial = [np.random.randint(0, x), np.random.randint(0, y)]
        # initialize environment
        state = initial
        env = environment.Environment(x, y, state, goal, walls, wall_penal)
        reward = 0

        # run episode
        for step in range(0, episode_length):

            # find state index
            state_index = state[0] * y + state[1]
            # choose an action
            action = learner.select_action(state_index, epsilon[index])
            # the agent moves in the environment
            result = env.move(action)
            # Q-learning update
            next_index = result[0][0] * y + result[0][1]
            learner.update(state_index, action, result[1], next_index, alpha[index], epsilon[index])
            # update state and reward
            reward += result[1]
            state = result[0]

        reward /= episode_length
        mean_rewards[index//ep_per_epoch] += reward/ep_per_epoch

        # periodically save the agent and print results
        if ((index + 1) % ep_per_epoch == 0):
            with open('agent.obj', 'wb') as agent_file:
                dill.dump(agent, agent_file)
            
            if log:
                print('Episodes '+str(index-99)+'-'+str(index+1)+': Mean reward per step is', \
                        np.round(mean_rewards[index//ep_per_epoch],4))
            
    return mean_rewards





# Function that train an agent using all the possible combinations
# of algorithm (Q-learning/SARSA) and policy (eps-greedy/softmax)
def compare_algorithms_and_policies(x, y, initial, goal, episodes, eps_decay_exp=False, 
                                    walls=None, wall_penal=None, discount=0.9, 
                                    ep_per_epoch=100, episode_length=50):
    # Test the 4 possible combinations of algorithm and policy
    for case in range(4):
        
        # Set algorithm
        sarsa = case//2 
        if sarsa:
            algorithm = 'SARSA'
        else:
            algorithm = 'Q-learning'
            
        # Set policy
        softmax = case%2
        if softmax:
            policy = 'Softmax'
        else:
            policy = 'eps-greedy'
        

        # Initialize the agent (grid with 'x*y' states and 5 possible actions)
        combination = algorithm +' + '+ policy
        print('\nTraining agent using ', algorithm+' + '+policy)
        learner = agent.Agent((x * y), 5, discount, max_reward=1, softmax=softmax, sarsa=sarsa)
        
        # Train model and store the Mean reward per step
        mean_rewards_per_step = train_agent(learner, x, y, initial, goal, episodes, eps_decay_exp,    \
                                            walls, wall_penal, log=False, discount=discount,          \
                                            ep_per_epoch=ep_per_epoch, episode_length=episode_length)
        plt.plot( np.arange(episodes//ep_per_epoch)*ep_per_epoch, mean_rewards_per_step, label=combination)
        print('Final value of mean reward per step is', np.round(mean_rewards_per_step[-1],4))
        
    # Plot evolution of mean reward per step during training of each combination
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward per step')
    plt.legend()
    plt.show()





# Function that builds a gif of one episode
# During this test, a greedy policy will be used
def play_episode_gif(gif_filename, learner, x, y, initial, goal,
                     walls=None, wall_penal=None, episode_length=50):

    # Start from a random state if it was set to None
    if initial == None:
        initial = [np.random.randint(0, x), np.random.randint(0, y)]
    # initialize environment
    state = initial
    env = environment.Environment(x, y, state, goal, walls, wall_penal)
    reward = 0

    # Define the environment matrix to plot it in the gif
    maze = np.zeros((x,y))
    maze[goal[0], goal[1]] = 0.5
    maze[initial[0], initial[1]] = 0.5
    if walls != None:
        maze[np.asarray(walls).T[0], np.asarray(walls).T[1]] = 1

    # Prepare temp folder that will contain the gif frames
    os.makedirs('tmp_gif_dir')
    still_steps = 0

    # run episode
    for step in range(0, episode_length):

        # Store frame of the current step (if the 
        # agent does not still for too long)
        if still_steps < 5:
            filename = 'tmp_gif_dir/image_'+str(step).zfill(3)+'.png'
            plt.imshow(maze, cmap='Greys')
            plt.plot(state[1], state[0], 'o', c='red', markersize=11)
            plt.savefig(filename)

        # find state index
        state_index = state[0] * y + state[1]
        # choose an action (greedy policy)
        action = learner.select_action(state_index, 0)
        # the agent moves in the environment
        result = env.move(action)
        # update state and reward
        reward += result[1]
        state = result[0]

        # If the agent didn't move, increase "still_steps" counter
        if action == 0:
            still_steps += 1
        else:
            still_steps = 0

    # Compute Mean reward per step
    reward /= episode_length
    print('Mean reward per step =', reward)

    # Order all filenames
    all_filenames = os.listdir('tmp_gif_dir')
    all_filenames = sorted(all_filenames)

    # Build the GIF
    images = []
    for filename in all_filenames:
        images.append(imageio.imread('tmp_gif_dir/'+filename))
  
    # Store the GIF
    imageio.mimsave(gif_filename, images, format='GIF', duration=0.5)
    print('GIF is stored as '+gif_filename)
    plt.close()
    
    # Delete temp files and folder
    for filename in all_filenames:
        os.remove('tmp_gif_dir/'+filename)
    os.rmdir('tmp_gif_dir/')


