
import dill
import numpy as np
import matplotlib.pyplot as plt
import agent
import environment


# Function that trains the agent in input
def train_agent(learner, x, y, initial, goal, episodes, eps_decay_exp, log=True,
                discount=0.9, ep_per_epoch=100, episode_length=50):
    
    # TODO alpha and epsilon profile
    alpha = np.ones(episodes) * 0.25
    #epsilon = np.linspace(0.8, 0.001,episodes)
    epsilon = np.linspace(0.3, 0.001,episodes)


    # perform the training
    mean_rewards = np.zeros(episodes//ep_per_epoch)
    for index in range(0, episodes):

        # start from a random state if it was set to None
        if initial == None:
            initial = [np.random.randint(0, x), np.random.randint(0, y)]
        # initialize environment
        state = initial
        env = environment.Environment(x, y, state, goal)
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
                print('Episodes '+str(index-99)+'-'+str(index+1)+': Mean reward per step is', np.round(mean_rewards[index//ep_per_epoch],4))
            
    return mean_rewards




# Function that train an agent using all the possible combinations
# of algorithm (Q-learning/SARSA) and policy (eps-greedy/softmax)
def compare_algorithms_and_policies(x, y, initial, goal, episodes, discount=0.9, ep_per_epoch=100):

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
        mean_rewards_per_step = train_agent(learner, x, y, initial, goal, episodes, log=False, \
                                            discount=discount, ep_per_epoch=ep_per_epoch)
        plt.plot( np.arange(episodes//ep_per_epoch)*ep_per_epoch, mean_rewards_per_step, label=combination)
        print('Final value of mean reward per step is', np.round(mean_rewards_per_step[-1],4))
        
    # Plot evolution of mean reward per step during training of each combination
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward per step')
    plt.legend()
    plt.show()