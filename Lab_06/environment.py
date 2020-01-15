import numpy as np

class Environment:

    state = []
    goal = []
    boundary = []
    action_map = {
        0: [0, 0],
        1: [0, 1],
        2: [0, -1],
        3: [1, 0],
        4: [-1, 0],
    }

    
    # Init function: I added the possibility to build some unsurmountable walls
    # inside the grid (if wall_penal is specified, walls become surmountable but
    # if the agent decides to do so it will penalized with that value as a reward)
    def __init__(self, x, y, initial, goal, walls=None, wall_penal=None):
        self.boundary = np.asarray([x, y])
        self.state = np.asarray(initial)
        self.goal = goal
        self.walls = walls
        self.wall_penal = wall_penal
    

    # the agent makes an action (0 is stay, 1 is up, 2 is down, 3 is right, 4 is left)
    def move(self, action):
        
        # start by default move
        reward = 0
        movement = self.action_map[action]
        
        # check if it is a goal
        if (action == 0 and (self.state == self.goal).all()):
            reward = 1
        next_state = self.state + np.asarray(movement)
        
        # Check if position is allowed or if the agent has to be penalized
        # (First if: check boundaries and unsurmountable walls
        #  Second if: check surmountable walls )
        if (self.check_boundaries(next_state)):
            reward = -1
        elif (self.wall_penal!=None) and (self.check_walls(next_state)):
            reward = self.wall_penal
            self.state = next_state
        else:
            self.state = next_state
        
        return [self.state, reward]


    # map action index to movement and check that new position is allowed
    def check_boundaries(self, state):
        out = len([num for num in state if num < 0])
        out += len([num for num in (self.boundary - np.asarray(state)) if num <= 0])
        if self.walls!=None and self.wall_penal==None:
            out += self.check_walls(state)
        return out > 0

    # Function that checks if the position is the same of a wall
    def check_walls(self, state):
        return np.any(np.all(np.asarray(state)==np.asarray(self.walls),axis=1)) 