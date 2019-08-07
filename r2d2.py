# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# connect to R2D2 
from client import DroidClient
droid = DroidClient()
droid.connect_to_droid('Q5-F43E')
droid.animate(5)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
import numpy as np 
import random 
import copy 
import robot_util

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class MarkovDecisionProcess(object): 
    def __init__(self, grid, rewards):
        self.grid = grid 
        self.rewards = rewards 
        self.grid_h, self.grid_w = np.array(grid).shape
        self.noise = 0.05 
        
        # get start state and terminal state 
        self.currentState = (0,0) # if no start state set in the board, then default (0,0) 
        self.terminalState = [] 
        for h in range(self.grid_h): 
            for w in range(self.grid_w): 
                if self.grid[h][w] == 'S': 
                    self.currentState = (h,w) # found a start state on the grid 
                if self.grid[h][w] == 'C' or self.grid[h][w] == 'F':
                    self.terminalState.append((h,w)) # got cake or fire, then end 
                    

    def getGridDim(self): 
        return self.grid_h, self.grid_w
    
    def getCurrentState(self): 
        return self.currentState
    
    def getStates(self):
        out = [] 
        for h in range(self.grid_h): 
            for w in range(self.grid_w): 
                # not wall 
                if self.grid[h][w] != 'W': 
                    out.append((h,w))
        return out 
    
    def isTerminal(self, state): 
        if state in self.terminalState: 
            return True 
        return False 
        
    def getPossibleActions(self, state):
        (h,w) = state
        possible_h = [h, h-1, h, h+1]
        possible_w = [w-1, w, w+1, w]   
        actions = ['left', 'up', 'right', 'down']
        out = [] 
        
        if self.isTerminal(state): 
            return out # no actions 
        
        for i in range(len(actions)): 
            this_h = possible_h[i]
            this_w = possible_w[i]
            if this_h >= 0 and this_h < self.grid_h and this_w >= 0 and this_w < self.grid_w \
            and self.grid[this_h][this_w] != 'W': # not wall 
                out.append(actions[i])
        return out 
    
    def getReward(self, state):
        (h,w) = state
        item = self.grid[h][w]
        r = self.rewards[item]
        return r 
                
    def getTransitionStatesAndProbs(self, state, action):
        (h,w) = state
        out = [] 
        if action in self.getPossibleActions(state):
            if action == 'left': 
                nextState1, prob1 = (h, w-1), 1-self.noise*2  # left 
                if nextState1 in self.getStates(): 
                    out.append((nextState1, prob1))
                else: 
                    out.append((state, prob1)) # stay 
                    
                nextState2, prob2 = (h-1, w), self.noise  # up 
                if nextState2 in self.getStates(): 
                    out.append((nextState2, prob2))
                else: 
                    out.append((state, prob2)) # stay 
                    
                nextState3, prob3 = (h+1, w), self.noise  # down 
                if nextState3 in self.getStates(): 
                    out.append((nextState3, prob3))
                else: 
                    out.append((state, prob3)) # stay 
                    
            elif action == 'up': 
                nextState1, prob1 = (h-1, w), 1-self.noise*2  # up 
                if nextState1 in self.getStates(): 
                    out.append((nextState1, prob1))
                else: 
                    out.append((state, prob1)) # stay 
                    
                nextState2, prob2 = (h, w-1), self.noise  # left 
                if nextState2 in self.getStates(): 
                    out.append((nextState2, prob2))
                else: 
                    out.append((state, prob2)) # stay 
                    
                nextState3, prob3 = (h, w+1), self.noise  # right 
                if nextState3 in self.getStates(): 
                    out.append((nextState3, prob3))
                else: 
                    out.append((state, prob3)) # stay 
                
            elif action == 'right': 
                nextState1, prob1 = (h, w+1), 1-self.noise*2  # right 
                if nextState1 in self.getStates(): 
                    out.append((nextState1, prob1))
                else: 
                    out.append((state, prob1)) # stay 
                    
                nextState2, prob2 = (h-1, w), self.noise  # up 
                if nextState2 in self.getStates(): 
                    out.append((nextState2, prob2))
                else: 
                    out.append((state, prob2)) # stay 
                    
                nextState3, prob3 = (h+1, w), self.noise  # down 
                if nextState3 in self.getStates(): 
                    out.append((nextState3, prob3))
                else: 
                    out.append((state, prob3)) # stay 
                    
            else: # action = 'down'
                nextState1, prob1 = (h+1, w), 1-self.noise*2  # down 
                if nextState1 in self.getStates(): 
                    out.append((nextState1, prob1))
                else: 
                    out.append((state, prob1)) # stay 
                    
                nextState2, prob2 = (h, w-1), self.noise  # left 
                if nextState2 in self.getStates(): 
                    out.append((nextState2, prob2))
                else: 
                    out.append((state, prob2)) # stay 
                    
                nextState3, prob3 = (h, w+1), self.noise  # right 
                if nextState3 in self.getStates(): 
                    out.append((nextState3, prob3))
                else: 
                    out.append((state, prob3)) # stay 
        return out 


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class PolicyAgent(object):
    def __init__(self, mdp, discount, iterations):
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.U = robot_util.Counter() # A Counter is a dict with default 0
        self.currentState = self.mdp.getCurrentState() 
        
        ##### modified policy iteration code ##### 
        # random initial policy 
        self.pi = {}
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s): 
                a = random.choice(self.mdp.getPossibleActions(s))
                self.pi[s] = a 
            
        pi_prev = None 
        while pi_prev != self.pi: 
            # policy evaluation 
            U_prime = robot_util.Counter()
            for idx in range(self.iterations): # update utility of each state k times with fixed pi 
                for s in self.mdp.getStates():
                    if not self.mdp.isTerminal(s): # terminal state no legal actions 
                        a = self.pi[s]
                        u_new = self.computeQValueFromValues(s, a)
                        U_prime[s] = u_new # save new utility to U_prime
                self.U = copy.deepcopy(U_prime) # update self.U 
            pi_prev = copy.deepcopy(self.pi) # save old pi to pi_prev 
            
            # policy improvement 
            for s in self.mdp.getStates(): 
                if not self.mdp.isTerminal(s): 
                    new_a = self.computeActionFromValues(s)
                    self.pi[s] = new_a

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    def getValue(self, state):
        if self.mdp.isTerminal(state):
            # return the fixed reward of the state 
            reward = self.mdp.getReward(state)
            return reward
        
        # not terminal state, return utilities 
        return self.U[state]
    
    def getCurrentState(self): 
        return self.currentState
    
    def computeQValueFromValues(self, state, action):
        Q_val = 0
        this_reward = self.mdp.getReward(state)
        for s_prime, p in self.mdp.getTransitionStatesAndProbs(state, action):
            Q_val += p * (this_reward + self.discount * self.getValue(s_prime))
        return Q_val

    def computeActionFromValues(self, state):
        # check if is terminal state
        if self.mdp.isTerminal(state):
            return None
        else:
            # pick the action with max Q(state, action)
            actions = self.mdp.getPossibleActions(state)
            best_action = None
            best_Q = -1E10
            
            for a in actions:
                this_Q = self.computeQValueFromValues(state, a)
                if this_Q > best_Q:
                    best_action = a
                    best_Q = this_Q
            return best_action

    
    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
    
    def walkOnGrid(self, currentState): 
        print('current state: ', currentState) 
        (current_h, current_w) = currentState 

        if self.mdp.isTerminal(currentState): 
            return None 
        
        # not terminal state 
        else: 
            optimal_action = self.getPolicy(currentState)
            print('optimal_action: ', optimal_action)  
            # update self.currentState
            if optimal_action == 'right': 
                self.currentState = (current_h, current_w+1)
            elif optimal_action == 'left':
                self.currentState = (current_h, current_w-1)
            elif optimal_action == 'up':
                self.currentState = (current_h-1, current_w)
            else: # 'down'
                self.currentState = (current_h+1, current_w)

            return optimal_action
        

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# PARAMETERS AND ENVIRONMENT 
# grid = [['S',' ',' ',' ', ' '],
#         [' ','W',' ','W', 'W'],
#         [' ','W',' ',' ', 'W'],
#         [' ','W','W',' ', ' '],
#         ['F',' ',' ',' ', 'C']]

grid = [['S',' ',' ',' ', ' '],
        [' ','W','W','W', 'W'],
        [' ',' ',' ',' ', 'W'],
        [' ','W','W',' ', ' '],
        ['F',' ',' ',' ', 'C']]

rewards = {
    "C": 10.0,
    "F": -20.0,  
    'S': -0.5,     
    ' ': -0.5
}

discount = 0.9
max_moves = 50 
iterations = 100 

rotate_deg = {
    'up': 0, 
    'right': 90, 
    'down': 180, 
    'left': 270
}

dir_speed = {
    'up': 0.09, 
    'right': 0.14, 
    'down': 0.095, 
    'left': 0.14
}
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
mdp = MarkovDecisionProcess(grid, rewards)

policy_agent = PolicyAgent(mdp, discount, iterations)

currentState = policy_agent.getCurrentState()

speed = 0.14 
roll_time = 2 

for i in range(max_moves):
    # droid.animate(2)
    # get the action 
    real_act = policy_agent.walkOnGrid(currentState) # this also internally updates self.currentState
    if real_act == None: 
        break; 
    
    # make the action 
    # if rotate deg is only related to face_direction 
    droid.roll(dir_speed[real_act], rotate_deg[real_act], roll_time) 

    # update state and direction 
    currentState = policy_agent.getCurrentState()
 
 