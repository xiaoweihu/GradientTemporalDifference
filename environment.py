import numpy

class Chain(object):
    """
    state: 0,1,2,...,49
    action: -1(left), +1(right)
    """
    def __init__(self, init_state=25):
        self.chain_size = 50
        self.state = numpy.random.randint(0, self.chain_size)
        self.slip_prob = 0.1
        self.goal_reward = 1
        self.goal = [9, 40]
        
    def reset(self, init_state=25):
        self.state = numpy.random.randint(0, self.chain_size)
        
    def observeState(self):
        return self.state
        
    def isAtGoal(self):
        return self.state in self.goal
    
    # return: action using the optimal policy
    def optimalPolicy(self):
        cur_state = self.observeState()
        goal = self.goal[0]
        if abs(cur_state-self.goal[0]) > abs(cur_state-self.goal[1]):
            goal = self.goal[1]
        elif abs(cur_state-self.goal[0]) == abs(cur_state-self.goal[1]):
            goal = self.goal[numpy.random.randint(0,2)]
        return 1 if goal>cur_state else -1
    
    # return: reward
    def takeAction(self, action):
        if not action in [-1, 1]:
            print "Invalid action: ", action
            raise StandardError
        if numpy.random.random() < self.slip_prob:
            action = -action
        self.state += action
        self.state = max(0, self.state)
        self.state = min(self.chain_size-1, self.state)
        if self.isAtGoal():
            return self.goal_reward
        else:
            return 0