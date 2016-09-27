import numpy

class TD(object):
    def __init__(self, num_states=50):
        self.num_steps = 0
        self.num_states = num_states
        self.state_values = numpy.zeros(num_states)
        self.state_values_temp = numpy.zeros(num_states)
        self.actions = [-1, 1]
        self.init_parameters()
        
    def init_parameters(self):
        self.epsilon = 0.2
        self.alpha = 0.01
        self.lmbda = 0
        self.gamma = 0.9
        
    def reset(self):
        self.num_steps = 0
        self.state_values = numpy.zeros(self.num_states)
        self.state_values_temp = numpy.zeros(self.num_states)
        self.init_parameters()
        
    def start(self):
        self.num_steps = 0
    
    def getOptimalAction(self, state):
        if state <0 or state >= self.num_states:
            print "Invalid state: ", state
            raise StandardError
        if state == 0:
            return 1
        elif state == self.num_states-1:
            return -1
        else:
            if self.state_values[state-1] > self.state_values[state+1]:
                return -1
            elif self.state_values[state-1] < self.state_values[state+1]:
                return 1
            else:
                return self.actions[numpy.random.randint(0, 2)]
                
    def egreedy(self, state):
        if numpy.random.random() < self.epsilon:
            return self.actions[numpy.random.randint(0, 2)]
        else:
            return self.getOptimalAction(state)
            
    def update(self, state, next_state, reward):
        self.num_steps += 1
        delta = reward + self.gamma*self.state_values[next_state] - self.state_values[state]
        self.state_values[state] += self.alpha * delta
        
    def avGTD(self, state, next_state, reward, alpha):
        self.num_steps += 1
        delta = reward + self.gamma*self.state_values_temp[next_state] - self.state_values_temp[state]
        self.state_values_temp[state] += alpha * delta
        self.state_values = float(self.num_steps)/float(self.num_steps+1.)*self.state_values + 1./float(self.num_steps+1.)*self.state_values_temp

        
        
    
    
    