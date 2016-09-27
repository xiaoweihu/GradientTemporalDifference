from environment import *
from agent import *
import numpy
import math
from pylab import *
import matplotlib.pyplot as plt

num_timesteps = 2000
num_runs = 10
num_methods = 2
true_values = numpy.zeros(50)
time_to_print = [20*i for i in range(101)]
errors = numpy.zeros((num_methods, num_runs, len(time_to_print)))

def L2error(state_values, true_values):
    diff = state_values - true_values
    return math.sqrt(numpy.dot(diff, diff))

env = Chain()
alg = TD()

# Get true state values following the optimal policy
for run in range(100):
    env.reset()
    alg.reset()
    for timestep in range(10000):
        if env.isAtGoal():
            env.reset()
        cur_state = env.observeState()
        #action = alg.egreedy(cur_state)
        action = env.optimalPolicy()
        reward = env.takeAction(action)
        next_state = env.observeState()
        alg.update(cur_state, next_state, reward)
    true_values += alg.state_values
true_values /= 100.
print "true values get!"


for iMethod in range(num_methods):            
    for run in range(num_runs):
        env.reset()
        alg.reset()
        i = 0
        print "Method: ", iMethod, "  run: ", run
        for timestep in range(num_timesteps+1):
            if env.isAtGoal():
                env.reset()
            cur_state = env.observeState()
            action = alg.egreedy(cur_state)
            reward = env.takeAction(action)
            next_state = env.observeState()
            if iMethod == 0: #fixed step size
                alg.update(cur_state, next_state, reward)
            elif iMethod == 1: # avGTD
                alg.avGTD(cur_state, next_state, reward, 0.1)
            if timestep == time_to_print[i]:
                errors[iMethod][run][i] = L2error(alg.state_values, true_values)
                i += 1

        
avgerror = numpy.mean(errors, axis = 1)
fig, ax = plt.subplots()

ax.plot(time_to_print, avgerror[0], label = "fixed")
ax.plot(time_to_print, avgerror[1], label = "avGTD")
ax.legend(loc=0)
ax.set_xlabel('Timesteps')
ax.set_ylabel('L2 error')
show()