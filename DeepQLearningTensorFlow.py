#-------------------------  Importing useful libraries ---------------------------------

# Please check the installation of
#   gym, keras, keras-rl.  (Installation on git window is prefered)


import numpy as np 
import gym 

from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten 
from keras.optimizers import Adam 

from rl.agents.dqn import DQNAgent 
from rl.policy import EpsGreedyQPolicy 
from rl.memory import SequentialMemory 


#---------------------------------------------------------------------------------------


# Building the environment 
environment_name = 'MountainCar-v0'
env = gym.make(environment_name) 
np.random.seed(0) 
env.seed(0) 

# Extracting the number of possible actions 
num_actions = env.action_space.n

# Building the learning agent

agent = Sequential() 
agent.add(Flatten(input_shape =(1, ) + env.observation_space.shape)) 
agent.add(Dense(16)) 
agent.add(Activation('relu')) 
agent.add(Dense(num_actions)) 
agent.add(Activation('linear')) 


# ---------------------  Finding the optimal strategy --------------------------------

# Building the model to find the optimal strategy 
strategy = EpsGreedyQPolicy() 
memory = SequentialMemory(limit = 10000, window_length = 1) 
dqn = DQNAgent(model = agent, nb_actions = num_actions, 
			memory = memory, nb_steps_warmup = 10, 
target_model_update = 1e-2, policy = strategy) 
dqn.compile(Adam(lr = 1e-3), metrics =['mae']) 

# Visualizing the training
nbr_steps = 20000
dqn.fit(env, nb_steps = nbr_steps, visualize = True, verbose = 2) 


# Testing the learning agent 
dqn.test(env, nb_episodes = 5, visualize = True) 

