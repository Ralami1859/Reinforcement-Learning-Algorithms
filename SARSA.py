import numpy as np
import gym

env = gym.make('FrozenLake-v0')
epsilon = 0.9
total_episodes = 10000
max_steps = 100
alpha = 0.85
gamma = 0.95

#Initializing the Q-matrix 
Q = np.zeros((env.observation_space.n, env.action_space.n)) 

#Function to choose the next action 
def choose_action(state): 
    if np.random.uniform(0, 1) < epsilon: 
        action = env.action_space.sample() 
    else: 
        action = np.argmax(Q[state, :]) 
    return action 
  
#Function to learn the Q-value 
def update(state1, state2, reward, action1, action2): 
    predict = Q[state1, action1] 
    target = reward + gamma * Q[state2, action2] 
    Q[state1, action1] = Q[state1, action1] + alpha * (target - predict)


#Initializing the reward 
reward=0
  
# Starting the SARSA learning 
for episode in range(total_episodes): 
    t = 0
    state1 = env.reset() 
    action1 = choose_action(state1) 
  
    while t < max_steps: 
        #Visualizing the training 
        env.render() 
          
        #Getting the next state 
        state2, reward, done, info = env.step(action1) 
  
        #Choosing the next action 
        action2 = choose_action(state2) 
          
        #Learning the Q-value 
        update(state1, state2, reward, action1, action2)  
        state1 = state2 
        action1 = action2 
          
        #Updating the respective values 
        t += 1
        reward += 1
          
        #If at the end of learning process 
        if done: 
            break


print ("Performace: ", reward/total_episodes) 
