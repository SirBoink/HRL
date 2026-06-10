import gymnasium as gym 
import numpy as np
import random
import tqdm

#enviroment 
env = gym.make("Taxi-v4")

#hyperparams
max_steps = 1000
n_eps = 100_000
lr = 0.005
df = 0.99 
e = 1.0 
e_decay = e / (n_eps/2)
e_min = 0.1 

#setting up q-value table 
q_value = np.zeros((env.observation_space.n, env.action_space.n))
location = env.unwrapped.locs
#fxn to select an option (e-greedy algo)
def sel_action(state): 

    if random.uniform(0,1) < e: 
        return env.action_space.sample()
    else: 
        return np.argmax(q_value[state, :])


# Q-learning algo update
def update_qvalue(state, action, reward, next_state): 

    best_action = np.argmax(q_value[next_state, :])

    td_target = reward + df * q_value[next_state, best_action]
    
    td_error = td_target - q_value[state, action]
    
    q_value[state, action] += lr * td_error


# training the agent 
for eps in tqdm.tqdm(range(n_eps)):

    state, info = env.reset()
    over = False

    for j in range(max_steps): 
        action = sel_action(state)

        next_state, reward, terminated, truncated, info = env.step(action)

        update_qvalue(state, action, reward, next_state)

        state = next_state

        over = terminated or truncated

        if over: 
            break

    e = max(e_min, e - e_decay)


#testing the agent 
env = gym.make("Taxi-v4", render_mode="human")

for eps in range(5):

    state, info = env.reset()
    over = False
    tr = 0
        
    for stps in range(max_steps):
    
        env.render()
        action = np.argmax(q_value[state, :])

        next_state, reward, terminated, truncated, info = env.step(action)

        over = terminated or truncated    

        tr += reward 

        state = next_state

        if over: 
            break

    if over:
        print(f"Episode: {eps + 1}, rewards: {tr}")

    
env.close()
















