"""
- gymnasium 'taxi-v4' smdp learning with options
- creates a class for learning options. 
- options are restricted to navigation only. no pickup/dropoff. q-value table created accordingly
- initiation set is excluded as I'm assuming you can start an option from anywhere, no state space constraint.
- options are limited to 25 primitive actions.
- intra-option policy is purely greedy, but option training employs an epsilon greedy algo (50% exploration - 50% exploitation)


- also creates an agent class for the upper-hierarchy 'manager' 
- agent can choos between prim actions and options 
- implements functions greedy, choose_action. they are self explanatory.
- smdp update implements the formula: Q(s,a) += lr * [R_cum + (gamma^k) * max_Q(s', a') - Q(s,a)]
- if k=1, the formula becomes normal q-learning formula. 
- evaluate agent tests the trained agent for 5 episodes. 
- random seed = 77 for reproducibility
- 'location = env.unwrapped.locs' for getting the red, blue, green and yellow coordinates. passenger and destination coords are not accessed. 
"""

import numpy as np
import gymnasium as gym
import tqdm

class Option:
    def __init__(self, name, goal_row, goal_col):
        self.name = name
        self.n_eps = 100_000
        self.max_steps = 25
        self.goal_row = goal_row
        self.goal_col = goal_col
        self.q_values = np.zeros((500, 4)) #only nav actions permitted
        self.lr = 0.1
        self.df = 0.9 
        self.e = 1.0
        self.e_decay = self.e / (self.n_eps / 2)
        self.e_min = 0.1


    def beta(self, state, env):
        decoded = list(env.unwrapped.decode(state))
        taxi_row, taxi_col = decoded[0], decoded[1]
        return taxi_row == self.goal_row and taxi_col == self.goal_col

    def pi(self, state):
        return np.argmax(self.q_values[state, :])

    def option_training(self, env):

        for eps in tqdm.tqdm(range(self.n_eps)):
            state, _ = env.reset()
            
            for steps in range(self.max_steps):

                if np.random.rand() < self.e:
                    action = np.random.choice([0, 1, 2, 3])
                else:
                    action = self.pi(state)
                    
                next_state, _, terminated, truncated, _ = env.step(action)
                
                reached_goal = self.beta(next_state, env)
                
                if reached_goal:
                    own_reward = 10
                    over = True
                else:
                    own_reward = -1
                    over = terminated or truncated
                
                #normal q learning
                best_action_next_step = np.argmax(self.q_values[next_state, :])
                td_target = own_reward + (0 if over else self.df * self.q_values[next_state][best_action_next_step])
                td_error = td_target - self.q_values[state][action]
                self.q_values[state][action] += self.lr * td_error
                
                state = next_state
                
                if over: 
                    break
            
            self.e = max(self.e_min, self.e - self.e_decay)


class Agent:
    def __init__(self, options):
        self.n_eps = 100_000
        self.num_actions = 6 + len(options)
        self.valid_actions = list(range(self.num_actions)) #this is redundant but kept for structure
        self.q_values = np.zeros((500, self.num_actions)) #basically 10 actions
        self.lr = 0.1
        self.df = 0.9  
        self.e = 1.0
        self.e_decay = self.e / (self.n_eps / 2)
        self.e_min = 0.1
        self.options = options

    def greedy(self, state):
        valid_q_values = [self.q_values[state, i] for i in self.valid_actions]
        return self.valid_actions[np.argmax(valid_q_values)]

    def choose_action(self, state):
        if np.random.rand() < self.e:
            return np.random.choice(self.valid_actions)
        else:
            return self.greedy(state)

    def smdp_update(self, state, behavior, cmltv_reward, next_state, k_steps, done):
        best_next_q = 0 if done else np.max(self.q_values[next_state, :])
        td_target = cmltv_reward + (self.df ** k_steps) * best_next_q
        td_error = td_target - self.q_values[state, behavior]
        self.q_values[state, behavior] += self.lr * td_error

    def train_agent(self, env):
        for eps in tqdm.tqdm(range(self.n_eps)):
            state, _ = env.reset()
            done = False
            
            while not done:
                behavior = self.choose_action(state)
                
                if behavior < 6:
                    next_state, reward, terminated, truncated, _ = env.step(behavior)
                    done = terminated or truncated
                    
                    self.smdp_update(state, behavior, reward, next_state, k_steps=1, done=done)
                    state = next_state
                    
                else:
                    opt = self.options[behavior - 6]
                    k = 0
                    cmltv_reward = 0
                    curr_state = state
                    
                    while True:
                        action = opt.pi(curr_state)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        
                        cmltv_reward += (self.df ** k) * reward
                        k += 1
                        
                        curr_state = next_state
                        
                        reached_goal = opt.beta(curr_state, env)
                        if reached_goal or done or k >= opt.max_steps:
                            break
                            
                    self.smdp_update(state, behavior, cmltv_reward, curr_state, k, done=done)
                    state = curr_state
                    
            self.e = max(self.e_min, self.e - self.e_decay)


def evaluate_agent(env, agent, n_eval_eps=5):
    successes = 0
    total_steps = 0
    
    max_step_limit = env.spec.max_episode_steps if env.spec.max_episode_steps else 50

    for eps in range(n_eval_eps):

        state, _ = env.reset(seed=42 + eps)
        done = False
        steps = 0
        episode_success = False
        
        while not done and steps < max_step_limit:
            behavior = agent.greedy(state)
            
            if behavior < 6:
                state, reward, terminated, truncated, _ = env.step(behavior)
                done = terminated or truncated
                steps += 1
                if reward == 20: 
                    episode_success = True
            else:
                opt = agent.options[behavior - 6]
                k = 0
                while True:
                    action = opt.pi(state)
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    k += 1
                    steps += 1
                    
                    if opt.beta(state, env) or done or k >= opt.max_steps:
                        break
        
        if episode_success:
            successes += 1
        total_steps += steps

    success_rate = (successes / n_eval_eps) * 100
    print(f"success rate: {success_rate:.2f}%")
    print(f"avg steps/episode: {total_steps / n_eval_eps:.2f}")


def main():
    np.random.seed(77)
    env = gym.make("Taxi-v4")
    env.action_space.seed(77)
    
    location = env.unwrapped.locs

    loc_names = ["to_R", "to_G", "to_Y", "to_B"]
    
    trained_options = []
    
    for name, (row, col) in zip(loc_names, location):
        opt = Option(name, row, col)
        opt.option_training(env)
        trained_options.append(opt)
        print(f"trained: {name}.\n")
        
    agent = Agent(trained_options)
    agent.train_agent(env)
    
    evaluate_agent(env, agent)
    

if __name__ == "__main__":
    main()
