import numpy as np
import gymnasium as gym
import tqdm
from gymnasium.wrappers import RecordVideo

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
        self.n_eps = 300_000
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

    def intra_option_update(self, env, state, action, reward, next_state, done, executing_behavior=None, executing_k=None):
        #Performs intra-option q leaarning updates for all options whose internal policies are consistent with the primitive action taken.
        for i, opt in enumerate(self.options):
            opt_behavior = 6 + i
            
            #off policy update condition
            if opt.pi(state) == action:
                reached_goal = opt.beta(next_state, env)
                
                #check if the option terminates at the next state
                opt_terminated = reached_goal or done
                if opt_behavior == executing_behavior and executing_k is not None:
                    if executing_k >= opt.max_steps: #we're technically bootstrapping from max Q(s',a') but its a practical measure to prevent infinite loops.
                        opt_terminated = True
                
                #calculating U(s', o)
                if done:
                    u_next = 0
                elif opt_terminated:
                    u_next = np.max(self.q_values[next_state, :])#option terminates
                else:
                    u_next = self.q_values[next_state, opt_behavior] # option continues
                
                #to update Q(s, o)
                td_target = reward + self.df * u_next
                td_error = td_target - self.q_values[state, opt_behavior]
                self.q_values[state, opt_behavior] += self.lr * td_error

    def train_agent(self, env):
        for eps in tqdm.tqdm(range(self.n_eps)):
            state, _ = env.reset()
            done = False
            
            while not done:
                behavior = self.choose_action(state)
                
                if behavior < 6:
                    next_state, reward, terminated, truncated, _ = env.step(behavior)
                    done = terminated or truncated
                    
                    #normal primitive action update
                    self.smdp_update(state, behavior, reward, next_state, k_steps=1, done=done)
                    
                    #intra option update; any consistent action is updated
                    self.intra_option_update(env, state, behavior, reward, next_state, done)
                    
                    state = next_state
                    
                else:
                    opt = self.options[behavior - 6]
                    k = 0
                    curr_state = state
                    
                    while True:
                        action = opt.pi(curr_state)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        k += 1
                        
                        #step by step update
                        self.intra_option_update(
                            env, curr_state, action, reward, next_state, done, 
                            executing_behavior=behavior, executing_k=k
                        )
                        
                        curr_state = next_state
                        
                        reached_goal = opt.beta(curr_state, env)
                        if reached_goal or done or k >= opt.max_steps:
                            break
                            
                    state = curr_state
                    
            self.e = max(self.e_min, self.e - self.e_decay)


def evaluate_agent(env, agent, n_eval_eps=500):
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
    
    #gym's native video recording
    env_render = gym.make("Taxi-v4", render_mode="rgb_array")
    env_render = RecordVideo(
        env_render, 
        video_folder=".", 
        name_prefix="taxi_agent", 
        episode_trigger=lambda e: True
    )
    
    state, _ = env_render.reset(seed=42)
    done = False
    
    while not done:
        behavior = agent.greedy(state)
        
        if behavior < 6:
            state, _, terminated, truncated, _ = env_render.step(behavior)
            done = terminated or truncated
        else:
            opt = agent.options[behavior - 6]
            k = 0
            while True:
                action = opt.pi(state)
                state, _, terminated, truncated, _ = env_render.step(action)
                done = terminated or truncated
                k += 1
                
                if opt.beta(state, env_render) or done or k >= opt.max_steps:
                    break

    env_render.close()

if __name__ == "__main__":
    main()
    