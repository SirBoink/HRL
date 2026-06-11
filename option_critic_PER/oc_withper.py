import gymnasium as gym
import mo_gymnasium as mo_gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
from tqdm import trange

class UnifiedRewardWrapper(gym.Wrapper):
    def __init__(self, env, bottleneck_coords, bottleneck_reward=2.0, goal_reward=1.5):
        super().__init__(env)
        self.bottleneck_coords = set(bottleneck_coords)
        self.bottleneck_reward = bottleneck_reward
        self.goal_reward = goal_reward
        self.visited_bottlenecks = set()    

    def reset(self, **kwargs):
        self.visited_bottlenecks.clear()
        obs, info = self.env.reset(**kwargs)
        return obs.astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs.astype(np.float32)
        
        # Scalarize base rewards
        item_r = float(np.sum(reward))
        scalar_reward = item_r
        
        # Bottleneck rewards
        b_r = 0.0
        agent_pos = (int(obs[0]), int(obs[1]))
        if agent_pos in self.bottleneck_coords and agent_pos not in self.visited_bottlenecks:
            self.visited_bottlenecks.add(agent_pos)
            b_r = self.bottleneck_reward
            scalar_reward += b_r
            
        # Goal reward
        g_r = 0.0
        if terminated and not truncated:
            g_r = self.goal_reward
            scalar_reward += g_r
            
        info['reward_components'] = {'item': item_r, 'bottleneck': b_r, 'goal': g_r}
        return obs, scalar_reward, terminated, truncated, info


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, state, option, action, reward, next_state, done):
        max_p = self.priorities.max() if self.buffer else 1.0
        experience = (state, option, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience

        self.priorities[self.pos] = max_p
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        batch = list(zip(*samples))
        return (np.array(batch[0]), np.array(batch[1]), np.array(batch[2]), 
                np.array(batch[3], dtype=np.float32), np.array(batch[4]), 
                np.array(batch[5], dtype=np.float32), indices, weights)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            self.priorities[idx] = float(error) + 1e-5
    
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, num_options, num_actions):
        super().__init__()
        self.num_options = num_options
        self.num_actions = num_actions
        
        self.shared = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU())
        self.action_head = nn.Linear(64, num_options * num_actions)
        self.termination_head = nn.Linear(64, num_options)

    def forward(self, state):
        x = self.shared(state)
        action_logits = self.action_head(x).view(-1, self.num_options, self.num_actions)
        action_dist = torch.softmax(action_logits, dim=-1)
        termination_prob = torch.sigmoid(self.termination_head(x))
        return action_dist, termination_prob

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, num_options):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_options)
        )

    def forward(self, state):
        return self.network(state)


def update_critic(actor, critic, target_critic, critic_opt, buffer, batch_size, gamma):
    if len(buffer.buffer) < batch_size:
        return

    states, options, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size)

    states_t = torch.FloatTensor(states)
    options_t = torch.LongTensor(options).unsqueeze(1)
    actions_t = torch.LongTensor(actions).unsqueeze(1)
    rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
    next_states_t = torch.FloatTensor(next_states)
    dones_t = torch.FloatTensor(dones).unsqueeze(1)
    weights_t = torch.FloatTensor(weights).unsqueeze(1)

    # Critic Calculations using Stable Target Network
    q_vals = critic(states_t)
    q_s_omega = q_vals.gather(1, options_t)

    with torch.no_grad():
        next_q_vals = target_critic(next_states_t)
        _, next_term_prob = actor(next_states_t)
        
        max_next_q, _ = next_q_vals.max(dim=1, keepdim=True)
        next_q_s_omega = next_q_vals.gather(1, options_t)
        next_term_omega = next_term_prob.gather(1, options_t)
        
        v_s_prime = (1 - next_term_omega) * next_q_s_omega + next_term_omega * max_next_q
        target_q = rewards_t + (1 - dones_t) * gamma * v_s_prime

    td_errors = torch.abs(target_q - q_s_omega).detach().numpy().flatten()
    buffer.update_priorities(indices, td_errors)

    critic_loss = (weights_t * (target_q - q_s_omega) ** 2).mean()
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()


def update_actor(actor, critic, target_critic, actor_opt, buffer, batch_size, gamma, reg_epsilon):
    """Off-policy actor update using PER samples with importance-sampling weights."""
    if len(buffer.buffer) < batch_size:
        return

    states, options, actions, rewards, next_states, dones, indices, weights = buffer.sample(batch_size)

    states_t = torch.FloatTensor(states)
    options_t = torch.LongTensor(options)          # (B,)
    actions_t = torch.LongTensor(actions)          # (B,)
    rewards_t = torch.FloatTensor(rewards)         # (B,)
    next_states_t = torch.FloatTensor(next_states)
    dones_t = torch.FloatTensor(dones)             # (B,)
    weights_t = torch.FloatTensor(weights)         # (B,)  IS weights

    # Compute targets and baseline Q-values (no grad for targets)
    with torch.no_grad():
        q_vals = critic(states_t)                  # (B, num_options)
        q_s_omega = q_vals[torch.arange(batch_size), options_t]   # (B,)
        max_q, _ = q_vals.max(dim=1)              # (B,)

        next_q_vals = target_critic(next_states_t) # (B, num_options)
        _, next_term_prob = actor(next_states_t)   # next_term_prob: (B, num_options)
        max_next_q, _ = next_q_vals.max(dim=1)    # (B,)
        next_q_s_omega = next_q_vals[torch.arange(batch_size), options_t]  # (B,)
        next_term_omega = next_term_prob[torch.arange(batch_size), options_t]  # (B,)

        v_s_prime = (1 - next_term_omega) * next_q_s_omega + next_term_omega * max_next_q
        target_q = rewards_t + (1 - dones_t) * gamma * v_s_prime  # (B,)
        advantage = (target_q - q_s_omega).detach()                # (B,)

    # Forward pass through actor (with grad)
    action_dist, term_prob = actor(states_t)   # (B, num_options, num_actions), (B, num_options)

    # Policy loss: weighted by IS weights and advantage
    opt_policies = action_dist[torch.arange(batch_size), options_t]   # (B, num_actions)
    log_action_prob = torch.log(opt_policies[torch.arange(batch_size), actions_t] + 1e-8)  # (B,)
    policy_loss = -(weights_t * log_action_prob * advantage).mean()

    # Entropy bonus (per sample, then weighted mean)
    entropy = -torch.sum(opt_policies * torch.log(opt_policies + 1e-8), dim=-1)  # (B,)
    entropy_loss = -(weights_t * entropy).mean()

    # Termination loss: weighted by IS weights
    term_omega = term_prob[torch.arange(batch_size), options_t]   # (B,)
    termination_advantage = (q_s_omega - max_q + reg_epsilon).detach()  # (B,)
    termination_loss = (weights_t * term_omega * termination_advantage * (1 - dones_t)).mean()

    actor_loss = policy_loss + termination_loss + 0.01 * entropy_loss

    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # Update PER priorities using the actor's TD errors
    actor_td_errors = torch.abs(advantage).detach().numpy().flatten()
    buffer.update_priorities(indices, actor_td_errors)

if __name__ == '__main__':
    bottlenecks = [(5, 2), (2, 5), (8, 5), (5, 8)] 
    base_env = mo_gym.make("four-room-v0")
    env = UnifiedRewardWrapper(base_env, bottleneck_coords=bottlenecks)
    
    num_options = 4
    num_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]
    gamma = 0.99
    batch_size = 32
    
    # Exploration & Regularization Schedules
    epsilon_start = 1.0
    epsilon_end = 0.05
    
    # Fix: Keep the dynamic deliberation cost baseline high enough to buffer against residual policy noise
    reg_epsilon_start = 0.10  
    reg_epsilon_end = 0.03    
    decay_duration = 800  
    
    actor = ActorNetwork(state_dim, num_options, num_actions)
    critic = CriticNetwork(state_dim, num_options)
    
    # Fix: Create and initialize the Target Critic Network
    target_critic = CriticNetwork(state_dim, num_options)
    target_critic.load_state_dict(critic.state_dict())
    
    actor_opt = optim.Adam(actor.parameters(), lr=0.001)
    critic_opt = optim.Adam(critic.parameters(), lr=0.002)
    buffer = PrioritizedReplayBuffer(capacity=5000)
    
    env_steps = 0
    gradient_updates = 0
    target_update_freq = 100 # Periodically sync target critic weights
    
    history_option_durations = []
    history_item_r = []
    history_bottleneck_r = []
    history_goal_r = []
    td_error_snapshots = {}
    
    pbar = trange(2000, desc="Training")
    for episode in pbar:
        epsilon = max(epsilon_end, epsilon_start - (epsilon_start - epsilon_end) * (episode / decay_duration))
        current_reg_epsilon = max(reg_epsilon_end, reg_epsilon_start - (reg_epsilon_start - reg_epsilon_end) * (episode / decay_duration))
        
        state, info = env.reset()
        done = False
        current_option = None
        
        ep_item_r = 0.0
        ep_bottleneck_r = 0.0
        ep_goal_r = 0.0
        
        current_option_duration = 0
        ep_option_durations = []
        
        while not done:
            env_steps += 1
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_dist, term_prob = actor(state_t)
            
            # 1. Option selection execution
            if current_option is None or random.random() < term_prob[0, current_option].item():
                if random.random() < epsilon:
                    current_option = random.randint(0, num_options - 1)
                else:
                    q_vals = critic(state_t)
                    current_option = torch.argmax(q_vals, dim=1).item()
                    
                if current_option_duration > 0:
                    ep_option_durations.append(current_option_duration)
                current_option_duration = 1
            else:
                current_option_duration += 1
                
            # 2. Sample action
            opt_policy = action_dist[0, current_option].detach().numpy()
            action = np.random.choice(num_actions, p=opt_policy)
            
            # 3. Environment Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            rc = info.get('reward_components', {'item': 0, 'bottleneck': 0, 'goal': 0})
            ep_item_r += rc['item']
            ep_bottleneck_r += rc['bottleneck']
            ep_goal_r += rc['goal']
            
            # --- 4. Off-Policy Actor & Critic Updates (via PER) ---
            buffer.push(state, current_option, action, reward, next_state, done)
            if len(buffer.buffer) >= batch_size:
                gradient_updates += 1

            update_actor(actor, critic, target_critic, actor_opt, buffer, batch_size, gamma, current_reg_epsilon)
            update_critic(actor, critic, target_critic, critic_opt, buffer, batch_size, gamma)
            
            # Fix: Periodically sync the target network
            if gradient_updates % target_update_freq == 0:
                target_critic.load_state_dict(critic.state_dict())
            
            state = next_state
            
        ep_option_durations.append(current_option_duration)
        history_option_durations.append(np.mean(ep_option_durations) if ep_option_durations else 0)
        history_item_r.append(ep_item_r)
        history_bottleneck_r.append(ep_bottleneck_r)
        history_goal_r.append(ep_goal_r)
        
        if episode in [100, 500, 1000, 1500, 1999]:
            valid_len = len(buffer.buffer)
            td_error_snapshots[f'priorities_ep_{episode}'] = buffer.priorities[:valid_len].copy()
            rewards_in_buffer = [exp[3] for exp in buffer.buffer[:valid_len]]
            td_error_snapshots[f'rewards_ep_{episode}'] = np.array(rewards_in_buffer)
    
        pbar.set_postfix(env_steps=env_steps, gradient_updates=gradient_updates, epsilon=f"{epsilon:.2f}")
    
    env.close()
    
    # Export metrics and weights
    torch.save(actor.state_dict(), "actor.pth")
    torch.save(critic.state_dict(), "critic.pth")
    np.savez("training_logs.npz", 
             option_durations=history_option_durations,
             item_rewards=history_item_r,
             bottleneck_rewards=history_bottleneck_r,
             goal_rewards=history_goal_r,
             **td_error_snapshots)
    print("Models and training logs saved successfully with Target Critic stabilizes.")