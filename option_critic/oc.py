import os
os.environ['PYGAME_DETECT_AVX2'] = '1'

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Bernoulli
import gymnasium as gym
import mo_gymnasium as mo_gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle

# Hyperparameters
NUM_OPTIONS = 4
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
XI = 0.01
ENTROPY_REG = 0.01
LR_ACTOR = 0.00025
LR_CRITIC = 0.01
NUM_EPISODES = 2000

class SharedNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(SharedNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.relu(self.fc2(x))

class ActorNetwork(nn.Module):
    def __init__(self, input_dim, num_options, num_actions, hidden_dim=64):
        super(ActorNetwork, self).__init__()
        self.shared = SharedNetwork(input_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, num_options * num_actions)
        self.termination_head = nn.Linear(hidden_dim, num_options)
        
        self.num_options = num_options
        self.num_actions = num_actions
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.shared(x)
        policy_logits = self.policy_head(features).view(-1, self.num_options, self.num_actions)
        pi_probs = self.softmax(policy_logits)
        beta_probs = self.sigmoid(self.termination_head(features))
        return pi_probs.squeeze(0), beta_probs.squeeze(0)

class CriticNetwork(nn.Module):
    def __init__(self, input_dim, num_options, hidden_dim=64):
        super(CriticNetwork, self).__init__()
        self.shared = SharedNetwork(input_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, num_options)

    def forward(self, x):
        features = self.shared(x)
        return self.q_head(features).squeeze(0)

def preprocess_state(state):
    if isinstance(state, np.ndarray):
        return torch.FloatTensor(state.flatten()).unsqueeze(0)
    elif isinstance(state, tuple) or isinstance(state, int):
        return torch.FloatTensor([state]).unsqueeze(0)
    return torch.FloatTensor(state).unsqueeze(0)

def plot_training_metrics(actor_losses, critic_losses, option_usage, option_durations):
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Network Losses Over Time
    axs[0].plot(actor_losses, label='Actor Loss', alpha=0.7)
    axs[0].plot(critic_losses, label='Critic Loss', alpha=0.7)
    axs[0].set_title('Network Losses over Episodes')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Plot 2: Option Usage Distribution
    axs[1].bar(range(NUM_OPTIONS), option_usage, color='skyblue', edgecolor='black')
    axs[1].set_title('Total Option Usage (Step Counts)')
    axs[1].set_xlabel('Option ID')
    axs[1].set_ylabel('Total Steps Active')
    axs[1].set_xticks(range(NUM_OPTIONS))
    axs[1].grid(axis='y', alpha=0.75)

    # Plot 3: Histogram of Option Durations
    axs[2].hist(option_durations, bins=20, color='salmon', edgecolor='black')
    axs[2].set_title('Distribution of Option Durations')
    axs[2].set_xlabel('Duration (Primitive Steps before Termination)')
    axs[2].set_ylabel('Frequency')
    axs[2].grid(axis='y', alpha=0.75)

    plt.tight_layout()
    plt.savefig('oc_training_metrics.png')
    plt.close()

def plot_evaluation_graphs(option_stats, all_episode_actions, n_episodes):
    names = list(option_stats.keys())
    calls = [stats["calls"] for stats in option_stats.values()]
    avg_steps = [stats["steps"] / stats["calls"] if stats["calls"] > 0 else 0 for stats in option_stats.values()]
    
    plt.figure(figsize=(8, 5))
    plt.bar(names, calls, color='steelblue')
    plt.title('Option Invocations')
    plt.ylabel('Number of Invocations')
    plt.tight_layout()
    plt.savefig('oc_option_invocations.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(names, avg_steps, color='darkseagreen')
    plt.title('Average Duration')
    plt.ylabel('Average Steps per Invocation')
    plt.tight_layout()
    plt.savefig('oc_option_durations.png')
    plt.close()

    # Timeline Plot
    color_map = {
        'Option 0': '#FF4444',
        'Option 1': '#44AA44',
        'Option 2': '#FFAA00',
        'Option 3': '#4488FF'
    }

    fig_height = max(6, min(60, n_episodes * 0.2))
    fig, ax = plt.subplots(figsize=(14, fig_height))

    for ep_idx, seq in enumerate(all_episode_actions):
        if not seq:
            continue
        start = 0
        current_label = seq[0]
        for i in range(1, len(seq)):
            if seq[i] != current_label:
                ax.barh(ep_idx, i - start, left=start, height=0.8,
                        color=color_map.get(current_label, 'black'),
                        edgecolor='none')
                start = i
                current_label = seq[i]
        ax.barh(ep_idx, len(seq) - start, left=start, height=0.8,
                color=color_map.get(current_label, 'black'),
                edgecolor='none')
        total_steps = len(seq)
        ax.text(total_steps, ep_idx, f'{total_steps}', va='center', ha='left', fontsize=8)

    y_labels = [f'Ep {i+1}' for i in range(n_episodes)]
    ax.set_yticks(range(n_episodes))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_ylim(-0.5, n_episodes - 0.5)
    ax.set_xlabel('Timesteps')
    ax.set_title(f'Option Usage per Timestep ({n_episodes} Episodes)')

    all_lengths = [len(seq) for seq in all_episode_actions if seq]
    if all_lengths:
        ax.set_xlim(0, max(all_lengths) * 1.05)

    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=color, label=name) for name, color in color_map.items()]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig('oc_episode_timeline.png', dpi=150)
    plt.close()

def evaluate_agent(env, actor_net, critic_net, n_eval_eps=10):
    total_steps = 0
    option_stats = {f"Option {i}": {"calls": 0, "steps": 0} for i in range(NUM_OPTIONS)}
    all_episode_actions = []
    
    max_step_limit = env.spec.max_episode_steps if env.spec.max_episode_steps else 1000

    for eps in tqdm(range(n_eval_eps), desc="Evaluating"):
        state, _ = env.reset(seed=42 + eps)
        state_tensor = preprocess_state(state)
        
        with torch.no_grad():
            q_vals = critic_net(state_tensor)
            current_option = torch.argmax(q_vals).item()

        done = False
        steps = 0
        episode_actions = []
        option_stats[f"Option {current_option}"]["calls"] += 1
        
        while not done and steps < max_step_limit:
            with torch.no_grad():
                pi_probs, beta_probs = actor_net(state_tensor)
            
            option_policy = pi_probs[current_option]
            action = torch.argmax(option_policy).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
            
            episode_actions.append(f"Option {current_option}")
            option_stats[f"Option {current_option}"]["steps"] += 1
            
            next_state_tensor = preprocess_state(next_state)
            
            if not done:
                with torch.no_grad():
                    next_pi_probs, next_beta_probs = actor_net(next_state_tensor)
                    next_q_vals = critic_net(next_state_tensor)
                
                if next_beta_probs[current_option].item() > 0.5:
                    current_option = torch.argmax(next_q_vals).item()
                    option_stats[f"Option {current_option}"]["calls"] += 1

            state_tensor = next_state_tensor

        total_steps += steps
        all_episode_actions.append(episode_actions)

    print("\n=== Execution Statistics ===")
    print(f"Average Steps/Episode: {total_steps / n_eval_eps:.2f}")

    plot_evaluation_graphs(option_stats, all_episode_actions, n_eval_eps)
    print("All statistics and timeline successfully saved.")

def train_option_critic():
    env = gym.make('four-room-v0', disable_env_checker=True)
    input_dim = np.prod(env.observation_space.shape) if hasattr(env.observation_space, 'shape') else 1
    num_actions = env.action_space.n

    actor_net = ActorNetwork(input_dim, NUM_OPTIONS, num_actions)
    critic_net = CriticNetwork(input_dim, NUM_OPTIONS)

    model_path = "oc_agent.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            checkpoint = pickle.load(f)
        actor_net.load_state_dict(checkpoint['actor'])
        critic_net.load_state_dict(checkpoint['critic'])
        print("Loaded existing Option-Critic agent from disk.")
    else:
        print("No saved agent found. Commencing training.")
        optimizer_actor = optim.RMSprop(actor_net.parameters(), lr=LR_ACTOR)
        optimizer_critic = optim.RMSprop(critic_net.parameters(), lr=LR_CRITIC)
    
        epsilon = EPSILON_START
    
        #tracking stats
        ep_actor_losses = []
        ep_critic_losses = []
        option_usage_counts = np.zeros(NUM_OPTIONS)
        option_durations = []
    
        pbar = tqdm(range(NUM_EPISODES), desc="Training Option-Critic")
    
        for episode in pbar:
            state, _ = env.reset()
            state_tensor = preprocess_state(state)
            
            with torch.no_grad():
                q_vals = critic_net(state_tensor)
                if np.random.rand() < epsilon:
                    current_option = np.random.randint(NUM_OPTIONS)
                else:
                    current_option = torch.argmax(q_vals).item()
    
            done = False
            total_reward = 0
            current_option_duration = 0
            ep_act_loss_sum = 0
            ep_crit_loss_sum = 0
            step_count = 0
    
            while not done:
                pi_probs, beta_probs = actor_net(state_tensor)
                option_policy = pi_probs[current_option]
                
                action_dist = Categorical(option_policy)
                action = action_dist.sample()
                
                next_state, reward, terminated, truncated, _ = env.step(action.item())
                
                # Scalarize the multi-objective reward array into a single float
                if isinstance(reward, np.ndarray):
                    reward = float(np.sum(reward))
                    
                done = terminated or truncated
                total_reward += reward
                next_state_tensor = preprocess_state(next_state)
    
                # Update usage metrics
                option_usage_counts[current_option] += 1
                current_option_duration += 1
    
                next_pi_probs, next_beta_probs = actor_net(next_state_tensor)
                next_q_vals = critic_net(next_state_tensor)
                q_vals = critic_net(state_tensor)
    
                beta_next = next_beta_probs[current_option]
    
                if done:
                    target_u = torch.tensor(0.0)
                else:
                    v_omega_next = torch.max(next_q_vals)
                    target_u = (1.0 - beta_next) * next_q_vals[current_option] + beta_next * v_omega_next
    
                target = reward + GAMMA * target_u.detach()
                td_error = target - q_vals[current_option]
    
                critic_loss = td_error.pow(2)
                
                optimizer_critic.zero_grad()
                critic_loss.backward()
                optimizer_critic.step()
                ep_crit_loss_sum += critic_loss.item()
    
                v_omega_next_detach = torch.max(next_q_vals).detach()
                advantage_omega = next_q_vals[current_option].detach() - v_omega_next_detach + XI
                
                termination_loss = next_beta_probs[current_option] * advantage_omega
    
                log_prob = action_dist.log_prob(action)
                policy_loss = -log_prob * td_error.detach()
                entropy = action_dist.entropy()
                
                actor_loss = policy_loss + termination_loss - ENTROPY_REG * entropy
                
                optimizer_actor.zero_grad()
                actor_loss.backward()
                optimizer_actor.step()
                ep_act_loss_sum += actor_loss.item()
    
                if not done:
                    termination_dist = Bernoulli(next_beta_probs[current_option])
                    if termination_dist.sample().item() == 1.0:
                        
                        # Record duration and reset for the new option
                        option_durations.append(current_option_duration)
                        current_option_duration = 0
                        
                        with torch.no_grad():
                            if np.random.rand() < epsilon:
                                current_option = np.random.randint(NUM_OPTIONS)
                            else:
                                current_option = torch.argmax(next_q_vals).item()
    
                state_tensor = next_state_tensor
                step_count += 1
    
            if current_option_duration > 0:
                option_durations.append(current_option_duration)
    
            epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
            
            avg_act_loss = ep_act_loss_sum / max(1, step_count)
            avg_crit_loss = ep_crit_loss_sum / max(1, step_count)
            
            ep_actor_losses.append(avg_act_loss)
            ep_critic_losses.append(avg_crit_loss)
            
            pbar.set_postfix({
                'Reward': f"{total_reward:.2f}",
                'A_Loss': f"{avg_act_loss:.4f}",
                'C_Loss': f"{avg_crit_loss:.4f}",
                'Eps': f"{epsilon:.2f}"
            })
    
        plot_training_metrics(ep_actor_losses, ep_critic_losses, option_usage_counts, option_durations)
        
        with open(model_path, "wb") as f:
            pickle.dump({
                'actor': actor_net.state_dict(),
                'critic': critic_net.state_dict()
            }, f)
        print("Saved trained Option-Critic agent to disk.")

    evaluate_agent(env, actor_net, critic_net, n_eval_eps=10)

if __name__ == "__main__":
    train_option_critic()