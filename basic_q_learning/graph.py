import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from smdpq import Option, Agent

# Suppress seaborn/matplotlib warnings for cleaner output
warnings.filterwarnings("ignore")

def extract_evaluation_data(env, agent, n_eval_eps=100):
    """Runs evaluation and extracts granular data for all visualizations."""
    raster_data = []          # (step, action) for a representative episode
    composition_data = []     # [{action_id: count, ...}] per episode
    option_durations = {i: [] for i in range(6, 6 + len(agent.options))}
    avg_duration_per_eps = []
    initiation_states = {i: np.zeros((5, 5)) for i in range(6, 6 + len(agent.options))}
    
    max_step_limit = env.spec.max_episode_steps if env.spec.max_episode_steps else 50
    raster_episode_captured = False

    for eps in range(n_eval_eps):
        state, _ = env.reset(seed=42 + eps)
        done = False
        steps = 0
        eps_composition = {i: 0 for i in range(agent.num_actions)}
        eps_option_durations = []
        eps_raster = []

        while not done and steps < max_step_limit:
            behavior = agent.greedy(state)
            eps_composition[behavior] += 1
            eps_raster.append((steps, behavior))
            
            if behavior < 6:
                state, reward, terminated, truncated, _ = env.step(behavior)
                done = terminated or truncated
                steps += 1
            else:
                opt = agent.options[behavior - 6]
                
                # Record initiation state for heatmap
                decoded = list(env.unwrapped.decode(state))
                taxi_row, taxi_col = decoded[0], decoded[1]
                initiation_states[behavior][taxi_row, taxi_col] += 1
                
                k = 0
                while True:
                    action = opt.pi(state)
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    k += 1
                    steps += 1
                    
                    if opt.beta(state, env) or done or k >= opt.max_steps:
                        break
                        
                # Record duration
                option_durations[behavior].append(k)
                eps_option_durations.append(k)

        composition_data.append(eps_composition)
        
        if eps_option_durations:
            avg_duration_per_eps.append(np.mean(eps_option_durations))
        else:
            avg_duration_per_eps.append(0)
            
        # Capture the first episode that utilized options for the raster plot
        if not raster_episode_captured and sum(eps_composition[b] for b in range(6, agent.num_actions)) > 0:
            raster_data = eps_raster
            raster_episode_captured = True

    return raster_data, composition_data, option_durations, avg_duration_per_eps, initiation_states


def plot_visualizations(agent, raster_data, composition_data, option_durations, avg_duration_per_eps, initiation_states):
    """Generates all 5 visualizations in a single figure."""
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(20, 15))
    
    behavior_labels = ["South", "North", "East", "West", "Pickup", "Dropoff"] + [opt.name for opt in agent.options]
    opt_labels = [opt.name for opt in agent.options]

    # 1. Action Raster Plot
    ax1 = plt.subplot(3, 2, 1)
    if raster_data:
        steps, behaviors = zip(*raster_data)
        ax1.scatter(steps, behaviors, marker='|', s=200, color='cyan')
        ax1.set_yticks(range(agent.num_actions))
        ax1.set_yticklabels(behavior_labels)
        ax1.set_xlabel("Sequential Time Step")
        ax1.set_title("Action Raster Plot (Representative Episode)")
        ax1.grid(axis='y', alpha=0.3)

    # 2. Stacked Bar Chart of Episode Composition
    ax2 = plt.subplot(3, 2, 2)
    episodes = range(len(composition_data))
    bottoms = np.zeros(len(composition_data))
    
    # Group primitives (0-5) into one category for cleaner visualization
    prim_counts = [sum(ep[b] for b in range(6)) for ep in composition_data]
    ax2.bar(episodes, prim_counts, label="Primitives (Combined)", color='gray')
    bottoms += prim_counts
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    for i in range(6, agent.num_actions):
        opt_counts = [ep[i] for ep in composition_data]
        ax2.bar(episodes, opt_counts, bottom=bottoms, label=behavior_labels[i], color=colors[i-6])
        bottoms += opt_counts
        
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Step Count")
    ax2.set_title("Episode Composition")
    ax2.legend(loc='upper right', fontsize='small')

    # 3. Box-and-Whisker Plot of Option Durations
    ax3 = plt.subplot(3, 2, 3)
    duration_data = [option_durations[i] for i in range(6, agent.num_actions)]
    ax3.boxplot(duration_data, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='white'),
                medianprops=dict(color='red'))
    ax3.set_xticks(range(1, len(opt_labels) + 1))
    ax3.set_xticklabels(opt_labels)
    ax3.set_ylabel("Primitive Steps (k)")
    ax3.set_title("Distribution of Option Durations")

    # 4. Binned Line Graph of Average Option Duration
    ax4 = plt.subplot(3, 2, 4)
    bin_size = 10
    if len(avg_duration_per_eps) >= bin_size:
        binned_avgs = [np.mean(avg_duration_per_eps[i:i+bin_size]) for i in range(0, len(avg_duration_per_eps), bin_size)]
        bin_edges = range(0, len(avg_duration_per_eps), bin_size)
        ax4.plot(bin_edges, binned_avgs, marker='o', color='yellow')
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Average Steps per Option")
        ax4.set_title(f"Average Option Duration (Bin Size: {bin_size})")
        ax4.grid(alpha=0.3)

    # 5. Heatmap of Option Initiation States (Aggregated)
    # We will plot the 4 heatmaps in the remaining span
    for i in range(4):
        ax_heat = plt.subplot(3, 4, 9 + i)
        sns.heatmap(initiation_states[6 + i], cmap="magma", cbar=False, ax=ax_heat, annot=True, fmt="g")
        ax_heat.set_title(f"Initiation: {opt_labels[i]}")
        ax_heat.set_xlabel("Col")
        if i == 0:
            ax_heat.set_ylabel("Row")
            
    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(77)
    env = gym.make("Taxi-v4")
    env.action_space.seed(77)
    
    # Train the exact same way as your original script
    print("Training models...")
    location = env.unwrapped.locs
    loc_names = ["to_R", "to_G", "to_Y", "to_B"]
    trained_options = []
    
    for name, (row, col) in zip(loc_names, location):
        opt = Option(name, row, col)
        opt.option_training(env)
        trained_options.append(opt)
        print(f"Trained: {name}")
        
    agent = Agent(trained_options)
    agent.train_agent(env)
    print("Agent trained. Extracting evaluation data and plotting...")
    
    # Extract data and plot
    data = extract_evaluation_data(env, agent, n_eval_eps=100)
    plot_visualizations(agent, *data)


if __name__ == "__main__":
    main()