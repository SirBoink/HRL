"""
same as without pickup/dropoff just allowing the NN to select all actions.

"""

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt 
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
import os
import pickle

PICKUP_NAMES = {0: 'R', 1: 'G', 2: 'Y', 3: 'B', 4: 'Taxi'}
DEST_NAMES   = {0: 'R', 1: 'G', 2: 'Y', 3: 'B'}


class Option:
    def __init__(self, name, goal_row, goal_col, env=None):
        self.name = name
        self.n_eps = 100_000
        self.max_steps = 25
        self.goal_row = goal_row
        self.goal_col = goal_col

        self.q_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 6)
        )
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

        self.df = 0.9
        self.e = 1.0
        self.e_decay = self.e / (self.n_eps / 2)
        self.e_min = 0.1

    @staticmethod
    def _get_taxi_position(state):
        dest = state % 4
        state //= 4
        pass_idx = state % 5
        state //= 5
        taxi_col = state % 5
        taxi_row = state // 5
        return torch.tensor([taxi_row, taxi_col], dtype=torch.float32)

    def beta(self, state, env):
        decoded = list(env.unwrapped.decode(state))
        taxi_row, taxi_col = decoded[0], decoded[1]
        return taxi_row == self.goal_row and taxi_col == self.goal_col

    def pi(self, state):
        state_t = self._get_taxi_position(state)
        with torch.no_grad():
            q_vals = self.q_net(state_t)
        return torch.argmax(q_vals).item()

    def option_training(self, env):
        for eps in tqdm.tqdm(range(self.n_eps)):
            state, _ = env.reset()
            for steps in range(self.max_steps):
                if np.random.rand() < self.e:
                    action = np.random.choice([0, 1, 2, 3, 4, 5])
                else:
                    action = self.pi(state)

                next_state, _, terminated, truncated, _ = env.step(action)
                reached_goal = self.beta(next_state, env)
                own_reward = 10 if reached_goal else -1
                over = reached_goal or terminated or truncated

                s_tensor = self._get_taxi_position(state)
                ns_tensor = self._get_taxi_position(next_state)

                q_pred = self.q_net(s_tensor)[action]
                with torch.no_grad():
                    next_q = self.q_net(ns_tensor).max() if not over else torch.tensor(0.0)
                td_target = own_reward + self.df * next_q

                loss = self.loss_fn(q_pred, td_target)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state
                if over:
                    break
            self.e = max(self.e_min, self.e - self.e_decay)


class Agent:
    def __init__(self, options):
        self.n_eps = 100_000
        self.num_actions = 6 + len(options)
        self.valid_actions = list(range(self.num_actions))
        self.q_values = np.zeros((500, self.num_actions))
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

    def intra_option_update(self, env, state, action, reward, next_state, done,
                            executing_behavior=None, executing_k=None):
        for i, opt in enumerate(self.options):
            opt_behavior = 6 + i
            if opt.pi(state) == action:
                reached_goal = opt.beta(next_state, env)
                opt_terminated = reached_goal or done
                if opt_behavior == executing_behavior and executing_k is not None:
                    if executing_k >= opt.max_steps:
                        opt_terminated = True
                if done:
                    u_next = 0
                elif opt_terminated:
                    u_next = np.max(self.q_values[next_state, :])
                else:
                    u_next = self.q_values[next_state, opt_behavior]
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
                    self.smdp_update(state, behavior, reward, next_state, k_steps=1, done=done)
                    self.intra_option_update(env, state, behavior, reward, next_state, done)
                    state = next_state
                else:
                    opt = self.options[behavior - 6]
                    cumulative_reward = 0.0
                    discount = 1.0
                    k = 0
                    curr_state = state
                    while True:
                        action = opt.pi(curr_state)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        cumulative_reward += discount * reward
                        discount *= self.df
                        k += 1
                        self.intra_option_update(
                            env, curr_state, action, reward, next_state, done,
                            executing_behavior=behavior, executing_k=k
                        )
                        curr_state = next_state
                        if opt.beta(curr_state, env) or done or k >= opt.max_steps:
                            break
                    self.smdp_update(state, behavior, cumulative_reward, curr_state, k, done)
                    state = curr_state
            self.e = max(self.e_min, self.e - self.e_decay)


def get_trained_agent(env, loc_names, location):
    model_path = "taxi_agent_withpd.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            agent = pickle.load(f)
        print("Loaded existing agent from disk.")
        return agent

    print("No saved agent found. Commencing training.")
    trained_options = []
    for name, (row, col) in zip(loc_names, location):
        opt = Option(name, row, col)
        opt.option_training(env)
        trained_options.append(opt)
        print(f"trained: {name}.")

    agent = Agent(trained_options)
    agent.train_agent(env)

    with open(model_path, "wb") as f:
        pickle.dump(agent, f)
    print("Saved trained agent to disk.")
    return agent


def evaluate_agent(env, agent, n_eval_eps=500):
    successes = 0
    total_steps = 0
    option_stats = {opt.name: {"calls": 0, "steps": 0, "completed": 0} for opt in agent.options}
    primitive_calls = 0
    max_step_limit = env.spec.max_episode_steps if env.spec.max_episode_steps else 50

    all_episode_actions = []
    episode_destinations = []
    episode_pickups = []

    for eps in tqdm.tqdm(range(n_eval_eps), desc="Evaluating"):
        state, _ = env.reset(seed=42 + eps)
        decoded = list(env.unwrapped.decode(state))
        dest_idx = decoded[3]
        pass_idx = decoded[2]
        episode_destinations.append(dest_idx)
        episode_pickups.append(pass_idx)

        done = False
        steps = 0
        episode_success = False
        episode_actions = []

        while not done and steps < max_step_limit:
            behavior = agent.greedy(state)
            if behavior < 6:
                state, reward, terminated, truncated, _ = env.step(behavior)
                done = terminated or truncated
                steps += 1
                primitive_calls += 1
                episode_actions.append("primitive")
                if reward == 20:
                    episode_success = True
            else:
                opt = agent.options[behavior - 6]
                option_stats[opt.name]["calls"] += 1
                k = 0
                while True:
                    action = opt.pi(state)
                    state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    k += 1
                    steps += 1
                    episode_actions.append(opt.name)
                    if opt.beta(state, env) or done or k >= opt.max_steps:
                        if opt.beta(state, env):
                            option_stats[opt.name]["completed"] += 1
                        break
                option_stats[opt.name]["steps"] += k
                if reward == 20:
                    episode_success = True

        if episode_success:
            successes += 1
        total_steps += steps
        all_episode_actions.append(episode_actions)

    print("\n=== Execution Statistics ===")
    print(f"Success Rate: {(successes / n_eval_eps) * 100:.2f}%")
    print(f"Average Steps/Episode: {total_steps / n_eval_eps:.2f}")
    print(f"Total Primitive Action Calls: {primitive_calls}")

    names = list(option_stats.keys())
    calls = [stats["calls"] for stats in option_stats.values()]
    avg_steps = [stats["steps"] / stats["calls"] if stats["calls"] > 0 else 0 for stats in option_stats.values()]
    completion_rates = [(stats["completed"] / stats["calls"] * 100) if stats["calls"] > 0 else 0 for stats in option_stats.values()]

    ylabels = ['Number of Invocations', 'Average Steps per Invocation', 'Completion Rate (%)']
    for title, data, color, fname, ylabel in zip(
        ['Option Invocations', 'Average Duration', 'Sub-Goal Completion Rate'],
        [calls, avg_steps, completion_rates],
        ['steelblue', 'darkseagreen', 'indianred'],
        ['option_invocations.png', 'option_durations.png', 'option_completion_rates.png'],
        ylabels
    ):
        plt.figure(figsize=(8, 5))
        plt.bar(names, data, color=color)
        plt.title(title)
        plt.ylabel(ylabel)
        if title == 'Sub-Goal Completion Rate':
            plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()

    plot_episode_timelines(
        all_episode_actions,
        n_episodes=n_eval_eps,
        destinations=episode_destinations,
        pickups=episode_pickups,
        output_path='episode_timeline.png'
    )

    print("All statistics and timeline successfully saved.")


def plot_episode_timelines(all_episode_actions, n_episodes,
                           destinations=None, pickups=None,
                           output_path='episode_timeline.png'):
    color_map = {
        'primitive': '#AAAAAA',
        'to_R':       '#FF4444',
        'to_G':       '#44AA44',
        'to_Y':       '#FFAA00',
        'to_B':       '#4488FF'
    }

    fig_height = max(6, min(60, n_episodes * 0.08))
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
        ax.text(total_steps, ep_idx, f'{total_steps}',
                va='center', ha='left', fontsize=5)

    if pickups is not None and destinations is not None:
        y_labels = []
        for i, (p, d) in enumerate(zip(pickups, destinations)):
            pick_str = PICKUP_NAMES.get(p, "?")
            dest_str = DEST_NAMES.get(d, "?")
            y_labels.append(f'Ep {i+1} ({pick_str}, {dest_str})')
    else:
        y_labels = [f'Ep {i+1}' for i in range(n_episodes)]

    ax.set_yticks(range(n_episodes))
    ax.set_yticklabels(y_labels, fontsize=4)
    ax.set_ylim(-0.5, n_episodes - 0.5)
    ax.set_xlabel('Timesteps')
    ax.set_title(f'Action / Option Usage per Timestep ({n_episodes} Episodes)')

    all_lengths = [len(seq) for seq in all_episode_actions if seq]
    if all_lengths:
        ax.set_xlim(0, max(all_lengths) * 1.05)

    import matplotlib.patches as mpatches
    legend_patches = [mpatches.Patch(color=color_map[name], label=name) for name in color_map]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_episode_detail(env, agent, episode_number, output_path=None):
    color_map = {
        'primitive': np.array([0.80, 0.80, 0.80]),
        'to_R':      np.array([0.90, 0.50, 0.50]),
        'to_G':      np.array([0.50, 0.80, 0.50]),
        'to_Y':      np.array([0.90, 0.75, 0.40]),
        'to_B':      np.array([0.50, 0.65, 0.90])
    }
    default_color = np.array([1.0, 1.0, 1.0])

    eps_index = episode_number - 1
    seed = 42 + eps_index
    state, _ = env.reset(seed=seed)
    decoded = list(env.unwrapped.decode(state))
    start_row, start_col = decoded[0], decoded[1]
    pass_idx = decoded[2]
    dest_idx = decoded[3]

    locs = env.unwrapped.locs
    landmark_names = ["R", "G", "Y", "B"]

    landmark_colors = {
        "R": np.array([0.8, 0.0, 0.0]),
        "G": np.array([0.0, 0.6, 0.0]),
        "Y": np.array([0.8, 0.5, 0.0]),
        "B": np.array([0.0, 0.0, 0.8])
    }

    pickup_pos = locs[pass_idx] if pass_idx < 4 else None
    dest_pos   = locs[dest_idx]

    done = False
    step = 0
    max_steps = 200
    path = [(start_row, start_col)]
    cell_labels = {}

    while not done and step < max_steps:
        behavior = agent.greedy(state)
        if behavior < 6:
            next_state, reward, terminated, truncated, _ = env.step(behavior)
            done = terminated or truncated
            step += 1
            nxt_dec = list(env.unwrapped.decode(next_state))
            nxt_row, nxt_col = nxt_dec[0], nxt_dec[1]
            cell_labels[step] = (nxt_row, nxt_col, "primitive")
            path.append((nxt_row, nxt_col))
            state = next_state
        else:
            opt = agent.options[behavior - 6]
            opt_name = opt.name
            k = 0
            while True:
                action = opt.pi(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                k += 1
                step += 1
                nxt_dec = list(env.unwrapped.decode(next_state))
                nxt_row, nxt_col = nxt_dec[0], nxt_dec[1]
                cell_labels[step] = (nxt_row, nxt_col, opt_name)
                path.append((nxt_row, nxt_col))
                state = next_state
                if opt.beta(state, env) or done or k >= opt.max_steps:
                    break

    grid_img = np.ones((5, 5, 3))
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(grid_img, origin='upper', extent=[0, 5, 5, 0])
    
    cell_policies = {}
    for s in sorted(cell_labels.keys()):
        row, col, lbl = cell_labels[s]
        if 0 <= row < 5 and 0 <= col < 5:
            if (row, col) not in cell_policies:
                cell_policies[(row, col)] = []
            if lbl not in cell_policies[(row, col)]:
                cell_policies[(row, col)].append(lbl)

    import matplotlib.patches as patches
    for (row, col), policies in cell_policies.items():
        n = len(policies)
        height = 1.0 / n
        for i, p in enumerate(policies):
            rect = patches.Rectangle((col, row + i * height), 1, height, facecolor=color_map.get(p, default_color))
            ax.add_patch(rect)

    # Add Taxi v4 walls
    ax.vlines(x=2, ymin=0, ymax=2, color='black', linewidth=3)
    ax.vlines(x=1, ymin=3, ymax=5, color='black', linewidth=3)
    ax.vlines(x=3, ymin=3, ymax=5, color='black', linewidth=3)

    ax.set_xticks(np.arange(0.5, 5, 1))
    ax.set_yticks(np.arange(0.5, 5, 1))
    ax.set_xticklabels(range(5))
    ax.set_yticklabels(range(5))
    ax.set_xticks(np.arange(0, 6, 1), minor=True)
    ax.set_yticks(np.arange(0, 6, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.grid(which='major', visible=False)

    for (r, c), name in zip(locs, landmark_names):
        clr = landmark_colors[name]
        display_char = name
        if pickup_pos is not None and (r, c) == pickup_pos:
            display_char = "P"
        elif (r, c) == dest_pos:
            display_char = "D"

        ax.text(c + 0.5, r + 0.5, display_char,
                ha='center', va='center',
                fontsize=20, fontweight='bold', color=tuple(clr),
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    ax.text(start_col + 0.5, start_row + 0.20, 'S', ha='center', va='center',
            fontsize=12, fontweight='bold', color='black',
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    def cell_center(row, col):
        return (col + 0.5, row + 0.5)

    edge_counts = {}
    in_place_counts = {}
    for i in range(1, len(path)):
        prev_row, prev_col = path[i-1]
        curr_row, curr_col = path[i]
        x1, y1 = cell_center(prev_row, prev_col)
        x2, y2 = cell_center(curr_row, curr_col)

        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)

        if norm > 1e-6:
            edge = (prev_row, prev_col, curr_row, curr_col)
            count = edge_counts.get(edge, 0)
            edge_counts[edge] = count + 1
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle='->', mutation_scale=12,
                                    color='black', linewidth=1.5, alpha=0.85)
            ax.add_patch(arrow)

            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            perp_x = -dy / norm
            perp_y =  dx / norm
            offset = 0.20 + count * 0.15
            label_x = mid_x + offset * perp_x
            label_y = mid_y + offset * perp_y
        else:
            count = in_place_counts.get((curr_row, curr_col), 0)
            in_place_counts[(curr_row, curr_col)] = count + 1
            label_x = x1 + 0.25
            label_y = y1 + 0.25 + count * 0.15

        lbl = cell_labels[i][2]
        c = tuple(color_map.get(lbl, default_color))
        ax.text(label_x, label_y, str(i),
                ha='center', va='center',
                fontsize=8, fontweight='bold', color=c,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[name], label=name) for name in color_map]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    pick_str = PICKUP_NAMES.get(pass_idx, "?")
    dest_str = DEST_NAMES.get(dest_idx, "?")
    ax.set_title(f'Passenger at {pick_str}, destination at {dest_str}')

    if output_path is None:
        output_path = f'episode_{episode_number}_detail.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Detail plot saved as {output_path}")


def plot_option_q_policies(agent, env, output_path='option_q_policies.png'):
    direction_names = {0: '↓', 1: '↑', 2: '→', 3: '←', 4: 'P', 5: 'D'}
    action_colors = {0: '#d62728',
                     1: '#2ca02c',
                     2: '#ff7f0e',
                     3: '#1f77b4',
                     4: '#9467bd',
                     5: '#8c564b'}

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for idx, opt in enumerate(agent.options):
        ax = axes[idx]
        policy_grid = np.full((5, 5), -1, dtype=int)
        value_grid  = np.full((5, 5), np.nan)
        for row in range(5):
            for col in range(5):
                state_t = torch.tensor([row, col], dtype=torch.float32)
                with torch.no_grad():
                    q_vals = opt.q_net(state_t)
                action = torch.argmax(q_vals).item()
                policy_grid[row, col] = action
                value_grid[row, col] = q_vals.max().item()

        cmap = plt.cm.colors.ListedColormap(
            [action_colors[i] for i in range(6)]
        )
        ax.imshow(policy_grid, cmap=cmap, vmin=0, vmax=5,
                  origin='upper', extent=[0, 5, 5, 0])

        for row in range(5):
            for col in range(5):
                arrow = direction_names[policy_grid[row, col]]
                ax.text(col + 0.5, row + 0.5, arrow,
                        ha='center', va='center', fontsize=18,
                        color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=2, foreground='black')])

        ax.plot(opt.goal_col + 0.5, opt.goal_row + 0.5,
                marker='*', markersize=18, color='gold',
                markeredgecolor='black', markeredgewidth=1.2,
                zorder=10)

        # Add Taxi v4 walls
        ax.vlines(x=2, ymin=0, ymax=2, color='black', linewidth=3)
        ax.vlines(x=1, ymin=3, ymax=5, color='black', linewidth=3)
        ax.vlines(x=3, ymin=3, ymax=5, color='black', linewidth=3)

        ax.set_xticks(np.arange(0.5, 5, 1))
        ax.set_yticks(np.arange(0.5, 5, 1))
        ax.set_xticklabels(range(5))
        ax.set_yticklabels(range(5))
        ax.set_xticks(np.arange(0, 6, 1), minor=True)
        ax.set_yticks(np.arange(0, 6, 1), minor=True)
        ax.grid(which='minor', color='white', linewidth=1.5)
        ax.grid(which='major', visible=False)
        ax.set_title(f"Option {opt.name}  (goal: [{opt.goal_row},{opt.goal_col}])",
                     fontsize=12)

    legend_elements = [plt.Line2D([0], [0], marker=r'$\downarrow$',
                                  color='w', label='Down',
                                  markerfacecolor=action_colors[0], markersize=15),
                       plt.Line2D([0], [0], marker=r'$\uparrow$',
                                  color='w', label='Up',
                                  markerfacecolor=action_colors[1], markersize=15),
                       plt.Line2D([0], [0], marker=r'$\rightarrow$',
                                  color='w', label='Right',
                                  markerfacecolor=action_colors[2], markersize=15),
                       plt.Line2D([0], [0], marker=r'$\leftarrow$',
                                  color='w', label='Left',
                                  markerfacecolor=action_colors[3], markersize=15),
                       plt.Line2D([0], [0], marker='P',
                                  color='w', label='Pickup',
                                  markerfacecolor=action_colors[4], markersize=15),
                       plt.Line2D([0], [0], marker='D',
                                  color='w', label='Dropoff',
                                  markerfacecolor=action_colors[5], markersize=15)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Option Q‑Network Policies (greedy action per taxi position)",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Option policy maps saved as {output_path}")


def _draw_episode_on_ax(ax, env, agent, episode_number):
    color_map = {
        'primitive': np.array([0.80, 0.80, 0.80]),
        'to_R':      np.array([0.90, 0.50, 0.50]),
        'to_G':      np.array([0.50, 0.80, 0.50]),
        'to_Y':      np.array([0.90, 0.75, 0.40]),
        'to_B':      np.array([0.50, 0.65, 0.90])
    }
    default_color = np.array([1.0, 1.0, 1.0])

    eps_index = episode_number - 1
    seed = 42 + eps_index
    state, _ = env.reset(seed=seed)
    decoded = list(env.unwrapped.decode(state))
    start_row, start_col = decoded[0], decoded[1]
    pass_idx = decoded[2]
    dest_idx = decoded[3]

    locs = env.unwrapped.locs
    landmark_names = ["R", "G", "Y", "B"]
    landmark_colors = {
        "R": np.array([0.8, 0.0, 0.0]),
        "G": np.array([0.0, 0.6, 0.0]),
        "Y": np.array([0.8, 0.5, 0.0]),
        "B": np.array([0.0, 0.0, 0.8])
    }

    pickup_pos = locs[pass_idx] if pass_idx < 4 else None
    dest_pos   = locs[dest_idx]

    done = False
    step = 0
    max_steps = 200
    path = [(start_row, start_col)]
    cell_labels = {}

    while not done and step < max_steps:
        behavior = agent.greedy(state)
        if behavior < 6:
            next_state, reward, terminated, truncated, _ = env.step(behavior)
            done = terminated or truncated
            step += 1
            nxt_dec = list(env.unwrapped.decode(next_state))
            nxt_row, nxt_col = nxt_dec[0], nxt_dec[1]
            cell_labels[step] = (nxt_row, nxt_col, "primitive")
            path.append((nxt_row, nxt_col))
            state = next_state
        else:
            opt = agent.options[behavior - 6]
            opt_name = opt.name
            k = 0
            while True:
                action = opt.pi(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                k += 1
                step += 1
                nxt_dec = list(env.unwrapped.decode(next_state))
                nxt_row, nxt_col = nxt_dec[0], nxt_dec[1]
                cell_labels[step] = (nxt_row, nxt_col, opt_name)
                path.append((nxt_row, nxt_col))
                state = next_state
                if opt.beta(state, env) or done or k >= opt.max_steps:
                    break

    grid_img = np.ones((5, 5, 3))
    ax.imshow(grid_img, origin='upper', extent=[0, 5, 5, 0])
    
    cell_policies = {}
    for s in sorted(cell_labels.keys()):
        row, col, lbl = cell_labels[s]
        if 0 <= row < 5 and 0 <= col < 5:
            if (row, col) not in cell_policies:
                cell_policies[(row, col)] = []
            if lbl not in cell_policies[(row, col)]:
                cell_policies[(row, col)].append(lbl)

    import matplotlib.patches as patches
    for (row, col), policies in cell_policies.items():
        n = len(policies)
        height = 1.0 / n
        for i, p in enumerate(policies):
            rect = patches.Rectangle((col, row + i * height), 1, height, facecolor=color_map.get(p, default_color))
            ax.add_patch(rect)

    # Add Taxi v4 walls
    ax.vlines(x=2, ymin=0, ymax=2, color='black', linewidth=3)
    ax.vlines(x=1, ymin=3, ymax=5, color='black', linewidth=3)
    ax.vlines(x=3, ymin=3, ymax=5, color='black', linewidth=3)

    ax.set_xticks(np.arange(0.5, 5, 1))
    ax.set_yticks(np.arange(0.5, 5, 1))
    ax.set_xticklabels(range(5))
    ax.set_yticklabels(range(5))
    ax.set_xticks(np.arange(0, 6, 1), minor=True)
    ax.set_yticks(np.arange(0, 6, 1), minor=True)
    ax.grid(which='minor', color='black', linewidth=0.5)
    ax.grid(which='major', visible=False)

    for (r, c), name in zip(locs, landmark_names):
        clr = landmark_colors[name]
        display_char = name
        if pickup_pos is not None and (r, c) == pickup_pos:
            display_char = "P"
        elif (r, c) == dest_pos:
            display_char = "D"
        ax.text(c + 0.5, r + 0.5, display_char,
                ha='center', va='center',
                fontsize=14, fontweight='bold', color=tuple(clr),
                path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    ax.text(start_col + 0.5, start_row + 0.20, 'S', ha='center', va='center',
            fontsize=10, fontweight='bold', color='black',
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    def cell_center(row, col):
        return (col + 0.5, row + 0.5)

    edge_counts = {}
    in_place_counts = {}
    for i in range(1, len(path)):
        prev_row, prev_col = path[i-1]
        curr_row, curr_col = path[i]
        x1, y1 = cell_center(prev_row, prev_col)
        x2, y2 = cell_center(curr_row, curr_col)
        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)
        if norm > 1e-6:
            edge = (prev_row, prev_col, curr_row, curr_col)
            count = edge_counts.get(edge, 0)
            edge_counts[edge] = count + 1
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle='->', mutation_scale=10,
                                    color='black', linewidth=1.0, alpha=0.7)
            ax.add_patch(arrow)
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            perp_x = -dy / norm
            perp_y =  dx / norm
            offset = 0.18 + count * 0.12
            label_x = mid_x + offset * perp_x
            label_y = mid_y + offset * perp_y
        else:
            count = in_place_counts.get((curr_row, curr_col), 0)
            in_place_counts[(curr_row, curr_col)] = count + 1
            label_x = x1 + 0.25
            label_y = y1 + 0.25 + count * 0.15
            
        lbl = cell_labels[i][2]
        c = tuple(color_map.get(lbl, default_color))
        ax.text(label_x, label_y, str(i),
                ha='center', va='center',
                fontsize=6, fontweight='bold', color=c,
                path_effects=[pe.withStroke(linewidth=1.5, foreground="black")])

    pick_str = PICKUP_NAMES.get(pass_idx, "?")
    dest_str = DEST_NAMES.get(dest_idx, "?")
    ax.set_title(f'Pick:{pick_str}  Dest:{dest_str}', fontsize=8)


def plot_episodes_grid(env, agent, n_episodes=12, rows=3, cols=4,
                       output_path='first_12_episodes_grid.png'):
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    axes = axes.flatten()
    for i in range(n_episodes):
        _draw_episode_on_ax(axes[i], env, agent, episode_number=i+1)
    for j in range(n_episodes, rows*cols):
        axes[j].axis('off')

    color_map_legend = {
        'primitive': (0.80, 0.80, 0.80),
        'to_R':      (0.90, 0.50, 0.50),
        'to_G':      (0.50, 0.80, 0.50),
        'to_Y':      (0.90, 0.75, 0.40),
        'to_B':      (0.50, 0.65, 0.90)
    }
    legend_patches = [plt.Rectangle((0,0),1,1, facecolor=color_map_legend[name],
                                    label=name) for name in color_map_legend]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("First 12 Evaluation Episodes – Hierarchical Agent",
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"12‑episode detail grid saved as {output_path}")


def main():
    np.random.seed(77)
    env = gym.make("Taxi-v4")
    env.action_space.seed(77)

    location = env.unwrapped.locs
    loc_names = ["to_R", "to_G", "to_Y", "to_B"]

    agent = get_trained_agent(env, loc_names, location)

    evaluate_agent(env, agent, n_eval_eps=500)

    plot_option_q_policies(agent, env, output_path='option_q_policies.png')

    plot_episodes_grid(env, agent, n_episodes=12, rows=3, cols=4,
                       output_path='first_12_episodes_grid.png')

    env_render = gym.make("Taxi-v4", render_mode="rgb_array")
    env_render = RecordVideo(env_render, video_folder=".",
                             name_prefix="taxi_agent",
                             episode_trigger=lambda e: True)
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
