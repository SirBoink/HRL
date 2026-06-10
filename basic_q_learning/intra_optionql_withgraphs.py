import numpy as np
import gymnasium as gym
import tqdm
from gymnasium.wrappers import RecordVideo
import pickle
import matplotlib.pyplot as plt
import os

# ============================================================
#  Mapping for passenger locations and destinations
# ============================================================
PICKUP_NAMES = {0: "R", 1: "G", 2: "Y", 3: "B", 4: "Taxi"}   # passenger start
DEST_NAMES   = {0: "R", 1: "G", 2: "Y", 3: "B"}              # destination

class Option:
    def __init__(self, name, goal_row, goal_col):
        self.name = name
        self.n_eps = 300_000
        self.max_steps = 25
        self.goal_row = goal_row
        self.goal_col = goal_col
        self.q_values = np.zeros((500, 4))
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
                own_reward = 10 if reached_goal else -1
                over = reached_goal or terminated or truncated
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
                    k = 0
                    curr_state = state
                    while True:
                        action = opt.pi(curr_state)
                        next_state, reward, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        k += 1
                        self.intra_option_update(
                            env, curr_state, action, reward, next_state, done,
                            executing_behavior=behavior, executing_k=k
                        )
                        curr_state = next_state
                        if opt.beta(curr_state, env) or done or k >= opt.max_steps:
                            break
                    state = curr_state
            self.e = max(self.e_min, self.e - self.e_decay)


def get_trained_agent(env, loc_names, location):
    model_path = "taxi_agent.pkl"
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


def evaluate_agent(env, agent, n_eval_eps=500, max_display=50, display_seed=None):
    """
    Evaluates for n_eval_eps episodes, prints statistics, and generates
    a timeline plot from max_display RANDOMLY CHOSEN episodes.
    Each episode row shows pickup → destination for context.
    """
    successes = 0
    total_steps = 0
    option_stats = {opt.name: {"calls": 0, "steps": 0, "completed": 0} for opt in agent.options}
    primitive_calls = 0
    max_step_limit = env.spec.max_episode_steps if env.spec.max_episode_steps else 50

    all_episode_actions = []
    episode_destinations = []
    episode_pickups = []           # NEW: store passenger start location

    for eps in tqdm.tqdm(range(n_eval_eps), desc="Evaluating"):
        state, _ = env.reset(seed=42 + eps)
        decoded = list(env.unwrapped.decode(state))
        dest_idx = decoded[3]      # destination
        pass_idx = decoded[2]      # pickup location (0-4)
        episode_destinations.append(dest_idx)
        episode_pickups.append(pass_idx)

        done = False
        steps = 0
        episode_success = False
        episode_actions = []

        while not done and steps < max_step_limit:
            behavior = agent.greedy(state)
            if behavior < 6:                     # primitive
                state, reward, terminated, truncated, _ = env.step(behavior)
                done = terminated or truncated
                steps += 1
                primitive_calls += 1
                episode_actions.append("primitive")
                if reward == 20:
                    episode_success = True
            else:                                # option
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

    # ---- Console statistics ----
    print("\n=== Execution Statistics ===")
    print(f"Success Rate: {(successes / n_eval_eps) * 100:.2f}%")
    print(f"Average Steps/Episode: {total_steps / n_eval_eps:.2f}")
    print(f"Total Primitive Action Calls: {primitive_calls}")

    # ---- Bar charts (unchanged) ----
    names = list(option_stats.keys())
    calls = [stats["calls"] for stats in option_stats.values()]
    avg_steps = [stats["steps"] / stats["calls"] if stats["calls"] > 0 else 0 for stats in option_stats.values()]
    completion_rates = [(stats["completed"] / stats["calls"] * 100) if stats["calls"] > 0 else 0 for stats in option_stats.values()]

    plt.figure(figsize=(8, 5))
    plt.bar(names, calls, color='steelblue')
    plt.title('Option Invocations per Evaluation')
    plt.xlabel('Options')
    plt.ylabel('Number of Calls')
    plt.tight_layout()
    plt.savefig('option_invocations.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(names, avg_steps, color='darkseagreen')
    plt.title('Average Duration of Executed Options')
    plt.xlabel('Options')
    plt.ylabel('Average Steps')
    plt.tight_layout()
    plt.savefig('option_durations.png')
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(names, completion_rates, color='indianred')
    plt.title('Sub-Goal Completion Rate')
    plt.xlabel('Options')
    plt.ylabel('Completion Rate (%)')
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig('option_completion_rates.png')
    plt.close()

    # ---- RANDOM selection of episodes for the timeline plot ----
    rng = np.random.default_rng(display_seed)      # separate seed for selection
    indices = rng.choice(n_eval_eps, size=max_display, replace=False)
    indices = np.sort(indices)                     # sort for easier reading (optional)

    plot_actions   = [all_episode_actions[i] for i in indices]
    plot_dest      = [episode_destinations[i] for i in indices]
    plot_pickups   = [episode_pickups[i] for i in indices]

    plot_episode_timelines(
        plot_actions,
        n_episodes=max_display,
        destinations=plot_dest,
        pickups=plot_pickups,
        output_path='episode_timeline.png'
    )

    print("Option statistics and episode timeline successfully saved as .png files.")


def plot_episode_timelines(all_episode_actions, n_episodes,
                           destinations=None, pickups=None,
                           output_path='episode_timeline.png'):
    """
    Horizontal bar chart. Each row labelled as: Ep X (A, B)
    where A = passenger start, B = destination.
    """
    color_map = {
        'primitive': '#AAAAAA',
        'to_R':       '#FF4444',
        'to_G':       '#44AA44',
        'to_Y':       '#FFAA00',
        'to_B':       '#4488FF'
    }

    fig_height = min(30, n_episodes * 0.15)
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
                va='center', ha='left', fontsize=7)

    # Build y-tick labels: "Ep X (A, B)"
    if pickups is not None and destinations is not None:
        y_labels = []
        for i, (p, d) in enumerate(zip(pickups, destinations)):
            pick_str = PICKUP_NAMES.get(p, "?")
            dest_str = DEST_NAMES.get(d, "?")
            y_labels.append(f'Ep {i+1} ({pick_str}, {dest_str})')
    elif destinations is not None:
        y_labels = [f'Ep {i+1} ({DEST_NAMES[d]}, ?)' for i, d in enumerate(destinations)]
    else:
        y_labels = [f'Ep {i+1}' for i in range(n_episodes)]

    ax.set_yticks(range(n_episodes))
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_xlabel('Timesteps')
    ax.set_title(f'Action / Option Usage per Timestep')

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
    for s in sorted(cell_labels.keys()):
        row, col, lbl = cell_labels[s]
        if 0 <= row < 5 and 0 <= col < 5:
            grid_img[row, col] = color_map.get(lbl, default_color)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(grid_img, origin='lower', extent=[0, 5, 0, 5])
    ax.set_xticks(np.arange(0.5, 5, 1))
    ax.set_yticks(np.arange(0.5, 5, 1))
    ax.set_xticklabels(range(5))
    ax.set_yticklabels(range(5))
    ax.grid(color='black', linewidth=0.5)

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

    ax.text(start_col + 0.5, start_row + 0.80, 'S', ha='center', va='center',
            fontsize=12, fontweight='bold', color='black',
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    def cell_center(row, col):
        return (col + 0.5, row + 0.5)

    for i in range(1, len(path)):
        prev_row, prev_col = path[i-1]
        curr_row, curr_col = path[i]
        x1, y1 = cell_center(prev_row, prev_col)
        x2, y2 = cell_center(curr_row, curr_col)

        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)

        if norm > 1e-6:
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                    arrowstyle='->', mutation_scale=12,
                                    color='black', linewidth=1.5, alpha=0.85)
            ax.add_patch(arrow)

            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            perp_x = -dy / norm
            perp_y =  dx / norm
            offset = 0.20
            label_x = mid_x + offset * perp_x
            label_y = mid_y + offset * perp_y
        else:
            label_x = x1 + 0.25
            label_y = y1 + 0.25

        ax.text(label_x, label_y, str(i),
                ha='center', va='center',
                fontsize=8, fontweight='bold', color='black',
                path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[name], label=name) for name in color_map]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    pick_str = PICKUP_NAMES.get(pass_idx, "?")
    dest_str = DEST_NAMES.get(dest_idx, "?")
    ax.set_title(f'Passenger at Red, destination at Yellow')

    if output_path is None:
        output_path = f'episode_{episode_number}_detail.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Detail plot saved as {output_path}")


def main():
    np.random.seed(77)
    env = gym.make("Taxi-v4")
    env.action_space.seed(77)

    location = env.unwrapped.locs
    loc_names = ["to_R", "to_G", "to_Y", "to_B"]

    agent = get_trained_agent(env, loc_names, location)

    # Evaluate on 500 episodes, randomly select 50 for the timeline,
    # using a separate seed (change if needed, or set to None for fully random)
    evaluate_agent(env, agent, n_eval_eps=500, max_display=50, display_seed=42)

    # ---- Record a single example episode ----
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
    
    plot_episode_detail(env, agent, episode_number=492)


if __name__ == "__main__":
    main()