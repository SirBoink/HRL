"""
pure_options.py
─────────────────────────────────────────────────────────────────────────────
1. The agent's primitive actions exclude pickup (4) and dropoff (5).
2. The agent is forced to execute pickup and dropoff via option actions.
3. Option policies are trained with actions 0-5 and terminate conditionally
   based on the passenger and destination state, using a 4D state representation
   [taxi_row, taxi_col, passenger_location, destination].
─────────────────────────────────────────────────────────────────────────────
"""

from dataclasses import dataclass
import os
import pickle
import sys
from typing import Any, Dict, List, Tuple
import gymnasium as gym
from loguru import logger
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

# Configure loguru logger to match "The Lumina Standard"
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    level="INFO"
)

# Centralized plotting styles following "The Lumina Standard"
PLOT_STYLE: Dict[str, Any] = {
    "font.family": "sans-serif",
    "axes.edgecolor": "#CCCCCC",
    "axes.linewidth": 0.8,
    "grid.color": "#EAEAEA",
    "grid.linewidth": 0.5,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF"
}

OPTION_COLOR_MAP: Dict[str, str] = {
    "primitive": "#888888",
    "to_R": "#FF5252",  # Vibrant Coral Red
    "to_G": "#4CAF50",  # Vibrant Emerald Green
    "to_Y": "#FFC107",  # Vibrant Amber Yellow
    "to_B": "#2196F3"   # Vibrant Dodger Blue
}

LANDMARK_COLOR_MAP: Dict[str, Tuple[float, float, float]] = {
    "R": (0.8, 0.0, 0.0),
    "G": (0.0, 0.6, 0.0),
    "Y": (0.8, 0.5, 0.0),
    "B": (0.0, 0.0, 0.8)
}

ACTION_COLOR_MAP: Dict[int, str] = {
    0: "#D32F2F",  # Down (Red)
    1: "#388E3C",  # Up (Green)
    2: "#F57C00",  # Right (Orange)
    3: "#1976D2",  # Left (Blue)
    4: "#7B1FA2",  # Pickup (Purple)
    5: "#5D4037"   # Dropoff (Brown)
}

DIRECTION_NAMES: Dict[int, str] = {
    0: "↓",
    1: "↑",
    2: "→",
    3: "←",
    4: "P",
    5: "D"
}


@dataclass
class OptionConfig:
    """
    Configuration parameters for Option training.
    """
    total_episodes: int = 100_000
    max_steps_per_episode: int = 25
    learning_rate: float = 0.001
    discount_factor: float = 0.9
    initial_epsilon: float = 1.0
    epsilon_minimum: float = 0.1
    hidden_dimension: int = 64


@dataclass
class AgentConfig:
    """
    Configuration parameters for top-level Agent training.
    """
    total_episodes: int = 100_000
    learning_rate: float = 0.1
    discount_factor: float = 0.9
    initial_epsilon: float = 1.0
    epsilon_minimum: float = 0.1
    model_save_path: str = "pure_options/taxi_agent_pure.pkl"


class TaxiAdapter:
    """
    Adapter for the Gymnasium Taxi-v4 environment.
    Exposes semantic states and handles environment-specific decoding.
    """
    LANDMARK_COORDINATES: Dict[str, Tuple[int, int]] = {
        "R": (0, 0),
        "G": (0, 4),
        "Y": (4, 0),
        "B": (4, 3)
    }

    LANDMARK_NAMES: Dict[int, str] = {
        0: "R",
        1: "G",
        2: "Y",
        3: "B",
        4: "Taxi"
    }

    def __init__(self, environment: gym.Env):
        self.environment = environment

    def decode_state(self, state: int) -> Dict[str, int]:
        """
        Decodes a raw state integer into semantic parameters.

        Parameters
        ----------
        state : int
            Raw environment state.

        Returns
        -------
        dict
            Semantic state mapping containing taxi_row, taxi_col,
            passenger_location, and destination.
        """
        decoded = list(self.environment.unwrapped.decode(state))
        return {
            "taxi_row": decoded[0],
            "taxi_col": decoded[1],
            "passenger_location": decoded[2],
            "destination": decoded[3]
        }

    def get_landmark_index(self, row: int, col: int) -> int:
        """
        Retrieves the landmark index corresponding to the given row and column.

        Parameters
        ----------
        row : int
            Taxi row.
        col : int
            Taxi column.

        Returns
        -------
        int
            Landmark index (0 to 3).
        """
        for index, name in enumerate(["R", "G", "Y", "B"]):
            if self.LANDMARK_COORDINATES[name] == (row, col):
                return index
        raise ValueError(f"Coordinates ({row}, {col}) do not match any landmark.")

    def get_nn_state(self, state: int) -> torch.Tensor:
        """
        Transforms a raw state integer into a tensor suitable for option networks.

        Parameters
        ----------
        state : int
            Raw environment state.

        Returns
        -------
        torch.Tensor
            4D tensor representing the state: [row, col, pass_loc, dest].
        """
        semantic = self.decode_state(state)
        return torch.tensor([
            semantic["taxi_row"],
            semantic["taxi_col"],
            semantic["passenger_location"],
            semantic["destination"]
        ], dtype=torch.float32)


class Option:
    """
    Represents an option trained to navigate to a landmark and optionally
    execute pickup or dropoff if needed.
    """
    def __init__(
        self,
        name: str,
        goal_row: int,
        goal_col: int,
        config: OptionConfig,
        adapter: TaxiAdapter
    ):
        self.name: str = name
        self.goal_row: int = goal_row
        self.goal_col: int = goal_col
        self.config: OptionConfig = config
        self.adapter: TaxiAdapter = adapter

        # Neural Network: 4 inputs -> 6 actions (0-5)
        self.q_network = nn.Sequential(
            nn.Linear(4, config.hidden_dimension),
            nn.ReLU(),
            nn.Linear(config.hidden_dimension, 6)
        )
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.loss_function = nn.MSELoss()

        self.epsilon: float = config.initial_epsilon
        self.epsilon_decay: float = config.initial_epsilon / (config.total_episodes / 2)
        self.epsilon_minimum: float = config.epsilon_minimum
        self.discount_factor: float = config.discount_factor

    def beta(self, state: int) -> bool:
        """
        Termination condition (beta) for the option.

        Parameters
        ----------
        state : int
            The raw environment state.

        Returns
        -------
        bool
            True if the option terminates, False otherwise.
        """
        semantic = self.adapter.decode_state(state)
        taxi_row = semantic["taxi_row"]
        taxi_col = semantic["taxi_col"]
        passenger_location = semantic["passenger_location"]
        destination = semantic["destination"]

        is_at_goal = (taxi_row == self.goal_row and taxi_col == self.goal_col)
        if not is_at_goal:
            return False

        landmark_index = self.adapter.get_landmark_index(self.goal_row, self.goal_col)

        # Case 1: Passenger is at this landmark (and not in taxi).
        # We must execute pickup before terminating, so do not terminate yet.
        if passenger_location == landmark_index:
            return False

        # Case 2: Passenger is in the taxi and this landmark is the destination.
        # We must execute dropoff before terminating, so do not terminate yet.
        if passenger_location == 4 and destination == landmark_index:
            return False

        return True

    def pi(self, state: int) -> int:
        """
        Policy (pi) of the option, selecting the greedy action.

        Parameters
        ----------
        state : int
            The raw environment state.

        Returns
        -------
        int
            Selected action (0 to 5).
        """
        state_tensor = self.adapter.get_nn_state(state)
        with torch.no_grad():
            action_values = self.q_network(state_tensor)
        return int(torch.argmax(action_values).item())

    def train_option(self, environment: gym.Env) -> None:
        """
        Trains the option's Q-network using standard Q-learning with epsilon-greedy.

        Parameters
        ----------
        environment : gym.Env
            The gym training environment.
        """
        logger.info(f"Starting training for option {self.name}...")
        for episode in tqdm.tqdm(range(self.config.total_episodes), desc=f"Training {self.name}"):
            state, _ = environment.reset()
            for step_index in range(self.config.max_steps_per_episode):
                if np.random.rand() < self.epsilon:
                    action = np.random.choice([0, 1, 2, 3, 4, 5])
                else:
                    action = self.pi(state)

                next_state, _, terminated, truncated, _ = environment.step(action)
                reached_goal = self.beta(next_state)
                
                own_reward = 10.0 if reached_goal else -1.0
                is_over = reached_goal or terminated or truncated

                state_tensor = self.adapter.get_nn_state(state)
                next_state_tensor = self.adapter.get_nn_state(next_state)

                predicted_q = self.q_network(state_tensor)[action]
                with torch.no_grad():
                    if not is_over:
                        next_q = self.q_network(next_state_tensor).max()
                    else:
                        next_q = torch.tensor(0.0)
                target_q = own_reward + self.discount_factor * next_q

                loss = self.loss_function(predicted_q, target_q)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                state = next_state
                if is_over:
                    break

            self.epsilon = max(self.epsilon_minimum, self.epsilon - self.epsilon_decay)
        logger.success(f"Option {self.name} training complete.")


class Agent:
    """
    Hierarchical agent utilizing SMDP and Intra-Option Q-learning.
    The agent is restricted from selecting primitive pickup and dropoff actions.
    """
    def __init__(self, options: List[Option], config: AgentConfig, adapter: TaxiAdapter):
        self.options: List[Option] = options
        self.config: AgentConfig = config
        self.adapter: TaxiAdapter = adapter

        # Action space size: 6 primitives + len(options) = 10
        self.num_actions: int = 6 + len(options)
        # Banned primitive actions: 4 (pickup) and 5 (dropoff).
        # Valid actions are [0, 1, 2, 3] and [6, 7, 8, 9].
        self.valid_actions: List[int] = [0, 1, 2, 3] + [6 + index for index in range(len(options))]

        # Q-table: 500 states x 10 actions
        self.q_values: np.ndarray = np.zeros((500, self.num_actions))

        self.learning_rate: float = config.learning_rate
        self.discount_factor: float = config.discount_factor
        self.epsilon: float = config.initial_epsilon
        self.epsilon_decay: float = config.initial_epsilon / (config.total_episodes / 2)
        self.epsilon_minimum: float = config.epsilon_minimum

    def greedy(self, state: int) -> int:
        """
        Selects the action with the highest Q-value among the valid actions.

        Parameters
        ----------
        state : int
            The current state.

        Returns
        -------
        int
            The selected valid action index.
        """
        valid_action_values = [self.q_values[state, action] for action in self.valid_actions]
        best_index = int(np.argmax(valid_action_values))
        return self.valid_actions[best_index]

    def choose_action(self, state: int) -> int:
        """
        Epsilon-greedy action selection.

        Parameters
        ----------
        state : int
            The current state.

        Returns
        -------
        int
            The selected action index.
        """
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(self.valid_actions))
        return self.greedy(state)

    def smdp_update(
        self,
        state: int,
        behavior: int,
        cumulative_reward: float,
        next_state: int,
        steps_taken: int,
        done: bool
    ) -> None:
        """
        Performs SMDP Q-learning update.

        Parameters
        ----------
        state : int
            The initial state.
        behavior : int
            The action or option executed.
        cumulative_reward : float
            The discounted cumulative reward received.
        next_state : int
            The state after execution.
        steps_taken : int
            The number of environment steps taken.
        done : bool
            Whether the episode has terminated.
        """
        if done:
            best_next_q = 0.0
        else:
            best_next_q = np.max([self.q_values[next_state, action] for action in self.valid_actions])

        td_target = cumulative_reward + (self.discount_factor ** steps_taken) * best_next_q
        td_error = td_target - self.q_values[state, behavior]
        self.q_values[state, behavior] += self.learning_rate * td_error

    def intra_option_update(
        self,
        environment: gym.Env,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        executing_behavior: int = None,
        executing_steps: int = None
    ) -> None:
        """
        Performs intra-option updates for all consistent options.

        Parameters
        ----------
        environment : gym.Env
            The environment instance.
        state : int
            The current state.
        action : int
            The primitive action executed.
        reward : float
            The reward received.
        next_state : int
            The next state.
        done : bool
            Whether the episode has terminated.
        executing_behavior : int, optional
            The option index currently executing, if any.
        executing_steps : int, optional
            The number of steps executed by the current option.
        """
        for index, option in enumerate(self.options):
            option_behavior = 6 + index

            # Check if option policy is consistent with the primitive action taken
            if option.pi(state) == action:
                reached_goal = option.beta(next_state)
                option_terminated = reached_goal or done

                # Check for max step limit termination
                if option_behavior == executing_behavior and executing_steps is not None:
                    if executing_steps >= option.config.max_steps_per_episode:
                        option_terminated = True

                if done:
                    u_next = 0.0
                elif option_terminated:
                    u_next = np.max([self.q_values[next_state, act] for act in self.valid_actions])
                else:
                    u_next = self.q_values[next_state, option_behavior]

                td_target = reward + self.discount_factor * u_next
                td_error = td_target - self.q_values[state, option_behavior]
                self.q_values[state, option_behavior] += self.learning_rate * td_error

    def train_agent(self, environment: gym.Env) -> None:
        """
        Trains the hierarchical agent using SMDP and Intra-Option Q-learning.

        Parameters
        ----------
        environment : gym.Env
            The gym training environment.
        """
        logger.info("Starting training for top-level agent...")
        for episode in tqdm.tqdm(range(self.config.total_episodes), desc="Training Agent"):
            state, _ = environment.reset()
            done = False
            while not done:
                behavior = self.choose_action(state)
                if behavior < 6:
                    # Primitive navigation action
                    next_state, reward, terminated, truncated, _ = environment.step(behavior)
                    done = terminated or truncated

                    self.smdp_update(state, behavior, reward, next_state, steps_taken=1, done=done)
                    self.intra_option_update(environment, state, behavior, reward, next_state, done)
                    state = next_state
                else:
                    # Option execution
                    option = self.options[behavior - 6]
                    cumulative_reward = 0.0
                    discount = 1.0
                    steps_taken = 0
                    current_state = state

                    while True:
                        action = option.pi(current_state)
                        next_state, reward, terminated, truncated, _ = environment.step(action)
                        done = terminated or truncated

                        cumulative_reward += discount * reward
                        discount *= self.discount_factor
                        steps_taken += 1

                        # Step-by-step intra-option update
                        self.intra_option_update(
                            environment,
                            current_state,
                            action,
                            reward,
                            next_state,
                            done,
                            executing_behavior=behavior,
                            executing_steps=steps_taken
                        )
                        current_state = next_state

                        if option.beta(current_state) or done or steps_taken >= option.config.max_steps_per_episode:
                            break

                    self.smdp_update(state, behavior, cumulative_reward, current_state, steps_taken, done)
                    state = current_state

            self.epsilon = max(self.epsilon_minimum, self.epsilon - self.epsilon_decay)
        logger.success("Agent training complete.")


def get_trained_agent(
    environment: gym.Env,
    config_option: OptionConfig,
    config_agent: AgentConfig,
    adapter: TaxiAdapter
) -> Agent:
    """
    Loads a trained agent from disk if it exists, otherwise trains a new one.

    Parameters
    ----------
    environment : gym.Env
        The gym environment.
    config_option : OptionConfig
        The option configuration.
    config_agent : AgentConfig
        The agent configuration.
    adapter : TaxiAdapter
        The environment adapter.

    Returns
    -------
    Agent
        The trained agent.
    """
    model_path = config_agent.model_save_path
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as file_handle:
                agent = pickle.load(file_handle)
            logger.info("Loaded existing agent from disk.")
            return agent
        except Exception:
            logger.exception("Failed to load agent from disk. Training from scratch.")

    logger.info("No saved agent found. Commencing training.")

    landmark_names = ["to_R", "to_G", "to_Y", "to_B"]
    landmark_coords = [(0, 0), (0, 4), (4, 0), (4, 3)]

    trained_options = []
    for name, (row, col) in zip(landmark_names, landmark_coords):
        option = Option(name, row, col, config_option, adapter)
        option.train_option(environment)
        trained_options.append(option)

    agent = Agent(trained_options, config_agent, adapter)
    agent.train_agent(environment)

    # Ensure directory exists before saving
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as file_handle:
        pickle.dump(agent, file_handle)
    logger.success("Saved trained agent to disk.")
    return agent


def plot_episode_timelines(
    all_episode_actions: List[List[str]],
    total_episodes: int,
    destinations: List[int] = None,
    pickups: List[int] = None,
    output_path: str = "pure_options/episode_timeline.png"
) -> None:
    """
    Creates a horizontal timeline bar chart showing action and option usage
    across all timesteps for all evaluation episodes.
    """
    plt.rcParams.update(PLOT_STYLE)
    figure_height = max(6.0, min(60.0, total_episodes * 0.08))
    fig, ax = plt.subplots(figsize=(14, figure_height))

    for episode_idx, sequence in enumerate(all_episode_actions):
        if not sequence:
            continue
        start_step = 0
        current_label = sequence[0]
        for step_idx in range(1, len(sequence)):
            if sequence[step_idx] != current_label:
                ax.barh(
                    episode_idx,
                    step_idx - start_step,
                    left=start_step,
                    height=0.8,
                    color=OPTION_COLOR_MAP.get(current_label, "black"),
                    edgecolor="none"
                )
                start_step = step_idx
                current_label = sequence[step_idx]
        ax.barh(
            episode_idx,
            len(sequence) - start_step,
            left=start_step,
            height=0.8,
            color=OPTION_COLOR_MAP.get(current_label, "black"),
            edgecolor="none"
        )
        total_steps = len(sequence)
        ax.text(
            total_steps,
            episode_idx,
            f" {total_steps}",
            va="center",
            ha="left",
            fontsize=6,
            color="#333333"
        )

    pickup_names = {0: "R", 1: "G", 2: "Y", 3: "B", 4: "Taxi"}
    destination_names = {0: "R", 1: "G", 2: "Y", 3: "B"}

    if pickups is not None and destinations is not None:
        y_labels = []
        for index, (pickup_val, dest_val) in enumerate(zip(pickups, destinations)):
            pickup_str = pickup_names.get(pickup_val, "?")
            dest_str = destination_names.get(dest_val, "?")
            y_labels.append(f"Ep {index + 1} ({pickup_str} -> {dest_str})")
    else:
        y_labels = [f"Ep {index + 1}" for index in range(total_episodes)]

    ax.set_yticks(range(total_episodes))
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.set_ylim(-0.5, total_episodes - 0.5)
    ax.set_xlabel("Timesteps")
    ax.set_title(f"Action / Option Usage per Timestep ({total_episodes} Episodes)")

    all_lengths = [len(seq) for seq in all_episode_actions if seq]
    if all_lengths:
        ax.set_xlim(0, max(all_lengths) * 1.05)

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=OPTION_COLOR_MAP[name], label=name)
        for name in OPTION_COLOR_MAP
    ]
    ax.legend(
        handles=legend_patches,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
        fontsize=8
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _draw_episode_on_ax(
    ax: plt.Axes,
    environment: gym.Env,
    agent: Agent,
    episode_number: int,
    adapter: TaxiAdapter
) -> None:
    """
    Helper function to draw an episode's path onto a matplotlib axis.
    """
    episode_index = episode_number - 1
    seed = 42 + episode_index
    state, _ = environment.reset(seed=seed)

    semantic = adapter.decode_state(state)
    start_row = semantic["taxi_row"]
    start_col = semantic["taxi_col"]
    passenger_location = semantic["passenger_location"]
    destination = semantic["destination"]

    landmark_names = ["R", "G", "Y", "B"]
    landmark_positions = environment.unwrapped.locs

    pickup_position = landmark_positions[passenger_location] if passenger_location < 4 else None
    destination_position = landmark_positions[destination]

    done = False
    step = 0
    max_steps = 200
    path = [(start_row, start_col)]
    cell_labels: Dict[int, Tuple[int, int, str]] = {}

    while not done and step < max_steps:
        behavior = agent.greedy(state)
        if behavior < 6:
            next_state, _, terminated, truncated, _ = environment.step(behavior)
            done = terminated or truncated
            step += 1
            next_semantic = adapter.decode_state(next_state)
            next_row, next_col = next_semantic["taxi_row"], next_semantic["taxi_col"]
            cell_labels[step] = (next_row, next_col, "primitive")
            path.append((next_row, next_col))
            state = next_state
        else:
            option = agent.options[behavior - 6]
            option_name = option.name
            steps_taken = 0
            while True:
                action = option.pi(state)
                next_state, _, terminated, truncated, _ = environment.step(action)
                done = terminated or truncated
                steps_taken += 1
                step += 1
                next_semantic = adapter.decode_state(next_state)
                next_row, next_col = next_semantic["taxi_row"], next_semantic["taxi_col"]
                cell_labels[step] = (next_row, next_col, option_name)
                path.append((next_row, next_col))
                state = next_state
                if option.beta(state) or done or steps_taken >= option.config.max_steps_per_episode:
                    break

    grid_image = np.ones((5, 5, 3))
    ax.imshow(grid_image, origin="upper", extent=[0, 5, 5, 0])

    cell_policies: Dict[Tuple[int, int], List[str]] = {}
    for step_num in sorted(cell_labels.keys()):
        row, col, label = cell_labels[step_num]
        if 0 <= row < 5 and 0 <= col < 5:
            if (row, col) not in cell_policies:
                cell_policies[(row, col)] = []
            if label not in cell_policies[(row, col)]:
                cell_policies[(row, col)].append(label)

    import matplotlib.patches as patches
    for (row, col), policies in cell_policies.items():
        count = len(policies)
        height = 1.0 / count
        for idx, policy_name in enumerate(policies):
            rect = patches.Rectangle(
                (col, row + idx * height),
                1.0,
                height,
                facecolor=OPTION_COLOR_MAP.get(policy_name, "#FFFFFF")
            )
            ax.add_patch(rect)

    # Draw Taxi-v4 walls
    ax.vlines(x=2, ymin=0, ymax=2, color="black", linewidth=3)
    ax.vlines(x=1, ymin=3, ymax=5, color="black", linewidth=3)
    ax.vlines(x=3, ymin=3, ymax=5, color="black", linewidth=3)

    ax.set_xticks(np.arange(0.5, 5, 1))
    ax.set_yticks(np.arange(0.5, 5, 1))
    ax.set_xticklabels(range(5))
    ax.set_yticklabels(range(5))
    ax.set_xticks(np.arange(0, 6, 1), minor=True)
    ax.set_yticks(np.arange(0, 6, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.5)
    ax.grid(which="major", visible=False)

    for idx, (row, col) in enumerate(landmark_positions):
        name = landmark_names[idx]
        color = LANDMARK_COLOR_MAP[name]
        display_char = name
        if pickup_position is not None and (row, col) == pickup_position:
            display_char = "P"
        elif (row, col) == destination_position:
            display_char = "D"

        ax.text(
            col + 0.5,
            row + 0.5,
            display_char,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color=color,
            path_effects=[pe.withStroke(linewidth=2.5, foreground="white")]
        )

    # Start symbol 'S'
    ax.text(
        start_col + 0.5,
        start_row + 0.20,
        "S",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="black",
        path_effects=[pe.withStroke(linewidth=2, foreground="white")]
    )

    def get_cell_center(row_val: int, col_val: int) -> Tuple[float, float]:
        return col_val + 0.5, row_val + 0.5

    edge_counts: Dict[Tuple[int, int, int, int], int] = {}
    in_place_counts: Dict[Tuple[int, int], int] = {}

    for idx in range(1, len(path)):
        prev_row, prev_col = path[idx - 1]
        curr_row, curr_col = path[idx]
        x1, y1 = get_cell_center(prev_row, prev_col)
        x2, y2 = get_cell_center(curr_row, curr_col)

        dx, dy = x2 - x1, y2 - y1
        norm = np.hypot(dx, dy)

        if norm > 1e-6:
            edge = (prev_row, prev_col, curr_row, curr_col)
            count = edge_counts.get(edge, 0)
            edge_counts[edge] = count + 1
            arrow = FancyArrowPatch(
                (x1, y1), (x2, y2),
                arrowstyle="->",
                mutation_scale=10,
                color="black",
                linewidth=1.0,
                alpha=0.7
            )
            ax.add_patch(arrow)

            mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            perp_x = -dy / norm
            perp_y = dx / norm
            offset = 0.18 + count * 0.12
            label_x = mid_x + offset * perp_x
            label_y = mid_y + offset * perp_y
        else:
            count = in_place_counts.get((curr_row, curr_col), 0)
            in_place_counts[(curr_row, curr_col)] = count + 1
            label_x = x1 + 0.25
            label_y = y1 + 0.25 + count * 0.15

        lbl = cell_labels[idx][2]
        color = OPTION_COLOR_MAP.get(lbl, "#FFFFFF")
        ax.text(
            label_x,
            label_y,
            str(idx),
            ha="center",
            va="center",
            fontsize=6,
            fontweight="bold",
            color=color,
            path_effects=[pe.withStroke(linewidth=1.5, foreground="black")]
        )

    pickup_names = {0: "R", 1: "G", 2: "Y", 3: "B", 4: "Taxi"}
    destination_names = {0: "R", 1: "G", 2: "Y", 3: "B"}

    pickup_str = pickup_names.get(passenger_location, "?")
    destination_str = destination_names.get(destination, "?")
    ax.set_title(f"Pick: {pickup_str}  Dest: {destination_str}", fontsize=8)


def plot_episodes_grid(
    environment: gym.Env,
    agent: Agent,
    adapter: TaxiAdapter,
    total_episodes: int = 12,
    rows: int = 3,
    cols: int = 4,
    output_path: str = "pure_options/first_12_episodes_grid.png"
) -> None:
    """
    Generates a grid visualization of evaluation episodes.
    """
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 3.5 * rows))
    axes = axes.flatten()

    for episode_idx in range(total_episodes):
        _draw_episode_on_ax(
            axes[episode_idx],
            environment,
            agent,
            episode_number=episode_idx + 1,
            adapter=adapter
        )

    for idx in range(total_episodes, rows * cols):
        axes[idx].axis("off")

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=OPTION_COLOR_MAP[name], label=name)
        for name in OPTION_COLOR_MAP
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.03)
    )
    fig.suptitle(
        "First 12 Evaluation Episodes - Hierarchical Agent (Pure Options)",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"12-episode detail grid saved to {output_path}")


def plot_option_q_policies(
    agent: Agent,
    adapter: TaxiAdapter,
    output_path: str = "pure_options/option_q_policies.png"
) -> None:
    """
    Visualizes the greedy policy of each option's internal Q-network
    over the 5x5 grid using arrows and heatmaps.
    """
    plt.rcParams.update(PLOT_STYLE)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for index, option in enumerate(agent.options):
        ax = axes[index]
        policy_grid = np.full((5, 5), -1, dtype=int)

        goal_landmark_index = adapter.get_landmark_index(option.goal_row, option.goal_col)
        passenger_location = goal_landmark_index
        destination = (goal_landmark_index + 1) % 4

        for row in range(5):
            for col in range(5):
                state_tensor = torch.tensor(
                    [row, col, passenger_location, destination],
                    dtype=torch.float32
                )
                with torch.no_grad():
                    q_values = option.q_network(state_tensor)
                policy_grid[row, col] = int(torch.argmax(q_values).item())

        cmap = ListedColormap(
            [ACTION_COLOR_MAP[i] for i in range(6)]
        )
        ax.imshow(policy_grid, cmap=cmap, vmin=0, vmax=5, origin='upper', extent=[0, 5, 5, 0])

        for row in range(5):
            for col in range(5):
                action = policy_grid[row, col]
                label = DIRECTION_NAMES[action]
                ax.text(
                    col + 0.5,
                    row + 0.5,
                    label,
                    ha='center',
                    va='center',
                    fontsize=18,
                    color='white',
                    fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=2, foreground='black')]
                )

        ax.plot(
            option.goal_col + 0.5,
            option.goal_row + 0.5,
            marker='*',
            markersize=18,
            color='gold',
            markeredgecolor='black',
            markeredgewidth=1.2,
            zorder=10
        )

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
        ax.set_title(
            f"Option {option.name} (Goal: [{option.goal_row},{option.goal_col}])",
            fontsize=12,
            fontweight='bold'
        )

    legend_elements = [
        plt.Line2D(
            [0], [0],
            marker=r'$\downarrow$',
            color='w',
            label='Down',
            markerfacecolor=ACTION_COLOR_MAP[0],
            markersize=15
        ),
        plt.Line2D(
            [0], [0],
            marker=r'$\uparrow$',
            color='w',
            label='Up',
            markerfacecolor=ACTION_COLOR_MAP[1],
            markersize=15
        ),
        plt.Line2D(
            [0], [0],
            marker=r'$\rightarrow$',
            color='w',
            label='Right',
            markerfacecolor=ACTION_COLOR_MAP[2],
            markersize=15
        ),
        plt.Line2D(
            [0], [0],
            marker=r'$\leftarrow$',
            color='w',
            label='Left',
            markerfacecolor=ACTION_COLOR_MAP[3],
            markersize=15
        ),
        plt.Line2D(
            [0], [0],
            marker='P',
            color='w',
            label='Pickup',
            markerfacecolor=ACTION_COLOR_MAP[4],
            markersize=12
        ),
        plt.Line2D(
            [0], [0],
            marker='D',
            color='w',
            label='Dropoff',
            markerfacecolor=ACTION_COLOR_MAP[5],
            markersize=12
        )
    ]
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=6,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02)
    )
    fig.suptitle(
        "Option Q-Network Policies (with Pickup / Dropoff Actions)",
        fontsize=14,
        fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.success(f"Option policy maps saved to {output_path}")


def evaluate_agent(
    environment: gym.Env,
    agent: Agent,
    adapter: TaxiAdapter,
    total_evaluation_episodes: int = 500
) -> None:
    """
    Evaluates the hierarchical agent over multiple episodes, computes
    performance statistics, and generates evaluation plots.
    """
    successes = 0
    total_steps = 0
    option_statistics = {
        option.name: {"calls": 0, "steps": 0, "completed": 0}
        for option in agent.options
    }
    primitive_calls = 0
    max_step_limit = environment.spec.max_episode_steps if environment.spec.max_episode_steps else 200

    all_episode_actions = []
    episode_destinations = []
    episode_pickups = []

    for episode in tqdm.tqdm(range(total_evaluation_episodes), desc="Evaluating"):
        state, _ = environment.reset(seed=42 + episode)
        semantic = adapter.decode_state(state)
        destination_index = semantic["destination"]
        passenger_index = semantic["passenger_location"]

        episode_destinations.append(destination_index)
        episode_pickups.append(passenger_index)

        done = False
        steps = 0
        episode_success = False
        episode_actions = []

        while not done and steps < max_step_limit:
            behavior = agent.greedy(state)
            if behavior < 6:
                state, reward, terminated, truncated, _ = environment.step(behavior)
                done = terminated or truncated
                steps += 1
                primitive_calls += 1
                episode_actions.append("primitive")
                if reward == 20:
                    episode_success = True
            else:
                option = agent.options[behavior - 6]
                option_statistics[option.name]["calls"] += 1
                steps_taken = 0
                while True:
                    action = option.pi(state)
                    state, reward, terminated, truncated, _ = environment.step(action)
                    done = terminated or truncated
                    steps_taken += 1
                    steps += 1
                    episode_actions.append(option.name)

                    if option.beta(state) or done or steps_taken >= option.config.max_steps_per_episode:
                        if option.beta(state):
                            option_statistics[option.name]["completed"] += 1
                        break
                option_statistics[option.name]["steps"] += steps_taken
                if reward == 20:
                    episode_success = True

        if episode_success:
            successes += 1
        total_steps += steps
        all_episode_actions.append(episode_actions)

    success_rate = (successes / total_evaluation_episodes) * 100.0
    average_steps = total_steps / total_evaluation_episodes

    logger.info("=== Execution Statistics ===")
    logger.info(f"Success Rate: {success_rate:.2f}%")
    logger.info(f"Average Steps/Episode: {average_steps:.2f}")
    logger.info(f"Total Primitive Action Calls: {primitive_calls}")

    names = list(option_statistics.keys())
    calls = [stats["calls"] for stats in option_statistics.values()]
    avg_steps = [
        stats["steps"] / stats["calls"] if stats["calls"] > 0 else 0
        for stats in option_statistics.values()
    ]
    completion_rates = [
        (stats["completed"] / stats["calls"] * 100) if stats["calls"] > 0 else 0
        for stats in option_statistics.values()
    ]

    ylabels = ['Number of Invocations', 'Average Steps per Invocation', 'Completion Rate (%)']
    for title, data, color, filename, ylabel in zip(
        ['Option Invocations', 'Average Duration', 'Sub-Goal Completion Rate'],
        [calls, avg_steps, completion_rates],
        ['steelblue', 'darkseagreen', 'indianred'],
        ['option_invocations.png', 'option_durations.png', 'option_completion_rates.png'],
        ylabels
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(names, data, color=color)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        if title == 'Sub-Goal Completion Rate':
            ax.set_ylim(0, 105)
        plt.tight_layout()
        output_filepath = f"pure_options/{filename}"
        plt.savefig(output_filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

    plot_episode_timelines(
        all_episode_actions,
        total_episodes=total_evaluation_episodes,
        destinations=episode_destinations,
        pickups=episode_pickups,
        output_path='pure_options/episode_timeline.png'
    )

    logger.success("All evaluation statistics and plots successfully saved.")


def main() -> None:
    """
    Main entry point for training and evaluating the pure options agent.
    """
    # Seed for reproducibility
    np.random.seed(77)
    torch.manual_seed(77)

    # Initialize environment
    environment = gym.make("Taxi-v4")
    environment.action_space.seed(77)

    # Instantiate configs
    config_option = OptionConfig()
    config_agent = AgentConfig()

    # Instantiate adapter
    adapter = TaxiAdapter(environment)

    # Train or load the agent
    agent = get_trained_agent(environment, config_option, config_agent, adapter)

    # Evaluate the agent
    evaluate_agent(environment, agent, adapter, total_evaluation_episodes=500)

    # Plot option policies
    plot_option_q_policies(agent, adapter, output_path='pure_options/option_q_policies.png')

    # Plot grid of first 12 episodes
    plot_episodes_grid(
        environment,
        agent,
        adapter,
        total_episodes=12,
        rows=3,
        cols=4,
        output_path='pure_options/first_12_episodes_grid.png'
    )


if __name__ == "__main__":
    main()
