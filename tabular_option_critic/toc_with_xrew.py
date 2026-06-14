"""
Module: toc_with_xrew.py
Algorithm: Option-Critic Architecture
Architectural Role: Tabular HRL training loop and FourRooms environment
Known Failure Modes:
1. Option Collapse: All options become identical or a single option dominates.
   Mitigation: Relying on varying initial conditions and random tie-breaking.
2. Premature Termination: Options terminate at every step if advantage is highly negative.
   Mitigation: Using sigmoid bounds and appropriate learning rates.
Monitoring Contract:
- Steps per episode must decrease to ~10-15 steps optimally.
- Option duration must span multiple environment steps indicating temporal abstraction.

Mathematical Reference:
Bacon, Harb, Precup (2017). The Option-Critic Architecture.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display required
import matplotlib.pyplot as plt
from scipy.special import expit, logsumexp
from tqdm import tqdm
import os
import copy
import matplotlib.patches as mpatches

# ==========================================
# 1. Environment: FourRooms
# ==========================================
class FourRooms:
    """
    FourRooms environment for hierarchical reinforcement learning.

    @observation:
        Shape: (1,). Semantics: Discrete state integer representing the 1D flattened index of a free cell.
    @action:
        Shape: (1,). Semantics: Discrete action integer in [0, 3] (UP, DOWN, LEFT, RIGHT).
    @reward:
        Sparse. +1.0 for reaching the goal state, 0.0 otherwise.
        
    @hrl_affordances:
        Natural temporal abstraction: 4 rooms connected by 4 bottlenecks (doorways).
        Worker space: 4D discrete movement.
        Manager space: Option selection, ideally corresponding to room-to-room traversal.
    """
    def __init__(self):
        layout = """\
wwwwwwwwwwwww
w     w     w
w     w     w
w           w
w     w     w
w     w     w
ww wwww     w
w     www www
w     w     w
w     w     w
w           w
w     w     w
wwwwwwwwwwwww
"""
        self.occupancy = np.array([list(map(lambda c: 1 if c=='w' else 0, line)) for line in layout.splitlines()])
        
        # Actions: 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        self.action_space = np.array([0, 1, 2, 3])
        self.observation_space = np.zeros(np.sum(self.occupancy == 0))
        self.directions = [np.array((-1,0)), np.array((1,0)), np.array((0,-1)), np.array((0,1))]

        self.rng = np.random.RandomState(1234)

        self.tostate = {}
        statenum = 0
        for i in range(13):
            for j in range(13):
                if self.occupancy[i,j] == 0:
                    self.tostate[(i,j)] = statenum
                    statenum += 1
        self.tocell = {v:k for k, v in self.tostate.items()}

        # All free states
        self._all_free = list(self.tostate.values())

        # --- Phase 1 Goal: Top-Right corner (same as toc_normal for direct comparison) ---
        self.goal_phase1 = self.tostate[(1, 11)]
        # --- Phase 2 Goal: Bottom-Right corner (adjacent corner) ---
        self.goal_phase2 = self.tostate[(11, 11)]

        # Start with Phase 1 goal
        self.goal = self.goal_phase1

        # Init states: all free except current goal
        self.init_states = [s for s in self._all_free if s != self.goal]

        # Identify Doorways automatically
        self.doorway_states = []
        for i in range(1, 12):
            for j in range(1, 12):
                if self.occupancy[i, j] == 0:
                    top, bottom = self.occupancy[i-1, j], self.occupancy[i+1, j]
                    left, right = self.occupancy[i, j-1], self.occupancy[i, j+1]
                    if (top == 1 and bottom == 1 and left == 0 and right == 0) or \
                       (top == 0 and bottom == 0 and left == 1 and right == 1):
                        self.doorway_states.append(self.tostate[(i, j)])

    def set_goal(self, goal_state):
        """Switch the goal and update init_states to exclude it."""
        self.goal = goal_state
        self.init_states = [s for s in self._all_free if s != self.goal]

    def reset(self):
        """Spawn at a random free state (excluding current goal)."""
        state = self.rng.choice(self.init_states)
        self.current_cell = self.tocell[state]
        return state

    def check_available_cells(self, cell):
        available_cells = []
        for action in range(len(self.action_space)):
            next_cell = tuple(cell + self.directions[action])
            if not self.occupancy[next_cell]:
                available_cells.append(next_cell)
        return available_cells

    def step(self, action):
        next_cell = tuple(self.current_cell + self.directions[action])

        if not self.occupancy[next_cell]:
            if self.rng.uniform() < 1/3:
                available_cells = self.check_available_cells(self.current_cell)
                self.current_cell = available_cells[self.rng.randint(len(available_cells))]
            else:
                self.current_cell = next_cell

        state = self.tostate[self.current_cell]
        done = (state == self.goal)
        return state, float(done), done, None


# ==========================================
# 2. RL Components
# ==========================================
class EpsGreedyPolicy():
    """
    Policy over options (Manager). Epsilon-greedy selection over Q_Omega.

    Role & Lifecycle:
    Persists across the entire training run. Updated via Q-learning.
    Evaluated at the start of each episode and upon option termination.

    Mutable State & Invariants:
    - `Q_Omega_table` (Tensor[nstates, noptions]): Value of each option in each state. Invariant: updated via TD learning.

    Initialization Contract:
    - `epsilon` (float): Probability of random exploration. Must be in [0, 1].
    """
    def __init__(self, rng, nstates, noptions, epsilon):
        self.rng = rng
        self.nstates = nstates
        self.noptions = noptions
        self.epsilon = epsilon
        self.Q_Omega_table = np.zeros((nstates, noptions))

    def Q_Omega(self, state, option=None):
        if option is None:
            return self.Q_Omega_table[state,:]
        else:
            return self.Q_Omega_table[state, option]

    def sample(self, state):
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.randint(self.noptions))
        else:
            return int(np.argmax(self.Q_Omega(state)))

class SoftmaxPolicy():
    """
    Intra-option policy (Worker). Softmax selection over Q_U.

    Role & Lifecycle:
    Persists across training. Updated via policy gradient (actor-critic).
    Evaluated at every environment step to choose primitive actions.

    Mutable State & Invariants:
    - `weights` (Tensor[nstates, nactions]): Logits for each action. Invariant: unbounded reals, updated via gradient ascent.

    Initialization Contract:
    - `temperature` (float): Scales logits before softmax. Must be > 0.
    """
    def __init__(self, rng, lr, nstates, nactions, temperature=1.0):
        self.rng = rng
        self.lr = lr
        self.nstates = nstates
        self.nactions = nactions
        self.temperature = temperature
        self.weights = np.zeros((nstates, nactions))

    def Q_U(self, state, action=None):
        if action is None:
            return self.weights[state,:]
        else:
            return self.weights[state, action]

    def pmf(self, state):
        exponent = self.Q_U(state) / self.temperature
        return np.exp(exponent - logsumexp(exponent))

    def sample(self, state):
        return int(self.rng.choice(self.nactions, p=self.pmf(state)))

    def update(self, state, action, Q_U):
        """
        Update intra-option policy weights.

        @math:
            Eq (5) from "The Option-Critic Architecture" (Bacon et al., 2017).
            ∇θ log π(a|s) * Q_U(s, o, a)
            Mapping: `actions_pmf` = π(a|s), `Q_U` = Q_U
        @shapes:
            input: state (scalar), action (scalar), Q_U (scalar)
        @provenance:
            - `Q_U` comes from the Critic's advantage estimate (Q_U - Q_Omega).
        @gradient:
            Flowing. Manual gradient update of logits.
        @stability:
            Subtracts expected Q_U (via pmf * Q_U) from all actions to reduce variance.
        @contract:
            None.
        """
        actions_pmf = self.pmf(state)  # Tensor[nactions] <Action probabilities for the given state>
        self.weights[state, :] -= self.lr * actions_pmf * Q_U
        self.weights[state, action] += self.lr * Q_U

class SigmoidTermination():
    """
    Option termination condition. Probability of terminating the current option.

    Role & Lifecycle:
    Persists across training. Evaluated after every environment step.
    
    Mutable State & Invariants:
    - `weights` (Tensor[nstates]): Logits for termination probability. Invariant: updated via policy gradient.
    
    Initialization Contract:
    - `lr` (float): Learning rate. Must be > 0.
    """
    def __init__(self, rng, lr, nstates):
        self.rng = rng
        self.lr = lr
        self.nstates = nstates
        self.weights = np.zeros((nstates,))

    def pmf(self, state):
        return expit(self.weights[state])

    def sample(self, state):
        return int(self.rng.uniform() < self.pmf(state))

    def gradient(self, state):
        return self.pmf(state) * (1.0 - self.pmf(state)), state

    def update(self, state, advantage):
        """
        Update termination probabilities.

        @math:
            Eq (7) from "The Option-Critic Architecture" (Bacon et al., 2017).
            ∇β * A_Omega(s, o)
            Mapping: `magnitude` = ∇β, `advantage` = A_Omega
        @shapes:
            input: state (scalar), advantage (scalar)
        @provenance:
            - `advantage` comes from Critic.A_Omega().
        @gradient:
            Flowing. Manual gradient update of termination logits.
        @stability:
            Negative sign in update because we want to minimize termination if advantage is positive.
        @contract:
            None.
        """
        magnitude, direction = self.gradient(state)
        self.weights[direction] -= self.lr * magnitude * advantage

class Critic():
    """
    Computes and caches value estimates for options and actions.

    Role & Lifecycle:
    Persists across training. Updated via TD errors.

    Mutable State & Invariants:
    - `Q_Omega_table` (Tensor[nstates, noptions]): Reference to the manager's Q-table.
    - `Q_U_table` (Tensor[nstates, noptions, nactions]): Value of taking action a in state s under option o.

    Initialization Contract:
    - `discount` (float): discount factor gamma. Must be in [0, 1].
    """
    def __init__(self, lr, discount, Q_Omega_table, nstates, noptions, nactions):
        self.lr = lr
        self.discount = discount
        self.Q_Omega_table = Q_Omega_table
        self.Q_U_table = np.zeros((nstates, noptions, nactions))

    def cache(self, state, option, action):
        self.last_state = state
        self.last_option = option
        self.last_action = action
        self.last_Q_Omega = self.Q_Omega(state, option)

    def Q_Omega(self, state, option=None):
        if option is None:
            return self.Q_Omega_table[state, :]
        else:
            return self.Q_Omega_table[state, option]

    def Q_U(self, state, option, action):
        return self.Q_U_table[state, option, action]

    def A_Omega(self, state, option=None):
        advantage = self.Q_Omega(state) - np.max(self.Q_Omega(state))
        if option is None:
            return advantage
        else:
            return advantage[option]

    def update_Qs(self, state, option, action, reward, done, terminations):
        """
        Update Q_Omega and Q_U via TD learning.

        @math:
            Eq (3) and (4) from "The Option-Critic Architecture".
            target = r + γ * ( (1 - β) * Q_Omega(s', o) + β * max_o' Q_Omega(s', o') )
            Mapping: `beta_omega` = β
        @shapes:
            input: reward (scalar)
        @provenance:
            - `terminations` are used to compute the expected value of the next state.
        @gradient:
            Flowing. Manual TD update.
        @stability:
            Uses `target - last_Q_Omega` as the TD error.
        @contract:
            None.
        """
        target = reward
        if not done:
            beta_omega = terminations[self.last_option].pmf(state)
            target += self.discount * ((1.0 - beta_omega)*self.Q_Omega(state, self.last_option) + \
                        beta_omega*np.max(self.Q_Omega(state)))

        tderror_Q_Omega = target - self.last_Q_Omega  # Tensor[1] <TD error for manager value function>
        self.Q_Omega_table[self.last_state, self.last_option] += self.lr * tderror_Q_Omega

        tderror_Q_U = target - self.Q_U(self.last_state, self.last_option, self.last_action)  # Tensor[1] <TD error for worker action-value function>
        self.Q_U_table[self.last_state, self.last_option, self.last_action] += self.lr * tderror_Q_U

        self.last_state = state
        self.last_option = option
        self.last_action = action
        if not done:
            self.last_Q_Omega = self.Q_Omega(state, option)


# ==========================================
# Diagnostics & Plotting
# ==========================================
class DiagnosticsLogger:
    def __init__(self, nstates, noptions):
        self.episodes = []
        self.snapshots = []
        self.visitation_counts = np.zeros(nstates)
        self.option_occupancy = np.zeros((nstates, noptions))
        self.option_switches = []
        self.termination_events = []
        
        self.doorway_visitation_counts = {}
        self.doorway_bonuses_per_episode = []
        self.option_durations_per_ep = []
        self.returns = []
        self.successes = []
        
        self.current_trace = []

        # Phase tracking
        self.phase1_episodes = []   # episode dicts for phase 1
        self.phase2_episodes = []   # episode dicts for phase 2
        
        dirs = [
            "results/learning_curves",
            "results/qomega_maps",
            "results/qomega_evolution",
            "results/beta_maps",
            "results/beta_evolution",
            "results/entropy_maps",
            "results/entropy_evolution",
            "results/specialization",
            "results/visitation",
            "results/occupancy",
            "results/episode_traces",
            "results/option_statistics",
            "results/doorway_analysis",
            "results/termination_maps",
            "results/learning_curves",
        ]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def log_step(self, episode, timestep, state, cell, option, action, reward,
                 qomega, qomega_all, advantage, advantage_all, beta, terminated,
                 q_u, option_duration):
        self.visitation_counts[state] += 1
        self.option_occupancy[state, option] += 1
        self.current_trace.append({
            'episode': episode,
            'timestep': timestep,
            'state': state,
            'cell': cell,
            'option': option,
            'action': action,
            'reward': reward,
            'qomega': qomega,
            'qomega_all': qomega_all,
            'advantage': advantage,
            'advantage_all': advantage_all,
            'beta': beta,
            'terminated': terminated,
            'q_u': q_u,
            'option_duration': option_duration
        })

    def log_episode(self, episode, switches, durations, doorway_bonuses, ret,
                    success, goal, phase):
        self.option_switches.append(switches)
        self.option_durations_per_ep.append(durations)
        self.doorway_bonuses_per_episode.append(doorway_bonuses)
        self.returns.append(ret)
        self.successes.append(success)
        
        ep_dict = {
            'episode': episode,
            'trace': self.current_trace,
            'success': success,
            'goal': goal,
            'phase': phase,
        }
        self.episodes.append(ep_dict)
        if phase == 1:
            self.phase1_episodes.append(ep_dict)
        else:
            self.phase2_episodes.append(ep_dict)
        self.current_trace = []

    def snapshot(self, episode, Q_Omega_table, Q_U_table, terminations, policies):
        self.snapshots.append({
            'episode': episode,
            'Q_Omega_table': copy.deepcopy(Q_Omega_table),
            'Q_U_table': copy.deepcopy(Q_U_table),
            'terminations': [copy.deepcopy(t.weights) for t in terminations],
            'policies': [copy.deepcopy(p.weights) for p in policies]
        })

class PlotManager:
    def __init__(self, env, logger, args, goal_change_episode):
        self.env = env
        self.logger = logger
        self.args = args
        self.goal_change_episode = goal_change_episode

    def _draw_grid_background(self, ax):
        """Draw the 4-rooms grid: walls as filled dark squares, free cells as light squares."""
        occ = self.env.occupancy
        for i in range(13):
            for j in range(13):
                if occ[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               color='#2c2c2c', zorder=1))
                else:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               color='#f0f0f0', zorder=1))
        ax.set_xlim(-0.5, 12.5)
        ax.set_ylim(12.5, -0.5)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(-0.5, 13, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 13, 1), minor=True)
        ax.grid(which='minor', color='#888888', linewidth=0.4, zorder=2)
        ax.tick_params(which='both', bottom=False, left=False,
                       labelbottom=False, labelleft=False)

    def plot_heatmap(self, data, title, filename, cmap='viridis', show_doorways=False,
                     highlight_goal=True, goal_state=None):
        """Render a per-state scalar as a colour overlay on the proper 4-rooms grid."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        self._draw_grid_background(ax)

        grid = np.full((13, 13), np.nan)
        for state in range(self.env.observation_space.shape[0]):
            i, j = self.env.tocell[state]
            grid[i, j] = data[state]

        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, cmap=cmap, origin='upper',
                       extent=(-0.5, 12.5, 12.5, -0.5), zorder=3)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        if highlight_goal:
            g = goal_state if goal_state is not None else self.env.goal
            gi, gj = self.env.tocell[g]
            ax.plot(gj, gi, 'r*', markersize=14, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.5, label='Goal')

        if show_doorways:
            for d in self.env.doorway_states:
                di, dj = self.env.tocell[d]
                ax.plot(dj, di, 'mD', markersize=7, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.5, label='Doorway')

        ax.set_title(title, fontsize=11, pad=8)
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)

    def plot_termination_maps_panel(self, option_terminations, noptions, label, goal_state):
        """Save a panel figure of all option termination maps."""
        nstates = self.env.observation_space.shape[0]
        fig, axes = plt.subplots(1, noptions, figsize=(5 * noptions, 5))
        if noptions == 1:
            axes = [axes]
        fig.suptitle(f'Termination Probability Maps — {label}', fontsize=14, fontweight='bold')

        for opt in range(noptions):
            ax = axes[opt]
            grid = np.full((13, 13), np.nan)
            for s in range(nstates):
                i, j = self.env.tocell[s]
                grid[i, j] = option_terminations[opt].pmf(s)

            for i in range(13):
                for j in range(13):
                    if self.env.occupancy[i, j] == 1:
                        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                                   color='#2c2c2c', zorder=1))

            masked = np.ma.masked_invalid(grid)
            im = ax.imshow(masked, cmap='Blues', origin='upper',
                           extent=(-0.5, 12.5, 12.5, -0.5), vmin=0, vmax=1, zorder=3)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Goal marker
            gi, gj = self.env.tocell[goal_state]
            ax.plot(gj, gi, 'r*', markersize=14, zorder=6,
                    markeredgecolor='white', markeredgewidth=0.5)

            # Doorway markers
            for d in self.env.doorway_states:
                di, dj = self.env.tocell[d]
                ax.plot(dj, di, 'mD', markersize=7, zorder=5,
                        markeredgecolor='white', markeredgewidth=0.5)

            ax.set_xlim(-0.5, 12.5)
            ax.set_ylim(12.5, -0.5)
            ax.set_aspect('equal')
            ax.set_title(f'Option {opt}', fontsize=11)
            ax.axis('off')

        plt.tight_layout()
        out_path = f'results/termination_maps/termination_maps_{label}.png'
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved termination maps: {out_path}")

    def generate_all_plots(self, critic, option_terminations, option_policies,
                           steps_history, dur_history):
        print("Generating diagnostic plots...")
        # Ensure all output directories exist (defensive — logger also does this)
        for d in [
            'results/qomega_maps', 'results/qomega_evolution',
            'results/beta_maps', 'results/beta_evolution',
            'results/entropy_maps', 'results/entropy_evolution',
            'results/specialization', 'results/visitation', 'results/occupancy',
            'results/episode_traces', 'results/option_statistics',
            'results/doorway_analysis', 'results/termination_maps',
            'results/learning_curves',
        ]:
            os.makedirs(d, exist_ok=True)
        nstates = self.env.observation_space.shape[0]
        noptions = self.args.noptions

        # ── Q_Omega heatmaps ──────────────────────────────────────────────
        for opt in range(noptions):
            self.plot_heatmap(critic.Q_Omega_table[:, opt], f'Q_Omega Option {opt}',
                              f'results/qomega_maps/option_{opt}_qomega.png')

        # ── Specialization & confidence ───────────────────────────────────
        spec = np.argmax(critic.Q_Omega_table, axis=1)
        self.plot_heatmap(spec, 'Option Specialization',
                          'results/specialization/specialization_map.png', cmap='tab10')
        
        sorted_q = np.sort(critic.Q_Omega_table, axis=1)
        conf = sorted_q[:, -1] - sorted_q[:, -2]
        self.plot_heatmap(conf, 'Option Confidence',
                          'results/specialization/option_confidence.png', cmap='magma')

        # ── Termination maps ─────────────────────────────────────────────
        for opt in range(noptions):
            beta_s = np.array([option_terminations[opt].pmf(s) for s in range(nstates)])
            self.plot_heatmap(beta_s, f'Termination Option {opt}',
                              f'results/beta_maps/option_{opt}_beta.png',
                              cmap='Blues', show_doorways=True)

        # Save combined panels for both goal phases
        self.plot_termination_maps_panel(option_terminations, noptions,
                                         label='final_phase2_goal',
                                         goal_state=self.env.goal_phase2)
        self.env.goal = self.env.goal_phase1   # temporarily set for plotting
        self.plot_termination_maps_panel(option_terminations, noptions,
                                         label='final_phase1_goal_ref',
                                         goal_state=self.env.goal_phase1)
        self.env.goal = self.env.goal_phase2   # restore

        # ── Entropy maps ──────────────────────────────────────────────────
        for opt in range(noptions):
            entropy = np.zeros(nstates)
            for s in range(nstates):
                pmf = option_policies[opt].pmf(s)
                entropy[s] = -np.sum(pmf * np.log(pmf + 1e-9))
            self.plot_heatmap(entropy, f'Policy Entropy Option {opt}',
                              f'results/entropy_maps/entropy_option_{opt}.png', cmap='Reds')

        # ── Visitation & occupancy ────────────────────────────────────────
        self.plot_heatmap(self.logger.visitation_counts, 'State Visitation',
                          'results/visitation/state_visitation.png', cmap='hot')

        for opt in range(noptions):
            self.plot_heatmap(self.logger.option_occupancy[:, opt],
                              f'Occupancy Option {opt}',
                              f'results/occupancy/occupancy_option_{opt}.png', cmap='YlGn')

        # ── Option statistics ─────────────────────────────────────────────
        plt.figure()
        plt.plot(self.logger.option_switches)
        plt.axvline(self.goal_change_episode, color='red', lw=2, ls='--',
                    label=f'Goal change (ep {self.goal_change_episode})')
        plt.legend()
        plt.title('Option Switches per Episode')
        plt.xlabel('Episode'); plt.ylabel('Switches')
        plt.savefig('results/option_statistics/option_switches.png')
        plt.close()

        avg_durations = [np.mean(d) if len(d) > 0 else 0 for d in self.logger.option_durations_per_ep]
        plt.figure()
        plt.plot(avg_durations)
        plt.axvline(self.goal_change_episode, color='red', lw=2, ls='--',
                    label=f'Goal change (ep {self.goal_change_episode})')
        plt.legend()
        plt.title('Average Option Duration per Episode')
        plt.xlabel('Episode'); plt.ylabel('Duration')
        plt.savefig('results/option_statistics/option_duration.png')
        plt.close()

        # ── Learning curves with phase boundary ───────────────────────────
        self._plot_learning_curves(steps_history, dur_history)

        # ── Evolution snapshots ───────────────────────────────────────────
        for snap in self.logger.snapshots:
            ep = snap['episode']
            q = snap['Q_Omega_table']
            spec_ep = np.argmax(q, axis=1)
            self.plot_heatmap(spec_ep, f'Specialization Ep {ep}',
                              f'results/qomega_evolution/specialization_ep_{ep}.png', cmap='tab10')

            for opt in range(noptions):
                t = SigmoidTermination(None, None, nstates)
                t.weights = snap['terminations'][opt]
                beta_ep = np.array([t.pmf(s) for s in range(nstates)])
                self.plot_heatmap(beta_ep, f'Beta Option {opt} Ep {ep}',
                                  f'results/beta_evolution/beta_ep_{ep}_option_{opt}.png', cmap='Blues')

                p = SoftmaxPolicy(None, None, nstates, self.env.action_space.shape[0])
                p.weights = snap['policies'][opt]
                ent_ep = np.zeros(nstates)
                for s in range(nstates):
                    pmf = p.pmf(s)
                    ent_ep[s] = -np.sum(pmf * np.log(pmf + 1e-9))
                self.plot_heatmap(ent_ep, f'Entropy Option {opt} Ep {ep}',
                                  f'results/entropy_evolution/entropy_ep_{ep}_option_{opt}.png', cmap='Reds')

        # ── Option responsibility ─────────────────────────────────────────
        total_occ = np.sum(self.logger.option_occupancy, axis=1, keepdims=True) + 1e-9
        resp = self.logger.option_occupancy / total_occ
        resp_map = np.argmax(resp, axis=1)
        self.plot_heatmap(resp_map, 'Option Responsibility',
                          'results/specialization/option_responsibility_map.png', cmap='tab10')

        # ── Doorway analysis ─────────────────────────────────────────────
        plt.figure()
        doorways = list(self.logger.doorway_visitation_counts.keys())
        counts = list(self.logger.doorway_visitation_counts.values())
        doorway_labels = [str(self.env.tocell[d]) for d in doorways]
        if doorways:
            plt.bar(doorway_labels, counts, color='steelblue')
        plt.title('Doorway Visitation Counts')
        plt.savefig('results/doorway_analysis/doorway_usage.png')
        plt.close()

        plt.figure()
        plt.plot(self.logger.doorway_bonuses_per_episode)
        plt.axvline(self.goal_change_episode, color='red', lw=2, ls='--',
                    label=f'Goal change (ep {self.goal_change_episode})')
        plt.legend()
        plt.title('Doorway Reward Curve')
        plt.savefig('results/doorway_analysis/doorway_reward_curve.png')
        plt.close()

        # ── Episode traces — Phase 1 & Phase 2 ───────────────────────────
        self.plot_traces_by_phase()

    def _plot_learning_curves(self, steps_history, dur_history):
        """Save the learning curve plot with phase boundary shading."""
        nepisodes = len(steps_history)
        window = max(1, nepisodes // 50)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('toc_with_xrew — With Bottleneck Reward', fontsize=15, fontweight='bold')

        ax1 = axes[0]
        ax1.set_title('Average Steps to Reach Goal', fontsize=14)
        ax1.set_xlabel('Episodes', fontsize=12)
        ax1.set_ylabel('Steps', fontsize=12)
        ax1.plot(steps_history, color='steelblue', lw=1.2, alpha=0.7, label='Steps')
        if len(steps_history) >= window:
            smooth = np.convolve(steps_history, np.ones(window)/window, mode='valid')
            ax1.plot(np.arange(window-1, len(steps_history)), smooth,
                     color='navy', lw=2.5, label=f'Smoothed (w={window})')
        ax1.axvline(self.goal_change_episode, color='red', lw=2, ls='--',
                    label=f'Goal change (ep {self.goal_change_episode})')
        ax1.axvspan(0, self.goal_change_episode, alpha=0.06, color='blue')
        ax1.axvspan(self.goal_change_episode, nepisodes, alpha=0.06, color='orange')
        ylim = ax1.get_ylim()
        ax1.text(self.goal_change_episode / 2, ylim[1] * 0.95,
                 'Phase 1\n(Goal: top-right)',
                 ha='center', va='top', fontsize=9, color='blue', alpha=0.8)
        ax1.text(self.goal_change_episode + (nepisodes - self.goal_change_episode) / 2,
                 ylim[1] * 0.95, 'Phase 2\n(Goal: bottom-right)',
                 ha='center', va='top', fontsize=9, color='darkorange', alpha=0.8)
        ax1.legend(fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.5)

        ax2 = axes[1]
        ax2.set_title('Average Option Duration', fontsize=14)
        ax2.set_xlabel('Episodes', fontsize=12)
        ax2.set_ylabel('Avg. Duration', fontsize=12)
        avg_durations = [np.mean(d) if len(d) > 0 else 0
                         for d in self.logger.option_durations_per_ep]
        ax2.plot(avg_durations, color='forestgreen', lw=1.2, alpha=0.7)
        if len(avg_durations) >= window:
            smooth_d = np.convolve(avg_durations, np.ones(window)/window, mode='valid')
            ax2.plot(np.arange(window-1, len(avg_durations)), smooth_d,
                     color='darkgreen', lw=2.5, label='Smoothed')
        ax2.axvline(self.goal_change_episode, color='red', lw=2, ls='--',
                    label=f'Goal change (ep {self.goal_change_episode})')
        ax2.axvspan(0, self.goal_change_episode, alpha=0.06, color='blue')
        ax2.axvspan(self.goal_change_episode, nepisodes, alpha=0.06, color='orange')
        ax2.legend(fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('results/learning_curves/learning_curve_xrew.png', bbox_inches='tight', dpi=150)
        plt.close()
        print("  Saved learning curves: results/learning_curves/learning_curve_xrew.png")

        # Per-phase steps graph
        fig2, ax = plt.subplots(figsize=(14, 5))
        ax.set_title('Steps per Episode — Both Phases (toc_with_xrew)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Steps', fontsize=12)
        p1 = steps_history[:self.goal_change_episode]
        p2 = steps_history[self.goal_change_episode:]
        ax.plot(np.arange(len(p1)), p1, color='steelblue', lw=1.2, alpha=0.7,
                label='Phase 1 (top-right goal)')
        ax.plot(np.arange(self.goal_change_episode, self.goal_change_episode + len(p2)),
                p2, color='darkorange', lw=1.2, alpha=0.7,
                label='Phase 2 (bottom-right goal)')
        ax.axvline(self.goal_change_episode, color='red', lw=2, ls='--',
                   label=f'Goal change (ep {self.goal_change_episode})')
        ax.legend(fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('results/learning_curves/steps_per_episode_both_phases_xrew.png',
                    bbox_inches='tight', dpi=150)
        plt.close()
        print("  Saved steps graph: results/learning_curves/steps_per_episode_both_phases_xrew.png")

    def plot_traces_by_phase(self):
        """Plot episode traces separately for Phase 1 and Phase 2."""
        for phase_label, eps_list in [('phase1', self.logger.phase1_episodes),
                                       ('phase2', self.logger.phase2_episodes)]:
            success_eps = [ep for ep in eps_list if ep['success']]
            if not success_eps:
                success_eps = [ep for ep in eps_list if ep.get('success', False)]
            if not success_eps:
                print(f"  No successful episodes to trace for {phase_label}.")
                continue

            n_traces = min(3, len(success_eps))
            chosen = list(np.random.choice(len(success_eps), n_traces, replace=False))
            sample_eps = [success_eps[i] for i in chosen]

            cmap = plt.get_cmap('tab10')
            opt_colors = [cmap(o / max(1, self.args.noptions - 1))
                          for o in range(self.args.noptions)]

            for idx, ep_data in enumerate(sample_eps):
                trace = ep_data['trace']
                ep_num = ep_data['episode']
                ep_goal = ep_data.get('goal', self.env.goal)

                fig, axes = plt.subplots(3, 3, figsize=(16, 15))
                phase_title = 'Phase 1 — Goal: top-right' if phase_label == 'phase1' \
                              else 'Phase 2 — Goal: bottom-right'
                fig.suptitle(
                    f'[{phase_title}]  Episode {ep_num}  —  {len(trace)} steps  |  '
                    f'Goal: {self.env.tocell[ep_goal]}',
                    fontsize=12, fontweight='bold'
                )

                # ── Panel 1: Grid trajectory ──────────────────────────────
                ax = axes[0, 0]
                self._draw_grid_background(ax)
                path = [t['cell'] for t in trace]
                if path:
                    # Colour-code path by option
                    for k in range(len(path) - 1):
                        opt = trace[k]['option']
                        r0, c0 = path[k]
                        r1, c1 = path[k+1]
                        ax.annotate("",
                                    xy=(c1, r1), xytext=(c0, r0),
                                    arrowprops=dict(arrowstyle="->",
                                                    color=opt_colors[opt],
                                                    lw=1.4, shrinkA=2, shrinkB=2),
                                    zorder=5)
                    # Termination markers
                    for t in trace:
                        if t['terminated']:
                            r, c = t['cell']
                            ax.plot(c, r, 'x', color='black', markersize=6,
                                    markeredgewidth=1.2, zorder=7)
                    # Start marker
                    r0, c0 = path[0]
                    ax.plot(c0, r0, 'o', color='limegreen', markersize=9,
                            zorder=8, markeredgecolor='white', markeredgewidth=0.8,
                            label='Start')
                    # Goal marker
                    gi, gj = self.env.tocell[ep_goal]
                    ax.plot(gj, gi, 'r*', markersize=14, zorder=8,
                            markeredgecolor='white', markeredgewidth=0.5, label='Goal')
                    # Doorway markers
                    for d in self.env.doorway_states:
                        di, dj = self.env.tocell[d]
                        ax.plot(dj, di, 'D', color='mediumpurple', markersize=6,
                                zorder=6, markeredgecolor='white', markeredgewidth=0.4,
                                label='Doorway')
                    # Option legend patches
                    handles, labels = ax.get_legend_handles_labels()
                    by_label = dict(zip(labels, handles))
                    for o in range(self.args.noptions):
                        by_label[f'Opt {o}'] = mpatches.Patch(color=opt_colors[o])
                    ax.legend(by_label.values(), by_label.keys(),
                              loc='upper right', fontsize=7, framealpha=0.85, ncol=2)
                ax.set_title('Grid Trajectory (colour = option)')

                # ── Panels 2–9: Timeseries ────────────────────────────────
                timesteps   = [t['timestep']        for t in trace]
                options     = [t['option']          for t in trace]
                actions     = [t['action']          for t in trace]
                qomegas     = [t['qomega']          for t in trace]
                advantages  = [t['advantage']       for t in trace]
                betas       = [t['beta']            for t in trace]
                rewards     = [t['reward']          for t in trace]
                terminations= [t['terminated']      for t in trace]
                durations   = [t['option_duration'] for t in trace]

                ax2 = axes[0, 1]
                ax2.scatter(timesteps, options, c=options, cmap='tab10',
                            vmin=0, vmax=max(1, self.args.noptions - 1), s=15, zorder=3)
                ax2.step(timesteps, options, where='post', color='grey', lw=0.8, alpha=0.5)
                ax2.set_yticks(range(self.args.noptions))
                ax2.set_xlabel('Timestep'); ax2.set_title('Option Timeline')

                ax3 = axes[0, 2]
                ax3.scatter(timesteps, actions, c=actions, cmap='tab10',
                            vmin=0, vmax=3, s=15, zorder=3)
                ax3.set_yticks([0, 1, 2, 3])
                ax3.set_yticklabels(['UP', 'DOWN', 'LEFT', 'RIGHT'], fontsize=8)
                ax3.set_xlabel('Timestep'); ax3.set_title('Action Timeline')

                axes[1, 0].plot(timesteps, qomegas, color='steelblue', lw=1.2)
                axes[1, 0].set_xlabel('Timestep'); axes[1, 0].set_title('Q_Omega(s, o_t)')

                axes[1, 1].plot(timesteps, advantages, color='darkorange', lw=1.2)
                axes[1, 1].axhline(0, color='grey', lw=0.8, ls='--')
                axes[1, 1].set_xlabel('Timestep'); axes[1, 1].set_title('Advantage A_Omega')

                axes[1, 2].plot(timesteps, betas, color='purple', lw=1.2)
                axes[1, 2].set_ylim(-0.05, 1.05)
                axes[1, 2].set_xlabel('Timestep'); axes[1, 2].set_title('Beta(s) — termination prob')

                ax7 = axes[2, 0]
                ax7.bar(timesteps, rewards, color='dodgerblue', width=0.8, alpha=0.7)
                door_ts = [t for t, r in zip(timesteps, rewards) if 0 < r < 1.0]
                door_rs = [r for r in rewards if 0 < r < 1.0]
                if door_ts:
                    ax7.scatter(door_ts, door_rs, color='magenta', zorder=5,
                                label='Doorway bonus')
                    ax7.legend(fontsize=7)
                ax7.set_xlabel('Timestep'); ax7.set_title('Reward Timeline')

                axes[2, 1].step(timesteps, terminations, where='post',
                                color='crimson', lw=1.2)
                axes[2, 1].set_ylim(-0.1, 1.1)
                axes[2, 1].set_xlabel('Timestep'); axes[2, 1].set_title('Termination Events')

                axes[2, 2].plot(timesteps, durations, color='teal', lw=1.2)
                axes[2, 2].set_xlabel('Timestep'); axes[2, 2].set_title('Option Duration')

                plt.tight_layout()
                out_path = f'results/episode_traces/{phase_label}_trace_{idx+1:03d}.png'
                plt.savefig(out_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
                print(f"  Saved {phase_label} trace {idx+1} (episode {ep_num}): {out_path}")


# ==========================================
# 3. Main Training Loop
# ==========================================
GOAL_CHANGE_EPISODE = 1000   # After this many episodes, the goal changes

def main(args):
    discount = 0.99
    lr_term = 0.25
    lr_intra = 0.25
    lr_critic = 0.5
    epsilon = 1e-1
    temperature = 1e-2
    SNAPSHOT_INTERVAL = 250
    DOORWAY_REWARD = 0.5

    env = FourRooms()
    nstates = env.observation_space.shape[0]
    nactions = env.action_space.shape[0]

    logger = DiagnosticsLogger(nstates, args.noptions)
    plotter = PlotManager(env, logger, args, GOAL_CHANGE_EPISODE)

    print("--- Starting Training ---")
    rng = np.random.RandomState(1234)

    option_policies = [SoftmaxPolicy(rng, lr_intra, nstates, nactions, temperature)
                       for _ in range(args.noptions)]
    option_terminations = [SigmoidTermination(rng, lr_term, nstates)
                           for _ in range(args.noptions)]
    policy_over_options = EpsGreedyPolicy(rng, nstates, args.noptions, epsilon)
    critic = Critic(lr_critic, discount, policy_over_options.Q_Omega_table,
                    nstates, args.noptions, nactions)

    # Start with Phase 1 goal
    env.set_goal(env.goal_phase1)
    current_phase = 1

    steps_history = []   # steps per episode (for learning curves)
    dur_history   = []   # avg option duration per episode

    pbar = tqdm(range(args.nepisodes), desc="Training")
    
    for episode in pbar:
        # ── Goal switch at GOAL_CHANGE_EPISODE ──────────────────────────
        if episode == GOAL_CHANGE_EPISODE and current_phase == 1:
            env.set_goal(env.goal_phase2)
            current_phase = 2
            print(f"\n  [Episode {episode}] Goal changed: "
                  f"{env.tocell[env.goal_phase1]} -> {env.tocell[env.goal_phase2]}")

        state = env.reset()
        # =====================================================================
        # >>> MANAGER TIME (Macro-step: t_k) <<<
        # Option selection based on epsilon-greedy policy over Q_Omega.
        # =====================================================================
        option = policy_over_options.sample(state)
        # =====================================================================
        # >>> WORKER TIME (Micro-step: t_k + i) <<<
        # Primitive action execution conditioned on the current option.
        # =====================================================================
        action = option_policies[option].sample(state)
        critic.cache(state, option, action)
        
        duration = 1
        option_switches = 0
        visited_doorways = set()
        episode_return = 0.0
        doorway_bonuses = 0
        durations = []
        
        for step in range(args.nsteps):
            prev_state = state
            state, env_reward, done, _ = env.step(action)
            
            reward = env_reward
            if state in env.doorway_states and state not in visited_doorways:
                reward += DOORWAY_REWARD
                visited_doorways.add(state)
                doorway_bonuses += 1
                logger.doorway_visitation_counts[state] = \
                    logger.doorway_visitation_counts.get(state, 0) + 1

            episode_return += reward

            terminated = option_terminations[option].sample(state)
            
            logger.log_step(
                episode=episode,
                timestep=step,
                state=prev_state,
                cell=env.tocell[prev_state],
                option=option,
                action=action,
                reward=reward,
                qomega=critic.Q_Omega(prev_state, option),
                qomega_all=critic.Q_Omega(prev_state).copy(),
                advantage=critic.A_Omega(prev_state, option),
                advantage_all=critic.A_Omega(prev_state).copy(),
                beta=option_terminations[option].pmf(state),
                terminated=terminated,
                q_u=critic.Q_U(prev_state, option, action),
                option_duration=duration
            )

            if terminated:
                durations.append(duration)
                # =====================================================================
                # >>> MANAGER TIME (Macro-step: t_k) <<<
                # Option selection based on epsilon-greedy policy over Q_Omega.
                # =====================================================================
                option = policy_over_options.sample(state)
                option_switches += 1
                duration = 1
            else:
                duration += 1
                
            # =====================================================================
            # >>> WORKER TIME (Micro-step: t_k + i) <<<
            # Primitive action execution conditioned on the current option.
            # =====================================================================
            action = option_policies[option].sample(state)
            
            critic.update_Qs(state, option, action, reward, done, option_terminations)
            
            Q_U = critic.Q_U(state, option, action)
            Q_U = Q_U - critic.Q_Omega(state, option)
            option_policies[option].update(state, action, Q_U)
            
            option_terminations[option].update(state, critic.A_Omega(state, option))
            
            if done: break
            
        durations.append(duration)
        logger.log_episode(episode, option_switches, durations, doorway_bonuses,
                           episode_return, done, goal=env.goal, phase=current_phase)

        steps_history.append(step)
        avg_dur = np.mean(durations) if durations else 0.0
        dur_history.append(avg_dur)
        
        if (episode + 1) % SNAPSHOT_INTERVAL == 0:
            logger.snapshot(episode + 1, critic.Q_Omega_table, critic.Q_U_table,
                            option_terminations, option_policies)

        pbar.set_postfix({'Phase': current_phase, 'Steps': step, 'Switches': option_switches})

    # Ensure final goal is Phase 2 for plotting
    env.set_goal(env.goal_phase2)
    plotter.generate_all_plots(critic, option_terminations, option_policies,
                               steps_history, dur_history)

    total_episodes = len(logger.episodes)
    success_rate = np.mean(logger.successes) if logger.successes else 0.0
    avg_return = np.mean(logger.returns) if logger.returns else 0.0
    
    all_durations = [d for ep_d in logger.option_durations_per_ep for d in ep_d]
    avg_duration = np.mean(all_durations) if all_durations else 0.0
    
    avg_switches = np.mean(logger.option_switches) if logger.option_switches else 0.0
    doorway_reward_freq = (np.mean([b > 0 for b in logger.doorway_bonuses_per_episode])
                           if logger.doorway_bonuses_per_episode else 0.0)
    
    total_occ = np.sum(logger.option_occupancy, axis=0)
    most_used = np.argmax(total_occ)
    least_used = np.argmin(total_occ)

    print("\n" + "="*40)
    print("FINAL TRAINING SUMMARY")
    print("="*40)
    print(f"Total episodes:           {total_episodes}")
    print(f"Success rate:             {success_rate:.2%}")
    print(f"Average return:           {avg_return:.2f}")
    print(f"Average option duration:  {avg_duration:.2f}")
    print(f"Average switches:         {avg_switches:.2f}")
    print(f"Doorway reward frequency: {doorway_reward_freq:.2%}")
    print(f"Most used option:         {most_used}")
    print(f"Least used option:        {least_used}")
    print("="*40)
    print("\n=== All plots saved to results/ ===")
    print("  Learning curves  -> results/learning_curves/")
    print("  Termination maps -> results/termination_maps/ & results/beta_maps/")
    print("  Episode traces   -> results/episode_traces/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nepisodes', type=int, default=2000, help='Number of episodes per run')
    parser.add_argument('--nsteps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--noptions', type=int, default=4, help='Number of options to learn')
    args = parser.parse_args()
    main(args)
