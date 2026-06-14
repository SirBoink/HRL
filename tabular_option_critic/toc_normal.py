import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import expit, logsumexp
from tqdm import tqdm

# ==========================================
# 1. Environment: FourRooms
# ==========================================
class FourRooms:
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

        # Phase 1 Goal: Top-Right corner
        self.goal_phase1 = self.tostate[(1, 11)]
        # Phase 2 Goal: Bottom-Right corner (adjacent corner)
        self.goal_phase2 = self.tostate[(11, 11)]

        self.goal = self.goal_phase1
        self._all_free = list(self.tostate.values())
        self.init_states = [s for s in self._all_free if s != self.goal]

        # Auto-detect doorway states
        self.doorway_states = []
        for i in range(1, 12):
            for j in range(1, 12):
                if self.occupancy[i, j] == 0:
                    top    = self.occupancy[i-1, j]
                    bottom = self.occupancy[i+1, j]
                    left   = self.occupancy[i, j-1]
                    right  = self.occupancy[i, j+1]
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
        '''
        Takes a step in the intended direction with 2/3 probability.
        Takes a step in the other directions with probability 1/3 (equally likely).
        '''
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
        actions_pmf = self.pmf(state)
        self.weights[state, :] -= self.lr * actions_pmf * Q_U
        self.weights[state, action] += self.lr * Q_U

class SigmoidTermination():
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
        magnitude, direction = self.gradient(state)
        self.weights[direction] -= self.lr * magnitude * advantage

class Critic():
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
        target = reward
        if not done:
            beta_omega = terminations[self.last_option].pmf(state)
            target += self.discount * ((1.0 - beta_omega)*self.Q_Omega(state, self.last_option) + \
                        beta_omega*np.max(self.Q_Omega(state)))

        tderror_Q_Omega = target - self.last_Q_Omega
        self.Q_Omega_table[self.last_state, self.last_option] += self.lr * tderror_Q_Omega

        tderror_Q_U = target - self.Q_U(self.last_state, self.last_option, self.last_action)
        self.Q_U_table[self.last_state, self.last_option, self.last_action] += self.lr * tderror_Q_U

        self.last_state = state
        self.last_option = option
        self.last_action = action
        if not done:
            self.last_Q_Omega = self.Q_Omega(state, option)


# ==========================================
# 3. Plotting Helpers
# ==========================================

def _draw_grid_background(ax, env):
    """Draw the proper 4-rooms layout: dark walls, light free cells, grid lines."""
    for i in range(13):
        for j in range(13):
            color = '#2c2c2c' if env.occupancy[i, j] == 1 else '#f5f5f5'
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       color=color, zorder=1))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(12.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(-0.5, 13, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 13, 1), minor=True)
    ax.grid(which='minor', color='#aaaaaa', linewidth=0.4, zorder=2)
    ax.tick_params(which='both', bottom=False, left=False,
                   labelbottom=False, labelleft=False)


def plot_termination_maps(env, option_terminations, noptions, out_dir, label):
    """Save a single panel PNG with one termination-probability heatmap per option."""
    os.makedirs(out_dir, exist_ok=True)
    nstates = env.observation_space.shape[0]

    fig, axes = plt.subplots(1, noptions, figsize=(5 * noptions, 5))
    if noptions == 1:
        axes = [axes]
    fig.suptitle(f'Termination Probability Maps — {label}', fontsize=14, fontweight='bold')

    for opt in range(noptions):
        ax = axes[opt]
        grid = np.full((13, 13), np.nan)
        for s in range(nstates):
            i, j = env.tocell[s]
            grid[i, j] = option_terminations[opt].pmf(s)

        for i in range(13):
            for j in range(13):
                if env.occupancy[i, j] == 1:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                               color='#2c2c2c', zorder=1))

        masked = np.ma.masked_invalid(grid)
        im = ax.imshow(masked, cmap='Blues', origin='upper',
                       extent=(-0.5, 12.5, 12.5, -0.5), vmin=0, vmax=1, zorder=3)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        gi, gj = env.tocell[env.goal]
        ax.plot(gj, gi, 'r*', markersize=14, zorder=6,
                markeredgecolor='white', markeredgewidth=0.5, label='Goal')
        for d in env.doorway_states:
            di, dj = env.tocell[d]
            ax.plot(dj, di, 'mD', markersize=7, zorder=5,
                    markeredgecolor='white', markeredgewidth=0.5)

        ax.set_xlim(-0.5, 12.5)
        ax.set_ylim(12.5, -0.5)
        ax.set_aspect('equal')
        ax.set_title(f'Option {opt}', fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, f'termination_maps_{label}.png')
    plt.savefig(out_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved termination maps: {out_path}")


def plot_episode_traces(env, episodes, noptions, out_dir, phase_label):
    """
    For each sampled successful episode produce a two-panel figure:
      Top    – 4-rooms grid with path segments colour-coded by active option.
      Bottom – horizontal option timeline with termination ticks.
    """
    os.makedirs(out_dir, exist_ok=True)

    success = [ep for ep in episodes if ep['success']]
    if not success:
        print(f"  No successful episodes to plot for {phase_label}.")
        return

    n       = min(3, len(success))
    sampled = [success[i] for i in np.random.choice(len(success), n, replace=False)]

    cmap       = plt.get_cmap('tab10')
    opt_colors = [cmap(o / max(1, noptions - 1)) for o in range(noptions)]

    for fig_idx, ep in enumerate(sampled):
        trace   = ep['trace']
        ep_num  = ep['episode']
        T       = len(trace)

        cells       = [t['cell']       for t in trace]
        options_seq = [t['option']     for t in trace]
        term_flags  = [t['terminated'] for t in trace]

        fig = plt.figure(figsize=(10, 11))
        phase_name = 'Phase 1 — Goal: top-right' if phase_label == 'phase1' \
                     else 'Phase 2 — Goal: bottom-right'
        fig.suptitle(
            f'[{phase_name}]  Episode {ep_num}  |  {T} steps  |  '
            f'{sum(term_flags)} terminations  |  Goal: {env.tocell[ep["goal"]]}',
            fontsize=12, fontweight='bold', y=0.98
        )

        ax_grid     = fig.add_axes([0.06, 0.22, 0.88, 0.72])
        ax_timeline = fig.add_axes([0.06, 0.06, 0.88, 0.10])

        # Grid panel
        _draw_grid_background(ax_grid, env)

        for k in range(T - 1):
            r0, c0 = cells[k]
            r1, c1 = cells[k + 1]
            ax_grid.annotate(
                "",
                xy=(c1, r1), xytext=(c0, r0),
                arrowprops=dict(arrowstyle="->", color=opt_colors[options_seq[k]],
                                lw=1.6, shrinkA=3, shrinkB=3),
                zorder=5
            )

        for t in trace:
            if t['terminated']:
                r, c = t['cell']
                ax_grid.plot(c, r, 'x', color='black', markersize=7,
                             markeredgewidth=1.4, zorder=7)

        for d in env.doorway_states:
            dr, dc = env.tocell[d]
            ax_grid.plot(dc, dr, 'D', color='mediumpurple', markersize=7,
                         markeredgecolor='white', markeredgewidth=0.6,
                         zorder=6, label='Doorway')

        r0, c0 = cells[0]
        ax_grid.plot(c0, r0, 'o', color='limegreen', markersize=11,
                     markeredgecolor='white', markeredgewidth=0.8,
                     zorder=8, label='Start')

        gi, gj = env.tocell[ep['goal']]
        ax_grid.plot(gj, gi, 'r*', markersize=16,
                     markeredgecolor='white', markeredgewidth=0.5,
                     zorder=8, label='Goal')

        handles, labels = ax_grid.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        for o in range(noptions):
            by_label[f'Option {o}'] = mpatches.Patch(color=opt_colors[o])
        by_label['Termination'] = plt.Line2D(
            [0], [0], marker='x', color='black',
            linestyle='None', markersize=7, markeredgewidth=1.4)
        ax_grid.legend(by_label.values(), by_label.keys(),
                       loc='upper right', fontsize=7.5, framealpha=0.9, ncol=2)
        ax_grid.set_title('Grid Trajectory (colour = active option)', fontsize=10, pad=6)

        # Timeline panel
        ax_timeline.set_xlim(0, T)
        ax_timeline.set_ylim(0, 1)
        ax_timeline.set_xlabel('Timestep', fontsize=9)
        ax_timeline.set_yticks([])
        ax_timeline.set_title('Option timeline  (▼ = termination)', fontsize=9, pad=4)

        seg_start = 0
        seg_opt   = options_seq[0]
        for k in range(1, T):
            if options_seq[k] != seg_opt or k == T - 1:
                end = k if options_seq[k] != seg_opt else k + 1
                ax_timeline.barh(0.5, end - seg_start, left=seg_start, height=0.8,
                                 color=opt_colors[seg_opt], alpha=0.85, align='center')
                seg_start = k
                seg_opt   = options_seq[k]

        for k, t in enumerate(trace):
            if t['terminated']:
                ax_timeline.axvline(k, color='black', lw=1.2, ymin=0.1, ymax=0.9)

        out_path = os.path.join(out_dir, f'{phase_label}_trace_{fig_idx + 1:03d}.png')
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"  Saved {phase_label} trace {fig_idx + 1} (episode {ep_num}, {T} steps)")


# ==========================================
# 4. Main Training & Plotting Loop
# ==========================================

GOAL_CHANGE_EPISODE = 1000   # After this many episodes the goal changes

def main(args):
    discount    = 0.99
    lr_term     = 0.25
    lr_intra    = 0.25
    lr_critic   = 0.5
    epsilon     = 1e-1
    temperature = 1e-2

    # Create the only three output directories this script uses
    for d in ['results/learning_curves', 'results/termination_maps', 'results/episode_traces']:
        os.makedirs(d, exist_ok=True)

    history = np.zeros((args.nruns, args.nepisodes, 2))
    option_terminations_list = []

    env     = FourRooms()
    nstates = env.observation_space.shape[0]
    nactions = env.action_space.shape[0]

    phase1_episodes_trace = []
    phase2_episodes_trace = []

    for run in range(args.nruns):
        print(f"--- Starting Run {run + 1}/{args.nruns} ---")
        rng = np.random.RandomState(1234 + run)

        option_policies     = [SoftmaxPolicy(rng, lr_intra, nstates, nactions, temperature)
                                for _ in range(args.noptions)]
        option_terminations = [SigmoidTermination(rng, lr_term, nstates)
                                for _ in range(args.noptions)]
        policy_over_options = EpsGreedyPolicy(rng, nstates, args.noptions, epsilon)
        critic = Critic(lr_critic, discount, policy_over_options.Q_Omega_table,
                        nstates, args.noptions, nactions)

        is_last_run = (run == args.nruns - 1)
        if is_last_run:
            phase1_episodes_trace = []
            phase2_episodes_trace = []

        env.set_goal(env.goal_phase1)
        current_phase = 1

        pbar = tqdm(range(args.nepisodes), desc="Training")

        for episode in pbar:
            if episode == GOAL_CHANGE_EPISODE and current_phase == 1:
                env.set_goal(env.goal_phase2)
                current_phase = 2
                if is_last_run:
                    print(f"\n  [Episode {episode}] Goal changed: "
                          f"{env.tocell[env.goal_phase1]} -> {env.tocell[env.goal_phase2]}")

            state  = env.reset()
            option = policy_over_options.sample(state)
            action = option_policies[option].sample(state)
            critic.cache(state, option, action)

            duration        = 1
            option_switches = 0
            avg_duration    = 0.0
            episode_trace   = []

            for step in range(args.nsteps):
                prev_state = state
                state, reward, done, _ = env.step(action)

                terminated = option_terminations[option].sample(state)

                if is_last_run:
                    episode_trace.append({
                        'cell'      : env.tocell[prev_state],
                        'option'    : option,
                        'action'    : action,
                        'reward'    : reward,
                        'beta'      : option_terminations[option].pmf(state),
                        'terminated': bool(terminated),
                    })

                if terminated:
                    option = policy_over_options.sample(state)
                    option_switches += 1
                    avg_duration += (1.0/option_switches) * (duration - avg_duration)
                    duration = 1

                action = option_policies[option].sample(state)

                critic.update_Qs(state, option, action, reward, done, option_terminations)

                Q_U = critic.Q_U(state, option, action) - critic.Q_Omega(state, option)
                option_policies[option].update(state, action, Q_U)

                option_terminations[option].update(state, critic.A_Omega(state, option))

                duration += 1
                if done:
                    break

            history[run, episode, 0] = step
            history[run, episode, 1] = avg_duration
            pbar.set_postfix({'Phase': current_phase, 'Steps': step, 'Switches': option_switches})

            if is_last_run:
                episode_trace.append({
                    'cell'      : env.tocell[state],
                    'option'    : option,
                    'action'    : action,
                    'reward'    : reward,
                    'beta'      : option_terminations[option].pmf(state),
                    'terminated': False,
                })
                ep_record = {
                    'episode': episode,
                    'trace'  : episode_trace,
                    'success': bool(done),
                    'goal'   : env.goal,
                    'phase'  : current_phase,
                }
                if current_phase == 1:
                    phase1_episodes_trace.append(ep_record)
                else:
                    phase2_episodes_trace.append(ep_record)

        option_terminations_list.append(option_terminations)

    print("\nTraining completed. Generating plots...")

    last_run_terminations = option_terminations_list[-1]
    steps_data = np.mean(history[:, :, 0], axis=0)
    dur_data   = np.mean(history[:, :, 1], axis=0)
    window     = max(1, args.nepisodes // 50)

    # ── PLOT 1: Learning curves ───────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('toc_normal — No Bottleneck Reward', fontsize=15, fontweight='bold')

    ax1 = axes[0]
    ax1.set_title('Steps to Reach Goal', fontsize=14)
    ax1.set_xlabel('Episodes', fontsize=12)
    ax1.set_ylabel('Steps', fontsize=12)
    ax1.plot(steps_data, color='steelblue', lw=1.2, alpha=0.7, label='Steps')
    if len(steps_data) >= window:
        smooth = np.convolve(steps_data, np.ones(window)/window, mode='valid')
        ax1.plot(np.arange(window-1, len(steps_data)), smooth,
                 color='navy', lw=2.5, label=f'Smoothed (w={window})')
    ax1.axvline(GOAL_CHANGE_EPISODE, color='red', lw=2, ls='--',
                label=f'Goal change (ep {GOAL_CHANGE_EPISODE})')
    ax1.axvspan(0, GOAL_CHANGE_EPISODE, alpha=0.06, color='blue')
    ax1.axvspan(GOAL_CHANGE_EPISODE, args.nepisodes, alpha=0.06, color='orange')
    ymax = ax1.get_ylim()[1]
    ax1.text(GOAL_CHANGE_EPISODE / 2, ymax * 0.95, 'Phase 1\n(Goal: top-right)',
             ha='center', va='top', fontsize=9, color='blue', alpha=0.8)
    ax1.text(GOAL_CHANGE_EPISODE + (args.nepisodes - GOAL_CHANGE_EPISODE) / 2,
             ymax * 0.95, 'Phase 2\n(Goal: bottom-right)',
             ha='center', va='top', fontsize=9, color='darkorange', alpha=0.8)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = axes[1]
    ax2.set_title('Average Option Duration', fontsize=14)
    ax2.set_xlabel('Episodes', fontsize=12)
    ax2.set_ylabel('Avg. Duration', fontsize=12)
    ax2.plot(dur_data, color='forestgreen', lw=1.2, alpha=0.7)
    if len(dur_data) >= window:
        smooth_d = np.convolve(dur_data, np.ones(window)/window, mode='valid')
        ax2.plot(np.arange(window-1, len(dur_data)), smooth_d,
                 color='darkgreen', lw=2.5, label='Smoothed')
    ax2.axvline(GOAL_CHANGE_EPISODE, color='red', lw=2, ls='--', label='Goal change')
    ax2.axvspan(0, GOAL_CHANGE_EPISODE, alpha=0.06, color='blue')
    ax2.axvspan(GOAL_CHANGE_EPISODE, args.nepisodes, alpha=0.06, color='orange')
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('results/learning_curves/learning_curve_normal.png', bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: results/learning_curves/learning_curve_normal.png")

    # ── PLOT 2: Steps per episode — colour-split by phase ────────────────
    fig2, ax = plt.subplots(figsize=(14, 5))
    ax.set_title('Steps per Episode — Both Phases (toc_normal)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Steps', fontsize=12)
    p1 = steps_data[:GOAL_CHANGE_EPISODE]
    p2 = steps_data[GOAL_CHANGE_EPISODE:]
    ax.plot(np.arange(len(p1)), p1,
            color='steelblue', lw=1.2, alpha=0.7, label='Phase 1 (top-right goal)')
    ax.plot(np.arange(GOAL_CHANGE_EPISODE, GOAL_CHANGE_EPISODE + len(p2)), p2,
            color='darkorange', lw=1.2, alpha=0.7, label='Phase 2 (bottom-right goal)')
    ax.axvline(GOAL_CHANGE_EPISODE, color='red', lw=2, ls='--',
               label=f'Goal change (ep {GOAL_CHANGE_EPISODE})')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/learning_curves/steps_per_episode_both_phases.png',
                bbox_inches='tight', dpi=150)
    plt.close()
    print("  Saved: results/learning_curves/steps_per_episode_both_phases.png")

    # ── PLOT 3: Termination maps — Phase 2 goal (final weights) ─────────
    env.set_goal(env.goal_phase2)
    plot_termination_maps(env, last_run_terminations, args.noptions,
                          out_dir='results/termination_maps', label='phase2_final')

    # ── PLOT 4: Termination maps — Phase 1 goal marker for reference ─────
    env.set_goal(env.goal_phase1)
    plot_termination_maps(env, last_run_terminations, args.noptions,
                          out_dir='results/termination_maps', label='phase1_ref')

    # ── PLOT 5: Episode traces — Phase 1 ────────────────────────────────
    print("\nPlotting Phase 1 episode traces (goal = top-right)...")
    env.set_goal(env.goal_phase1)
    plot_episode_traces(env, phase1_episodes_trace, args.noptions,
                        out_dir='results/episode_traces', phase_label='phase1')

    # ── PLOT 6: Episode traces — Phase 2 ────────────────────────────────
    print("Plotting Phase 2 episode traces (goal = bottom-right)...")
    env.set_goal(env.goal_phase2)
    plot_episode_traces(env, phase2_episodes_trace, args.noptions,
                        out_dir='results/episode_traces', phase_label='phase2')

    print("\n=== All plots saved to results/ ===")
    print("  results/learning_curves/")
    print("  results/termination_maps/")
    print("  results/episode_traces/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nruns',     type=int, default=1,    help='Number of independent runs')
    parser.add_argument('--nepisodes', type=int, default=2000, help='Episodes per run')
    parser.add_argument('--nsteps',   type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--noptions', type=int, default=4,    help='Number of options')
    args = parser.parse_args()
    main(args)