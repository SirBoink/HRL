"""
plot_it.py  –  Scientific Visualization Suite for Option-Critic + PER
======================================================================
Plots produced:
  Plot A : Per-option Q-value heatmaps on the four-rooms grid  (4 sub-plots)
  Plot B : 10 random-episode scientific traces saved to ./episode_traces/
  Plot C : PER priority landscape & batch sampling overlay on the grid
  Plot D : Per-step option-critic parameter tracking along an episode
"""

import os, random, math
import numpy as np
import matplotlib
matplotlib.use("Agg")                        # headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines  as mlines
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib.collections import LineCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import torch
import mo_gymnasium as mo_gym
from opps import UnifiedRewardWrapper, ActorNetwork, CriticNetwork

# ── Aesthetic constants ──────────────────────────────────────────────────────
DARK_BG   = "#0D1117"
PANEL_BG  = "#161B22"
GRID_DARK = "#1C2128"
GRID_LITE = "#21262D"
WALL_CLR  = "#30363D"
ACCENT    = "#58A6FF"
TEXT_CLR  = "#E6EDF3"
MUTED     = "#8B949E"

OPTION_COLORS  = ["#FF6B6B", "#4ECDC4", "#FFD93D", "#A855F7"]
OPTION_CMAPS   = ["Reds", "Greens", "YlOrRd", "Purples"]
OPTION_NAMES   = ["Option 0", "Option 1", "Option 2", "Option 3"]

BOTTLENECK_COORDS = [(5, 2), (2, 5), (8, 5), (5, 8)]
GRID_SIZE = 13

# ── Matplotlib global style ──────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    PANEL_BG,
    "axes.edgecolor":    WALL_CLR,
    "axes.labelcolor":   TEXT_CLR,
    "text.color":        TEXT_CLR,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "axes.titlecolor":   TEXT_CLR,
    "grid.color":        WALL_CLR,
    "grid.linewidth":    0.5,
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})


# ── Environment & model helpers ──────────────────────────────────────────────

def make_env():
    base = mo_gym.make("four-room-v0")
    return UnifiedRewardWrapper(base, bottleneck_coords=BOTTLENECK_COORDS)


def load_models(env, actor_path="actor.pth", critic_path="critic.pth"):
    num_options = 4
    actor  = ActorNetwork(env.observation_space.shape[0], num_options, env.action_space.n)
    critic = CriticNetwork(env.observation_space.shape[0], num_options)
    if os.path.exists(actor_path) and os.path.exists(critic_path):
        actor.load_state_dict(torch.load(actor_path,  map_location="cpu", weights_only=True))
        critic.load_state_dict(torch.load(critic_path, map_location="cpu", weights_only=True))
        print("✓ Model weights loaded.")
    else:
        print("⚠  Weights not found – using random initialisation.")
    actor.eval(); critic.eval()
    return actor, critic


# ── Grid-drawing utility ─────────────────────────────────────────────────────

def draw_grid(ax, env, title="", show_legend=True, alpha_wall=0.95, checkerboard=True):
    """Draw the four-rooms grid on *ax* with dark/light tiles, walls, goal, start."""
    uw    = env.unwrapped
    walls = set(map(tuple, uw.occupied))
    goal  = tuple(uw.goal)
    start = tuple(uw.initial[0])

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) in walls:
                color = WALL_CLR
            elif checkerboard:
                color = GRID_DARK if (r + c) % 2 == 0 else GRID_LITE
            else:
                color = PANEL_BG
            rect = mpatches.FancyBboxPatch(
                (c - 0.5, r - 0.5), 1, 1,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor=DARK_BG, linewidth=0.4,
                zorder=0
            )
            ax.add_patch(rect)

    # Goal star
    ax.plot(goal[1], goal[0], marker="*", color="#FFD700", markersize=18,
            markeredgecolor=DARK_BG, markeredgewidth=1.0,
            zorder=10, label="Goal" if show_legend else "")
    # Start circle
    ax.plot(start[1], start[0], marker="o", color="#00E5FF", markersize=12,
            markeredgecolor=DARK_BG, markeredgewidth=1.0,
            zorder=10, label="Start" if show_legend else "")
    # Door/bottleneck squares
    for idx, (br, bc) in enumerate(BOTTLENECK_COORDS):
        ax.plot(bc, br, marker="D", color="#FF9F43", markersize=9, alpha=0.85,
                markeredgecolor=DARK_BG, markeredgewidth=0.8,
                zorder=9, label="Bottleneck" if (show_legend and idx == 0) else "")

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)
    ax.set_xticks(range(GRID_SIZE)); ax.set_xticklabels(range(GRID_SIZE), fontsize=6)
    ax.set_yticks(range(GRID_SIZE)); ax.set_yticklabels(range(GRID_SIZE), fontsize=6)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT_CLR, pad=8)


def probe_state(env, actor, critic, r, c, base_state):
    """Return (action_dist, term_prob, q_vals) tensors at grid cell (r,c)."""
    s = base_state.copy(); s[0] = r; s[1] = c
    st = torch.FloatTensor(s).unsqueeze(0)
    with torch.no_grad():
        ad, tp = actor(st)
        qv     = critic(st)
    return ad[0], tp[0], qv[0]   # all shape (num_options,...) or (num_actions,)


def scan_grid(env, actor, critic):
    """Return dicts of numpy arrays indexed [r,c] for all free cells."""
    uw    = env.unwrapped
    walls = set(map(tuple, uw.occupied))
    base, _ = env.reset()

    q_maps    = [np.full((GRID_SIZE, GRID_SIZE), np.nan) for _ in range(4)]
    t_maps    = [np.full((GRID_SIZE, GRID_SIZE), np.nan) for _ in range(4)]
    adv_maps  = [np.full((GRID_SIZE, GRID_SIZE), np.nan) for _ in range(4)]
    pol_U     = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    pol_V     = [[[] for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) in walls:
                continue
            ad, tp, qv = probe_state(env, actor, critic, r, c, base)
            for opt in range(4):
                q_maps[opt][r, c]   = qv[opt].item()
                t_maps[opt][r, c]   = tp[opt].item()
                # advantage = Q(s,ω) - max_ω' Q(s,ω')
                adv_maps[opt][r, c] = (qv[opt] - qv.max()).item()
    return q_maps, t_maps, adv_maps


# ════════════════════════════════════════════════════════════════════════════
#  PLOT A  –  Per-option Q-value heatmaps
# ════════════════════════════════════════════════════════════════════════════

def plot_A_qvalue_heatmaps(env, actor, critic, filename="plotA_option_qvalues.png"):
    print("  Scanning grid for Q-values …")
    q_maps, t_maps, adv_maps = scan_grid(env, actor, critic)

    fig = plt.figure(figsize=(24, 12), facecolor=DARK_BG)
    fig.suptitle("Per-Option Q-Value Landscapes on the Four-Rooms Grid",
                 fontsize=20, fontweight="bold", color=TEXT_CLR, y=0.98)

    outer = gridspec.GridSpec(1, 4, figure=fig, wspace=0.35)

    uw    = env.unwrapped
    walls = set(map(tuple, uw.occupied))

    base, _ = env.reset()

    for opt in range(4):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[opt], hspace=0.08, height_ratios=[3, 1])
        ax_main = fig.add_subplot(inner[0])
        ax_adv  = fig.add_subplot(inner[1])

        # --- Main: Q-value heatmap ---
        q  = q_maps[opt]
        vmax = np.nanmax(np.abs(q)) or 1.0
        vmin = np.nanmin(q)

        draw_grid(ax_main, env, show_legend=(opt == 0), checkerboard=True)
        im = ax_main.imshow(
            q, origin="upper", extent=[-0.5, GRID_SIZE-0.5, GRID_SIZE-0.5, -0.5],
            cmap="plasma", vmin=vmin, vmax=np.nanmax(q), alpha=0.72, zorder=1
        )
        # Termination contours
        t = t_maps[opt]
        valid = ~np.isnan(t)
        if valid.any():
            ax_main.contour(
                np.arange(GRID_SIZE), np.arange(GRID_SIZE), t,
                levels=[0.3, 0.6, 0.85], colors=["white"],
                linewidths=[0.7, 1.1, 1.6], alpha=0.55, zorder=3
            )

        # Q value text annotation per cell
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                if not np.isnan(q[r, c]):
                    ax_main.text(c, r, f"{q[r,c]:.2f}",
                                 ha="center", va="center", fontsize=4.2,
                                 color="white", alpha=0.85, zorder=4)

        divider = make_axes_locatable(ax_main)
        cax = divider.append_axes("right", size="4%", pad=0.06)
        cb  = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        cb.set_label("Q(s, ω)", fontsize=8, color=TEXT_CLR)

        ax_main.set_title(
            f"Option {opt}  —  Q(s, ω{opt})\n"
            f"[contours = termination β probability]",
            fontsize=10, fontweight="bold",
            color=OPTION_COLORS[opt], pad=6
        )
        if opt == 0:
            ax_main.legend(loc="upper right", fontsize=6,
                           facecolor=PANEL_BG, edgecolor=WALL_CLR,
                           labelcolor=TEXT_CLR, markerscale=0.7)

        # --- Advantage histogram (marginal) ---
        adv_vals = adv_maps[opt][~np.isnan(adv_maps[opt])].ravel()
        ax_adv.hist(adv_vals, bins=20, color=OPTION_COLORS[opt], alpha=0.8,
                    edgecolor=DARK_BG, linewidth=0.5)
        ax_adv.axvline(0, color="white", linewidth=0.9, linestyle="--", alpha=0.6)
        ax_adv.set_xlabel("Advantage A(s, ω)", fontsize=7, color=TEXT_CLR)
        ax_adv.set_ylabel("Cells", fontsize=7, color=TEXT_CLR)
        ax_adv.tick_params(labelsize=6)
        ax_adv.set_facecolor(GRID_DARK)

    plt.savefig(filename, dpi=160, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ Saved {filename}")


# ════════════════════════════════════════════════════════════════════════════
#  PLOT B  –  10 random-episode scientific traces
# ════════════════════════════════════════════════════════════════════════════

def _run_episode(env, actor, critic, max_steps=500, greedy=False):
    """Run one episode, collecting rich per-step data."""
    state, _ = env.reset()
    done      = False
    opt       = None
    steps     = []
    gamma     = 0.99

    for t in range(max_steps):
        st = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            ad, tp = actor(st)
            qv     = critic(st)

        terminated_option = False
        if opt is None or random.random() < tp[0, opt].item():
            if greedy:
                opt = int(qv.argmax())
            else:
                opt = int(qv.argmax())   # greedy for evaluation traces
            terminated_option = True

        pol      = ad[0, opt].numpy()
        action   = int(np.argmax(pol))          # greedy action for trace clarity
        q_curr   = qv[0, opt].item()
        max_q    = qv.max().item()
        advantage= q_curr - max_q               # always ≤ 0
        beta_val = tp[0, opt].item()
        entropy  = float(-np.sum(pol * np.log(pol + 1e-8)))

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Compute TD error approximation (single-step)
        ns = torch.FloatTensor(next_state).unsqueeze(0)
        with torch.no_grad():
            nqv    = critic(ns)
            _, ntp = actor(ns)
            nq_w   = nqv[0, opt].item()
            nt_w   = ntp[0, opt].item()
            v_next = (1 - nt_w) * nq_w + nt_w * nqv.max().item()
            target_q = reward + (1 - int(done)) * gamma * v_next
        td_error = abs(target_q - q_curr)

        rc = info.get("reward_components", {"item": 0, "bottleneck": 0, "goal": 0})

        steps.append({
            "t":           t,
            "r":           int(state[0]), "c":  int(state[1]),
            "nr":          int(next_state[0]), "nc": int(next_state[1]),
            "option":      opt,
            "action":      action,
            "q_curr":      q_curr,
            "max_q":       max_q,
            "advantage":   advantage,
            "beta":        beta_val,
            "entropy":     entropy,
            "td_error":    td_error,
            "reward":      reward,
            "item_r":      rc["item"],
            "bn_r":        rc["bottleneck"],
            "goal_r":      rc["goal"],
            "terminated_option": terminated_option,
            "q_all":       qv[0].numpy().copy(),
        })

        state = next_state
        if done:
            break

    return steps


def _colored_path(ax, xs, ys, values, cmap="plasma", lw=2.8, alpha=0.85, zorder=5):
    """Draw a path whose segments are colored by *values*."""
    pts   = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segs  = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm  = mcolors.Normalize(vmin=np.min(values), vmax=np.max(values))
    lc    = LineCollection(segs, cmap=cmap, norm=norm,
                           linewidth=lw, alpha=alpha, zorder=zorder)
    lc.set_array(np.array(values))
    ax.add_collection(lc)
    return lc


def plot_B_episode_traces(env, actor, critic, n=10, out_dir="episode_traces"):
    os.makedirs(out_dir, exist_ok=True)
    print(f"  Generating {n} episode traces → {out_dir}/")

    for ep_idx in range(n):
        steps = _run_episode(env, actor, critic)
        T     = len(steps)
        ts    = np.arange(T)

        # ── Layout: grid on top, 5 metric panels below ──────────────────────
        fig = plt.figure(figsize=(22, 18), facecolor=DARK_BG)
        fig.suptitle(
            f"Episode {ep_idx+1}  |  {T} steps  |  "
            f"total reward = {sum(s['reward'] for s in steps):.3f}",
            fontsize=15, fontweight="bold", color=TEXT_CLR, y=0.99
        )

        gs_top = gridspec.GridSpec(1, 2, figure=fig,
                                   top=0.92, bottom=0.52,
                                   left=0.04, right=0.97, wspace=0.30)
        gs_bot = gridspec.GridSpec(5, 1, figure=fig,
                                   top=0.46, bottom=0.04,
                                   left=0.08, right=0.97, hspace=0.05)

        ax_grid  = fig.add_subplot(gs_top[0])
        ax_qgrid = fig.add_subplot(gs_top[1])

        ax_q   = fig.add_subplot(gs_bot[0])
        ax_adv = fig.add_subplot(gs_bot[1], sharex=ax_q)
        ax_bet = fig.add_subplot(gs_bot[2], sharex=ax_q)
        ax_ent = fig.add_subplot(gs_bot[3], sharex=ax_q)
        ax_td  = fig.add_subplot(gs_bot[4], sharex=ax_q)

        # ── LEFT GRID: trajectory coloured by option ─────────────────────────
        draw_grid(ax_grid, env, title="Trajectory  (colour = option)", show_legend=True)

        # Option segments
        xs = [s["c"] + np.random.uniform(-0.12, 0.12) for s in steps]
        ys = [s["r"] + np.random.uniform(-0.12, 0.12) for s in steps]
        opt_ids = [s["option"] for s in steps]

        for i in range(T - 1):
            ax_grid.plot(
                [xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=OPTION_COLORS[opt_ids[i]],
                linewidth=2.0, alpha=0.80, solid_capstyle="round", zorder=5
            )

        # Termination markers
        for i, s in enumerate(steps):
            if s["terminated_option"] and i > 0:
                ax_grid.scatter(xs[i], ys[i], marker="X", s=90,
                                color="white", zorder=8, linewidths=0.7,
                                edgecolors=OPTION_COLORS[s["option"]])
        # Start/end of episode
        ax_grid.scatter(xs[0],  ys[0],  s=120, color="#00E5FF", zorder=9,
                        edgecolors=DARK_BG, linewidths=1)
        ax_grid.scatter(xs[-1], ys[-1], s=120, marker="*", color="#FFD700", zorder=9,
                        edgecolors=DARK_BG, linewidths=1)

        legend_handles = [
            mlines.Line2D([], [], color=OPTION_COLORS[o], lw=3, label=OPTION_NAMES[o])
            for o in range(4)
        ] + [
            mlines.Line2D([], [], color="white", marker="X", linestyle="None",
                          markersize=9, markeredgecolor="#888", label="Option switch"),
        ]
        ax_grid.legend(handles=legend_handles, loc="upper right", fontsize=7,
                       facecolor=PANEL_BG, edgecolor=WALL_CLR, labelcolor=TEXT_CLR)

        # ── RIGHT GRID: TD-error heat trail ──────────────────────────────────
        draw_grid(ax_qgrid, env, title="TD-Error Heat Trail  (brighter = higher error)",
                  show_legend=False)

        td_vals = [s["td_error"] for s in steps]
        norm_td = mcolors.Normalize(vmin=0, vmax=max(max(td_vals), 1e-6))
        cmap_td = matplotlib.colormaps.get_cmap("inferno")

        for i in range(T - 1):
            c_val = cmap_td(norm_td(td_vals[i]))
            ax_qgrid.plot(
                [xs[i], xs[i+1]], [ys[i], ys[i+1]],
                color=c_val, linewidth=2.8, alpha=0.88,
                solid_capstyle="round", zorder=5
            )

        # Q-value annotations at each cell visited (de-duped)
        seen = {}
        for s in steps:
            key = (s["r"], s["c"])
            if key not in seen:
                seen[key] = s["q_curr"]
                ax_qgrid.text(s["c"], s["r"], f"{s['q_curr']:.2f}",
                              ha="center", va="center", fontsize=4.8,
                              color="white", alpha=0.85, zorder=6)

        sm = cm.ScalarMappable(cmap=cmap_td, norm=norm_td)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax_qgrid, fraction=0.03, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=MUTED)
        cb.set_label("|TD error|", fontsize=8, color=TEXT_CLR)

        # ── BOTTOM PANELS: per-step metrics ──────────────────────────────────
        q_curr   = [s["q_curr"]    for s in steps]
        max_q    = [s["max_q"]     for s in steps]
        adv      = [s["advantage"] for s in steps]
        beta_s   = [s["beta"]      for s in steps]
        ent_s    = [s["entropy"]   for s in steps]
        td_s     = [s["td_error"]  for s in steps]
        opts     = [s["option"]    for s in steps]
        term_ts  = [i for i, s in enumerate(steps) if s["terminated_option"] and i > 0]

        def shade_options(axx):
            """Shade background bands by current option."""
            seg_start = 0
            for i in range(1, T):
                if opts[i] != opts[i-1] or i == T - 1:
                    end = i if i < T - 1 else T
                    axx.axvspan(seg_start, end, alpha=0.12,
                                color=OPTION_COLORS[opts[seg_start]], zorder=0)
                    seg_start = i
            for tt in term_ts:
                axx.axvline(tt, color="white", linewidth=0.6, alpha=0.35, zorder=1)

        # Q values
        shade_options(ax_q)
        ax_q.plot(ts, q_curr, color=ACCENT,       lw=1.8, label="Q(s, ω_t)", zorder=3)
        ax_q.plot(ts, max_q,  color="#FF6B6B",    lw=1.4, linestyle="--",
                  label="max_ω Q(s,ω)", zorder=3, alpha=0.8)
        ax_q.fill_between(ts, q_curr, max_q, color="#FF6B6B", alpha=0.12, zorder=2)
        ax_q.set_ylabel("Q Value", fontsize=8, color=TEXT_CLR)
        ax_q.legend(loc="upper right", fontsize=7,
                    facecolor=PANEL_BG, edgecolor=WALL_CLR, labelcolor=TEXT_CLR)
        ax_q.tick_params(labelbottom=False, labelsize=7)

        # Advantage
        shade_options(ax_adv)
        ax_adv.bar(ts, adv, color=[OPTION_COLORS[o] for o in opts],
                   alpha=0.75, width=0.9, zorder=3)
        ax_adv.axhline(0, color="white", lw=0.7, linestyle="--", alpha=0.4)
        ax_adv.set_ylabel("Advantage\nA(s,ω)", fontsize=8, color=TEXT_CLR)
        ax_adv.tick_params(labelbottom=False, labelsize=7)

        # Termination probability β
        shade_options(ax_bet)
        ax_bet.plot(ts, beta_s, color="#FFD93D", lw=1.8, zorder=3)
        ax_bet.fill_between(ts, 0, beta_s, color="#FFD93D", alpha=0.20, zorder=2)
        ax_bet.set_ylim(0, 1.05)
        ax_bet.set_ylabel("β (term prob)", fontsize=8, color=TEXT_CLR)
        ax_bet.tick_params(labelbottom=False, labelsize=7)

        # Entropy
        shade_options(ax_ent)
        ax_ent.plot(ts, ent_s, color="#A855F7", lw=1.8, zorder=3)
        ax_ent.fill_between(ts, 0, ent_s, color="#A855F7", alpha=0.20, zorder=2)
        ax_ent.set_ylabel("Policy Entropy", fontsize=8, color=TEXT_CLR)
        ax_ent.tick_params(labelbottom=False, labelsize=7)

        # TD error
        shade_options(ax_td)
        ax_td.bar(ts, td_s, color="#FF9F43", alpha=0.85, width=0.9, zorder=3)
        ax_td.set_ylabel("|TD Error|", fontsize=8, color=TEXT_CLR)
        ax_td.set_xlabel("Time step  t", fontsize=9, color=TEXT_CLR)
        ax_td.tick_params(labelsize=7)

        # ── Shared annotations ────────────────────────────────────────────────
        for ax in [ax_q, ax_adv, ax_bet, ax_ent, ax_td]:
            ax.set_facecolor(GRID_DARK)
            ax.set_xlim(0, T - 1)
            for sp in ["top", "right"]:
                ax.spines[sp].set_visible(False)

        # Reward event rugs
        for s in steps:
            if s["bn_r"] > 0:
                ax_td.axvline(s["t"], color="#FF9F43", lw=2.5, alpha=0.9, zorder=4)
            if s["goal_r"] > 0:
                ax_td.axvline(s["t"], color="#FFD700", lw=3.0, alpha=0.95, zorder=5)

        out_path = os.path.join(out_dir, f"episode_{ep_idx+1:02d}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()
        print(f"    ✓ Saved {out_path}  ({T} steps)")


# ════════════════════════════════════════════════════════════════════════════
#  PLOT C  –  PER priority landscape on the grid
# ════════════════════════════════════════════════════════════════════════════

def plot_C_per_visualization(env, actor, critic,
                              logs_path="training_logs.npz",
                              filename="plotC_per_priority.png"):
    print("  Building PER visualisation …")
    uw    = env.unwrapped
    walls = set(map(tuple, uw.occupied))

    # ── Panel layout: 2×3 ───────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 14), facecolor=DARK_BG)
    fig.suptitle(
        "Prioritized Experience Replay  —  Priority Landscape & Batch Dynamics",
        fontsize=17, fontweight="bold", color=TEXT_CLR, y=0.98
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.42,
                           top=0.93, bottom=0.06, left=0.05, right=0.97)

    # ── C1: TD-error heatmap from a live rollout ─────────────────────────────
    ax_td_grid = fig.add_subplot(gs[0, 0])
    draw_grid(ax_td_grid, env,
              title="Live TD-Error Density Map\n(mean |TD| per cell, 3 rollouts)",
              show_legend=True)

    td_map   = np.zeros((GRID_SIZE, GRID_SIZE))
    td_count = np.zeros((GRID_SIZE, GRID_SIZE))

    for _ in range(3):
        steps = _run_episode(env, actor, critic)
        for s in steps:
            td_map[s["r"], s["c"]]   += s["td_error"]
            td_count[s["r"], s["c"]] += 1

    with np.errstate(invalid="ignore"):
        td_mean = np.where(td_count > 0, td_map / td_count, np.nan)

    im1 = ax_td_grid.imshow(
        td_mean, origin="upper",
        extent=[-0.5, GRID_SIZE-0.5, GRID_SIZE-0.5, -0.5],
        cmap="inferno", alpha=0.70, zorder=2
    )
    cb1 = fig.colorbar(im1, ax=ax_td_grid, fraction=0.035, pad=0.02)
    cb1.ax.tick_params(labelsize=7, colors=MUTED)
    cb1.set_label("Mean |TD error|", fontsize=8, color=TEXT_CLR)

    # ── C2: Priority distribution over training snapshots ───────────────────
    ax_prio_hist = fig.add_subplot(gs[0, 1])
    ax_prio_hist.set_facecolor(GRID_DARK)
    ax_prio_hist.set_title("Priority Distribution\nacross Training Snapshots",
                            fontsize=10, fontweight="bold", color=TEXT_CLR)
    ax_prio_hist.set_xlabel("|TD Error| (priority proxy)", fontsize=8, color=TEXT_CLR)
    ax_prio_hist.set_ylabel("Density", fontsize=8, color=TEXT_CLR)

    if os.path.exists(logs_path):
        logs    = np.load(logs_path)
        ep_keys = sorted([k for k in logs.files if k.startswith("priorities_ep_")])
        palette = sns.color_palette("coolwarm", len(ep_keys))
        for i, key in enumerate(ep_keys):
            prios = logs[key] - 1e-5
            prios = prios[prios > 0]
            ep_num = key.split("_")[-1]
            ax_prio_hist.hist(prios, bins=40, alpha=0.55,
                              color=palette[i], density=True,
                              label=f"Ep {ep_num}", histtype="stepfilled",
                              edgecolor="none")
        ax_prio_hist.legend(fontsize=7, facecolor=PANEL_BG,
                            edgecolor=WALL_CLR, labelcolor=TEXT_CLR)

        # ── C3: Cumulative sampling probability (log scale) ──────────────────
        ax_cdf = fig.add_subplot(gs[0, 2])
        ax_cdf.set_facecolor(GRID_DARK)
        ax_cdf.set_title("PER Sampling Prob (CDF)\nvs TD-Error Rank",
                          fontsize=10, fontweight="bold", color=TEXT_CLR)

        alpha_per = 0.6
        for i, key in enumerate(ep_keys):
            prios = logs[key]
            probs = prios ** alpha_per
            probs /= probs.sum()
            ranks = np.argsort(np.argsort(-probs))   # rank by descending probability
            sorted_probs = np.sort(probs)[::-1]
            cum = np.cumsum(sorted_probs)
            ax_cdf.plot(np.arange(len(cum)) / len(cum) * 100,
                        cum, lw=1.8, color=palette[i],
                        label=f"Ep {key.split('_')[-1]}")
        ax_cdf.axhline(0.5, lw=0.8, linestyle="--", color="white", alpha=0.4)
        ax_cdf.set_xlabel("Top-k% samples", fontsize=8, color=TEXT_CLR)
        ax_cdf.set_ylabel("Cum. sampling prob", fontsize=8, color=TEXT_CLR)
        ax_cdf.legend(fontsize=7, facecolor=PANEL_BG,
                      edgecolor=WALL_CLR, labelcolor=TEXT_CLR)
        ax_cdf.tick_params(labelsize=7)

        # ── C4: Priority evolution over training episodes (mean±std) ─────────
        ax_evo = fig.add_subplot(gs[1, 0:2])
        ax_evo.set_facecolor(GRID_DARK)
        ax_evo.set_title("Priority Statistics Across Snapshots",
                          fontsize=10, fontweight="bold", color=TEXT_CLR)

        ep_nums, means, stds, p90s = [], [], [], []
        for key in ep_keys:
            prios = logs[key]
            prios_valid = prios[prios > 1e-5] - 1e-5
            ep_nums.append(int(key.split("_")[-1]))
            means.append(float(np.mean(prios_valid)))
            stds.append(float(np.std(prios_valid)))
            p90s.append(float(np.percentile(prios_valid, 90)))

        ep_nums = np.array(ep_nums)
        means   = np.array(means)
        stds    = np.array(stds)
        p90s    = np.array(p90s)

        ax_evo.plot(ep_nums, means, "o-", color=ACCENT, lw=2, label="Mean priority", zorder=3)
        ax_evo.fill_between(ep_nums, means - stds, means + stds,
                             color=ACCENT, alpha=0.20, zorder=2, label="±1 std")
        ax_evo.plot(ep_nums, p90s, "s--", color="#FF9F43", lw=1.5,
                    label="90th percentile", zorder=3)
        ax_evo.set_xlabel("Training episode", fontsize=8, color=TEXT_CLR)
        ax_evo.set_ylabel("|TD Error|", fontsize=8, color=TEXT_CLR)
        ax_evo.legend(fontsize=7, facecolor=PANEL_BG,
                      edgecolor=WALL_CLR, labelcolor=TEXT_CLR)
        ax_evo.tick_params(labelsize=7)

    else:
        print("  ⚠  training_logs.npz not found – skipping log-based panels.")

    # ── C5: IS-weight landscape (sample a mock batch and map weights) ─────────
    ax_is_grid = fig.add_subplot(gs[1, 2])
    draw_grid(ax_is_grid, env,
              title="IS-Weight Map\n(lower weight = higher priority = more sampled)",
              show_legend=False)

    is_map  = np.full((GRID_SIZE, GRID_SIZE), np.nan)
    vis_map = np.zeros((GRID_SIZE, GRID_SIZE))
    vis_cnt = np.zeros((GRID_SIZE, GRID_SIZE))

    # Use TD mean as priority proxy; IS weight ∝ (N·p)^{-β}
    beta_is = 0.4
    N       = GRID_SIZE * GRID_SIZE  # approx
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if not np.isnan(td_mean[r, c]):
                prio = td_mean[r, c] + 1e-5
                is_map[r, c] = prio ** beta_is   # ∝ 1/weight (higher = sampled more)

    im5 = ax_is_grid.imshow(
        is_map, origin="upper",
        extent=[-0.5, GRID_SIZE-0.5, GRID_SIZE-0.5, -0.5],
        cmap="YlOrRd", alpha=0.70, zorder=2
    )
    cb5 = fig.colorbar(im5, ax=ax_is_grid, fraction=0.035, pad=0.02)
    cb5.ax.tick_params(labelsize=7, colors=MUTED)
    cb5.set_label("Sampling weight ∝ p^β", fontsize=8, color=TEXT_CLR)

    for ax in fig.get_axes():
        ax.tick_params(colors=MUTED)

    plt.savefig(filename, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ Saved {filename}")


# ════════════════════════════════════════════════════════════════════════════
#  PLOT D  –  Per-step option-critic parameter tracking on the grid
# ════════════════════════════════════════════════════════════════════════════

def plot_D_parameter_tracking(env, actor, critic,
                               filename="plotD_parameter_tracking.png"):
    print("  Generating option-critic parameter tracking …")

    steps = _run_episode(env, actor, critic)
    T     = len(steps)
    ts    = np.arange(T)

    fig = plt.figure(figsize=(26, 16), facecolor=DARK_BG)
    fig.suptitle(
        "Option-Critic Architecture  —  Per-Step Parameter Landscape (Single Episode)",
        fontsize=17, fontweight="bold", color=TEXT_CLR, y=0.99
    )

    # Layout: top-left = grid, top-right = 4 Q-option lines,
    #         bottom = 5 parameter panels stacked
    gs_top = gridspec.GridSpec(1, 2, figure=fig,
                               top=0.93, bottom=0.52,
                               left=0.04, right=0.97, wspace=0.30)
    gs_bot = gridspec.GridSpec(5, 1, figure=fig,
                               top=0.46, bottom=0.03,
                               left=0.07, right=0.97, hspace=0.06)

    ax_grid  = fig.add_subplot(gs_top[0])
    ax_qall  = fig.add_subplot(gs_top[1])

    ax_p1 = fig.add_subplot(gs_bot[0])
    ax_p2 = fig.add_subplot(gs_bot[1], sharex=ax_p1)
    ax_p3 = fig.add_subplot(gs_bot[2], sharex=ax_p1)
    ax_p4 = fig.add_subplot(gs_bot[3], sharex=ax_p1)
    ax_p5 = fig.add_subplot(gs_bot[4], sharex=ax_p1)

    opts    = [s["option"]    for s in steps]
    q_curr  = [s["q_curr"]   for s in steps]
    max_q   = [s["max_q"]    for s in steps]
    adv     = [s["advantage"] for s in steps]
    beta_s  = [s["beta"]     for s in steps]
    ent_s   = [s["entropy"]  for s in steps]
    td_s    = [s["td_error"] for s in steps]
    rew_s   = [s["reward"]   for s in steps]
    term_ts = [i for i, s in enumerate(steps) if s["terminated_option"] and i > 0]

    xs = [s["c"] + np.random.uniform(-0.12, 0.12) for s in steps]
    ys = [s["r"] + np.random.uniform(-0.12, 0.12) for s in steps]

    # ── Grid: β-coloured trajectory ─────────────────────────────────────────
    draw_grid(ax_grid, env,
              title="Episode Trajectory\n(line colour = β termination prob)",
              show_legend=True)

    norm_b = mcolors.Normalize(vmin=0, vmax=1)
    cmap_b = cm.get_cmap("coolwarm")
    for i in range(T - 1):
        ax_grid.plot(
            [xs[i], xs[i+1]], [ys[i], ys[i+1]],
            color=cmap_b(norm_b(beta_s[i])),
            linewidth=2.8, alpha=0.85, solid_capstyle="round", zorder=5
        )
    # Annotate termination events on the grid
    for i in term_ts:
        ax_grid.scatter(xs[i], ys[i], s=80, marker="X",
                        color="white", edgecolors=OPTION_COLORS[opts[i]],
                        linewidths=1.0, zorder=8)
        ax_grid.text(xs[i]+0.25, ys[i]-0.25, f"ω{opts[i]}",
                     fontsize=5.5, color=OPTION_COLORS[opts[i]], zorder=9)

    sm_b = cm.ScalarMappable(cmap=cmap_b, norm=norm_b)
    sm_b.set_array([])
    cb_b = fig.colorbar(sm_b, ax=ax_grid, fraction=0.03, pad=0.02)
    cb_b.ax.tick_params(labelsize=7, colors=MUTED)
    cb_b.set_label("β(s, ω_t)", fontsize=8, color=TEXT_CLR)

    # ── Right: all 4 Q(s,ω) over time ───────────────────────────────────────
    ax_qall.set_facecolor(GRID_DARK)
    ax_qall.set_title("Q(s, ω) for All Options Over Time",
                       fontsize=10, fontweight="bold", color=TEXT_CLR)
    for opt in range(4):
        q_opt = [s["q_all"][opt] for s in steps]
        ax_qall.plot(ts, q_opt, color=OPTION_COLORS[opt],
                     lw=1.8, label=OPTION_NAMES[opt], alpha=0.85)
    # Mark active option at each step
    for i, s in enumerate(steps):
        ax_qall.scatter(i, s["q_all"][s["option"]], s=18,
                        color=OPTION_COLORS[s["option"]], zorder=5,
                        edgecolors=DARK_BG, linewidths=0.3)
    for tt in term_ts:
        ax_qall.axvline(tt, color="white", lw=0.5, alpha=0.25)
    ax_qall.set_xlabel("Time step t", fontsize=9, color=TEXT_CLR)
    ax_qall.set_ylabel("Q value", fontsize=9, color=TEXT_CLR)
    ax_qall.legend(loc="upper right", fontsize=7,
                   facecolor=PANEL_BG, edgecolor=WALL_CLR, labelcolor=TEXT_CLR)
    ax_qall.tick_params(labelsize=7)

    # ── helper ───────────────────────────────────────────────────────────────
    def shade_and_vlines(axx):
        seg_start = 0
        for i in range(1, T):
            if opts[i] != opts[i-1] or i == T - 1:
                end = i if i < T - 1 else T
                axx.axvspan(seg_start, end, alpha=0.10,
                            color=OPTION_COLORS[opts[seg_start]], zorder=0)
                seg_start = i
        for tt in term_ts:
            axx.axvline(tt, color="white", lw=0.55, alpha=0.30, zorder=1)
        axx.set_facecolor(GRID_DARK)
        axx.tick_params(labelsize=7)
        for sp in ["top", "right"]:
            axx.spines[sp].set_visible(False)

    # P1: Q(s,ω_t) vs max Q
    shade_and_vlines(ax_p1)
    ax_p1.plot(ts, q_curr, color=ACCENT,    lw=2.0, label="Q(s, ω_t)")
    ax_p1.plot(ts, max_q,  color="#FF6B6B", lw=1.5, linestyle="--",
               label="max_ω Q(s,ω)", alpha=0.8)
    ax_p1.fill_between(ts, q_curr, max_q, color="#FF6B6B", alpha=0.12)
    ax_p1.set_ylabel("Q Value", fontsize=8, color=TEXT_CLR)
    ax_p1.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG,
                 edgecolor=WALL_CLR, labelcolor=TEXT_CLR)
    ax_p1.tick_params(labelbottom=False)

    # P2: Advantage A(s,ω_t)
    shade_and_vlines(ax_p2)
    ax_p2.bar(ts, adv, color=[OPTION_COLORS[o] for o in opts],
              alpha=0.75, width=0.9)
    ax_p2.axhline(0, color="white", lw=0.7, linestyle="--", alpha=0.4)
    ax_p2.set_ylabel("Advantage\nA(s,ω)", fontsize=8, color=TEXT_CLR)
    ax_p2.tick_params(labelbottom=False)

    # P3: Termination probability β
    shade_and_vlines(ax_p3)
    ax_p3.plot(ts, beta_s, color="#FFD93D", lw=2.0, label="β(s, ω_t)")
    ax_p3.fill_between(ts, 0, beta_s, color="#FFD93D", alpha=0.20)
    ax_p3.set_ylim(0, 1.08)
    ax_p3.set_ylabel("β  (term)", fontsize=8, color=TEXT_CLR)
    ax_p3.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG,
                 edgecolor=WALL_CLR, labelcolor=TEXT_CLR)
    ax_p3.tick_params(labelbottom=False)

    # P4: Policy entropy H(π_ω)
    shade_and_vlines(ax_p4)
    ax_p4.plot(ts, ent_s, color="#A855F7", lw=2.0, label="H(π_ω_t)")
    ax_p4.fill_between(ts, 0, ent_s, color="#A855F7", alpha=0.20)
    ax_p4.set_ylabel("Entropy\nH(π_ω)", fontsize=8, color=TEXT_CLR)
    ax_p4.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG,
                 edgecolor=WALL_CLR, labelcolor=TEXT_CLR)
    ax_p4.tick_params(labelbottom=False)

    # P5: TD error + reward events
    shade_and_vlines(ax_p5)
    ax_p5.bar(ts, td_s, color="#FF9F43", alpha=0.80, width=0.9, label="|TD error|")
    # Reward event markers
    for i, s in enumerate(steps):
        if s["bn_r"] > 0:
            ax_p5.axvline(i, color="#FF9F43", lw=2.5, alpha=0.95, zorder=4)
            ax_p5.text(i, max(td_s) * 0.92, "BN", fontsize=4.5,
                       color="#FF9F43", ha="center", zorder=5)
        if s["goal_r"] > 0:
            ax_p5.axvline(i, color="#FFD700", lw=3.0, alpha=0.98, zorder=5)
            ax_p5.text(i, max(td_s) * 0.92, "★", fontsize=7,
                       color="#FFD700", ha="center", zorder=6)
    ax_p5.set_ylabel("|TD Error|", fontsize=8, color=TEXT_CLR)
    ax_p5.set_xlabel("Time step  t", fontsize=9, color=TEXT_CLR)
    ax_p5.legend(loc="upper right", fontsize=7, facecolor=PANEL_BG,
                 edgecolor=WALL_CLR, labelcolor=TEXT_CLR)

    plt.savefig(filename, dpi=155, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  ✓ Saved {filename}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Option-Critic + PER  —  Scientific Visualisation Suite")
    print("=" * 60)

    env = make_env()
    actor, critic = load_models(env)

    print("\n[A]  Per-option Q-value heatmaps …")
    plot_A_qvalue_heatmaps(env, actor, critic)

    print("\n[B]  10 random-episode scientific traces …")
    plot_B_episode_traces(env, actor, critic, n=10)

    print("\n[C]  PER priority landscape & batch dynamics …")
    plot_C_per_visualization(env, actor, critic)

    print("\n[D]  Per-step option-critic parameter tracking …")
    plot_D_parameter_tracking(env, actor, critic)

    env.close()
    print("\n✓ All plots generated successfully.")


if __name__ == "__main__":
    main()
