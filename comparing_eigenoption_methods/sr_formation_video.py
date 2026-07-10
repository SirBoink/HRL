import argparse
import json
import os
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MPL_CACHE_DIR = Path("/tmp/eigenoptions_mpl_cache")
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import TwoSlopeNorm

from even_better_comparisons import (
    COLORS,
    FourRoomsEnv,
    build_random_policy_transition_matrix,
    build_transition_table,
    draw_env_grid,
    progress,
    set_seed,
    sr_eigenpairs,
    values_to_grid,
)


TITLE = "Successor Representation Formation in FourRooms"


@dataclass
class TailTransition:
    episode: int
    start: tuple
    end: tuple


@dataclass
class SRSnapshot:
    step: int
    episode: int
    current_state_idx: int
    current_coord: tuple
    trajectory_tail: list
    psi: np.ndarray
    visited_count: int
    coverage_pct: float
    relative_error: float
    current_sr_row: np.ndarray
    eigenvector: np.ndarray
    eigen_corr: float


@dataclass
class RenderContext:
    env: FourRoomsEnv
    num_states: int
    matrix_vmax: float
    future_vmax: float
    eigen_vmax: float
    reference_rank: int


class TabularSuccessorRepresentation:
    def __init__(self, num_states, gamma, alpha):
        self.num_states = int(num_states)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.psi = np.zeros((self.num_states, self.num_states), dtype=np.float64)

    def update(self, state_idx, next_state_idx):
        state_idx = int(state_idx)
        next_state_idx = int(next_state_idx)
        target = np.zeros(self.num_states, dtype=np.float64)
        target[state_idx] = 1.0
        target += self.gamma * self.psi[next_state_idx]
        self.psi[state_idx] += self.alpha * (target - self.psi[state_idx])


def build_snapshot_schedule(total_steps, smoke_test=False):
    total_steps = int(total_steps)
    if smoke_test:
        schedule = [{"start": 0, "end": total_steps, "every": 50}]
        steps = set(range(0, total_steps + 1, 50))
        steps.add(total_steps)
        return sorted(steps), schedule

    schedule = [
        {"start": 0, "end": min(200, total_steps), "every": 10},
        {"start": 200, "end": min(2000, total_steps), "every": 50},
        {"start": 2000, "end": total_steps, "every": 200},
    ]

    steps = {0, total_steps}
    first_end = min(200, total_steps)
    steps.update(range(10, first_end + 1, 10))

    if total_steps > 200:
        second_end = min(2000, total_steps)
        steps.update(range(250, second_end + 1, 50))

    if total_steps > 2000:
        steps.update(range(2200, total_steps + 1, 200))

    return sorted(step for step in steps if 0 <= step <= total_steps), schedule


def orient_vector(vector):
    oriented = np.asarray(vector, dtype=np.float64).copy()
    if oriented.size == 0:
        return oriented
    anchor = int(np.argmax(np.abs(oriented)))
    if oriented[anchor] < 0.0:
        oriented *= -1.0
    return oriented


def normalize_vector(vector):
    vector = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


def choose_reference_rank(eigenvalues, requested_rank=None, max_candidate_rank=24):
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if requested_rank is not None:
        rank = int(requested_rank)
        if rank <= 0 or rank >= len(eigenvalues):
            raise ValueError("--reference-rank must be in [1, num_states - 1]; rank 0 is the near-constant mode.")
        return rank

    if len(eigenvalues) <= 2:
        return 1

    upper = min(int(max_candidate_rank), len(eigenvalues) - 2)
    candidates = np.arange(2, upper + 1, dtype=np.int64)
    if candidates.size == 0:
        return 1

    left_gap = np.abs(eigenvalues[candidates - 1] - eigenvalues[candidates])
    right_gap = np.abs(eigenvalues[candidates] - eigenvalues[candidates + 1])
    separation = np.minimum(left_gap, right_gap)
    return int(candidates[int(np.argmax(separation))])


def matched_nontrivial_eigenvector(psi_hat, reference_vector):
    if np.linalg.norm(psi_hat) < 1e-12:
        return np.zeros_like(reference_vector), 0.0

    # The learned SR is not exactly symmetric during TD learning. We symmetrize
    # for a real eigendecomposition, then match to a fixed exact reference and
    # correct sign so near-degenerate modes do not arbitrarily flip in the video.
    psi_sym = 0.5 * (psi_hat + psi_hat.T)
    eigenvalues, eigenvectors_col = np.linalg.eigh(psi_sym)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors_col[:, order].T

    candidates = eigenvectors[1:]
    if len(candidates) == 0:
        return np.zeros_like(reference_vector), 0.0

    correlations = candidates @ reference_vector
    best_idx = int(np.argmax(np.abs(correlations)))
    corr = float(correlations[best_idx])
    vector = candidates[best_idx].copy()
    if corr < 0.0:
        vector *= -1.0
        corr *= -1.0
    return vector, corr


def validate_uniform_transition_model(env):
    transitions = build_transition_table(env)
    p_from_table = np.zeros((env.num_states, env.num_states), dtype=np.float64)
    for state_idx in range(env.num_states):
        for next_idx in transitions[state_idx]:
            p_from_table[state_idx, next_idx] += 0.25

    p_random = build_random_policy_transition_matrix(env)
    if not np.allclose(p_from_table, p_random):
        raise RuntimeError("Uniform-random transition matrix does not match the primitive transition table.")

    for state_idx, state in env.idx_to_state.items():
        for action in range(4):
            env.set_state(state)
            _, _, terminated, _, _ = env.step(action)
            next_idx = env.get_state_idx()
            if terminated or next_idx != transitions[state_idx, action]:
                raise RuntimeError("FourRoomsEnv.step no longer matches build_transition_table in exploration mode.")

    return p_random


def trajectory_segments(tail):
    segments = []
    current_segment = []
    current_episode = None

    for item in tail:
        if item.episode != current_episode:
            if current_segment:
                segments.append(current_segment)
            current_segment = [item.start, item.end]
            current_episode = item.episode
            continue

        if not current_segment or current_segment[-1] != item.start:
            current_segment.append(item.start)
        current_segment.append(item.end)

    if current_segment:
        segments.append(current_segment)
    return segments


def make_snapshot(
    step,
    episode,
    env,
    learner,
    tail,
    visited,
    exact_sr,
    exact_norm,
    reference_vector,
):
    current_state_idx = int(env.get_state_idx())
    current_coord = tuple(int(x) for x in env.current_state)
    psi_copy = learner.psi.copy()
    relative_error = float(np.linalg.norm(psi_copy - exact_sr) / exact_norm)
    eigenvector, eigen_corr = matched_nontrivial_eigenvector(psi_copy, reference_vector)
    visited_count = int(np.count_nonzero(visited))

    return SRSnapshot(
        step=int(step),
        episode=int(episode),
        current_state_idx=current_state_idx,
        current_coord=current_coord,
        trajectory_tail=trajectory_segments(tail),
        psi=psi_copy,
        visited_count=visited_count,
        coverage_pct=100.0 * visited_count / env.num_states,
        relative_error=relative_error,
        current_sr_row=psi_copy[current_state_idx].copy(),
        eigenvector=eigenvector.copy(),
        eigen_corr=float(eigen_corr),
    )


def collect_snapshots(args, snapshot_steps):
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = FourRoomsEnv(goal_state=None, terminate_on_goal=False, max_steps=args.horizon)
    validate_uniform_transition_model(env)

    exact_values, exact_vectors, exact_sr = sr_eigenpairs(env, args.gamma)
    if exact_sr.shape != (env.num_states, env.num_states):
        raise RuntimeError(f"Expected SR shape {(env.num_states, env.num_states)}, got {exact_sr.shape}.")

    reference_rank = choose_reference_rank(exact_values, args.reference_rank)
    reference_vector = normalize_vector(orient_vector(exact_vectors[reference_rank]))
    exact_norm = float(np.linalg.norm(exact_sr))
    learner = TabularSuccessorRepresentation(env.num_states, args.gamma, args.alpha)

    tail = deque(maxlen=int(args.tail_length))
    visited = np.zeros(env.num_states, dtype=bool)
    snapshots = []
    snapshot_lookup = set(snapshot_steps)

    env.reset(seed=args.seed)
    episode = 1
    visited[env.get_state_idx()] = True

    if 0 in snapshot_lookup:
        snapshots.append(
            make_snapshot(0, episode, env, learner, tail, visited, exact_sr, exact_norm, reference_vector)
        )

    iterator = progress(range(1, int(args.total_steps) + 1), desc="Collecting SR snapshots", unit="step")
    for step in iterator:
        state_idx = int(env.get_state_idx())
        start_state = tuple(int(x) for x in env.current_state)
        action = int(rng.integers(0, 4))

        _, _, _, truncated, _ = env.step(action)
        next_state_idx = int(env.get_state_idx())
        end_state = tuple(int(x) for x in env.current_state)

        learner.update(state_idx, next_state_idx)
        visited[next_state_idx] = True
        tail.append(TailTransition(episode=episode, start=start_state, end=end_state))

        if step in snapshot_lookup:
            snapshots.append(
                make_snapshot(step, episode, env, learner, tail, visited, exact_sr, exact_norm, reference_vector)
            )

        if truncated:
            episode += 1
            env.reset()
            visited[env.get_state_idx()] = True

    context = RenderContext(
        env=env,
        num_states=env.num_states,
        matrix_vmax=max(float(np.max(exact_sr)), 1e-12),
        future_vmax=max(float(np.max(exact_sr)), 1e-12),
        eigen_vmax=max(float(np.max(np.abs(reference_vector))), 1e-12),
        reference_rank=reference_rank,
    )
    return snapshots, context


def set_panel_title(ax, title):
    ax.set_title(title, loc="left", color=COLORS["text"], pad=7, fontweight="semibold")


def draw_agent(ax, coord, size=90):
    row, col = coord
    ax.scatter(
        col,
        row,
        s=size,
        marker="o",
        color="#F59E0B",
        edgecolor="white",
        linewidth=1.1,
        zorder=12,
    )


def draw_snapshot(fig, snapshot, context):
    env = context.env
    fig.clear()
    grid = fig.add_gridspec(
        2,
        2,
        left=0.055,
        right=0.965,
        bottom=0.105,
        top=0.885,
        wspace=0.24,
        hspace=0.28,
    )
    ax_explore = fig.add_subplot(grid[0, 0])
    ax_matrix = fig.add_subplot(grid[0, 1])
    ax_future = fig.add_subplot(grid[1, 0])
    ax_eigen = fig.add_subplot(grid[1, 1])

    draw_env_grid(ax_explore, env, goal=None)
    set_panel_title(ax_explore, "Uniform Random Exploration")
    for segment in snapshot.trajectory_tail:
        if len(segment) < 2:
            continue
        xs = [coord[1] for coord in segment]
        ys = [coord[0] for coord in segment]
        ax_explore.plot(
            xs,
            ys,
            color=COLORS["sr"],
            linewidth=2.0,
            alpha=0.62,
            solid_capstyle="round",
            zorder=6,
        )
    draw_agent(ax_explore, snapshot.current_coord, size=96)

    matrix_im = ax_matrix.imshow(
        snapshot.psi,
        origin="upper",
        cmap="magma",
        vmin=0.0,
        vmax=context.matrix_vmax,
        interpolation="nearest",
        aspect="auto",
    )
    set_panel_title(ax_matrix, r"Estimated SR Matrix $\hat{\Psi}$")
    ax_matrix.set_xlabel("predicted future occupancy state (column)")
    ax_matrix.set_ylabel("current state (row)")
    ax_matrix.set_xticks([0, context.num_states // 2, context.num_states - 1])
    ax_matrix.set_yticks([0, context.num_states // 2, context.num_states - 1])
    cbar_matrix = fig.colorbar(matrix_im, ax=ax_matrix, fraction=0.046, pad=0.035)
    cbar_matrix.set_label("discounted occupancy")

    future_grid = values_to_grid(env, snapshot.current_sr_row)
    future_im = ax_future.imshow(
        future_grid,
        origin="upper",
        cmap="viridis",
        vmin=0.0,
        vmax=context.future_vmax,
        interpolation="nearest",
    )
    draw_env_grid(ax_future, env, goal=None)
    draw_agent(ax_future, snapshot.current_coord, size=82)
    set_panel_title(ax_future, "Future Occupancy from Current State")
    cbar_future = fig.colorbar(future_im, ax=ax_future, fraction=0.046, pad=0.035)
    cbar_future.set_label("row of SR estimate")

    eigen_grid = values_to_grid(env, snapshot.eigenvector)
    eigen_norm = TwoSlopeNorm(vmin=-context.eigen_vmax, vcenter=0.0, vmax=context.eigen_vmax)
    eigen_im = ax_eigen.imshow(
        eigen_grid,
        origin="upper",
        cmap="coolwarm",
        norm=eigen_norm,
        interpolation="nearest",
    )
    draw_env_grid(ax_eigen, env, goal=None)
    set_panel_title(ax_eigen, "Emerging Eigenpurpose")
    ax_eigen.text(
        0.02,
        0.98,
        f"matched exact rank {context.reference_rank}\ncorr = {snapshot.eigen_corr:.3f}",
        transform=ax_eigen.transAxes,
        ha="left",
        va="top",
        fontsize=8.2,
        color=COLORS["text"],
        bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#C9CED6", alpha=0.92),
        zorder=20,
    )
    cbar_eigen = fig.colorbar(eigen_im, ax=ax_eigen, fraction=0.046, pad=0.035)
    cbar_eigen.set_label("matched eigenvector value")

    fig.suptitle(TITLE, fontsize=16, fontweight="bold", color=COLORS["text"], y=0.965)
    fig.text(
        0.5,
        0.045,
        (
            f"Primitive transitions: {snapshot.step:,} | Episode: {snapshot.episode} | "
            f"Visited states: {snapshot.visited_count} / {context.num_states} | "
            f"Relative SR error: {snapshot.relative_error:.3f} | "
            f"Eigenvector correlation: {snapshot.eigen_corr:.3f}"
        ),
        ha="center",
        va="center",
        fontsize=9.2,
        color=COLORS["text"],
    )
    fig.text(
        0.5,
        0.018,
        "Online TD learning under uniform random primitive actions",
        ha="center",
        va="center",
        fontsize=8.6,
        color=COLORS["muted"],
    )


def save_final_frame(path, snapshot, context):
    fig = plt.figure(figsize=(12.6, 9.0), facecolor="white")
    draw_snapshot(fig, snapshot, context)
    fig.savefig(path, bbox_inches="tight", facecolor="white", dpi=180)
    plt.close(fig)


def write_mp4(path, snapshots, context, fps):
    ffmpeg_found = shutil.which("ffmpeg") is not None and animation.writers.is_available("ffmpeg")
    if not ffmpeg_found:
        print("ffmpeg was not found by matplotlib; MP4 export skipped. Install ffmpeg to create the MP4.")
        return False, ffmpeg_found

    fig = plt.figure(figsize=(12.6, 9.0), facecolor="white")
    writer = animation.FFMpegWriter(
        fps=int(fps),
        codec="libx264",
        bitrate=2400,
        metadata={"title": TITLE, "artist": "sr_formation_video.py"},
    )
    try:
        with writer.saving(fig, str(path), dpi=150):
            iterator = progress(snapshots, desc="Writing MP4", unit="frame")
            for snapshot in iterator:
                draw_snapshot(fig, snapshot, context)
                writer.grab_frame(facecolor="white")
    except Exception as exc:
        print(f"MP4 export failed: {exc}")
        plt.close(fig)
        return False, ffmpeg_found

    plt.close(fig)
    print(f"Saved MP4: {path}")
    return True, ffmpeg_found


def write_gif(path, snapshots, context, fps):
    if not animation.writers.is_available("pillow"):
        print("Pillow writer is unavailable; GIF export skipped.")
        return False

    fig = plt.figure(figsize=(12.6, 9.0), facecolor="white")
    writer = animation.PillowWriter(fps=int(fps))
    try:
        with writer.saving(fig, str(path), dpi=110):
            iterator = progress(snapshots, desc="Writing GIF", unit="frame")
            for snapshot in iterator:
                draw_snapshot(fig, snapshot, context)
                writer.grab_frame(facecolor="white")
    except Exception as exc:
        print(f"GIF export failed: {exc}")
        plt.close(fig)
        return False

    plt.close(fig)
    print(f"Saved GIF: {path}")
    return True


def write_metadata(path, args, schedule, snapshots, context, ffmpeg_found, mp4_written, gif_written):
    final = snapshots[-1]
    metadata = {
        "seed": int(args.seed),
        "gamma": float(args.gamma),
        "alpha": float(args.alpha),
        "total_primitive_transitions": int(args.total_steps),
        "rollout_horizon": int(args.horizon),
        "snapshot_schedule": schedule,
        "fps": int(args.fps),
        "number_of_frames": int(len(snapshots)),
        "num_states": int(context.num_states),
        "final_state_coverage": {
            "visited": int(final.visited_count),
            "total": int(context.num_states),
            "percentage": float(final.coverage_pct),
        },
        "final_relative_sr_error": float(final.relative_error),
        "final_eigenvector_correlation": float(final.eigen_corr),
        "selected_reference_eigenvector_rank": int(context.reference_rank),
        "ffmpeg_found": bool(ffmpeg_found),
        "mp4_written": bool(mp4_written),
        "gif_requested": bool(args.gif),
        "gif_written": bool(gif_written),
        "behavior_policy": "uniform_random_primitive_actions",
    }
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata: {path}")


def parse_args():
    parser = argparse.ArgumentParser(description=TITLE)
    parser.add_argument("--total-steps", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=0.90)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--tail-length", type=int, default=45)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference-rank", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("better_plots/sr_formation"))
    parser.add_argument("--gif", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        args.total_steps = min(args.total_steps, 200)
        args.fps = min(args.fps, 5)

    if args.total_steps <= 0:
        raise ValueError("--total-steps must be positive.")
    if args.alpha <= 0.0:
        raise ValueError("--alpha must be positive.")
    if not 0.0 <= args.gamma < 1.0:
        raise ValueError("--gamma must be in [0, 1).")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_steps, schedule = build_snapshot_schedule(args.total_steps, smoke_test=args.smoke_test)
    snapshots, context = collect_snapshots(args, snapshot_steps)

    mp4_path = args.output_dir / "sr_formation_uniform_random.mp4"
    gif_path = args.output_dir / "sr_formation_uniform_random.gif"
    final_frame_path = args.output_dir / "sr_formation_final_frame.png"
    metadata_path = args.output_dir / "sr_formation_metadata.json"

    save_final_frame(final_frame_path, snapshots[-1], context)
    print(f"Saved final frame: {final_frame_path}")

    mp4_written, ffmpeg_found = write_mp4(mp4_path, snapshots, context, args.fps)
    gif_written = False
    if args.gif:
        gif_written = write_gif(gif_path, snapshots, context, args.fps)

    write_metadata(metadata_path, args, schedule, snapshots, context, ffmpeg_found, mp4_written, gif_written)
    if not mp4_written:
        print("Primary MP4 artifact was not created; see the ffmpeg message above.")


if __name__ == "__main__":
    main()
