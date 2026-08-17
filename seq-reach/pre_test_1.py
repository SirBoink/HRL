# %% [markdown]
"""
1. Can a linear readout of a frozen controller's hidden state reproduce the
    true discounted future occupancy `psi(x) = W x`?
2. Fit on one set of rollouts, score on a disjoint set, noise free, tail excluded, moving steps only.
"""
# %%
import sys
from pathlib import Path

#pointing to right path for some dependecies
_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ROOT = next(p for p in (_here, *_here.parents) if (p / "sf_boundaries.py").exists())
sys.path.insert(0, str(ROOT))
POC_DIR = ROOT / "poc"

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import motornet
import numpy as np
import torch
from motornet.effector import RigidTendonArm26
from motornet.environment import RandomTargetReach
from motornet.muscle import RigidTendonHillMuscle
from motornet.policy import PolicyGRU
from tqdm import tqdm

import sf_boundaries as sf #only for the phi, nothing else.

print(f"motornet {motornet.__version__}, gymnasium {gym.__version__}, "
      f"torch {torch.__version__}, matplotlib {matplotlib.__version__}")

assert sf.REGRESS_ON_PHI, "the h vs phi assumes REGRESS_ON_PHI"


# %% [markdown]
# ## Parameters

# %%
HIDDEN_DIM = 64
LR = 1e-3
N_BATCH = 3000
BATCH_SIZE = 32
TRAIN_EP = 1.0  # s

EVAL_EP = 6.0  # s
EVAL_BATCH = 128

GAMMAS = (0.9, 0.95, 0.97, 0.98, 0.99, 0.995)
N_TAIL = 5  # score only steps this many horizons from the end; bias ~ e**-5 < 1%
V_MOVE = 0.05  # m/s, above this the hand counts as moving

R2_GO = 0.9
CERT_RANGE = (0.95, 0.98)
MIN_VALID = 200
SUCCESS_DIST = 0.03  # m

POLICY_SEED = 0
TRAIN_SEED = 1
TEST_SEED = 2
CLOUD_SEED = 0

DEVICE = torch.device("cpu")
CKPT = POC_DIR / "stage0_policy.pt"

C_FULL, C_PHI, C_TRUTH, C_PRED, C_GREY = "#2a78d6", "#eb6834", "#898781", "#7b52ab", "#b8b6b0"


# %% [markdown]
# ## Plant and controller

# %%
def build_env(differentiable, max_ep_duration):
  effector = RigidTendonArm26(muscle=RigidTendonHillMuscle())
  return RandomTargetReach(effector=effector, differentiable=differentiable,
                           max_ep_duration=max_ep_duration)


torch.manual_seed(POLICY_SEED)
train_env = build_env(differentiable=True, max_ep_duration=TRAIN_EP)
DT = train_env.dt  # s
policy = PolicyGRU(train_env.observation_space.shape[0], HIDDEN_DIM,
                   train_env.n_muscles, DEVICE)
print(f"obs dim {train_env.observation_space.shape[0]}, muscles {train_env.n_muscles}, "
      f"hidden {HIDDEN_DIM}, dt {DT} s")


# %% [markdown]
# ## Train, then freeze
#
# Skipped if `poc/stage0_policy.pt` exists

# %%
def train():
  optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
  losses = []
  progress = tqdm(range(N_BATCH), desc="training", unit="batch")
  for _ in progress:
    hidden = policy.init_hidden(BATCH_SIZE)
    obs, info = train_env.reset(options={"batch_size": BATCH_SIZE})
    fingertip = [info["states"]["fingertip"][:, None, :]]
    goals = [info["goal"][:, None, :]]

    terminated = truncated = False
    while not (terminated or truncated):
      action, hidden = policy(obs, hidden)
      obs, _, terminated, truncated, info = train_env.step(action)
      fingertip.append(info["states"]["fingertip"][:, None, :])
      goals.append(info["goal"][:, None, :])

    dist = torch.abs(torch.cat(fingertip, dim=1) - torch.cat(goals, dim=1)).sum(dim=-1)
    loss = dist.mean()
    optimizer.zero_grad()
    loss.backward()
    # Gradients through the unrolled plant explode without a clip.
    torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()
    losses.append(loss.item())
    progress.set_postfix(loss=f"{losses[-1]:.4f}")
  return losses


if CKPT.exists():
  state = torch.load(CKPT, weights_only=False)
  policy.load_state_dict(state["policy"])
  losses = state["losses"]
  print(f"loaded frozen policy from {CKPT} ({len(losses)} training batches)")
else:
  losses = train()
  torch.save({"policy": policy.state_dict(), "losses": losses,
              "config": {"HIDDEN_DIM": HIDDEN_DIM}}, CKPT)
  print(f"final training loss {losses[-1]:.4f}, saved to {CKPT}")

policy.eval()  # frozen from here on


# %%
fig, ax = plt.subplots(figsize=(6, 3.2), constrained_layout=True)
ax.plot(losses, color=C_FULL, lw=1)
ax.set(xlabel="training batch", ylabel="loss (mean per-step L1 distance, m)",
       title="Training loss")
ax.set_yscale("log")
fig.savefig(POC_DIR / "pre_test_1_training.png", dpi=140)
plt.show()


# %% [markdown]
# ## Does the frozen policy reach and stop?

# %%
eval_env = build_env(differentiable=False, max_ep_duration=EVAL_EP)

# The Monte-Carlo target is only valid against a noise-free rollout.
for name in ("obs_noise", "action_noise", "proprioception_noise", "vision_noise"):
  noise = np.asarray(getattr(eval_env, name), dtype=float)
  assert np.all(noise == 0), f"{name} is non-zero; Monte-Carlo ground truth invalid"
print("noise guard passed: obs / action / proprioception / vision noise all zero")

hidden_te, xy_te = sf.collect(eval_env, policy, EVAL_BATCH, seed=TEST_SEED)
T = len(xy_te)

# collect() returns hidden and fingertip only; the goal is fixed per episode, so one
# reset with the same seed recovers it.
_, info0 = eval_env.reset(seed=TEST_SEED, options={"batch_size": EVAL_BATCH})
goal_te = torch.as_tensor(info0["goal"])

final_dist = torch.linalg.norm(xy_te[-1] - goal_te, dim=-1)  # (batch,)
speed_te = torch.linalg.norm(xy_te[1:] - xy_te[:-1], dim=-1) / DT  # (T-1, batch)
speed_te = torch.cat([speed_te[:1], speed_te], dim=0)  # pad to length T
moving_te = speed_te > V_MOVE
reach_secs = moving_te.sum(dim=0).float() * DT

reach_ok = final_dist.median().item() < SUCCESS_DIST
print(f"median final distance {final_dist.median():.4f} m (reach success: {reach_ok})")
print(f"median time moving {reach_secs.median():.2f} s of {EVAL_EP} s "
      f"-> matched horizon ~ gamma {1 - DT / reach_secs.median():.3f}")


# %%
# plots
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
for b in range(min(8, EVAL_BATCH)):
  ax0.plot(xy_te[:, b, 0], xy_te[:, b, 1], color=C_FULL, lw=0.8, alpha=0.7)
  ax0.scatter(*xy_te[0, b], color=C_GREY, s=18, zorder=3)
  ax0.scatter(*goal_te[b], color=C_TRUTH, marker="x", s=40, zorder=3)
ax0.set(xlabel="x (m)", ylabel="y (m)", title="Reaches (o start, x target)")
ax0.set_aspect("equal")

t_axis = np.arange(T) * DT
for b in range(min(8, EVAL_BATCH)):
  ax1.plot(t_axis, speed_te[:, b], color=C_FULL, lw=0.8, alpha=0.7)
ax1.axhline(V_MOVE, color=C_PHI, ls="--", lw=1, label=f"V_MOVE = {V_MOVE} m/s")
ax1.set(xlabel="time (s)", ylabel="fingertip speed (m/s)", title="Move, then stop")
ax1.legend()
fig.savefig(POC_DIR / "pre_test_1_sanity.png", dpi=140)
plt.show()


# %% [markdown]
# ## Detectors and the two rollout sets

# %%
eval_env.reset(seed=CLOUD_SEED, options={"batch_size": 1})
cloud = eval_env.joint2cartesian(
  eval_env.effector.draw_random_uniform_states(4000)).chunk(2, dim=-1)[0]
centers, width = sf.detector_grid(cloud)
print(f"{centers.shape[0]} detectors, width {width:.3f} m over the reachable cloud")

hidden_tr, xy_tr = sf.collect(eval_env, policy, EVAL_BATCH, seed=TRAIN_SEED)
phi_tr = sf.phi(xy_tr, centers, width)
phi_te = sf.phi(xy_te, centers, width)
print(f"fit rollouts {tuple(phi_tr.shape)}, scored rollouts {tuple(phi_te.shape)}")


# %% [markdown]
# ## Fit on train, score on held out test

# %%
def masked_fit_quality(pred, truth, mask):

  # Pooled R square and median relative L2 error over the masked (time, batch) steps.

  p, m = pred[mask], truth[mask]  # (n_valid, n_detectors)
  if len(p) < MIN_VALID:
    return float("nan"), float("nan"), len(p)
  ss_res = ((p - m) ** 2).sum()
  ss_tot = ((m - m.mean(dim=0)) ** 2).sum()
  r2 = (1 - ss_res / ss_tot).item()
  rel = (torch.linalg.norm(p - m, dim=-1) / torch.linalg.norm(m, dim=-1)).median().item()
  return r2, rel, len(p)


t_index = torch.arange(T)[:, None]  # (T, 1), broadcasts over the batch
hidden_zero_tr = torch.zeros_like(hidden_tr)
hidden_zero_te = torch.zeros_like(hidden_te)

results = {}
for g in GAMMAS:
  horizon = 1 / (1 - g)  # steps
  tail_ok = (T - 1 - t_index) >= N_TAIL * horizon  # (T, 1)
  mask = moving_te & tail_ok  # (T, batch)

  w_full = sf.fit_psi(hidden_tr, phi_tr, g)
  pred_full = sf.psi(hidden_te, phi_te, w_full)
  w_phi = sf.fit_psi(hidden_zero_tr, phi_tr, g)
  pred_phi = sf.psi(hidden_zero_te, phi_te, w_phi)
  truth = sf.monte_carlo_psi(phi_te, g)

  r2_full, rel_full, n = masked_fit_quality(pred_full, truth, mask)
  r2_phi, rel_phi, _ = masked_fit_quality(pred_phi, truth, mask)
  worst_bias = (g ** (T - 1 - t_index).float()).expand_as(mask)[mask]
  worst_bias = worst_bias.max().item() if mask.any() else float("nan")
  results[g] = dict(horizon=horizon, n=n, worst_bias=worst_bias, r2_full=r2_full,
                    rel_full=rel_full, r2_phi=r2_phi, rel_phi=rel_phi)

header = f"{'gamma':>7} {'horizon':>8} {'n_valid':>8} {'tailbias':>9} " \
         f"{'R2[h,phi]':>10} {'R2 phi':>8} {'rel[h,phi]':>11} {'rel phi':>8}"
print(header)
print("-" * len(header))
for g, r in results.items():
  print(f"{g:>7} {r['horizon']:>8.0f} {r['n']:>8d} {r['worst_bias']:>9.1%} "
        f"{r['r2_full']:>10.3f} {r['r2_phi']:>8.3f} {r['rel_full']:>11.1%} "
        f"{r['rel_phi']:>8.1%}")


# %% [markdown]
# ## Seeing it track

# %%
g_show = 0.98
w_show = sf.fit_psi(hidden_tr, phi_tr, g_show)
pred_show = sf.psi(hidden_te, phi_te, w_show)
truth_show = sf.monte_carlo_psi(phi_te, g_show)
horizon = 1 / (1 - g_show)
mask_show = moving_te & ((T - 1 - t_index) >= N_TAIL * horizon)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
b = 0  # one held-out episode
top = truth_show[:, b].sum(dim=0).topk(4).indices  # its most-visited detectors
for k, d in enumerate(top):
  ax0.plot(t_axis, truth_show[:, b, d], color=C_TRUTH, lw=1.4,
           label="Monte-Carlo truth" if k == 0 else None)
  ax0.plot(t_axis, pred_show[:, b, d], color=C_PRED, lw=1.1, ls="--",
           label="linear readout W h" if k == 0 else None)
ax0.set(xlabel="time (s)", ylabel=f"discounted occupancy (gamma={g_show})",
        title="Readout vs truth, top detectors")
ax0.legend()

p = pred_show[mask_show].flatten().numpy()
m = truth_show[mask_show].flatten().numpy()
ax1.scatter(m, p, s=2, alpha=0.15, color=C_FULL, edgecolors="none")
lims = [min(m.min(), p.min()), max(m.max(), p.max())]
ax1.plot(lims, lims, color=C_GREY, lw=1, ls="--")
ax1.set(xlabel="Monte-Carlo truth", ylabel="linear readout W h", aspect="equal",
        title=f"Held-out, moving steps (R2={results[g_show]['r2_full']:.3f})")
fig.savefig(POC_DIR / "pre_test_1_tracking.png", dpi=140)
plt.show()


# %% [markdown]
# ## Decision panel

# %%
plotted = [g for g in GAMMAS if not np.isnan(results[g]["r2_full"])]
r2_full = [results[g]["r2_full"] for g in plotted]
r2_phi = [results[g]["r2_phi"] for g in plotted]

fig, ax = plt.subplots(figsize=(6.4, 4), constrained_layout=True)
ax.axvspan(*CERT_RANGE, color=C_GREY, alpha=0.18, label="certified range")
ax.axhline(R2_GO, color=C_GREY, ls="--", lw=1, label=f"GO threshold R2={R2_GO}")
ax.plot(plotted, r2_full, "-o", color=C_FULL, label="readout from [h, phi]")
ax.plot(plotted, r2_phi, "-s", color=C_PHI, label="diagnostic: phi only")
ax.set(xlabel="gamma (discount)", ylabel="held-out R^2 on moving steps",
       title="Successor-feature readout accuracy",
       ylim=(min(0, min(r2_phi) - 0.05), 1.02))
ax.legend(loc="lower left")
fig.savefig(POC_DIR / "pre_test_1_decision.png", dpi=140)
plt.show()

"""
# %% [markdown]
# just a normal check to see if R_square values are above the threshold we set, realistically means nothing.

# %%
judged = [g for g in GAMMAS if CERT_RANGE[0] <= g <= CERT_RANGE[1]
          and results[g]["n"] >= MIN_VALID]
r2_ok = bool(judged) and all(results[g]["r2_full"] >= R2_GO for g in judged)
go = reach_ok and r2_ok

gap = [f"{g}: [h,phi] {results[g]['rel_full']:.0%} vs phi {results[g]['rel_phi']:.0%}"
       for g in judged]
print(f"  policy reaches and settles (median final < {SUCCESS_DIST} m): {reach_ok}")
print(f"  readout R2 >= {R2_GO} across gamma in {CERT_RANGE}: {r2_ok}")
print(f"  judged at gamma = {judged}")
print(f"  diagnostic, relative error [h,phi] vs phi-only: {gap}")
print()
print(f"Pre Test 1: {'GO' if go else 'NO-GO'}")

"""
