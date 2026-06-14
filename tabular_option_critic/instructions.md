specification for the coding agent. It is written against the current `so.py` architecture. 

---

# OBJECTIVE

Transform the current tabular Option-Critic FourRooms implementation into a diagnostics-heavy research version that:

1. Adds a doorway exploration reward.
2. Logs complete trajectories.
3. Produces publication-quality plots.
4. Produces evolution snapshots through training.
5. Saves everything into a structured results directory.
6. Does NOT add PER.

---

# PART 1 — ADD DOORWAY DISCOVERY REWARD

## Goal

Reward first-time doorway visitation within an episode.

Reward:

```python
DOORWAY_REWARD = 0.5
```

Only once per doorway per episode.

---

## Identify Doorways

The FourRooms layout has four room-connecting hallway cells.

Add a helper:

```python
self.doorway_states = [...]
```

Store them as state indices, not coordinates.

Use:

```python
env.tostate[(row,col)]
```

to convert.

The coder should automatically detect doorways from geometry rather than hardcoding if possible.

Doorway = traversable cell connecting two rooms through a wall opening.

---

## Episode-level bookkeeping

At episode start:

```python
visited_doorways = set()
```

---

## Modify environment reward

Current:

```python
reward = float(done)
```

Replace with:

```python
reward = float(done)

if state in doorway_states and state not in visited_doorways:
    reward += 0.5
    visited_doorways.add(state)
```

Must happen before critic update.

---

# PART 2 — CREATE DIAGNOSTICS LOGGER

Create class:

```python
class DiagnosticsLogger
```

Responsibilities:

```python
episodes
snapshots
visitation_counts
option_occupancy
option_switches
termination_events
```

---

## Per-step logging

Every timestep log:

```python
{
    episode,
    timestep,

    state,
    cell,

    option,
    action,

    reward,

    qomega,
    qomega_all,

    advantage,
    advantage_all,

    beta,

    terminated,

    q_u,

    option_duration
}
```

---

## Snapshot logging

Every:

```python
SNAPSHOT_INTERVAL = 250
```

episodes:

Store:

```python
Q_Omega_table

Q_U_table

termination weights

policy weights
```

deep copied.

---

# PART 3 — RESULTS DIRECTORY STRUCTURE

Automatically create:

```text
results/

├── learning_curves/

├── qomega_maps/

├── qomega_evolution/

├── beta_maps/

├── beta_evolution/

├── entropy_maps/

├── specialization/

├── visitation/

├── occupancy/

├── episode_traces/

├── option_statistics/

└── doorway_analysis/
```

---

# PART 4 — QOMEGA HEATMAPS

After training:

For each option:

```python
QΩ(s,o)
```

Generate:

```text
option_0_qomega.png
option_1_qomega.png
...
```

13×13 heatmaps.

Walls masked.

Colorbar required.

Goal cell highlighted.

---

# PART 5 — OPTION SPECIALIZATION MAP

Generate:

```python
argmax(QΩ(s,:))
```

for every state.

Plot:

```text
specialization_map.png
```

Each color = option ID.

Shows which option dominates each state.

---

## Confidence map

Also compute:

```python
max(QΩ) - second_best(QΩ)
```

Plot:

```text
option_confidence.png
```

This is more informative than raw argmax.

---

# PART 6 — TERMINATION HEATMAPS

Already partially exists.

Replace with:

For each option:

```python
β(s)
```

Plot separately.

Save instead of show.

Add:

```python
goal marker
doorway markers
```

---

# PART 7 — POLICY ENTROPY MAPS

For each option:

Compute:

```python
H(pi)

= -sum(p*log(p))
```

for every state.

Generate:

```text
entropy_option_0.png
...
```

Interpretation:

Low entropy:

```text
deterministic option
```

High entropy:

```text
uncertain option
```

---

# PART 8 — VISITATION HEATMAP

Track:

```python
state_visits[state]
```

during training.

Generate:

```text
state_visitation.png
```

Shows where learning spent time.

---

# PART 9 — OPTION OCCUPANCY MAPS

Track:

```python
option_occupancy[state, option]
```

Whenever option active:

```python
+= 1
```

Generate:

For each option:

```text
occupancy_option_0.png
...
```

This is one of the most valuable diagnostics.

---

# PART 10 — OPTION SWITCH ANALYSIS

Track:

```python
option switches per episode
```

Plot:

```text
option_switches.png
```

Also:

```python
average option duration
```

Plot:

```text
option_duration.png
```

---

# PART 11 — EPISODE TRACE VISUALIZATION

Randomly sample:

```python
10 successful episodes
```

after training.

Generate:

```text
trace_001.png
...
trace_010.png
```

---

## Trace Figure Layout

Use 3x3 grid.

### Panel 1

Grid trajectory.

Show:

```python
start = green
goal = red
path = blue
```

Arrows between states.

---

### Panel 2

Option timeline.

```python
option vs timestep
```

Color coded.

---

### Panel 3

Action timeline.

```python
up/down/left/right
```

---

### Panel 4

QΩ(s,o_t)

over time.

---

### Panel 5

Advantage

```python
AΩ
```

over time.

---

### Panel 6

β(s)

over time.

---

### Panel 7

Reward timeline.

Show doorway rewards distinctly.

---

### Panel 8

Termination events.

Binary signal.

---

### Panel 9

Option durations.

Segment lengths.

---

# PART 12 — EVOLUTION SNAPSHOTS

Using stored snapshots.

Generate evolution figures.

---

## QΩ evolution

Every snapshot:

```python
argmax(QΩ)
```

Save:

```text
specialization_ep_250.png
specialization_ep_500.png
...
```

---

## β evolution

Every snapshot:

Generate option termination maps.

Save:

```text
beta_ep_250_option_0.png
...
```

---

## Entropy evolution

Every snapshot:

Generate entropy maps.

Save:

```text
entropy_ep_250_option_0.png
...
```

---

# PART 13 — OPTION RESPONSIBILITY ANALYSIS

Generate:

```python
fraction_of_time_option_selected
```

per state.

Normalize occupancy.

Create:

```text
option_responsibility_map.png
```

This reveals learned spatial decomposition.

---

# PART 14 — DOORWAY ANALYSIS

Track:

```python
doorway visitation counts
```

during training.

Generate:

```text
doorway_usage.png
```

Bar chart:

```python
doorway_id vs visits
```

---

## Doorway reward acquisition

Track:

```python
doorway bonuses per episode
```

Generate:

```text
doorway_reward_curve.png
```

---

# PART 15 — SAVE EVERYTHING

Replace every:

```python
plt.show()
```

with:

```python
plt.savefig(...)
plt.close()
```

No interactive plotting.

---

# PART 16 — FINAL TRAINING SUMMARY

At end print:

```text
Total episodes
Success rate

Average return

Average option duration

Average switches

Doorway reward frequency

Most used option

Least used option
```

---

# PART 17 — IMPORTANT IMPLEMENTATION DETAILS

The logger should be completely separated from learning.

Recommended classes:

```python
DiagnosticsLogger

PlotManager
```

Training loop should only call:

```python
logger.log_step(...)
logger.snapshot(...)
```

Everything else should happen after training.

Do not modify:

```python
Critic update equations
Option update equations
Termination update equations
```

except for adding the doorway reward into the reward signal.

The reward shaping should be the only behavioral change; all other modifications should be diagnostics and visualization.
