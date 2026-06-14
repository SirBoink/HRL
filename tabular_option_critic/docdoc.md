# Documentation Design Document v3
## For Reinforcement Learning & Hierarchical RL Codebases

**Version:** 3.0  
**Target Audience:** AI Agents generating, auditing, or refactoring code documentation  
**Core Maxims:** 
1. *Code is read far more often than it is written.* 
2. *Documentation is the UI of the codebase.* 
3. *In RL/HRL, hidden state, time, and gradients are invisible in the code; documentation must make them visible.*

---

## 1. Philosophical Foundation: The "Readability-First" Paradigm

RL and HRL implementations are notoriously difficult to interpret because the execution flow is driven by mathematical recursion, temporal abstraction, and detached computation graphs. V2 of this document focused on *adding* information. V3 focuses on **Signal-to-Noise Ratio**. 

Documentation must not merely exist; it must actively reduce the cognitive load on the reader. To achieve this, AI agents must adhere to five immutable principles:

1. **Provenance over Presence:** Do not just state what a tensor is; state *where it came from* and *what it influences*. (e.g., `old_log_probs` is not just a shape; it is the "behavior cloning anchor from the previous policy").
2. **Pitfall Anticipation:** RL is rife with silent failures (dead options, gradient kill-switches, reward hacking). Documentation must proactively explain *why* an implementation does not fall into these traps.
3. **Visual Signposting:** In HRL, time-scales blur. Use structural, visual markers in comments to delineate temporal context (Macro vs. Micro).
4. **Contractual Interfaces:** In HRL, the Manager and Worker have a strict contract (e.g., goal bounds, effort guarantees). This contract must be documented as strictly as a REST API.
5. **Narrative Consistency:** Variable names in comments must exactly match the mathematical variable mapping. If the paper calls it $\beta$, the code must call it `beta` or `term_prob`, and the docstring must explicitly link them: `beta (term_prob)`.

---

## 2. Strict Annotation Schema (The Grammar of Readability)

To eliminate ambiguity and enable machine-parseable documentation, AI agents must use the following strict tag schema in docstrings and inline comments. 

### 2.1 Docstring Tag Schema (EBNF-ish)
All public functions/classes must use these exact tags in this order:

```python
"""
[One-line imperative summary].

[Extended narrative description - the "Why"].

@math:
    Eq (X) from [Paper], [Year].
    Mapping: code_var = math_var
@shapes:
    input: (B, T, D_in)
    output: (B, T, D_out)
@provenance:
    Where inputs are typically generated, and where outputs are consumed.
@gradient:
    [Flowing / Isolated / No Grad]. Explicitly state what parameters receive gradients.
@timescale:
    [Environment Step / Manager Macro-Step / Option Termination]
@stability:
    Numerical tricks used (e.g., log-sum-exp, epsilon clipping) and the mathematical reason.
@contract: (HRL Only)
    Guarantees made about this function's inputs/outputs across hierarchy levels.
"""
```

### 2.2 Inline Tensor Grammar
Tensors must be annotated with their semantic meaning, not just their shape.

**Syntax:** `# Tensor[Shape] <Semantic Meaning>`
```python
# BAD: advantages = advantages.unsqueeze(-1)  # Shape: (B, T, 1)
# GOOD: advantages = advantages.unsqueeze(-1)  # Tensor[B, T, 1] <Broadcastable GAE advantage for policy ratio scaling>
```

### 2.3 HRL Temporal Block Markers
In HRL, do not rely on sparse inline comments. Use block-level visual markers to shift the reader's temporal context.

```python
# =====================================================================
# >>> MANAGER TIME (Macro-step: t_k) <<<
# Executing high-level goal generation. Occurs every `k` worker steps.
# =====================================================================
goal = manager_policy(state) 

# =====================================================================
# >>> WORKER TIME (Micro-step: t_k + i) <<<
# Executing primitive actions conditioned on manager's goal.
# =====================================================================
action = worker_policy(state, goal)
```

---

## 3. Module-Level Documentation (The Context Layer)

The module docstring is the map. It must prevent the reader from ever asking, "Why does this file exist?"

**Mandatory Fields:**
- **Algorithm & Variant:** (e.g., *SAC with Automatic Entropy Tuning*, *Option-Critic with Double Q-Learning*)
- **Architectural Role:** Where this sits in the data pipeline (Data Collection → Buffer → Update).
- **Known Failure Modes:** What breaks if this code is altered naively?
- **Monitoring Contract:** What metrics *must* be logged from this module to verify health?

**Template:**
```python
"""
Module: algorithms/hiro_manager.py
Algorithm: HIRO (Data-Efficient Hierarchical Reinforcement Learning)
Architectural Role: High-Level Policy / Goal Proposer

Summary:
Generates latent sub-goals for the low-level worker every `k` environment steps.
Trained off-policy using transitional goals relabeled for hindsight consistency.

Known Failure Modes:
1. Goal Explosion: If manager outputs are unbounded, worker gradients explode.
   Mitigation: Output layer uses `tanh` bounded to [-1, 1], rescaled by `max_goal`.
2. Hindsight Inefficiency: If relabeled goals are not distinct, manager learns nothing.
   Mitigation: Relabeling samples goals uniformly from achieved states.

Monitoring Contract:
- `manager/goal_norm`: L2 norm of proposed goals. Crash to 0 = manager is ignored.
- `manager/intrinsic_reward`: Must remain negative (distance penalty).

Mathematical Reference:
HIRO, Nachum et al. (2018). Implements Equation (3) for goal relabeling 
and Equation (5) for manager Q-learning update.
"""
```

---

## 4. Class-Level Documentation (The State Layer)

Classes in RL are stateful. Unmanaged state (e.g., RNN hidden states, running normalization means) is the #1 cause of irreproducibility. Documentation must enforce state invariants.

**Mandatory Sections:**
- **Role & Lifecycle:** When is it created, and how long does it persist?
- **Mutable State & Invariants:** Explicitly list attributes that change and their mathematical invariant.
- **Initialization Contract:** What *must* be true about the constructor arguments?

**Template:**
```python
class RunningMeanStd(nn.Module):
    """
    Computes online running mean and standard deviation for observation normalization.

    Role & Lifecycle:
    Persists across the entire training run. Updated on every environment step.
    Used to normalize states before passing to the policy network.

    Mutable State & Invariants:
    - `mean` (Tensor[D]): Running mean. Invariant: converges to true data mean.
    - `var` (Tensor[D]): Running variance. Invariant: always >= 0.
    - `count` (float): Number of items seen. Invariant: monotonically increasing.

    Initialization Contract:
    - `epsilon` (float): Added to variance to prevent divide-by-zero. Must be > 1e-8.
    """
```

---

## 5. Function-Level Documentation (The Execution Layer)

This is where mathematical rigor meets code. The `@math`, `@provenance`, and `@gradient` tags are non-negotiable.

**Example:**
```python
def compute_ppo_clip_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float
) -> torch.Tensor:
    """
    Calculate the clipped surrogate objective for PPO.

    @math:
        Eq (7) from "Proximal Policy Optimization Algorithms" (Schulman et al., 2017).
        ratio = exp(log π(a|s) - log π_old(a|s))
        L_clip = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
        Mapping: `log_probs` = log π, `old_log_probs` = log π_old, `advantages` = A, `clip_range` = ε

    @shapes:
        input: log_probs (B, 1), old_log_probs (B, 1), advantages (B, 1)
        output: (1,) - scalar loss to be minimized

    @provenance:
        - `old_log_probs` come from the rollout buffer (detached from current graph).
        - Output is aggregated with value and entropy losses for the final backward pass.

    @gradient:
        Flowing. Gradients propagate exclusively through `log_probs` (the current policy).
        `old_log_probs` and `advantages` are treated as constants.

    @stability:
        Ratio is computed in log-space (`exp(logp - old_logp)`) rather than dividing 
        probabilities to avoid division by zero and floating point underflow.

    @contract:
        `advantages` must be zero-centered (advantage normalization) for stable clipping.
    """
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()  # Tensor[B, 1] <Policy ratio (π / π_old)>

    # Tensor[B, 1] <Clipped ratio, bounded to [1-ε, 1+ε] to prevent destructive large updates>
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    
    # Tensor[B, 1] <Take the pessimistic bound of the surrogate objective>
    surr1 = ratio * advantages
    surr2 = clipped_ratio * advantages
    loss = -torch.min(surr1, surr2).mean()
    
    return loss
```

---

## 6. RL/HRL Inline Micro-Patterns

### 6.1 The `detach()` Audit Trail
Every `.detach()`, `.stop_gradient()`, or `with torch.no_grad():` must follow the **Action-Reason-Effect** pattern.

```python
# BAD: next_state_value = self.critic(next_state).detach()
# GOOD: 
# Action: Detach target critic
# Reason: Prevent gradient flow from Q-loss into the target network
# Effect: Target network is updated exclusively via Polyak (soft) averaging
next_state_value = self.target_critic(next_state).detach() 
```

### 6.2 The `reshape` / `view` Justification
RL code often collapses time for batch processing. This destroys interpretability if not explained.

```python
# BAD: advantages = advantages.view(-1)
# GOOD: 
# Why: Flatten time and batch dimensions to match the 1D expectation of the MSE loss function.
# Note: We do not unsqueeze because value_loss expects unnormalized 1D inputs.
advantages = advantages.view(-1) # Tensor[B*T] <Flattened GAE advantages>
```

### 6.3 HRL Goal Semantics
Never document a goal as just "goal". Document its *interpretation space*.

```python
# BAD: goal = self.manager(state)
# GOOD: 
# Tensor[B, goal_dim] <Sub-goal in Cartesian coordinates [x,y] relative to agent>
# Contract: Bounded by tanh to [-5, 5] meters.
goal = self.manager(state)
```

---

## 7. Environment & Network Domain Rules

### 7.1 Environments: The "Affordance" Doc
Environments must document their *hierarchical affordances*—how they are meant to be decomposed.

```python
class KitchenEnv(gym.Env):
    """
    Franka kitchen manipulation environment.

    @observation:
        Shape: (60,). Semantics: 9 joint angles + 9 joint velocities + 21 object positions + 21 object velocities.
    @action:
        Shape: (9,). Semantics: Continuous joint velocity commands, range [-1, 1].
    @reward:
        Sparse. +1 for each of 4 sub-tasks completed (microwave, kettle, light, slide).
        
    @hrl_affordances:
        Natural temporal abstraction: Sub-tasks are sequentially independent.
        Worker space: 9D joint control.
        Manager space: Object-level manipulation goals (21D position targets).
    """
```

### 7.2 Networks: The "Initialization & Activation" Doc
Poor initialization kills RL training. Document it explicitly.

```python
class ActorCritic(nn.Module):
    """
    @architecture:
        Shared encoder -> separate policy/value heads.
        Encoder: MLP(layers=[256, 256], activation=ReLU)
        Policy Head: Linear(256, act_dim) -> Tanh (for bounded actions)
        Value Head: Linear(256, 1) -> Identity

    @initialization:
        Orthogonal initialization for all weights.
        - Encoder gain: sqrt(2) (standard for ReLU)
        - Policy Head gain: 0.01 (crucial: starts policy near uniform random)
        - Value Head gain: 1.0
        Biases initialized to 0.
    """
```

---

## 8. Anti-Patterns (The Readability Killers)

1. **The "Orphaned Loss"**: Returning `loss = -advantage * log_prob` without specifying if it is minimized or maximized. Always prefix returns: `Returns: loss (to be minimized)`.
2. **The "Phantom Update"**: Updating a target network or EMA inside a `forward()` pass without an explicit `# Side-effect:` comment.
3. **The "Ungrounded Epsilon"**: `eps = 1e-8`. Instead: `eps = 1e-8 # Stability: prevents divide-by-zero in variance normalization`.
4. **The "Time Traveler"**: Using `t` and `t+1` in comments but indexing with `[i]` and `[i-1]`. Always align math and index: `# s_{t+1} is states[i], s_t is states[i-1]`.

---

## 9. AI Agent Execution Protocol (The Pipeline)

To ensure rigor and adherence to this document, the AI agent must process files in the following sequential pipeline:

**Phase 1: Contextualization**
1. Scan filename, imports, and class inheritance.
2. Identify the exact algorithm and paper.
3. Identify the hierarchy level (Manager/Worker/Orchestrator/Primitive).

**Phase 2: Provenance Tracing**
1. Trace all inputs and outputs of public functions.
2. Identify all `.detach()` and `no_grad` blocks. Determine *why* they exist.
3. Trace tensor shapes through `view`, `reshape`, and `permute`. Note the semantic reason for the reshape.

**Phase 3: Drafting (Applying the Schema)**
1. Generate Module, Class, and Function docstrings strictly adhering to the Tag Schema (`@math`, `@shapes`, `@gradient`, etc.).
2. Apply HRL Temporal Block Markers where control flow shifts between time-scales.
3. Apply the `Tensor[Shape] <Semantics>` inline grammar to all intermediate variables.

**Phase 4: Readability Pruning (The Signal-to-Noise Filter)**
1. Review generated documentation. Remove redundant phrasing.
2. Ensure comments explain *Why* and *When*, not just *What*.
3. Verify that mathematical notation in `@math` exactly matches variable names in the code.
4. Ensure all "Known Failure Modes" and "@stability" tags are accurate for the algorithm.

**Phase 5: Validation Check**
Run the self-check against the V3 checklist (Section 10).

---

## 10. Strict Validation Checklist

Before outputting the documented file, the AI agent must assert `TRUE` for every item:

- [ ] **Module:** Does it state the exact algorithm, paper, architectural role, and known failure modes?
- [ ] **Class:** Are mutable state invariants and initialization contracts explicitly documented?
- [ ] **Function Docstrings:** Do all public functions include `@math`, `@shapes`, `@provenance`, `@gradient`, and `@stability` tags?
- [ ] **Math Mapping:** Is there an exact `code_var = math_var` mapping in the `@math` tag?
- [ ] **Inline Tensors:** Does every tensor creation/mutation use the `Tensor[Shape] <Semantics>` grammar?
- [ ] **Reshape/Detach:** Does *every* `view`, `reshape`, `detach`, or `no_grad` have an inline comment explaining the *Action-Reason-Effect*?
- [ ] **HRL Time-Scales:** Are `>>> MANAGER TIME <<<` and `>>> WORKER TIME <<<` visual block markers used to separate temporal contexts?
- [ ] **HRL Contracts:** Are the bounds, semantics, and guarantees of sub-goals/intrinsic rewards explicitly defined?
- [ ] **Loss Directionality:** Are all returned losses explicitly marked as "(to be minimized)" or "(to be maximized)"?
- [ ] **Zero Magic Numbers:** Are all hardcoded floats/constants replaced with named constants and annotated with default, range, and sensitivity?