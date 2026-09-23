# SWARMMIND_TECHNICAL_SPEC.md

**SWARMMIND — Multi-Agent Robot Swarm Learning and Coordination Simulator**

| Field | Value |
|---|---|
| Document | Authoritative technical specification (Phase 0 output) |
| Version | 1.0 |
| Date | 2026-09-23 |
| Status | APPROVED FOR PHASE 1 once the user replies `CONTINUE` |
| Author role | Principal architect (Opus 5.5, Phase 0) |
| Audience | Claude Code implementation sessions (Sonnet / Haiku / Opus), the student author, and the project's assessors |

### How to read this document

- **MUST** / **MUST NOT**: binding. Deviating needs a written justification in the phase completion report, and for ML methodology a recommendation to escalate to Opus.
- **SHOULD**: the default. Deviate only for a concrete, stated reason.
- **MAY**: optional.
- Numbers marked *(provisional)* are starting values. Phase 3 calibrates them and Phase 6 freezes them in `docs/EXPERIMENTS.md`. **After Phase 7 starts, frozen values MUST NOT change**, because changing them would invalidate the comparison.
- Section numbers match the 44 items requested in the Phase 0 brief. Section 38 holds the per-phase implementation plan. Sections 39–44 summarise and operationalise it.

---

## Table of contents

1. Executive project definition
2. Final project scope
3. Research question
4. Testable hypotheses
5. System architecture
6. Environment specification
7. PettingZoo/Gymnasium architecture
8. Observation space
9. Global state
10. Action space
11. Partial observability
12. CTDE methodology
13. Reward design
14. Random baseline (and reference heuristic)
15. IPPO methodology
16. MAPPO methodology
17. Parameter sharing
18. Curriculum
19. Communication strategy
20. Procedural maps
21. Training methodology
22. Evaluation methodology
23. Metrics
24. Coordination metrics
25. Generalization experiments
26. Behaviour analysis
27. Replay architecture
28. Backend architecture
29. Frontend architecture
30. Professional UI/UX system
31. Repository structure
32. Configuration strategy
33. Testing strategy
34. Reproducibility
35. Computational requirements
36. Failure modes and debugging
37. Security and reliability
38. Detailed implementation phases
39. Model recommendation for every phase
40. Effort recommendation for every phase
41. Skills/connectors for every phase
42. Opus escalation criteria
43. Validation gate for every phase
44. Claude Code execution protocol

Appendix A: Default configuration files
Appendix B: References
Appendix C: Glossary

---

## 1. Executive project definition

SWARMMIND is a **research-grade 2D grid simulator plus a learning pipeline**. In it, a homogeneous team of 2–6 robots learns, through Multi-Agent Reinforcement Learning (MARL), to **search an initially unknown map, find targets and rescue them**, while avoiding obstacles and each other. The environment gives each robot **only local, partial observations**.

The project has three deliverables of equal importance:

1. **A correct MARL study.** Random vs. IPPO vs. MAPPO under identical conditions, trained on procedurally generated maps, evaluated on held-out maps with multiple seeds, and reported with confidence intervals. The architecture isolates one independent variable: **the information available to the critic during training**.
2. **Measurable coordination analysis.** Duplicate exploration, workload distribution, redundant target pursuit and the other metrics in §24 decide whether cooperation occurred. How the robots look while they move does not count as evidence.
3. **A professional research product.** A FastAPI + React application where a user can watch live or replayed swarm behaviour (Swarm Lab), launch training runs, compare algorithms using **real** results only, and inspect behaviour as it evolves across training checkpoints.

**Guiding principle:** a small system that is correct beats a large one that is plausible. If a feature threatens MARL correctness, reproducibility or the schedule, cut it.

**What SWARMMIND is NOT:** a scripted or pathfinding demo, a dashboard of fabricated numbers, an LLM wrapper, or a physics simulator. All behaviour shown as "learned" MUST come from a trained policy checkpoint that the evaluation pipeline produced.

---

## 2. Final project scope

### 2.1 Core scope (must be completed)

| Area | Included |
|---|---|
| Environment | Discrete grid, static obstacles, static targets, 2–6 robots, simultaneous moves, deterministic conflict rules, disc-shaped sensing, per-robot belief maps, communication modes A (none) and B (shared map), procedural maps with validity guarantees, fixed held-out evaluation maps |
| Algorithms | Random policy (baseline), IPPO (parameter-shared actor, local critic), MAPPO (parameter-shared actor, centralized agent-specific critic) |
| Reference | Greedy nearest-frontier heuristic, used **only** for T_max calibration and as a contextual reference line. It is never presented as learned behaviour |
| Training | Config-driven, seeded, checkpoint + resume, CPU and CUDA, periodic validation |
| Evaluation | Fixed test maps, 5 training seeds per algorithm, mean ± std ± 95% CI, paired design |
| Analysis | Mission metrics, coordination metrics, heatmaps, coverage curves, discovery timelines, checkpoint-evolution replays |
| Backend | FastAPI REST + WebSocket; sessions, models, experiments, replays, training jobs |
| Frontend | React + Vite + TypeScript; Swarm Lab, Train Swarm, Algorithm Comparison, Swarm Analysis, Experiment History, Settings |
| Tests | Environment, MARL, evaluation, information-leak, backend, frontend, integration |

### 2.2 Stretch scope (only after Phase 7 passes, in this priority order)

1. Communication mode C (limited-range map merging) plus experiment H5 (CTDE benefit under reduced information).
2. Generalization to different team sizes (G3) and map sizes (G4).
3. Curriculum training for L3/L4, if direct training fails there.
4. Occlusion-aware sensing (line of sight).
5. QMIX, only if a value-decomposition comparison answers a question the report needs (§16.9).

### 2.3 Explicitly out of scope

Continuous physics, kinematics, SLAM and localization noise, 3D, real hardware, dynamic obstacles, energy models, heterogeneous robots, learned communication (only a design note, §19.4), LLM agents, cloud deployment, multi-user accounts.

---

## 3. Research question

### 3.1 Improved research question

> **Under partial observability and a shared team objective, does giving the critic centralized training-time information (MAPPO) produce more efficient cooperative search-and-rescue on unseen maps than independent learners (IPPO) that have an identical decentralized actor? And do both learn behaviours that measurably outperform a non-learning random policy on mission success and exploration efficiency?**

### 3.2 Why this formulation is better than the starting proposal

- **It isolates one variable.** IPPO and MAPPO in SWARMMIND share the actor network, observation, action space, reward, hyperparameters, sample budget, seeds and maps. **Only the critic's input differs** (local observation vs. local observation + agent-specific global state). Any difference between them can therefore be attributed to centralized critic information. This is enforced by a bitwise-parity test (§33, Phase 5).
- **"More efficiently" becomes measurable.** It is tied to pre-registered metrics: success rate, steps-to-completion and coverage per agent-step.
- **"Unseen maps"** makes generalization part of the claim and prevents train/test contamination.
- **It does not assume the answer.** The literature (Yu et al. 2022; de Witt et al. 2020) shows IPPO is often competitive with MAPPO. A null result is a valid, reportable finding.

---

## 4. Testable hypotheses

All hypotheses are evaluated on the **held-out test map set of the primary level L2** (10×10, 3 robots, 2 targets, communication mode B) unless stated otherwise. The **unit of analysis is the training seed**: each seed contributes one number, the mean over its evaluation episodes. Episodes are not independent samples of an algorithm, and treating them as such would be pseudo-replication.

| ID | Hypothesis (null in brackets) | Primary metric | Decision rule |
|---|---|---|---|
| **H1** | IPPO and MAPPO each achieve higher mission success than Random. [No difference.] | Success rate | Supported if the 95% CI of (learned − random), computed over seeds, excludes 0 for that algorithm |
| **H2** | MAPPO's performance differs from IPPO's. Two-sided; no direction is assumed. [No difference.] | Success rate; secondary: steps-to-completion, validation success AUC (sample efficiency) | Welch 95% CI of the difference in per-seed means excludes 0. Report the effect size (Hedges' g) either way |
| **H3** | Learned policies reduce duplicate exploration relative to Random, and MAPPO differs from IPPO on it. [No difference.] | Duplicate Exploration Ratio at matched coverage, DER@50 (§24) | Same CI rule. H3 counts as "coordination evidence" only when it holds together with H1 |
| **H4** | Policies trained on procedural maps keep most of their in-distribution performance on unseen, differently structured maps. [Large drop.] | Relative success retention = success(G2) / success(G1) | Descriptive: report retention with CI. No threshold is pre-assumed; any claim is worded from the measured value |
| **H5** *(stretch)* | The MAPPO − IPPO gap is larger without communication (mode A) than with a shared map (mode B). [Same gap.] | Difference-in-differences of success | CI of the interaction term excludes 0 |

### 4.1 Pre-registration (mandatory, Phase 6)

Before any Phase 7 run, `docs/EXPERIMENTS.md` MUST record: the primary metric, the levels, seeds, budgets, evaluation map sets, the checkpoint rule (the final checkpoint is primary), the statistical procedure, and the frozen hyperparameters. Any later change is reported as a deviation. This prevents tuning on test results.

### 4.2 Claims policy

- Never write "MAPPO is superior" unless H2 is supported in that direction.
- Never write "emergent cooperation" unless the coordination criteria in §24.4 are met.
- Report negative and null results with the same prominence as positive ones.

---

## 5. System architecture

### 5.1 Layered architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│ FRONTEND (React + Vite + TS)                                            │
│  pages/  features/  components/  canvas renderer  charts  design tokens │
│  api/ (typed client generated from OpenAPI)   stores/ (zustand)         │
└───────────────▲──────────────────────────────────▲──────────────────────┘
                │ REST (JSON)                      │ WebSocket (frames, metrics)
┌───────────────┴──────────────────────────────────┴──────────────────────┐
│ BACKEND (FastAPI) — thin adapters only                                  │
│  routers/ → services/ (SessionManager, ModelRegistry, RunIndex,         │
│              ReplayStore, JobRunner)  → schemas/ (pydantic)             │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ Python imports (never re-implements ML/env logic)
┌───────────────▼──────────────────────────────────────────────────────────┐
│ marl/ (single source of truth)                                          │
│  envs/core ← envs/pettingzoo wrapper, envs/vector (batched)             │
│  networks → algorithms/{common, ippo, mappo, baselines}                 │
│  buffers → training → evaluation → analysis → replay                    │
│  utils (seeding, config, paths, versions)                               │
└───────────────┬──────────────────────────────────────────────────────────┘
                │ files
┌───────────────▼──────────────────────────────────────────────────────────┐
│ STORAGE (filesystem, no DB)                                              │
│ configs/  results/runs/…  results/studies/…  results/replays/…  models/ │
└──────────────────────────────────────────────────────────────────────────┘
        ▲
        │ CLI scripts (train.py, evaluate.py, run_study.py …) — also used by
        │ backend JobRunner as subprocesses
```

### 5.2 Dependency rules (enforced by `tests/test_import_boundaries.py`)

| Module | May import | MUST NOT import |
|---|---|---|
| `marl.envs.core` | numpy, `marl.utils` | torch, `marl.algorithms`, `marl.networks`, backend |
| `marl.envs.pettingzoo_env`, `marl.envs.vector` | `marl.envs.core`, pettingzoo, gymnasium | `marl.algorithms` |
| `marl.networks.actor` | torch, `marl.networks.encoders` | `marl.envs.core.global_state`, `marl.networks.critics` |
| `marl.algorithms.*` | networks, buffers, envs (types only), utils | backend |
| `marl.evaluation`, `marl.analysis`, `marl.replay` | envs, algorithms (policy loading), utils | backend, training internals |
| `backend.*` | `marl.*` public API | torch internals directly (it goes through `marl.algorithms.policy`) |

These rules are enforced with a pytest AST scan rather than an extra dependency. Circular imports are forbidden. No source file may exceed about 400 lines without justification in the phase report.

### 5.3 Single code path principle

The **same** `SwarmWorld` (core), the **same** `ActorPolicy.act()`, and the **same** metric functions run in training, evaluation, backend live sessions and replay recording. There is exactly one implementation of each of these: observation building, reward, conflict resolution, action masking, and policy inference. Parity tests (§33.5) enforce this.

---

## 6. Environment specification

### 6.1 Formal model

The task is a **Decentralized POMDP** ⟨𝓘, 𝓢, {𝓐ᵢ}, P, {Rᵢ}, {Ωᵢ}, O, γ⟩ with N homogeneous agents (robots), a discrete state (map, robot positions, target status, belief maps, time), simultaneous discrete actions, deterministic transitions (given the map seed) and per-agent rewards. The positive task rewards are fully shared; a small per-agent collision cost is added (§13).

### 6.2 World entities

| Entity | Representation | Semantics |
|---|---|---|
| Grid | `H × W`, int8 array `terrain[y, x] ∈ {FREE=0, OBSTACLE=1}` | Origin top-left. `x` is the column, `y` is the row. Cells outside the grid behave as obstacles |
| Robot `i` | `pos[i] = (y, x)`, int | Occupies exactly one FREE cell. Robots never overlap |
| Target `k` | `target_pos[k]`, `status ∈ {HIDDEN, DISCOVERED, RESCUED}` | Non-blocking. Static. Rescued targets disappear |
| Team knowledge | `known_team: bool[H,W]` | Cells observed by any robot so far (used for rewards, metrics and mode B) |
| Belief `i` | `belief[i]: {known: bool[H,W], terrain_known: int8[H,W], targets_known: set, rescued_known: set, visited_self: bool[H,W], visited_team_known: bool[H,W]}` | What robot `i` is entitled to know (§11) |
| Time | `t ∈ {0,…,T_max}` | Step counter |

### 6.3 Task

**Mission:** rescue all `M` targets as quickly as possible. A target is **discovered** the first time any robot senses its cell. It is **rescued** when a robot occupies its cell at the end of a step's movement phase. **Exploration is instrumental.** It is rewarded as shaping with a fixed budget (§13.3), and coverage is reported as a secondary metric.

### 6.4 Step order (deterministic, one call to `step(actions)`)

1. **Validate** actions. Each `aᵢ ∈ {0..4}`, otherwise raise `ValueError`, because an out-of-range action is a programming error.
2. **Propose** destinations `dᵢ = posᵢ + δ(aᵢ)`.
3. **Terrain check.** If `dᵢ` is out of bounds or an OBSTACLE, set `dᵢ ← posᵢ` and flag `bumped_wall[i]`. Under action masking this cannot happen for a policy that respects the mask; the environment asserts it in debug mode.
4. **Robot conflict resolution** using the **no-follow rule** (§6.5). This single pass is order-independent.
5. **Move** robots. Update `visited_self`, the visit counts and the distance counters.
6. **Rescue.** For each robot on an unrescued target cell, mark the target RESCUED and credit `rescued_by = i`.
7. **Sense.** For every robot, compute its sensed cell set (§11.2). Update `known_team`, each `belief[i]` according to the communication mode (§19), and target discovery (HIDDEN → DISCOVERED, `discovered_by`, `t_discovered`).
8. **Rewards** (§13).
9. **Termination and truncation:** `success = all targets RESCUED`. `terminated = success`. `truncated = (t+1 == T_max) and not success`. Success takes precedence at `t+1 == T_max`.
10. **Observations** (§8), action masks (§10.2), infos. Increment `t`.

**Invariant:** a target cannot be rescued before it is discovered, because a robot always senses its 4-neighbourhood (sensor radius ≥ 1) before moving into it. The environment asserts `t_discovered ≤ t_rescued`.

### 6.5 Multi-agent movement semantics — the no-follow rule

Let `occ0` be the set of robot positions **at the start of the step**. After the terrain check:

- **Rule R1 (occupied cell):** if `dᵢ ≠ posᵢ` and `dᵢ ∈ occ0`, robot `i` is **blocked** (it stays put), even if the occupant is leaving this step. This rule covers swaps, following and rotation cycles.
- **Rule R2 (contested empty cell):** if two or more moving robots have the same `dᵢ ∉ occ0`, **all** of them are blocked. There is no priority or tie-breaking by index, so the rule is symmetric and fair.
- Blocked robots stay in cells that were theirs in `occ0`, so no new conflicts can arise and **one pass suffices**.

| Situation | Outcome | `collision[i]` counted for |
|---|---|---|
| Move into free, uncontested cell | moves | — |
| Move into wall/obstacle/out-of-bounds | stays | `bumped_wall[i]` (invalid move) |
| Move into cell occupied by a waiting robot | mover stays | mover only |
| Two robots → same empty cell | both stay | both |
| Swap A↔B | both stay | both |
| Follow (A moves into B's cell while B moves away) | A stays, B moves | A |
| 3- or 4-cycle rotation | all stay | all movers |
| WAIT | stays | — (a waiting robot is never penalized) |

*Rationale:* this is the most conservative rule that can be expressed in a single pass. It is deterministic, independent of agent index and trivial to unit-test. Its cost (following in a corridor takes an extra step) applies equally to every algorithm.

### 6.6 Episode configuration

| Parameter | Key | Type / range | Default (L2) |
|---|---|---|---|
| Map width, height | `width`, `height` | int 6–20 | 10, 10 |
| Robots | `n_robots` | int 1–6 | 3 |
| Targets | `n_targets` | int 1–5 | 2 |
| Obstacle density | `obstacle_density` | float 0–0.35 | 0.15 |
| Max episode length | `max_steps` | int | 150 *(provisional)* |
| Sensor radius (Euclidean) | `sensor_radius` | float ≥ 1 | 2.0 |
| Observation window radius | `obs_radius` | int | 5 (window 11×11) |
| Communication mode | `comm_mode` | `none` \| `shared_map` \| `limited_range` | `shared_map` |
| Communication radius | `comm_radius` | float | 4.0 (mode C only) |
| Spawn mode | `spawn` | `clustered` \| `scattered` | `clustered` |
| Min target distance from spawn (BFS) | `target_min_dist` | int | 3 |
| Action masking | `action_masking` | bool | true |
| Map source | `map_source` | `procedural` \| `fixed:<set>/<id>` | `procedural` |
| Seed | passed to `reset(seed=…)` | int | — |

### 6.7 Difficulty levels (named configs)

| Level | Grid | Robots | Targets | Density | `max_steps` *(prov.)* | Comm | Role |
|---|---|---|---|---|---|---|---|
| **L1** | 8×8 | 2 | 1 | 0.10 | 100 | B | Debugging, learning gates |
| **L2** | 10×10 | 3 | 2 | 0.15 | 150 | B | **Primary study level** |
| **L3** | 12×12 | 4 | 3 | 0.20 | 200 | B | Secondary scale study |
| **L4** | 12×12 | 4 (test 4–6) | 3 | 0.20 | 200 | A or C | Stretch: communication constraints (H5) |

---

## 7. PettingZoo/Gymnasium architecture

### 7.1 Decision

The design uses **three layers over one core**:

```
marl/envs/core/  (pure NumPy, no framework)
   SwarmWorld.reset(seed | MapSpec) → WorldState
   SwarmWorld.step(actions: int[N]) → StepResult (arrays)
   SwarmWorld.actor_obs(i) / all_actor_obs() → (grid[N,C,K,K], vec[N,D], mask[N,5])
   SwarmWorld.global_state() → (grid[Cg,H,W], vec[4]);  agent_state_channels(i)
        │
        ├── marl/envs/pettingzoo_env.py : SwarmParallelEnv(pettingzoo.ParallelEnv)
        │      dict API, used by tests, backend sessions and external tooling
        │
        └── marl/envs/vector.py : SyncVectorSwarm(E copies of SwarmWorld)
               batched arrays [E, N, …], auto-reset with final-obs capture,
               used by training and evaluation for speed
```

*Justification:* PettingZoo's Parallel API is the standard for simultaneous-move MARL and provides `parallel_api_test` for conformance. Converting dicts on every step is slow for training, however, so the trainer uses the batched wrapper. Both wrappers are thin; the **parity test** (same seed + same actions → identical observations, rewards and flags through both paths) guarantees there is no divergence between training and inference.

### 7.2 PettingZoo contract

- `possible_agents = ["robot_0", …, "robot_{N-1}"]`
- `observation_space(agent) = gymnasium.spaces.Dict({"grid": Box(0,1,(8,K,K),float32), "vec": Box(0,1,(10,),float32), "action_mask": MultiBinary(5)})`
- `action_space(agent) = Discrete(5)`
- `state_space = Box(0,1,(Cg,H,W),float32)` and `state()` returns the shared global grid. The agent-specific critic input is assembled in `marl.algorithms` from `state()` + `agent_state_channels(i)` + the agent's observation.
- `reset(seed=None, options={"map": MapSpec?, "level": str?}) → (obs, infos)`
- `step(actions) → (obs, rewards, terminations, truncations, infos)`. On episode end, all agents terminate or truncate together and `agents` becomes `[]`.
- `render()` returns an ASCII string (`render_mode="ansi"`). Visual rendering is the frontend's job.
- Per-agent `infos[agent]`: `{"moved", "blocked_by_robot", "bumped_wall", "rescued", "discovered", "explore_credit"}`. Team info goes in every agent's info under `"team"`: `{"t", "coverage", "n_discovered", "n_rescued", "success"}`.

### 7.3 Vectorized wrapper contract

- `reset(seeds: list[int]) → Batch`
- `step(actions: int[E,N]) → Batch(obs_grid[E,N,C,K,K], obs_vec[E,N,D], mask[E,N,5], state_grid[E,Cg,H,W], state_vec[E,4], reward[E,N], terminated[E], truncated[E], final: FinalInfo | None)`
- **Auto-reset:** when env `e` finishes, the wrapper records `final.obs/state` for `e` (needed to bootstrap truncated episodes, §15.6), records the episode metrics, then resets `e` with the next training-map seed from that env's seed stream. The returned `obs` for `e` is the *new* episode's first observation.
- Global-state grids MUST have the same `H×W` within one vector env. Mixed map sizes are only allowed at evaluation time, one map size per call.

---

## 8. Observation space (actor — LOCAL OBSERVATION)

### 8.1 Spatial tensor `grid ∈ [0,1]^{8×K×K}`, K = 2·obs_radius + 1 = 11

The window is **egocentric**: robot `i` is always at the centre cell `(obs_radius, obs_radius)`. Every channel is computed **only from `belief[i]` and robot i's current sensing**.

| Ch | Name | Value | Source |
|---|---|---|---|
| 0 | `obstacle` | 1 if the cell is known to be an obstacle **or lies out of bounds** | belief[i] / padding |
| 1 | `known_free` | 1 if known FREE | belief[i] |
| 2 | `unknown` | 1 if in bounds and never known | belief[i] |
| 3 | `sensed_now` | 1 if inside robot i's sensor disc this step | current sensing |
| 4 | `visited_self` | 1 if robot i has ever occupied the cell | own odometry |
| 5 | `visited_known` | 1 if the cell is known to have been visited by any robot (mode A: equals ch 4; mode B: team union; mode C: merged beliefs) | belief[i] |
| 6 | `teammate` | 1 where a teammate is **currently sensed** | current sensing only (never stale) |
| 7 | `target` | 1 where a **known, not-known-rescued** target lies | belief[i] (targets are static, so remembered positions remain valid) |

Channels 0–2 form a one-hot partition for in-bounds cells; out-of-bounds cells are `obstacle=1`, `known_free=0`, `unknown=0`.

### 8.2 Vector `vec ∈ [0,1]^{10}`

| Idx | Feature | Legitimacy |
|---|---|---|
| 0–1 | `x/(W−1)`, `y/(H−1)` | Own localization (assumed, e.g. odometry) |
| 2 | `1 − t/T_max` | Onboard clock + known mission deadline |
| 3–7 | one-hot of own previous action (all zeros at t=0) | Own memory |
| 8 | `blocked_last_step` (robot or wall) | Own bump sensor |
| 9 | `known_rescued / M` | Mission briefing gives M; rescues are known per belief |

**Agent identity is NOT included** by default. `obs.agent_id_onehot: true` exists only as an ablation (§17).

### 8.3 Normalization and dtype

All features are already in `[0,1]` by construction. **No running observation normalization** is used, because running statistics are a classic source of train/inference divergence. Tensors are `float32` in the network. The buffer MAY store the grid as `uint8` to save memory and cast at batch time.

### 8.4 Visible vs. invisible semantics

- *Visible now*: cells in `sensed_now`. Their terrain, teammates and targets are ground truth at this instant.
- *Remembered*: cells known from earlier sensing (or shared, mode B/C). Terrain and targets are reliable because they are static. Teammate positions are **never** remembered (ch 6 is current sensing only).
- *Unknown*: never observed. The observation carries no information about the true contents.
- **Hidden targets are never encoded anywhere in the actor's observation.** This is tested by counterfactual invariance (§12.4).

### 8.5 Shapes (reference)

`grid: float32[8, 11, 11]` (968 values), `vec: float32[10]`, `action_mask: int8[5]`.

---

## 9. Global state (critic input — TRAINING ONLY)

### 9.1 Global spatial tensor `S ∈ [0,1]^{Cg×H×W}`, Cg = 9 (allocentric, full map)

| Ch | Name | Privileged? |
|---|---|---|
| 0 | ground-truth obstacle | yes, including unknown regions |
| 1 | reachable free cell | yes |
| 2 | team-known cell | no (team-level) |
| 3 | **all** unrescued targets (including HIDDEN) | **yes** |
| 4 | discovered, unrescued targets | no |
| 5 | robot occupancy (all robots) | yes (exact positions of all teammates) |
| 6 | team visit count, `min(count/10, 1)` | partly |
| 7 | **agent-specific:** this agent's position | no |
| 8 | **agent-specific:** this agent's belief-known mask | no (equals ch 2 in mode B) |

Channels 0–6 are shared by all agents (`env.state()`). Channels 7–8 are agent-specific (`agent_state_channels(i)`).

### 9.2 Global vector `g ∈ [0,1]^4`

`[1 − t/T_max, n_rescued/M, coverage, n_discovered/M]`

### 9.3 Agent-specific critic input (MAPPO)

`x_i = (S_shared ⊕ S_agent_i, g, o_i)`, where `o_i` is agent i's **full actor observation** (grid + vec). This follows the "agent-specific global state" recommendation of Yu et al. (2022). Including `o_i` guarantees that the critic conditions on a **superset** of the actor's information, so the value function can represent the actor's information state.

### 9.4 Size handling

The global encoder ends in `AdaptiveAvgPool2d(4,4)`, so the critic accepts any `H×W ≥ 6` without padding. The critic is never used at test time, so its size handling does not affect generalization claims.

---

## 10. Action space

### 10.1 Discrete(5)

| a | Name | δ = (dy, dx) |
|---|---|---|
| 0 | WAIT | (0, 0) |
| 1 | UP | (−1, 0) |
| 2 | DOWN | (+1, 0) |
| 3 | LEFT | (0, −1) |
| 4 | RIGHT | (0, +1) |

### 10.2 Action masking

- `mask[i, a] = 0` iff moving in direction `a` leads out of bounds or into an **obstacle robot i currently senses** (the 4-neighbourhood is always sensed). WAIT is always valid.
- The mask uses **only local sensing**, so there is no leakage (tested in §12.4).
- Cells occupied by other robots are **never masked**. Conflicts depend on the teammates' simultaneous choices, and learning to avoid them is part of the coordination problem.
- The masked categorical distribution sets `logits[mask==0] = −1e9` (a finite value, avoiding NaN from `-inf·0` in entropy). The mask **MUST be stored in the rollout buffer and re-applied identically during PPO updates.**
- `action_masking: false` exists as an ablation and for the invalid-action metric. In that mode a bump means the robot stays, `bumped_wall=1`, and it receives the collision penalty.

### 10.3 Invalid actions

| Case | Behaviour |
|---|---|
| Action ∉ {0..4} or wrong dtype/shape | `ValueError` (programming error) |
| Masked action submitted anyway (e.g. an external caller ignoring the mask) | Treated as a bump. It is counted in `invalid_actions`. The penalty applies |
| Missing agent key in the PettingZoo dict | `KeyError` |

---

## 11. Partial observability

### 11.1 Sources of partial observability

1. Limited sensor radius (`r_s = 2`, a Euclidean disc of 13 cells).
2. Limited observation window (11×11) over a belief map that may be incomplete.
3. Teammates are visible only within sensor range.
4. Mode-dependent knowledge sharing (A < C < B).

### 11.2 Sensing model

`sensed(i) = {(y,x) in bounds : (y−yᵢ)² + (x−xᵢ)² ≤ r_s²}`. **v1 has no occlusion**, a documented simplification. Line-of-sight sensing via Bresenham is a stretch option (`sensing.occlusion: true`). Sensing is exact (noise-free).

### 11.3 Memory without recurrence

A feedforward actor on a purely momentary view would oscillate, because it cannot remember where it has been. SWARMMIND handles memory **explicitly and legitimately** with an onboard belief map that the observation window reads (channels 0–2, 4, 5, 7), plus the previous-action and blocked features. This avoids recurrent policies. RNN/GRU actors are **out of scope** for the core project: they would add complexity to the buffer (sequence chunks, hidden-state storage) with little expected benefit given explicit belief maps.

### 11.4 What robots cannot know

The positions of hidden targets; the terrain of unobserved cells; the positions of unsensed teammates; in mode A, teammates' discoveries and rescues (unless sensed); the global coverage fraction.

---

## 12. CTDE methodology

### 12.1 Is CTDE appropriate?

Yes. Execution MUST be decentralized, because each robot acts on its own sensors. Training happens in simulation, where the full state is available. Using that state in the **critic only** can reduce the variance of the advantage estimates and improve credit assignment under a shared reward, without changing what the deployed actor needs. This is exactly MAPPO's setting.

### 12.2 Information boundary table

| Information | Actor (train) | Actor (execution) | IPPO critic | MAPPO critic | Reward fn | Metrics |
|---|---|---|---|---|---|---|
| Own local observation `o_i` | ✔ | ✔ | ✔ | ✔ | — | ✔ |
| Own action mask (local) | ✔ | ✔ | — | — | — | — |
| Ground-truth terrain (unknown regions) | ✘ | ✘ | ✘ | ✔ | ✔ | ✔ |
| Hidden target positions | ✘ | ✘ | ✘ | ✔ | ✔ | ✔ |
| All robot positions | ✘ (sensed only) | ✘ | ✘ | ✔ | ✔ | ✔ |
| Team-known map | mode B ✔ / A ✘ / C partial | same | via `o_i` | ✔ | ✔ | ✔ |
| Other agents' actions | ✘ | ✘ | ✘ | ✘ (v1) | ✔ | ✔ |
| Rewards, returns, advantages | ✘ (never an input) | ✘ | targets | targets | — | ✔ |
| Agent ID | ✘ (default) | ✘ | ✘ | ✔ (position channel) | — | ✔ |

**Rewards may be computed from global information** because rewards are a training signal and are never an actor input.

### 12.3 What exists at execution time

Only `ActorPolicy` (encoder + policy head) together with `SwarmWorld.actor_obs(i)` and the mask. Exported models (`models/<name>/actor.pt`) contain **only actor weights**. The backend and evaluator load actor-only files.

### 12.4 Leak-prevention tests (MUST exist and pass from Phase 2/5 onward)

| Test | Mechanism |
|---|---|
| `test_obs_counterfactual_hidden_target` | Build two world states that differ only in a HIDDEN target's position (outside all robots' knowledge). All actor observations and masks MUST be bit-identical |
| `test_obs_counterfactual_unknown_terrain` | The same, flipping an obstacle in a never-observed region |
| `test_obs_counterfactual_unsensed_teammate` | Moving an unsensed teammate does not change `o_i` (mode A/B, ch 6) |
| `test_mode_a_isolation` | In mode A, robot j's discoveries outside i's sensing do not appear in `o_i` |
| `test_mask_local_only` | The mask is invariant to non-sensed terrain changes |
| `test_actor_import_boundary` | AST scan: `networks/actor.py` and `algorithms/common/policy.py` do not import `global_state` or critic modules |
| `test_actor_forward_signature` | `ActorPolicy.forward(grid, vec, mask)` has no other tensor inputs |
| `test_actor_only_inference` | Loading `actor.pt` without any critic file runs full episodes |
| `test_critic_sees_global` (sanity, reverse direction) | The MAPPO critic's output changes when a hidden target moves, proving the global state actually reaches it |
| Statistical leak probe (Phase 5 & 15) | **Pre-discovery target-direction bias:** before a target is discovered, the fraction of moves that reduce BFS distance to it MUST be statistically indistinguishable from the same fraction for a decoy "phantom" target placed uniformly at random. A significant bias indicates leakage |

---

## 13. Reward design

### 13.1 Choice: shared team reward plus individual collision cost

- **Positive mission rewards are fully shared** (every robot receives the same value). The task is cooperative, and shared reward removes the incentive to compete for cells and targets. This is also the setting in which the IPPO vs. MAPPO credit-assignment question is meaningful.
- **Collision cost is individual**, because a collision is a local safety event attributable to the robot that attempted the move (the same convention as MPE `simple_spread`).
- *Ablation (stretch, only if free-riding appears):* `reward.team_share β = 0.5` mixes shared and individually credited positive rewards.

### 13.2 Reward function (per agent i, per step t)

```
r_i,t =  w_x · ΔK_t                                   (team exploration shaping)
       + w_d · D_t                                    (targets newly discovered by team)
       + w_r · Q_t                                    (targets rescued this step, by anyone)
       + 1[success_t] · ( w_c + w_x · (|G| − |K_{t+1}|) )   (completion bonus + exploration payout)
       − w_s                                          (time cost)
       − w_b · b_i,t                                  (i blocked by robot, or bumped wall)
```

- `K_t` is the team-known in-bounds cell set before step t's sensing, `ΔK_t = |K_{t+1}| − |K_t|`, and `G` is the set of all in-bounds cells.
- The cells known at reset (`K_0`) are never rewarded.

| Weight | Default | Rationale |
|---|---|---|
| `w_x` | `1/|G|` | The total exploration budget per episode is ≤ 1.0 regardless of map size |
| `w_d` | 0.25 | Small bonus for finding a target |
| `w_r` | 1.0 | Main task reward |
| `w_c` | 2.0 | Mission completion |
| `w_s` | 0.01 | Time pressure. The same for WAIT and MOVE, so there is no special incentive to camp or to move |
| `w_b` | 0.05 | Discourages conflicts without dominating the other terms |

Reward scale at L2: per-agent return is roughly in `[−1.6, +5.5]`. Value normalization (§16.4) makes the absolute scale immaterial to the critic.

### 13.3 Exploration budget with terminal payout (anti-farming design)

A naive "+w per new cell" combined with "episode ends at success" creates an incentive to **delay the final rescue in order to farm exploration reward**. Rough estimate for L2: about 0.07/step gained versus 0.04/step lost, so farming would pay. SWARMMIND prevents this: at success, all remaining unexplored cells are **paid out**. The total exploration reward of any successful episode is therefore the constant `w_x·(|G| − |K_0|)`, so delaying completion can only lose (time cost + discounting). This is the undiscounted special case of potential-based shaping with `Φ(s) = w_x·|K(s)|` and a constant terminal potential. Truncated (failed) episodes get no payout.

**Invariant test:** for every successful episode, `Σ_t exploration_component == w_x·(|G| − |K_0|)` to within 1e-9.

### 13.4 Reward-hacking analysis

| Risk | Mechanism of the hack | Safeguard | Detection metric |
|---|---|---|---|
| Re-triggering exploration | Revisit cells for reward | Reward only on first team-knowledge (monotone set) | Invariant test: explore reward ≤ budget |
| Oscillation | A↔B loops | No revisit reward; time cost; previous-action feature | Oscillation rate (§23) |
| Camping / idling | WAIT forever | Time cost applies equally; exploration and rescue are the only positive signals | Idle fraction |
| Delaying last rescue to farm coverage | See §13.3 | Terminal payout | Time from last discovery to last rescue |
| One robot does everything (free-riding) | Shared reward hides laziness | Parameter sharing (the same policy for all robots makes systematic laziness state-dependent, not identity-dependent); β-mix ablation | Workload Gini, max share, idle per robot |
| Intentional collisions | None beneficial | Collisions never move a robot and always cost | Collision rate |
| Ignoring targets for coverage | Coverage is worth more than rescue | Exploration budget ≤ 1.0 < one rescue + completion | Rescue latency after discovery |
| Rushing targets, ignoring exploration | Rescue-only behaviour | Not a hack: rescuing is the objective. Coverage is secondary | Coverage@success |
| Clustering | Staying near teammates is safe | Simultaneous co-observation pays once; teammates are visible (ch 6) | DER, mean inter-robot distance |
| Exploiting truncation | Treating the time limit as terminal | Correct truncation bootstrap (§15.6) | Value at `t≈T_max` |

**Required logging:** every reward **component** is logged separately (training metrics + evaluation). A rising total return with a flat success rate is the primary hacking alarm (§36).

---

## 14. Random baseline (and reference heuristic)

### 14.1 Random policy (required baseline)

- At every step each robot samples **uniformly from its valid (masked) actions**, including WAIT. Under `action_masking: false` it samples uniformly over all 5.
- Seeded: `torch.Generator` or `np.random.Generator` per evaluation episode, derived from the evaluation seed.
- It has no training. In statistical tables, "Random seed s" means evaluation sampling seed `s ∈ {0..4}`, so its variance is reported on the same footing as the learned methods.
- It is implemented through the same `Policy` interface (`act(obs_batch) → actions`) as the learned policies.

### 14.2 Reference heuristic: greedy nearest-frontier (non-learning, context only)

- Per robot, independently: if the belief contains a known unrescued target, BFS to the nearest one. Otherwise BFS to the nearest frontier (a known-free cell adjacent to an unknown cell). Ties are broken deterministically (lexicographic cell order). If unreachable, WAIT. There is no explicit coordination.
- **Purposes:** (a) calibrating `max_steps` (§20.6); (b) a solvability check within the time limit; (c) a context line in comparison charts, labelled "Reference heuristic (hand-coded, not learned)".
- It MUST NOT be used as the behaviour of any "learned" view in the UI, and the report must not claim that learned policies are "better than state of the art" based on it. If learned policies underperform it, report that honestly.

---

## 15. IPPO methodology

### 15.1 Definition

IPPO here means **independent PPO learners with a parameter-shared actor and a parameter-shared *local* critic** `V_ψ(o_i)`, following de Witt et al. (2020). Every robot treats the others as part of the environment. The critic sees only what the actor sees.

### 15.2 Networks

```
LocalEncoder (one class; separate instances for actor and critic)
  grid[8,11,11] → Conv3×3(8→32, p1) → ReLU
               → Conv3×3(32→64, stride 2, p1) → ReLU      # 6×6
               → Conv3×3(64→64, p1) → ReLU
               → Flatten(2304) → Linear(2304→256) → ReLU
  vec[10]      → Linear(10→64) → ReLU
  concat(320)  → Linear(320→256) → ReLU                    → h ∈ R^256

ActorPolicy   = LocalEncoder → Linear(256→5) → masked Categorical
IPPO critic   = LocalEncoder' → Linear(256→1)
```

- Initialization: orthogonal. Conv/hidden gain √2; policy head gain 0.01, so the initial policy is near-uniform over valid actions; value head gain 1.0. Biases are 0.
- About 0.73 M parameters per encoder.
- **Separate** actor and critic networks and **separate** Adam optimizers (no shared trunk), identical for IPPO and MAPPO.

### 15.3 Action distribution

`Categorical(logits = masked_logits)`, where `masked_logits = logits.masked_fill(mask==0, −1e9)`. Entropy is computed on the masked distribution. `log_prob` of the stored action is computed with the **stored** mask.

### 15.4 Rollout storage (`buffers/rollout_buffer.py`)

The layout is `[T, E, N, …]` (time, env, agent). Stored per step:

| Field | Shape | Notes |
|---|---|---|
| `obs_grid` | `[T,E,N,8,K,K]` uint8 | |
| `obs_vec` | `[T,E,N,10]` float32 | |
| `mask` | `[T,E,N,5]` bool | |
| `critic_in` | IPPO: none (it reuses obs). MAPPO: `state_grid[T,E,Cg,H,W]` uint8 + agent-specific channels built at batch time + `state_vec[T,E,4]` | |
| `action` | `[T,E,N]` int64 | |
| `logp_old` | `[T,E,N]` float32 | **Stored at collection time**, never recomputed |
| `value_old` | `[T,E,N]` float32 | normalized space if ValueNorm is on |
| `reward` | `[T,E,N]` float32 | after truncation folding (§15.6) |
| `terminated`, `done` | `[T,E]` bool | `done = terminated or truncated` |
| `active` | `[T,E,N]` bool | all True in v1; exists for generality (variable teams) |

### 15.5 GAE (per env, per agent, along time only)

```
next_value[t] = V(s_{t+1})            if t < T−1
              = V(s_last)             if t = T−1   (bootstrap at rollout end)
δ_t = r_t + γ · next_value[t] · (1 − terminated_t or truncated_t) − V(s_t)
A_t = δ_t + γλ · (1 − done_t) · A_{t+1}
R̂_t = A_t + V(s_t)
```

Values are **denormalized** before GAE when ValueNorm is on. GAE MUST be computed on the `[T,E,N]` tensor **before** flattening; flattening first mixes agents and envs, which is a classic bug.

### 15.6 Truncation handling (folding)

When env `e` is truncated at step t, compute `v_final = V(final_obs_e)` with the current critic, **denormalized**, and set `r_t ← r_t + γ·v_final`. The step is then treated as `done` (no further bootstrap), because the bootstrap has been folded into the reward. Terminated steps get no bootstrap. A unit test compares this against a hand-computed expected advantage.

### 15.7 PPO objective

```
ρ_t      = exp(logπ_θ(a_t|o_t) − logp_old_t)
L_clip   = − mean_active[ min(ρ_t·Â_t, clip(ρ_t, 1−ε, 1+ε)·Â_t) ]
L_ent    = − c_ent · mean_active[ H(π_θ(·|o_t)) ]
L_actor  = L_clip + L_ent
L_critic = 0.5 · mean_active[ (V_ψ(x_t) − norm(R̂_t))² ]        (value clipping OFF by default)
```

- **Advantage normalization:** over the **whole batch** of active samples (all envs, agents, steps): `Â = (A − mean)/(std + 1e-8)`. This is computed once per update, before minibatching.
- **Gradient clipping:** `clip_grad_norm_` on each network separately.
- **Diagnostics logged per update:** policy loss, value loss, entropy, approx KL (`mean((ρ−1) − log ρ)`), clip fraction, explained variance of the value predictions, gradient norms, learning rates, SPS.

### 15.8 Hyperparameters (shared by IPPO and MAPPO)

| Hyperparameter | Default | Modest tuning range |
|---|---|---|
| Parallel envs `E` | 16 | 8, 16 |
| Rollout length `T` | 128 | 64, 128 |
| Env steps per update | 2048 (6144 agent samples at N=3) | — |
| PPO epochs | 5 | 5, 10 |
| Minibatches | 4 | 2, 4 |
| Clip ε | 0.2 | 0.1, 0.2 |
| γ | 0.99 | — |
| GAE λ | 0.95 | — |
| Actor LR | 3e-4 | 1e-4, 3e-4, 5e-4 |
| Critic LR | 3e-4 | 1e-4, 3e-4, 5e-4 |
| Adam ε | 1e-5 | — |
| Entropy coef | 0.01 (constant) | 0.003, 0.01, 0.02 |
| Max grad norm | 0.5 | 0.5, 10 |
| Value normalization | on | on/off (ablation only) |
| Value clipping | off | — |
| Target KL | none (logged only) | 0.02 (early stop, only if instability) |
| LR schedule | constant | linear decay (only if instability) |
| Training budget | L1: 1 M, L2: 3 M, L3: 6 M env steps | ±50% |

**Tuning policy:** the defaults are used for both algorithms. If tuning is needed, **both** algorithms get the **same** grid, at most 8 configurations × 2 seeds × 1 M steps on the **validation** maps of L2, and the best configuration per algorithm is frozen. The tuning budget is reported. Test maps are never used for tuning.

### 15.9 Training loop (algorithm-agnostic; shared by IPPO and MAPPO)

```
init seeds; vec_env = SyncVectorSwarm(E, level, seed_stream(run_seed))
actor, critic = build(algo_cfg); opt_a, opt_c = Adam(...)
obs = vec_env.reset()
for update in range(n_updates):
    for t in range(T):
        with no_grad:
            a, logp = actor.act(obs.grid, obs.vec, obs.mask, generator)
            v = critic.value(critic_input(obs))          # IPPO: local; MAPPO: global+local
        nxt = vec_env.step(a)
        fold truncation bootstrap for envs with nxt.truncated
        buffer.add(obs, a, logp, v, nxt.reward, nxt.terminated, nxt.done)
        obs = nxt
    last_v = critic.value(critic_input(obs))
    adv, ret = gae(buffer, last_v, γ, λ, value_norm)
    value_norm.update(ret); ret_n = value_norm.normalize(ret)
    adv = normalize(adv over active samples)
    for epoch in range(K):
        for mb in random_minibatches(flatten(buffer)):
            update actor (L_actor), update critic (L_critic); clip grads
    log metrics; periodic validation eval; periodic checkpoint
```

### 15.10 Checkpointing

The checkpoint `results/runs/<run>/checkpoints/step_<envsteps>.pt` contains: actor state, critic state, both optimizer states, ValueNorm stats, `global_env_steps`, `update_idx`, RNG states (python, numpy global, every env Generator state, torch CPU/CUDA, the policy sampling generator), the resolved config, a schema version, and the git commit. `latest.pt` is written atomically (write to a temp file, then rename). `torch.load(..., weights_only=True)` is used where possible. Checkpoints are saved at 0%, 10%, 20% … 100% of the budget (the 0% checkpoint is the untrained policy, needed for Swarm Evolution).

---

## 16. MAPPO methodology (primary ML contribution)

### 16.1 Definition

MAPPO = PPO with a **parameter-shared decentralized actor** `π_θ(a|o_i)` (identical to IPPO) and a **parameter-shared centralized critic** `V_φ(x_i)` on the agent-specific global input `x_i` (§9.3), trained with CTDE.

```
o_i ──► ActorPolicy (shared θ) ──► π(·|o_i) ──► a_i          [train + execution]

(S_shared ⊕ S_agent_i) ──► GlobalEncoder ─┐
o_i ─────────────────────► LocalEncoder' ─┼─► MLP ──► V_φ(x_i)  [training only]
g ────────────────────────────────────────┘
```

### 16.2 Critic architecture

```
GlobalEncoder: [9,H,W] → Conv3×3(9→32,p1) ReLU → Conv3×3(32→64,s2,p1) ReLU
             → Conv3×3(64→64,p1) ReLU → AdaptiveAvgPool2d(4,4) → Flatten(1024)
             → Linear(1024→256) ReLU
LocalEncoder': same architecture as the actor's encoder, separate weights → 256
Fusion: concat(256, 256, g[4]) → Linear(516→256) ReLU → Linear(256→1)
```

### 16.3 Controlled difference from IPPO

The only config difference is `critic.input: local | agent_specific_global`. Actor architecture, initialization, optimizer, hyperparameters, rollout, GAE, losses and budgets are identical. **Parity test:** running the MAPPO code path with `critic.input: local` MUST reproduce IPPO bit-for-bit for 2 updates on CPU with the same seed. This proves that nothing else differs.

**Known confound:** the MAPPO critic has more parameters. *Optional control (stretch):* "IPPO-wide", whose local critic is widened to a matched parameter count. Report the parameter counts in the results either way.

### 16.4 Value normalization

A running mean/std of returns (`ValueNorm`, β-debiased EMA, as in the MAPPO reference implementation) is used for **both** algorithms (to keep the comparison fair). The critic predicts normalized values. Stored `value_old` is normalized. GAE uses denormalized values. Order per update: `update(returns) → normalize(returns)` for the targets.

### 16.5 Rollout representation and batch construction

The same `[T,E,N]` buffer is used. The shared state grid is stored once per `(t,e)`, not per agent. The agent-specific channels (position one-hot, belief mask) are built at batch time from stored positions and belief masks, which saves N× memory. After GAE, flatten to `B = T·E·N` samples, shuffle, and split into minibatches. Each sample carries `(o_i, mask_i, a_i, logp_old_i, x_i, R̂_i, Â_i, active_i)`.

### 16.6 Agent masking and episode masking

- `active[t,e,n]` multiplies every loss term and every normalization statistic. In v1 it is always 1. It must still be implemented and tested with an artificial mask so that variable team sizes work later.
- Episode boundaries are handled by `done` in GAE. Truncation is folded (§15.6). No computation propagates across an auto-reset.

### 16.7 Theoretical caveat (must appear in the report)

A critic conditioned on the true state `s` rather than on the actor's information can be **biased** as a value estimate of an observation-conditioned policy in a POMDP (Lyu et al. 2021/2023). A state-dependent baseline leaves the policy gradient unbiased in expectation, but bootstrapped targets can inherit this bias, and variance can increase. Mitigation: agent-specific input including `o_i`. **Interpretation rule:** if MAPPO does not beat IPPO, this is a known, citable possibility, not evidence of a bug by itself.

### 16.8 Common MAPPO implementation mistakes (checklist for Phase 5 and Phase 15)

1. Feeding global state (or anything derived from it) to the actor, including via the action mask.
2. Computing GAE after flattening agents and envs together.
3. Bootstrapping from the auto-reset observation instead of the true final observation.
4. Treating truncation as termination (or the reverse).
5. Not storing the action mask, so update-time log-probs mismatch collection-time log-probs.
6. Recomputing `logp_old` after the policy has changed.
7. Including inactive/padded samples in advantage normalization or loss means.
8. Mixing normalized and denormalized values in GAE or targets.
9. Using `-inf` logits, which produces NaN entropy.
10. Concatenating all agents' observations as the critic input: this does not scale with N and breaks parameter sharing across team sizes.
11. An agent-specific input that omits "which agent am I", so the critic cannot tell agents apart in the shared state.
12. Different sample budgets, seeds, maps or hyperparameters between IPPO and MAPPO.
13. Counting agent-steps as env-steps (or the reverse) when comparing budgets.
14. Deterministic (argmax) evaluation of symmetric shared policies causing symmetric deadlock (§22.3).
15. Evaluating on training maps, or selecting checkpoints by test performance.
16. Summing the shared reward over agents in one place and averaging it in another.

### 16.9 QMIX (optional, not planned)

QMIX needs off-policy replay, recurrent Q-networks and a mixing network, which roughly doubles the ML surface area. It answers a different question (value factorization vs. centralized critic). **Recommend only if** Phase 7 completes early, H1/H2 are settled, and the report explicitly needs a value-based comparison. Otherwise do not implement it.

---

## 17. Parameter sharing

| Aspect | Decision |
|---|---|
| Actor | **One shared actor for all robots.** Robots are homogeneous, observations are egocentric, and the action semantics are identical |
| Critic | Shared (IPPO local critic, MAPPO centralized critic) |
| Robot identity | **Not needed** for the actor. Robots differ in their observations (positions, beliefs, teammates in view), and stochastic sampling breaks symmetry. The MAPPO critic gets identity implicitly through the agent-position channel. `agent_id_onehot` is an ablation only |
| Batching | Every `(t, e, n)` is one sample. With N=3 the batch has 3× the samples per env step, so learning data scales with team size |
| Different swarm sizes | Without IDs the actor is size-agnostic, so a policy trained with N=3 can run with N=2…6 (G3). The critic is not needed at test time |
| Advantages | Sample efficiency; consistent behaviour; generalization over N; far fewer parameters |
| Disadvantages | Cannot represent role specialization tied to identity. Symmetric states produce symmetric actions (deadlock risk under greedy evaluation). The same policy may do both "lead" and "follow" roles depending on state, which can blur workload attribution |

---

## 18. Curriculum

### 18.1 Position in the study

**Main experiments train directly on the target level (no curriculum)**, so that IPPO and MAPPO see exactly the same data distribution. A curriculum is an optional tool for L3/L4, used only if direct training fails the Phase 7 learning gate at those levels.

### 18.2 Performance-gated promotion (when used)

- Levels proceed L1 → L2 → L3 → L4. The architecture transfers unchanged (egocentric, size-agnostic actor; adaptive-pool critic).
- **Promote when** the rolling validation success rate is ≥ 0.80 **and** the median steps-to-completion is ≤ 0.8 × `max_steps` over **2 consecutive** validation evaluations (each on 50 validation maps), **and** at least 20% of that stage's budget has been spent.
- **Never promote on elapsed time.** If a stage exhausts its budget without meeting the criteria, the run stops and records `curriculum_stalled`.
- The same criteria and budgets apply to all algorithms. "Env steps to promotion" is itself reported as a metric.
- Optimizer state and ValueNorm carry across stages. ValueNorm is reset only if it is shown to be necessary.

---

## 19. Communication strategy

| Mode | Key | What robots share | Status |
|---|---|---|---|
| **A** | `none` | Nothing. Each belief is built from own sensing only | Core (implemented in Phase 2; used in H5) |
| **B** | `shared_map` | Idealized broadcast of **map knowledge** (terrain, targets, rescued status, visited cells) to all robots each step. Teammate **positions are not shared** (sensor range only) | **Core default** |
| **C** | `limited_range` | At the end of each step, robots within `comm_radius` (Euclidean), transitively over the connected components of the comm graph, merge beliefs (set union). Comm links are recorded for the UI | Stretch |
| **D** | learned | — | Out of scope (§19.4) |

### 19.1 Justification for B as the default

B makes coordination learnable with a feedforward actor: robots can see where the team has explored and choose other frontiers. Partial observability remains, because teammates are visible only in sensor range and the window is local. Map broadcast is low-bandwidth and tolerant of delay (the information is static), which makes it a defensible abstraction.

### 19.2 Relationship to CTDE

In every mode the actor sees **only its belief** under that mode's rules. The centralized critic always sees the ground truth. Mode A maximizes the information gap between actor and critic, which is why H5 predicts that MAPPO's benefit is largest there.

### 19.3 Tests

`test_mode_b_sharing` (robot j's discovery appears in i's belief on the same step); `test_mode_c_merge_transitive` (A–B–C chain in range merges all three); `test_mode_c_out_of_range_isolated`.

### 19.4 Learned communication (design note only)

If ever attempted: add a discrete message action head (vocabulary 4–8 tokens) whose tokens are delivered to robots in range the next step, trained end-to-end by PPO. Evidence of genuine communication MUST include **message-intervention tests** (replacing messages with noise or shuffling them degrades performance significantly) and mutual information between messages and the sender's private observations. Scripted messages are forbidden.

---

## 20. Procedural maps

### 20.1 Generator (`marl/envs/core/mapgen.py`), deterministic given `(level_cfg, seed)`

1. `rng = np.random.default_rng(seed)`.
2. Obstacles: each in-bounds cell is an obstacle with probability `ρ`.
3. **Connectivity:** find the largest 4-connected FREE component, fill every other FREE cell as OBSTACLE ("pocket filling"), so that **all free cells are reachable**. Reject (retry) if the component is < 60% of the cells.
4. **Spawn (clustered):** choose a uniformly random free anchor, then BFS to take the N nearest free cells (ties by lexicographic order) as start positions. Scattered mode samples N distinct free cells uniformly.
5. **Targets:** sample M distinct free cells with BFS distance ≥ `target_min_dist` from every start. Reject if impossible.
6. **Validation:** BFS reachability from every start to every target (asserted). No overlaps. Record the actual density after pocket filling.
7. **Retries:** attempt `k` uses seed `hash64(seed, k)`. After 100 failed attempts, raise `MapGenerationError` (the level config is infeasible).

### 20.2 `MapSpec` (serializable)

`{id, width, height, terrain (row strings '.#'), robot_starts, targets, generator: {level, seed, attempt} | "handmade", hash}`, where `hash = sha256(canonical JSON of terrain+starts+targets)`.

### 20.3 Splits (hard-separated)

| Split | Source | Size per level | Used for |
|---|---|---|---|
| **Train** | Procedural, seeds from the run's `SeedSequence` stream, drawn from `[0, 2^31)` | unbounded | Training only |
| **Validation** | Procedural, seeds `1_000_000 + i` | 50 | Periodic evaluation during training, tuning, checkpoint diagnostics, curriculum gating |
| **Test (in-distribution)** | Procedural, seeds `2_000_000 + i` | 100 | Final reported results only |
| **Test (structured, G2)** | Handmade `.txt` maps: rooms, corridors, open field, maze-like, bottleneck | 10–20 | Generalization (H4), showcase replays |

- Validation and test sets are generated **once** by `scripts/make_eval_maps.py` and saved as JSON with their hashes in `configs/maps/`. These files are committed and immutable. Regenerating them requires a version bump.
- **Isolation:** the training generator loads the frozen hash set and **rejects** any training map whose hash is in validation ∪ test (test: `test_train_rejects_eval_hashes`). Config validation forbids `map_source` pointing at `test` in any training config. The final test evaluation script refuses to run on a run whose `manifest.status != "completed"` and logs every access.

### 20.4 Structured handmade maps

Stored in `configs/maps/structured/*.txt`: `#` obstacle, `.` free, `R` robot start, `T` target. They are validated by the same reachability checks.

### 20.5 Solvability

Every map is solvable by construction. **Solvable within `max_steps`** is checked empirically: the reference heuristic's success rate on validation maps MUST be ≥ 0.95 (§20.6).

### 20.6 `max_steps` calibration (Phase 3; frozen in Phase 6)

`max_steps(level) = round_up_to_10( 3 × P90(heuristic completion steps on validation maps) )`, bounded below by the provisional value in §6.7 if that is larger. The random-policy success rate and the heuristic success rate are recorded. If random success is ≥ 0.6 on L2, the level is too easy to show learning. The level is then adjusted (raise density or targets) **before** freezing, and the change is documented.

---

## 21. Training methodology

### 21.1 Pipeline

`scripts/train.py --config configs/experiments/<exp>.yaml --algo mappo --level L2 --seed 3 [--set key=value …] [--resume <run_dir>]`

1. Resolve the config (§32), validate it, compute `config_hash`.
2. Create the run dir `results/runs/<study>/<algo>/<level>/s<seed>-<YYYYMMDDTHHMMSS>-<hash8>/`.
3. Write `manifest.json` (status `running`), `config.resolved.yaml` and `env_versions.json`.
4. Seed everything (§34). Build the vector env, networks and optimizers.
5. Loop (§15.9). Stream metrics to `metrics.jsonl`, one JSON line per update. Validation eval goes to `eval_val.jsonl`.
6. Checkpoints at every 10% of the budget plus `latest.pt` every N updates.
7. On completion: final checkpoint, `status: completed`, then export the actor (`actor.pt` + `meta.json`).
8. On exception or signal: `status: failed|interrupted` with the traceback. `latest.pt` makes resume possible.

### 21.2 What is recorded (manifest)

`run_id, study, algo, level, seed, env_config, algo_config (all hyperparameters), total_env_steps, total_agent_steps, updates, start/end timestamps, wall time, device, torch/numpy/pettingzoo/gymnasium/python versions, platform, git commit + dirty flag, config_hash, checkpoints[], final_val_metrics, status`.

### 21.3 Validation during training

Every `eval_interval_updates` (default 20, about 41k env steps): 50 validation maps × 1 episode, stochastic policy, fixed evaluation seeds. Log success, coverage, steps, collisions and return components. This never touches test maps.

### 21.4 Resume

`--resume` restores everything in the checkpoint (§15.10) and continues **exactly**. Test: on CPU, 4 updates straight must equal 2 updates + save + resume + 2 updates, bitwise on parameters.

### 21.5 Device policy

- `device: auto` selects CUDA if available, else CPU. On Apple Silicon, **CPU is the default**. MPS is allowed only if a Phase 4 benchmark shows a speedup for these small CNNs (often it does not).
- Environment stepping always runs on CPU/NumPy. Rollout inference runs batched `[E·N]` per step.
- Multi-seed parallelism uses **separate processes**, each with `torch.set_num_threads(1)` (or 2), rather than one process with many threads.

---

## 22. Evaluation methodology

### 22.1 Protocol

| Item | Specification |
|---|---|
| Maps | Level test set (100 maps) and structured set (G2) |
| Episodes | For each map, 2 episodes with evaluation sampling seeds `{0, 1}` → 200 episodes per (checkpoint, level) |
| Policy mode | **Stochastic sampling (primary).** Greedy argmax is secondary and reported separately |
| Checkpoint | **Final checkpoint (primary).** "Best on validation" is secondary and always labelled |
| Training seeds | 5 per algorithm (seeds 0–4). Random: 5 evaluation sampling seeds |
| Pairing | All methods run on the same maps with the same evaluation seeds (**common random numbers**). Training seed s uses the same training-map stream for IPPO and MAPPO |
| Output | `results/runs/<run>/eval/test_<level>.json`: per-episode records plus a per-run aggregate |

### 22.2 Statistics

- Per seed: the mean of each metric over its 200 episodes.
- Across seeds (n=5): mean, sample std, **95% t-interval**. Bootstrap CIs are optional and labelled as unreliable at n=5.
- Pairwise algorithm comparisons: the difference in means with a **Welch** 95% CI and Hedges' g. With 5 seeds the power is low, so "no significant difference" is **not** evidence of equivalence. Word conclusions accordingly.
- Learning curves: x-axis = **environment steps**. Show the mean over seeds with a 95% CI band, plus thin per-seed lines.
- Sample efficiency: the normalized area under the validation-success curve.
- Optional: IQM with stratified bootstrap (rliable).

### 22.3 Deterministic-evaluation pitfall

With a shared policy, two robots in mirror-symmetric states pick the same argmax action and can deadlock (repeated R2 conflicts). Stochastic evaluation is therefore the primary protocol. The greedy result is reported for completeness.

### 22.4 "Genuine learning" evidence checklist (required in the Phase 7 report)

1. The 0% checkpoint's metrics are about equal to Random's (CIs overlap).
2. Validation success rises over training for ≥ 4/5 seeds.
3. Final learned policy > Random on test (H1).
4. **Input ablation:** zeroing channel 6 (teammates) or channel 5 (visited_known) at evaluation changes coordination metrics measurably, which shows the policy uses that information.
5. The leak probe (§12.4) is negative.

---

## 23. Metrics

Notation: episode length `T_ep`; robots `i=1..N`; targets `M`; reachable free cells `F`; `S_i` = the set of cells robot i has sensed itself; `K` = team-known cells; `p_i,t` = position.

| Metric | Definition | Unit |
|---|---|---|
| Episode return | Mean over agents of `Σ_t r_i,t`, plus **each component** separately | reward units |
| Mission success | `1[all M rescued before T_max]` | rate |
| Targets discovered / rescued | `n_disc`, `n_resc`; rates `/M` | count, rate |
| Coverage | `|K ∩ F| / |F|` at episode end; `C(t)` over time | % |
| Visited coverage | `|∪_i visited_i| / |F|` | % |
| Steps to completion | `T_ep` over successful episodes; also **censored mean** with failures counted as `T_max` (always report success rate beside it) | steps |
| Collisions | `Σ_{i,t} blocked_by_robot_i,t`; also per 100 agent-steps | count |
| Invalid actions | `Σ bumped_wall` (0 by construction when masking is on) | count |
| Distance travelled | `d_i = Σ_t 1[p_i,t+1 ≠ p_i,t]`; team `Σ d_i` | cells |
| Idle fraction | `idle_i = 1 − d_i/T_ep` | % |
| Oscillation rate | `Σ_t 1[p_i,t+1 = p_i,t−1 ≠ p_i,t] / Σ_t 1[moved]` | % |
| Exploration efficiency | `|K_T ∩ F| − |K_0 ∩ F|` divided by `N·T_ep` | cells / agent-step |
| Rescue efficiency | `n_resc / (N·T_ep)` | targets / agent-step |
| Rescue latency | `t_rescued_k − t_discovered_k` (mean) | steps |
| Workload share | `w_i = credit_i / Σ_j credit_j`, where credit = newly team-known cells attributed to i (split among simultaneous observers) | fraction |
| Workload Gini | `G = Σ_i Σ_j |w_i − w_j| / (2N Σ w)`; also `max_i w_i` | [0, (N−1)/N] |

All metrics are computed in `marl/analysis/metrics.py` from a standard **EpisodeTrace** (the replay frames, §27), so replay, evaluation and backend always agree. Arbitrary composite "intelligence scores" are forbidden.

---

## 24. Coordination metrics

### 24.1 Duplicate Exploration Ratio (DER)

`DER = 1 − |∪_i S_i| / Σ_i |S_i|`. It is 0 when there is no overlap and approaches `(N−1)/N` when all robots sense the same cells. It is confounded by episode length, so SWARMMIND also reports **DER@50**: DER computed at the first step where coverage ≥ 50% (a matched-coverage comparison). The DER-vs-coverage curve is also plotted.

### 24.2 Other coordination measures

| Measure | Definition | Coordination signal |
|---|---|---|
| Visitation overlap | `|{c : visited by ≥2 robots}| / |∪ visited|` | Lower means better division |
| Redundant pursuit | After target k is discovered and until it is rescued, the mean number of robots per step whose BFS distance to k strictly decreases | ≈1 means allocation; >1 means herding |
| Target allocation optimality | Fraction of targets rescued by the robot that was BFS-nearest at discovery time | Descriptive |
| Mean inter-robot distance | `mean_t mean_{i<j} ‖p_i − p_j‖₂` and minimum distance | Dispersion |
| Dispersion speed | Steps until the mean inter-robot distance first exceeds `r_s` | Spreading out from a clustered spawn |
| Workload Gini / idle | §23 | Division of labour |
| Collision rate | §23 | Implicit conflict avoidance |

### 24.3 Isolation null model (stretch, strongly recommended)

For each test map, run each robot's policy **alone** (N=1, same start cell, same map) and union the resulting sensed sets. This gives `DER_iso`, the overlap expected with **no interaction at all**. A coordinated team should achieve `DER_team < DER_iso` on the same maps. This separates "robots spread out because of teammates and shared knowledge" from "robots take different paths by chance".

### 24.4 Criteria for claiming "coordination/cooperation emerged"

All of the following must hold for a learned algorithm, across seeds, with CIs:
1. H1 is supported (it succeeds more than Random).
2. DER@50 is lower than Random's **and** lower than the isolation null (if run).
3. Redundant pursuit is lower than Random's.
4. The input ablation of the teammate or visited channels degrades coordination metrics.

Otherwise the report says: "differences in behaviour were observed; evidence of learned coordination is inconclusive."

---

## 25. Generalization experiments

| ID | Train | Test | Fair because | Priority |
|---|---|---|---|---|
| **G1** | L2 procedural | L2 held-out procedural test maps | Same distribution; unseen instances | Core |
| **G2** | L2 procedural | Structured handmade 10×10 maps (rooms/corridors/maze) | Same size, N, M and observation; only layout statistics shift | Core (H4) |
| **G3** | L2 (N=3) | L2 maps with N=2,4,5 | The actor has no IDs and the observation is egocentric, so it is defined for any N. `max_steps` stays the same | Stretch |
| **G4** | L2 (10×10) | 12×12, 14×14 with `max_steps` scaled by area | Egocentric window is size-agnostic. **Caveat:** normalized position features change meaning; report as exploratory | Stretch |
| **G5** | L2 (ρ=0.15) | ρ = 0.05, 0.25 | Same everything else | Stretch |

Rules: no fine-tuning; the same checkpoints (final); the same evaluation seeds; report **absolute** performance and **retention** relative to G1; Random is evaluated on the same shifted sets as the floor.

---

## 26. Behaviour analysis

| Analysis | Data | Visualization | Question answered |
|---|---|---|---|
| Robot trajectories | Replay frames | Canvas polylines per robot, fading by age | How do robots move and split up? |
| Exploration heatmap | Visit counts aggregated over K episodes on one map | Sequential single-hue heatmap over the grid | Where does the swarm spend time? Does it neglect regions? |
| Collision heatmap | Blocked events by cell | Heatmap overlay | Where do conflicts happen (corridors, spawn)? |
| Coverage over time | `C(t)` | Line chart: mean + 95% CI per algorithm, x = step | How fast is the map known? |
| Target discovery timeline | `t_discovered`, `t_rescued` per target | Gantt-style strip per episode | Search latency vs. rescue latency |
| Workload distribution | `w_i`, `d_i` | Per-robot bars (sorted), Gini annotation | Do all robots contribute? |
| Idle time | `idle_i` | Per-robot bars | Free-riding? |
| Inter-robot distance | Mean/min distance over time | Line chart | Dispersion and clustering dynamics |
| Communication links (mode C) | Comm edges per step | Canvas dotted lines | When do robots share info? |
| Swarm Evolution | Checkpoints at 0%, 10%, 50%, 100% on the same map + seed | 4 synchronized replay canvases + metric row | Visible learning progression |
| DER vs. coverage | Per-step DER and C(t) | Line chart | Coordination at matched coverage |

The Python side (`marl/analysis/`) produces the data as JSON plus static matplotlib figures for the written report. The frontend renders the interactive versions from the **same JSON**.

---

## 27. Replay architecture

### 27.1 Recording

`marl/replay/recorder.py` wraps any episode run (evaluation, backend session, or `scripts/record_replays.py`) and emits a **Replay** file. Frames come from the environment's own `StepResult`. They are never re-simulated by the frontend.

### 27.2 Schema (`replay.schema_version = 1`, pydantic models exported to JSON Schema and TypeScript)

```json
{
  "schema_version": 1,
  "replay_id": "L2-test-0042_mappo_s3_step3000000_eval0",
  "created_at": "ISO-8601",
  "env": { "level": "L2", "config": { "...": "resolved EnvConfig" } },
  "map": { "id": "...", "hash": "...", "width": 10, "height": 10,
           "terrain": ["..#.......", "..."], "robot_starts": [[y,x],...],
           "targets": [[y,x],...] },
  "policy": { "kind": "mappo|ippo|random|heuristic", "run_id": "...",
              "checkpoint_step": 3000000, "stochastic": true, "eval_seed": 0 },
  "initial": { "known_team": "<RLE bitmask>", "belief_known": ["<RLE>", ...] },
  "frames": [
    { "t": 1,
      "pos": [[y,x],...], "action": [..], "moved": [..],
      "blocked_by_robot": [..], "bumped_wall": [..],
      "newly_known": [[y,x],...],             // team delta
      "belief_delta": [[[y,x],...], ...],     // per robot; only for comm modes A/C
      "discovered": [k,...], "rescued": [[k, robot_i],...],
      "reward": [..], "reward_components": {"explore":..,"discover":..,"rescue":..,"complete":..,"time":..,"collision":[..]},
      "comm_edges": [[i,j],...]               // mode C only
    }
  ],
  "outcome": { "success": true, "steps": 87, "metrics": { "...": "§23/24 metrics" } }
}
```

- Delta encoding keeps an L2 episode at about 30–120 KB of JSON (gzip ≈ 10–25 KB). Files are stored as `.json.gz`.
- **Determinism test:** replaying `(map, env seed, recorded actions)` through `SwarmWorld` reproduces every frame exactly.

### 27.3 Storage and index

`results/replays/<study>/<run_id>/<replay_id>.json.gz` plus `index.json` (id, map, policy, step, outcome). Swarm Evolution replays are recorded by `scripts/record_replays.py --evolution` for 0/10/50/100% checkpoints on 5 showcase maps.

### 27.4 Frontend replay controls

Play, Pause, Step ±1, speed (0.25×–8×), timeline scrubber with event markers (discovery ◆, rescue ★, collision ✕), checkpoint selector (Evolution view), and toggles for layers. The frontend reconstructs the belief/known map by accumulating deltas up to frame t, with a keyframe cache every 25 frames for O(1) scrubbing.

---

## 28. Backend architecture

### 28.1 Structure

```
backend/app/
  main.py              # app factory, CORS, lifespan (scan results/, build indexes)
  settings.py          # pydantic-settings: RESULTS_DIR, MODELS_DIR, HOST=127.0.0.1, limits
  api/routers/         # health, env, sessions, models, experiments, replays, training, studies
  api/ws/              # sim_ws.py, train_ws.py
  services/
    session_manager.py # live SwarmWorld + ActorPolicy sessions (TTL, max concurrent)
    model_registry.py  # discovers models/ and results/runs/*/actor.pt
    run_index.py       # indexes manifests, metrics, eval JSON
    replay_store.py    # lists/loads replays (path-confined)
    job_runner.py      # launches scripts/train.py as subprocess; tracks PID/status
    study_service.py   # reads aggregated study results
  schemas/             # pydantic request/response models (mirroring marl types)
```

Routers contain **no ML or environment logic**. They validate input, call a service, and return a schema. Services call the public `marl` API.

### 28.2 REST endpoints (all under `/api`)

| Method & path | Purpose |
|---|---|
| `GET /health` | Status, versions, device, counts of runs/models |
| `GET /levels` | Level configs (L1–L4) |
| `GET /maps?set=validation|test|structured` | Fixed map metadata (+ terrain) |
| `POST /sessions` | Create a live sim session `{level|env_config, map: procedural{seed}|fixed{id}, policy: {kind, model_id?}, stochastic, seed}` → `session_id` + initial frame |
| `POST /sessions/{id}/reset` | Reset with a new seed/map |
| `POST /sessions/{id}/step` | `{n: 1..50}` → frames |
| `GET /sessions/{id}` | Current frame + episode metrics |
| `DELETE /sessions/{id}` | Close |
| `GET /models`, `GET /models/{id}` | Model registry (algo, level, seed, steps, test metrics if available) |
| `GET /experiments` | Runs list (filter by study/algo/level/status) |
| `GET /experiments/{run_id}` | Manifest + summary |
| `GET /experiments/{run_id}/metrics?from_update=` | Training metrics series |
| `GET /experiments/{run_id}/eval` | Evaluation results |
| `GET /studies`, `GET /studies/{id}` | Aggregated comparison tables + curves (Phase 6 output) |
| `GET /replays?run_id=&map_id=` , `GET /replays/{id}` | Replay index/content (gzip passthrough) |
| `GET /analysis/{run_id}/heatmap?map_id=&kind=visits|collisions` | Precomputed analysis JSON |
| `POST /training/jobs` | Launch training (bounded, validated params) → job_id/run_id |
| `GET /training/jobs/{id}` , `POST /training/jobs/{id}/cancel` | Status / cancel |

### 28.3 WebSockets

- `WS /ws/sessions/{id}`. Client → server: `{"cmd":"play"|"pause"|"step"|"reset"|"speed","value":…}`. Server → client: `{"type":"frame", …Frame}` at the requested rate (≤ 30 steps/s), `{"type":"episode_end", metrics}`, `{"type":"error", …}`. The server owns the simulation clock; the client only renders.
- `WS /ws/training/{run_id}`. The server tails `metrics.jsonl` / `eval_val.jsonl` and pushes new lines (`{"type":"update", …}` and `{"type":"val_eval", …}`) plus status changes. Polling fallback: `GET /experiments/{run_id}/metrics?from_update=`.

### 28.4 Model/inference parity

`SessionManager` uses `marl.algorithms.policy.load_actor(path)` and `SwarmWorld`. **Test:** a backend session with (map, seed, stochastic, eval_seed) produces the identical action/frame sequence as `marl.evaluation.run_episode` with the same inputs.

### 28.5 Training jobs

Training runs as a **separate subprocess** (`python scripts/train.py …`), never inside the API process. At most 1 concurrent job by default (configurable). The job's stdout goes to the run dir, and the API reads status from the manifest. Training parameters submitted from the UI are bounded by the same pydantic schema as the CLI config.

---

## 29. Frontend architecture

### 29.1 Stack decisions

| Choice | Justification |
|---|---|
| React 18/19 + Vite + **TypeScript** | Typed contracts for Replay/Frame/metrics prevent silent UI data bugs. The types are generated from backend OpenAPI and JSON Schema (`openapi-typescript`, dev-only) |
| Bootstrap 5.3 (CSS only; no Bootstrap JS unless a component needs it) | Grid, reboot and form primitives, **re-themed via SWARMMIND tokens** (Bootstrap CSS variables mapped to tokens) so it does not look like stock Bootstrap |
| Custom CSS (CSS Modules or BEM-scoped files) + `tokens.css` | Centralized design tokens; no inline magic numbers |
| **Canvas 2D** for the simulator | Hundreds of cells × layers × 60 fps interpolation; SVG/DOM per cell would be slow and noisy |
| Recharts | Standard charts (lines with CI bands, bars, scatter) |
| Zustand | Tiny store for high-frequency sim/replay state that the canvas reads outside React renders; avoids re-render storms |
| react-router | Page routing |
| lucide-react | One consistent icon set, used sparingly |
| Motion (optional) | Only for panel/page transitions where CSS is insufficient |
| Vitest + Testing Library | Unit/component tests; Playwright MAY be added in Phase 14 for E2E |

### 29.2 Source layout

```
frontend/src/
  app/            App.tsx, routes.tsx, AppShell (nav + layout), providers
  design/         tokens.css, bootstrap-theme.css, typography.css, README.md
  api/            client.ts (fetch wrapper), types.gen.ts (generated), endpoints/*.ts, ws.ts
  stores/         simStore.ts (live/replay playback), uiStore.ts (layers, preferences)
  features/
    simulation/   SimulationCanvas/, renderers/ (terrain, belief, paths, sensors, robots,
                  targets, overlays), interpolation.ts, frameReducer.ts (delta accumulation)
    replay/       ReplayControls, Timeline, EvolutionGrid
    training/     TrainForm, LiveTrainingCharts, JobStatus
    comparison/   ComparisonTable, MetricDotPlot, LearningCurves
    analysis/     Heatmap, CoverageChart, DiscoveryTimeline, WorkloadChart, DistanceChart
    experiments/  ExperimentTable, RunDetail
  components/     MetricCard, Panel, Toolbar, Select, Toggle, Tabs, EmptyState,
                  ErrorState, LoadingState, Tooltip, StatusDot, KeyboardHint
  pages/          SwarmLab, TrainSwarm, AlgorithmComparison, SwarmAnalysis,
                  ExperimentHistory, Settings
  mocks/          ONLY place for mock data (see §29.4)
  lib/            format.ts (units, %), colors.ts (robot palette from tokens), stats.ts
```

### 29.3 Separation of concerns

- **Presentation** (`components/`) holds no fetching.
- **State** lives in `stores/` and hooks.
- **API communication** lives in `api/` only.
- **Simulation rendering** (`features/simulation/renderers/*`) is a set of pure functions `(ctx, frameState, viewport, tokens) → void`, one per layer, testable with a mock canvas.
- **Chart logic** (`features/*/charts`) receives prepared series and does no aggregation. Aggregation happens on the backend or in `lib/stats.ts`.

### 29.4 Real-data rule enforcement

- Mocks exist only in `src/mocks/`, only behind `import.meta.env.VITE_USE_MOCKS === "true"`, and every mocked view shows a visible "MOCK DATA" badge.
- `scripts/check_no_mocks.mjs` (run in `npm run build` and CI) fails if any non-mock file imports from `mocks/` directly, or if a production build has `VITE_USE_MOCKS=true`.
- Phase 14 gate: zero mock usage and the production bundle contains no mock modules.

---

## 30. Professional UI/UX system

### 30.1 Critique of the reference image (what to keep, what to change)

| Keep | Change | Why |
|---|---|---|
| Dark left navigation + light workspace | Collapse nav to 64 px icons on narrower screens; remove the decorative robot illustration | Space goes to the simulator |
| Central grid arena, coloured robots, sensor discs, dashed paths, star targets | Make the canvas the **dominant** element (≥ 55% of viewport width on desktop); move controls into a compact toolbar | The reference gives the arena about 35% width, with equal-weight panels competing for attention |
| Live mission stats | One **mission strip** above the canvas instead of an overlay card plus a duplicate metrics panel | Removes duplication |
| Robot status table | Remove "Energy" (not modelled); replace intent labels like "Moving to Target" with **observable facts** (Moving / Blocked / Waiting / Rescued T2 @ t=41) | The policy has no explicit intents; labelling intents would fabricate cognition |
| Training progress chart | Split the dual-axis reward + success chart into **small multiples** with separate y-axes | Dual axes mislead |
| Algorithm comparison, analysis, history pages | Keep, backed by real data only | Required functionality |
| "System Architecture", "Development Phases", "Recommended Models" panels | **Remove from the product**; they belong in README/docs | They are documentation, not product functions |
| Red robot | Robot palette avoids red and green, which are reserved for danger/collision and success | Semantic clarity |

### 30.2 Information architecture

```
SWARM LAB (default route "/")        — live + replay simulator (flagship)
TRAIN SWARM ("/train")               — launch & monitor training runs
ALGORITHM COMPARISON ("/compare")    — study results: Random vs IPPO vs MAPPO
SWARM ANALYSIS ("/analysis")         — heatmaps, timelines, workload, Swarm Evolution
EXPERIMENT HISTORY ("/experiments")  — runs table, run detail
SETTINGS ("/settings")               — backend URL, playback defaults, reduced motion, display prefs
```

### 30.3 Design tokens (`frontend/src/design/tokens.css`; the only source of raw values)

**Typography.** UI font: *Inter* (fallback system-ui). Data/number font: *JetBrains Mono* with tabular figures.

| Token | Size / line-height / weight | Use |
|---|---|---|
| `--sm-font-xs` | 11/16/500 | Axis ticks, legends |
| `--sm-font-sm` | 12/16/400 | Table cells, captions |
| `--sm-font-md` | 14/20/400 | Body (base) |
| `--sm-font-lg` | 16/24/600 | Panel titles |
| `--sm-font-xl` | 20/28/600 | Page titles |
| `--sm-font-2xl` | 24/32/600 | Key metric numerals |

**Spacing (4 px base).** `--sm-space-0: 0; -1: 2px; -2: 4px; -3: 8px; -4: 12px; -5: 16px; -6: 24px; -7: 32px; -8: 48px`.

**Radius.** `--sm-radius-cell: 2px; -control: 4px; -panel: 6px; -pill: 999px` (status dots only). No large rounded cards.

**Elevation.** `--sm-elev-0: none` (flat + 1px border, the default for panels); `--sm-elev-1: 0 1px 2px rgba(15,27,45,.06)` (floating toolbar); `--sm-elev-2: 0 4px 12px rgba(15,27,45,.12)` (menus, popovers, tooltips). No glows.

**Surfaces.**

| Token | Hex | Use |
|---|---|---|
| `--sm-bg-app` | `#F4F6F9` | App background |
| `--sm-bg-panel` | `#FFFFFF` | Panels |
| `--sm-bg-sunken` | `#EEF1F5` | Inputs, table headers, canvas frame |
| `--sm-bg-nav` | `#0F1B2D` | Left navigation |
| `--sm-border` | `#DCE1E8` | Default borders |
| `--sm-text` / `--sm-text-muted` / `--sm-text-on-dark` | `#111827` / `#5B6576` / `#E6EAF0` | Text |

**Accent and semantic colours.** Primary `--sm-accent: #2563EB` (hover `#1D4ED8`); secondary `--sm-accent-2: #0E7490` (teal, secondary series); `--sm-success: #15803D`; `--sm-warning: #B45309`; `--sm-danger: #B91C1C`; `--sm-info: #0369A1`. Text on these colours meets WCAG AA.

**Algorithm colours (charts).** Random `#6B7280` (grey), IPPO `#0E7490` (teal), MAPPO `#2563EB` (blue), Heuristic reference `#111827` dashed line.

**Robot identity colours (Okabe–Ito derived, CVD-safe, avoiding the semantic red/green).** R1 `#0072B2`, R2 `#D55E00`, R3 `#009E73`, R4 `#CC79A7`, R5 `#56B4E9`, R6 `#7A5195`. Robots are **always** labelled (R1…) and never distinguished by colour alone.

**Simulation layer tokens.**

| Token | Value | Rendering |
|---|---|---|
| `--sm-cell-free` | `#FFFFFF` | Known free cell |
| `--sm-cell-unknown` | `#CBD2DB` + 45° hatch (`#BCC4CF`, 1px, 6px pitch) | Fog of war |
| `--sm-cell-obstacle` | `#374151` | Solid |
| `--sm-cell-visited` | robot colour @ 10% alpha (last visitor), or `#E3EDFB` for team view | Visited tint |
| `--sm-grid-line` | `#E5E7EB` 1px | Cell borders (hidden when cell size < 14 px) |
| `--sm-target` | `#F2B600` fill, `#7A5A00` 1.5px outline, star shape | Known target; rescued → hollow grey star for 1 s, then removed |
| `--sm-target-hidden` (ground-truth view only) | star outline, dashed, 40% | Only in "Ground truth" debug layer, labelled |
| `--sm-sensor` | robot colour, fill 8% alpha, stroke 35% alpha 1px dashed | Sensor disc |
| `--sm-path` | robot colour, 2px, alpha fades 1.0 → 0.15 over last 30 steps | Trajectory |
| `--sm-comm-link` | `#64748B` 1px dotted | Mode C links |
| `--sm-collision` | `--sm-danger` ring | Collision flash |

### 30.4 Swarm Lab layout (desktop ≥ 1280 px)

```
┌──────┬───────────────────────────────────────────────────────────┬───────────────────┐
│ NAV  │ TOOLBAR  Policy[MAPPO · L2 · s3 · 3.0M ▾] Map[Test-042 ▾]   │                   │
│ 64/  │          ◀◀ ▶/❚❚ ▶| ↺   Speed[1× ▾]   Mode[Live|Replay]   ⚙ │  RIGHT RAIL 320px │
│ 220px├───────────────────────────────────────────────────────────┤  Tabs:            │
│      │ MISSION STRIP  t 84/150 │ Rescued 1/2 │ Coverage 63% │     │  [Robots][Live]   │
│      │                Collisions 2 │ ● RUNNING                   │  [Layers]         │
│      ├───────────────────────────────────────────────────────────┤                   │
│      │                                                           │  Robots: id, pos, │
│      │               SIMULATION CANVAS (square, fit)             │  state (fact),    │
│      │                                                           │  distance, idle % │
│      │                                                           │  Live: coverage   │
│      │                                                           │  C(t) sparkline,  │
│      ├───────────────────────────────────────────────────────────┤  reward components│
│      │ TIMELINE (replay) with event markers ◆ ★ ✕  /  event log   │  Layers: toggles  │
└──────┴───────────────────────────────────────────────────────────┴───────────────────┘
```

- **Scenario drawer** (⚙): level, grid size, robots, targets, density, comm mode, map seed. Opens over the right rail; it never shrinks the canvas.
- **Layer toggles:** Sensor radius · Paths · Visited cells · Unknown (fog) · Communication links · Ground truth (debug, off by default, labelled "Not visible to robots").
- **Keyboard:** `Space` play/pause, `→` step, `←` step back (replay), `R` reset, `1–6` focus robot, `L` toggle layers panel.
- **Robot focus:** clicking a robot highlights it, dims the others to 40%, and shows its **own observation window** outline (11×11). This lets users see exactly what that robot knows, which makes partial observability tangible.

### 30.5 Responsive behaviour

| Width | Layout |
|---|---|
| ≥ 1440 | Nav 220 + canvas + right rail 320 |
| 1024–1439 | Nav 64 (icons + tooltips) + canvas + right rail 300 |
| 768–1023 | Nav 64; right rail becomes a bottom tab sheet; toolbar wraps into two rows |
| < 768 | Single column; canvas full-width square; controls in a sticky bottom bar; analysis pages stack. Mobile is supported but not the primary target |

### 30.6 Other pages (key layout decisions)

- **Train Swarm:** left column holds the config form (algorithm, level, seed, total steps, lr, entropy, clip, rollout length, comm mode), each field with bounds and helper text. The right area has live small-multiple charts: success, coverage, targets rescued, collisions, episode return (raw faint + rolling mean), policy loss, value loss, entropy, approx-KL, explained variance. Plus a job status card with SPS and ETA computed from measured SPS. Only metrics that exist in `metrics.jsonl` are rendered.
- **Algorithm Comparison:** study selector → results table (mean ± std, 95% CI, n seeds) → dot-and-whisker plots per metric (per-seed dots + mean + CI) → learning curves (mean ± CI band, per-seed thin lines) → "Hypothesis status" panel quoting the pre-registered decision rules and the computed outcome, with no editorializing. The heuristic reference is shown as a labelled dashed line.
- **Swarm Analysis:** run + map selector; heatmap grid (visits/collisions); coverage-over-time; DER vs. coverage; discovery timeline; workload/idle bars; inter-robot distance; **Swarm Evolution** 2×2 synchronized replays (0/10/50/100%) with a shared timeline and per-panel mission strip.
- **Experiment History:** sortable, filterable table (status, algo, level, seed, steps, final val success, date), with row → run detail (config, manifest, curves, checkpoints, replays).
- **Settings:** API base URL, default playback speed, reduced-motion override, cell coordinate display, theme (light only in v1).

### 30.7 Micro-interactions (each communicates state)

| Interaction | Spec | Reduced motion |
|---|---|---|
| Robot movement | Tween between cells, duration = `min(frame_interval × 0.8, 160ms)`, ease-out | Instant |
| Blocked move | 2px nudge toward the attempted direction and back, 120ms | Static ✕ glyph for 1 frame |
| Collision | Danger ring expands 0 → 0.6 cell and fades, 400ms | Static ring for 1 frame |
| Target discovery | Star scales 0.6 → 1.0 with a single gold pulse ring, 500ms | Instant appear |
| Target rescue | Star → hollow grey, fade out over 800ms | Instant change |
| Sensor radius toggle | Opacity 0 ↔ 1, 150ms (CSS) | Instant |
| Comm link | Line opacity pulse once when a merge happens, 300ms | Static |
| Sim state change (running/paused/ended) | Status dot colour + label change; the mission strip flashes subtly on episode end, 200ms | Label only |
| Chart updates | Recharts animation ≤ 200ms, **disabled** for live-streaming series (avoids jitter) | Off |
| Panel/page transitions | 150ms fade/slide via CSS; Motion only where layout animation is needed | Off |

Nothing loops indefinitely except the "running" status dot, which pulses gently at 1.5 s.

### 30.8 Data-visualization rules

- Every chart has a title, axis labels **with units**, a legend (if more than one series), tooltips with exact values, and loading, empty and error states.
- The x-axis for learning curves is **environment steps**, formatted 0, 500k, 1M, ….
- Rates have fixed y-domains `[0, 100%]`. No truncated axes on bar charts; bars always start at 0. Dot plots may zoom, but show the axis break explicitly.
- Raw noisy measurements (faint, 1px) are shown alongside the rolling mean (solid, 2px). The rolling window size is stated in the caption.
- No dual y-axes. No 3D. No gradients inside data marks.
- Categorical colours come from tokens. Heatmaps use a sequential single-hue scale (light → `--sm-accent`) with a colour bar and numeric scale.
- Every chart offers "View data" (a table) for accessibility.

### 30.9 Accessibility

WCAG 2.2 AA contrast for text and essential graphics; focus-visible rings (`2px --sm-accent`); every control is keyboard reachable; `aria-live="polite"` on the mission strip for status changes (throttled); robots are distinguished by label + colour; `prefers-reduced-motion` is honoured; minimum 32 px hit targets in toolbars.

### 30.10 Design approval gate

Phase 9 ends with a design review. **The user approves the design system and the Swarm Lab spec before Phase 11 starts.**

---

## 31. Repository structure

```
swarmmind/
├── SWARMMIND_TECHNICAL_SPEC.md        # this document (authoritative)
├── README.md                          # quickstart, architecture diagram, results link
├── pyproject.toml                     # package "swarmmind" (marl + backend), deps, ruff, mypy, pytest
├── Makefile                           # setup, lint, test, test-slow, dev-backend, dev-frontend, study, report
├── .gitignore  .editorconfig  .python-version (3.11)
├── marl/
│   ├── __init__.py
│   ├── envs/
│   │   ├── core/
│   │   │   ├── config.py              # EnvConfig (pydantic), LevelRegistry
│   │   │   ├── world.py               # SwarmWorld: state, reset, step orchestration
│   │   │   ├── dynamics.py            # propose/terrain-check/no-follow conflict resolution
│   │   │   ├── sensing.py             # disc offsets, sensed sets
│   │   │   ├── belief.py              # per-robot beliefs, comm modes A/B/C
│   │   │   ├── rewards.py             # reward components (§13)
│   │   │   ├── observations.py        # actor obs + action mask (LOCAL ONLY)
│   │   │   ├── global_state.py        # critic global state (TRAINING ONLY)
│   │   │   ├── mapgen.py              # procedural generation + validation
│   │   │   ├── maps_io.py             # MapSpec, load/save, hashing, eval-set loading
│   │   │   └── types.py               # dataclasses: StepResult, WorldState, EpisodeTrace
│   │   ├── pettingzoo_env.py          # SwarmParallelEnv
│   │   └── vector.py                  # SyncVectorSwarm (batched, auto-reset, final obs)
│   ├── networks/
│   │   ├── encoders.py                # LocalEncoder, GlobalEncoder
│   │   ├── actor.py                   # ActorPolicy (+ masked categorical)
│   │   ├── critics.py                 # LocalCritic, CentralCritic
│   │   └── init.py                    # orthogonal init helpers
│   ├── algorithms/
│   │   ├── common/
│   │   │   ├── policy.py              # Policy protocol; load_actor(); act()
│   │   │   ├── distributions.py       # MaskedCategorical
│   │   │   ├── gae.py                 # GAE + truncation folding
│   │   │   ├── ppo_update.py          # losses, minibatching, diagnostics
│   │   │   └── value_norm.py
│   │   ├── ippo/agent.py              # builds critic_input=local
│   │   ├── mappo/agent.py             # builds critic_input=agent_specific_global
│   │   ├── baselines/random_policy.py
│   │   ├── baselines/frontier_heuristic.py
│   │   └── registry.py
│   ├── buffers/rollout_buffer.py
│   ├── training/
│   │   ├── trainer.py                 # algorithm-agnostic loop (§15.9)
│   │   ├── checkpoint.py              # save/load/resume, atomic writes, actor export
│   │   ├── run_dir.py                 # run dir + manifest lifecycle
│   │   ├── logger.py                  # JSONL metrics logger
│   │   └── curriculum.py              # (stretch)
│   ├── evaluation/
│   │   ├── evaluator.py               # run_episode(), evaluate_checkpoint()
│   │   ├── eval_sets.py               # frozen map sets, isolation guard
│   │   └── stats.py                   # t-CI, Welch, Hedges g, bootstrap, AUC
│   ├── analysis/
│   │   ├── metrics.py                 # §23 from EpisodeTrace
│   │   ├── coordination.py            # §24 incl. isolation null
│   │   ├── heatmaps.py
│   │   ├── aggregate.py               # study-level aggregation → JSON
│   │   └── figures.py                 # matplotlib figures for report
│   ├── replay/{schema.py, recorder.py, loader.py}
│   └── utils/{seeding.py, config_loader.py, paths.py, versions.py, hashing.py}
├── configs/
│   ├── env/{L1.yaml, L2.yaml, L3.yaml, L4.yaml}
│   ├── algo/{ippo.yaml, mappo.yaml, random.yaml, heuristic.yaml}
│   ├── experiments/{smoke.yaml, main_L2.yaml, scale_L3.yaml, comm_L4.yaml}
│   └── maps/{validation/, test/, structured/}   # frozen, committed
├── scripts/
│   ├── train.py  evaluate.py  run_study.py  aggregate_results.py
│   ├── make_eval_maps.py  record_replays.py  bench_env.py  export_actor.py
│   └── export_schemas.py                        # JSON Schema/OpenAPI for TS types
├── backend/app/ …                                # §28
├── frontend/ …                                   # §29 (package.json, vite.config.ts, src/)
├── tests/
│   ├── env/  marl/  evaluation/  analysis/  replay/  backend/  integration/
│   ├── test_import_boundaries.py
│   └── conftest.py                               # tiny configs, fixed seeds, fixtures
├── models/                                       # curated exported actors for the demo + registry.json
├── results/                                      # runs/, studies/, replays/ (gitignored except study summaries)
└── docs/
    ├── ARCHITECTURE.md  EXPERIMENTS.md (pre-registration)  RESULTS.md (generated)
    ├── design/{DESIGN_SYSTEM.md, UI_SPEC.md, screens/}
    └── phase_reports/PHASE_<n>.md                # copy of each completion report
```

---

## 32. Configuration strategy

- **Format:** YAML files validated into **pydantic v2** models (`EnvConfig`, `AlgoConfig`, `TrainConfig`, `ExperimentConfig`). The same models are reused by FastAPI.
- **Composition:** an experiment YAML references `env: L2` and `algo: mappo` plus overrides. CLI overrides use `--set algo.lr_actor=1e-4` (dotted keys, type-checked). No Hydra; the loader is about 100 lines.
- **Validation:** ranges (e.g. `0 ≤ obstacle_density ≤ 0.35`, `n_robots ≤ 6`, `sensor_radius ≥ 1`); cross-field checks (the robots + targets fit in free space; `obs_radius ≥ sensor_radius`; training configs cannot reference test maps).
- **Resolved config** is written to every run dir, with `config_hash = sha256(canonical_json(resolved))[:8]` embedded in the run name.
- **Frozen study configs** (Phase 6) live in `configs/experiments/` and are referenced by `docs/EXPERIMENTS.md`. Changing a frozen file after Phase 7 starts is detectable by hash.
- No secrets exist in this project. `.env` is used only for `VITE_API_BASE_URL` and backend host/port.

---

## 33. Testing strategy

### 33.1 Organization and commands

- pytest markers: `unit` (default, < 60 s total), `slow` (learning smoke tests, minutes), `integration`.
- `make test` runs unit + integration. `make test-slow` runs the learning smoke tests. `npm test` runs Vitest. `npm run build` runs the type check and mock guard.
- Coverage targets: `marl/envs` ≥ 90% lines; `marl/algorithms`, `buffers`, `evaluation`, `analysis` ≥ 85%; backend ≥ 80%.

### 33.2 Environment tests (`tests/env/`)

Move into a free cell; wall/out-of-bounds bump; vertex conflict (both blocked); swap blocked; follow blocked; rotation cycle blocked; mover into a waiting robot; waiting robot never penalized; **fuzz: 10k random joint steps across 50 maps keep all positions unique and on free cells**; rescue marks, removes and credits; discovery precedes rescue; success termination; truncation at T_max; success precedence at T_max; deterministic `reset(seed)` (identical state hash); step determinism under identical actions; mapgen connectivity/reachability/min-distance/no-overlap/density bounds; retry determinism; `MapGenerationError` on infeasible configs; eval-hash rejection; obs shapes/dtypes/ranges; ch 0–2 partition; self at the window centre; out-of-bounds padding; counterfactual leak tests (§12.4); mask correctness and locality; comm modes A/B/C; reward invariants (exploration budget; no double counting on simultaneous discovery; team components identical across agents; collision penalty only on blocked movers); `pettingzoo.test.parallel_api_test`; PettingZoo ↔ vector wrapper parity; global state contains hidden targets (critic sanity).

### 33.3 MARL tests (`tests/marl/`)

GAE vs. hand-computed values for 3 cases (termination mid-rollout, truncation mid-rollout, rollout-end bootstrap); truncation folding with ValueNorm; no GAE leakage across agents (a unique reward on one agent influences only that agent's advantages); advantage normalization ignores inactive samples; PPO clip: zero gradient when `ρ` is outside `[1−ε,1+ε]` in the clipped direction; masked categorical (zero probability on masked actions, finite entropy, correct renormalization); stored-mask replay (log-probs at update equal those at collection before any gradient step); buffer layout/order; ValueNorm update/normalize/denormalize round-trip; checkpoint save/load round-trip; **resume equivalence** (bitwise, CPU); actor-only inference; actor import boundary; **IPPO ≡ MAPPO(critic=local) bitwise parity**; MAPPO critic input shapes for N ∈ {2,3,4} and map sizes 8/10/12.

**Learning smoke tests (slow, fixed seeds, generous thresholds):**
(a) 6×6, 1 robot, 1 target, ρ=0: IPPO success ≥ 0.9 within 200k env steps.
(b) The same for MAPPO.
(c) L1: validation success increases by ≥ 0.2 absolute from the 0% checkpoint within 500k steps (both algorithms).

### 33.4 Evaluation and analysis tests

Every metric in §23/§24 is checked against a hand-crafted EpisodeTrace with known answers (DER, Gini, oscillation, idle, redundant pursuit, DER@50); evaluation reproducibility (same checkpoint + seeds → identical JSON); random baseline determinism; eval-set immutability (the hashes of files in `configs/maps/` match the recorded manifest); statistics functions against scipy reference values; the aggregation script on a synthetic study directory.

### 33.5 Integration tests

- Model → env: load an exported actor and run 5 episodes on L1 without error; metrics are within valid ranges.
- Backend: FastAPI `TestClient` for every endpoint (happy path + validation errors + path traversal attempts); WebSocket session stream; **backend ↔ evaluator parity**.
- Replay: record → load → re-simulate from actions gives identical frames.
- Frontend: Vitest component tests with fixture data *from real recorded files* (copied into `tests/fixtures/` after Phase 7); TS types regenerated from OpenAPI compile cleanly (contract test); E2E smoke in Phase 14 (browser automation opens Swarm Lab, starts a session, verifies frames advance).

---

## 34. Reproducibility

- `marl/utils/seeding.py`:
  - `seed_everything(seed)` sets `random.seed`, `np.random.seed` (legacy global, for third-party code only), `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `PYTHONHASHSEED` (documented; it only takes effect for subprocesses).
  - `make_streams(seed) -> SeedStreams` uses `np.random.SeedSequence(seed).spawn(k)` to create independent streams: per-env map seeds, per-env dynamics RNG (currently unused because dynamics are deterministic; reserved), a torch policy-sampling `Generator`, and network init.
- The environment uses only its own `np.random.Generator`. **Global numpy RNG in env code is forbidden** (enforced by a grep test).
- Gymnasium action-space seeding: `action_space(agent).seed(derived)` in the PettingZoo wrapper, used only by random-sampling utilities.
- Optional `deterministic: true` sets `torch.use_deterministic_algorithms(True)` and `torch.backends.cudnn.benchmark=False`. CPU runs are bitwise reproducible (tested). CUDA runs may differ in low-order bits (documented; reported variance comes from seeds, not from nondeterminism).
- Every run records software versions and the git commit. `pyproject.toml` pins major/minor versions, and a lock file is generated (`pip freeze > requirements.lock` or `uv lock`).
- Results are produced only by scripts. `docs/RESULTS.md` is **generated** by `scripts/aggregate_results.py` and never hand-edited.

---

## 35. Computational requirements

> All numbers are **estimates to be replaced by measurements** from `scripts/bench_env.py` (Phase 3) and the first training runs (Phase 4). They are recorded in `docs/EXPERIMENTS.md`.

| Component | Estimate (Apple M-series / modern x86 laptop CPU) |
|---|---|
| Env throughput, single process (L2, N=3) | 3k–10k env steps/s (pure NumPy; per-step sensing uses precomputed disc offsets) |
| IPPO training throughput (CPU, E=16) | ≈ 1.5k–3k env steps/s including updates |
| MAPPO | ≈ 20–40% slower than IPPO (global encoder, larger buffer) |
| One L1 run (1 M steps) | 6–15 min CPU |
| One L2 run (3 M steps) | 20–45 min CPU; 15–30 min with a consumer CUDA GPU |
| **Main study** (L2: IPPO + MAPPO × 5 seeds × 3 M) | 30 M env steps ≈ 3–7 h sequential CPU; ≈ 1–2 h with 4–6 parallel single-thread processes |
| Evaluation (200 episodes × 10 runs × test sets) | Minutes |
| L3 study (6 M × 10 runs) | 2–3× the L2 study (optional) |
| Memory | < 1.5 GB per training process |
| Disk | ≈ 5–15 MB per run (checkpoints) + replays; the study total is < 2 GB |

**If compute is limited, drop in this order:** L3 study → G3/G4/G5 → mode A/H5 → reduce seeds from 5 to 3 (report the reduced power) → reduce L2 budget to 2 M. **Never drop:** Random baseline, the L2 IPPO vs. MAPPO comparison, held-out test evaluation, ≥ 3 seeds.

---

## 36. Failure modes and debugging

| Failure | Symptoms | Debugging strategy |
|---|---|---|
| **Agents never learn** | Validation success flat at random level; entropy stays at max | (1) Run smoke test (a) (1 robot, open map). If that fails, the bug is in the PPO core, not MARL. (2) Check reward sign/scale and per-component logs. (3) Check that advantages are non-zero, logp_old matches logp_new before the first gradient step, and the mask is applied at update. (4) Overfit test: a single fixed map should reach ~100% success. (5) Check that explained variance rises above 0 |
| **Reward hacking** | Return ↑, success ↔ or ↓; a single component dominates | Inspect the component breakdown; watch replays of the highest-return episodes; check invariants (§13.3). Fix the reward *design* rather than tuning weights blindly. **Escalate to Opus** |
| **Agents cluster** | High DER, low inter-robot distance, many R2 conflicts | Check teammates are visible (ch 6) and team-visited information (ch 5) under mode B; check clustered spawn dispersion speed; compare with the isolation null; examine whether collision penalty is too weak or strong |
| **Agents ignore targets** | High coverage, low rescue, long rescue latency | Verify target channel (ch 7) is populated after discovery; verify rescue reward fires; check that the exploration budget is below the rescue value |
| **One agent does all the work** | Gini high, some robots idle | Per-robot idle stats by spawn position; test β=0.5 mixed reward ablation (equal for both algorithms); check agent-ID not accidentally enabled |
| **Policy collapse** | Entropy → 0 early; success drops; KL spikes | Lower LR, raise entropy coefficient, enable target_kl early stop, reduce epochs; check for NaN from masked logits |
| **Training instability** | Oscillating curves; value loss explodes | ValueNorm on? Grad clipping applied? Check reward scale, advantage normalization, critic LR |
| **Poor generalization** | Train/val gap large, G2 drop large | Check the train map diversity (seed stream actually varies); ensure no accidental fixed map; use a larger budget; report honestly |
| **MAPPO does not outperform IPPO** | CIs overlap | **Not a bug by itself** (§16.7). Verify the parity test, the critic sanity test and explained variance (MAPPO's critic should fit better). Report as a finding. **Only escalate if** MAPPO is *worse* and its explained variance is *lower* than IPPO's, which suggests a critic bug |
| **Critic leaks into execution** | Pre-discovery target bias; implausibly fast discovery | Run §12.4 tests and the statistical probe; inspect the actor's import graph; **escalate to Opus immediately** |
| **Evaluation contamination** | Test ≥ validation performance consistently; suspiciously high generalization | Verify eval-hash rejection; check the manifest log of test-set access; regenerate a fresh test set with a new seed range to confirm; **escalate** |
| **Truncation bootstrap bug** | Value at t≈T_max unrealistic; performance drops as T_max changes | GAE unit tests with truncation; plot value vs. time |
| **Symmetric deadlock** | Greedy eval shows oscillation/R2 repeat | Use stochastic eval (primary); report the greedy numbers separately |
| **Slow training** | SPS far below estimate | Profile with `bench_env.py`; vectorize sensing via precomputed offsets; set torch threads = 1 per process; avoid per-step Python dict conversions (use the vector wrapper) |

---

## 37. Security and reliability

- **Local-first:** the backend binds to `127.0.0.1` by default, and CORS allows only the configured dev origin.
- **Path safety:** every ID (`run_id`, `replay_id`, `model_id`, `map_id`) is validated with a strict regex (`^[A-Za-z0-9._-]{1,128}$`). Paths are resolved and required to stay within `RESULTS_DIR` or `MODELS_DIR` (`Path.resolve().is_relative_to`). Path traversal tests are included.
- **Safe deserialization:** `torch.load(..., weights_only=True)`; only files from the registry are loaded; no user-uploaded checkpoints in v1.
- **Resource limits:** max live sessions (4), session idle TTL (10 min), max `n` steps per request (50), WebSocket rate cap (30 frames/s), max 1 training job, and bounded training parameters (e.g. ≤ 10 M steps from the UI).
- **Subprocesses:** the training job launches with an argument list (no shell=True), with a fixed script path and validated arguments only.
- **Robustness:** atomic writes for manifests and checkpoints; the run index tolerates partial or corrupt runs (marks them `invalid`, never crashes); the frontend handles backend-unavailable states gracefully.
- **Integrity:** eval map hash manifest; study results carry the list of run IDs and config hashes they aggregate.

---

## 38. Detailed implementation phases

Each phase follows the protocol in §44. "Skills/connectors" lists capabilities **observed in the Phase 0 environment on 2026-09-23**. At the start of every phase they **MUST be re-verified**, and anything unavailable is skipped without claiming it was used.

---

### PHASE 0 — System architecture ✅ (this document)

- **Output:** `SWARMMIND_TECHNICAL_SPEC.md`. Model: Opus 5.5, High. No implementation. **STOP.**

---

### PHASE 1 — Repository + tooling

- **OBJECTIVE:** A clean, runnable skeleton with reproducibility utilities, config validation, testing and lint infrastructure, and minimal backend/frontend shells. **No ML or env logic.**
- **DEPENDENCIES:** Phase 0.
- **FILES/MODULES:** `pyproject.toml`, `Makefile`, `.gitignore`, `.python-version`, `README.md`, `marl/__init__.py`, `marl/utils/{seeding,config_loader,paths,versions,hashing}.py`, `marl/envs/core/config.py` (EnvConfig/levels schema only), `configs/env/L1–L4.yaml`, `configs/algo/*.yaml` (values from §15.8), `backend/app/{main.py,settings.py,api/routers/health.py}`, `frontend/` (Vite React TS app with AppShell, routes to empty pages, `design/tokens.css` placeholder), `tests/{conftest.py,test_seeding.py,test_config.py,test_import_boundaries.py}`, `tests/backend/test_health.py`, `docs/ARCHITECTURE.md` (short; links to the spec).
- **IMPLEMENTATION TASKS:**
  1. Create a Python **3.11** venv (`python3.11 -m venv .venv`). Do not use the system default Python if it is 3.13+/3.14, where torch/pettingzoo wheels may be missing.
  2. Pin dependencies: numpy, torch (CPU wheel is fine), pettingzoo, gymnasium, pydantic v2, pyyaml, fastapi, uvicorn, pandas, scipy, matplotlib; dev: pytest, pytest-cov, ruff, mypy, httpx.
  3. Implement seeding and SeedStreams (§34), the config loader with `--set` overrides and validation, and the versions collector.
  4. FastAPI `GET /api/health`.
  5. Frontend: `npm create vite` (react-ts), add bootstrap, react-router, zustand, recharts, lucide-react; AppShell with nav items (§30.2) and placeholder pages; a health indicator calling the backend. On macOS, if npm stalls, use `NODE_OPTIONS=--no-network-family-autoselection`. Use npm, not pnpm.
  6. `git init`. Propose the first commit to the user; commit only with approval.
- **TESTS:** seeding reproducibility (identical sequences for random/numpy/torch/SeedStreams); config validation (valid levels load, invalid values rejected, `--set` overrides typed); import-boundary scanner runs (trivially passing); health endpoint; `npm run build` and `tsc --noEmit`; `ruff check`, `mypy marl/utils`.
- **VALIDATION GATE:** `make lint test` passes; `npm run build` passes; the backend serves `/api/health` and the frontend shell displays "backend: ok"; no ML code present.
- **EXPECTED OUTPUT:** A skeleton repository; a Phase 1 report.
- **KNOWN RISKS:** Python version/wheel mismatch; npm network stall; accidental over-building (implementing env code early).
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** Medium.
- **SKILLS/CONNECTORS:** built-in browser preview (`mcp__Claude_Browser__preview_start`) to verify the shell renders; nothing else needed.
- **OPUS ESCALATION:** none expected.
- **STOP CONDITION:** Gate passes → completion report → STOP.

---

### PHASE 2 — Multi-agent environment

- **OBJECTIVE:** A correct, deterministic, fully tested `SwarmWorld` core with PettingZoo and vector wrappers, observations, global state, masks, rewards, comm modes A/B (C optional) and procedural maps.
- **DEPENDENCIES:** Phase 1.
- **FILES/MODULES:** `marl/envs/core/{world,dynamics,sensing,belief,rewards,observations,global_state,mapgen,maps_io,types}.py`, `marl/envs/{pettingzoo_env,vector}.py`, `tests/env/*`.
- **IMPLEMENTATION TASKS:** Implement §6 (step order, no-follow rule), §8 (8-channel obs, 10-dim vec), §9 (9-channel global state + vec), §10 (mask), §11 (sensing, belief), §13 (rewards with exploration payout, per-component output), §19 modes A/B (C if time permits and cheap), §20.1–20.2 (mapgen, MapSpec, hashing), §7 (both wrappers, auto-reset with final obs), EpisodeTrace emission, and ASCII render.
- **TESTS:** All of §33.2, including the counterfactual leak tests, `parallel_api_test`, wrapper parity, reward invariants and fuzz.
- **VALIDATION GATE:** All env tests pass; `marl/envs` coverage ≥ 90%; `parallel_api_test` passes; fuzz (10k steps) passes; leak tests pass; single-process throughput measured and reported (no threshold, but < 1k env steps/s must be flagged).
- **EXPECTED OUTPUT:** A usable environment; ASCII renders of 3 sample maps included in the report.
- **KNOWN RISKS:** Subtle conflict-rule bugs; off-by-one in the egocentric window; x/y transposition; rewards double-counting simultaneous discovery; auto-reset returning the wrong final observation.
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** High.
- **SKILLS/CONNECTORS:** `engineering:testing-strategy` (optional, for the test plan); none required.
- **OPUS ESCALATION:** only for a *conceptual* issue with observation design, transition semantics, partial observability or reward methodology (e.g., the spec proves contradictory or an invariant cannot hold). Not for failing tests or bugs.
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 3 — Validation + random baseline + metrics + replay foundation

- **OBJECTIVE:** Prove the environment is sane and solvable, create the frozen evaluation map sets, implement the baselines, the core episode metrics and the replay recorder/loader, and calibrate `max_steps`.
- **DEPENDENCIES:** Phase 2.
- **FILES/MODULES:** `marl/algorithms/baselines/{random_policy,frontier_heuristic}.py`, `marl/algorithms/common/policy.py` (Policy protocol), `marl/analysis/metrics.py` (§23), `marl/replay/{schema,recorder,loader}.py`, `marl/evaluation/{evaluator,eval_sets}.py` (episode runner, no learned policies yet), `scripts/{make_eval_maps,bench_env,evaluate}.py`, `configs/maps/{validation,test,structured}/`, `tests/{evaluation,analysis,replay}/*`.
- **IMPLEMENTATION TASKS:**
  1. Stress tests: 1k procedural maps per level (generation success rate, density distribution, reachability).
  2. Generate and freeze validation/test sets and 10–20 structured maps, with a hash manifest.
  3. Random policy; frontier heuristic.
  4. Metrics from EpisodeTrace.
  5. Replay recorder, schema v1, loader, re-simulation determinism check.
  6. `bench_env.py`.
  7. **Calibration:** run heuristic + random on validation sets for L1–L3 → set `max_steps` by §20.6 → write a provisional `docs/EXPERIMENTS.md` section "Calibration" including the random baseline numbers and the **Phase 4 learning gate thresholds** (derived: L1 IPPO final validation success ≥ max(0.85, random + 0.25) and censored mean steps ≤ 0.7 × random's).
- **TESTS:** §33.4 metric tests; random determinism; eval-set immutability; replay round-trip and re-simulation; mapgen stress assertions.
- **VALIDATION GATE:** Heuristic success ≥ 0.95 on each level's validation set within the calibrated `max_steps`; random success on L2 < 0.6 (else adjust the level per §20.6 and document); all metric tests pass; the replay of a random episode loads and re-simulates identically; map set hashes are committed.
- **EXPECTED OUTPUT:** Calibration table (random vs. heuristic per level), frozen map sets, env benchmark numbers, sample replay files.
- **KNOWN RISKS:** Heuristic too strong or weak for meaningful T_max; metric definitional errors; replay size bloat.
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** Medium.
- **SKILLS/CONNECTORS:** `dataviz` skill (only if producing calibration plots); `data:statistical-analysis` (optional).
- **OPUS ESCALATION:** if calibration shows the task is degenerate (random nearly always succeeds, or heuristic cannot succeed at any reasonable T_max) and a level redesign affecting methodology is needed.
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 4 — PPO core + IPPO

- **OBJECTIVE:** A correct, algorithm-agnostic PPO training stack (buffer, GAE, losses, ValueNorm, trainer, checkpoint/resume, actor export, validation evaluation), instantiated as IPPO, that demonstrably learns on L1.
- **DEPENDENCIES:** Phase 3.
- **FILES/MODULES:** `marl/networks/*` (actor, LocalEncoder, LocalCritic), `marl/algorithms/common/{distributions,gae,ppo_update,value_norm,policy}.py`, `marl/algorithms/ippo/agent.py`, `marl/buffers/rollout_buffer.py`, `marl/training/{trainer,checkpoint,run_dir,logger}.py`, `scripts/{train,export_actor}.py`, `configs/experiments/smoke.yaml`, `tests/marl/*`.
- **IMPLEMENTATION TASKS:** §15 in full; the evaluator supports learned checkpoints; stochastic/greedy modes; periodic validation; checkpoint schedule including 0%; resume; manifest; SPS logging; device selection and a one-off CPU vs. MPS/CUDA micro-benchmark.
- **TESTS:** §33.3 unit tests (GAE, truncation, masking, clipping, buffer, ValueNorm, checkpoint, resume equivalence, actor-only inference, import boundary); smoke tests (a) and (c) for IPPO.
- **VALIDATION GATE:** All unit tests pass; smoke (a) passes; **L1 IPPO, 3 seeds × 1 M steps: at least 2/3 seeds meet the pre-written Phase 3 learning thresholds**; 0% checkpoint ≈ random (CIs overlap); measured SPS recorded.
- **EXPECTED OUTPUT:** Learning curves (validation success, return components, entropy, KL, explained variance) for L1; an exported actor; a replay of a trained L1 episode.
- **KNOWN RISKS:** GAE/truncation bugs; mask mismatch; silent shape broadcasting (e.g., `[B]` vs. `[B,1]` in value loss); slow SPS due to dict conversions.
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** High.
- **SKILLS/CONNECTORS:** `code-review` skill at low/medium on the PPO core diff before closing the phase (cheap self-check); `dataviz` for curves.
- **OPUS ESCALATION:** smoke test (a) fails after the §36 "agents never learn" checklist has been worked through; GAE/PPO mathematics appear inconsistent with the spec; policy collapse with no identifiable implementation cause. **Do not escalate** for shape errors, import errors or failing unit tests with clear causes.
- **STOP CONDITION:** Gate passes (or a documented FAIL with an escalation recommendation) → report → STOP.

---

### PHASE 5 — MAPPO (centralized critic, CTDE)

- **OBJECTIVE:** Implement the centralized agent-specific critic and MAPPO on top of the shared PPO core, prove the CTDE information boundaries, prove IPPO/MAPPO differ only in critic input, and demonstrate learning on L1.
- **DEPENDENCIES:** Phase 4.
- **FILES/MODULES:** `marl/networks/{encoders.GlobalEncoder, critics.CentralCritic}`, `marl/algorithms/mappo/agent.py`, buffer extension for state storage + agent-specific channel construction, `tests/marl/test_mappo_*.py`, `tests/marl/test_parity_ippo_mappo.py`, `marl/analysis/leak_probe.py`.
- **IMPLEMENTATION TASKS:** §16 in full (critic, AS state construction at batch time, ValueNorm usage, masking); the **parity test**; critic sanity test; statistical leak probe (pre-discovery target-direction bias) run on L1-trained MAPPO and IPPO; review against the §16.8 mistakes checklist, recording each item as checked in the report.
- **TESTS:** Parity (bitwise, 2 updates, CPU); critic input shapes across N and map sizes; critic sees global; all §12.4 tests; smoke tests (b) and (c) for MAPPO.
- **VALIDATION GATE:** All tests pass; the leak probe is negative for both algorithms; **L1 MAPPO, 3 seeds × 1 M: ≥ 2/3 seeds meet the same learning thresholds as IPPO**; MAPPO critic explained variance reported alongside IPPO's.
- **EXPECTED OUTPUT:** MAPPO training on L1; a side-by-side preliminary L1 curve (clearly labelled *preliminary, not the study*); the completed §16.8 checklist.
- **KNOWN RISKS:** Actor accidentally receiving state; agent-specific channel misaligned with agent order; memory blow-up storing state per agent; critic LR imbalance.
- **RECOMMENDED MODEL:** **Opus 5.5.** **EFFORT:** High.
- **SKILLS/CONNECTORS:** `code-review` (high) on the MAPPO diff.
- **OPUS ESCALATION:** already Opus. If Opus identifies a need to change the spec's CTDE design, it documents the change as a spec amendment in the report and asks the user.
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 6 — Experiment infrastructure

- **OBJECTIVE:** Multi-seed study orchestration, final test evaluation, aggregation with statistics, result persistence, and the **pre-registration** of the study.
- **DEPENDENCIES:** Phase 5.
- **FILES/MODULES:** `scripts/{run_study,aggregate_results,record_replays}.py`, `marl/evaluation/stats.py`, `marl/analysis/{aggregate,figures}.py`, `configs/experiments/{main_L2,scale_L3,comm_L4}.yaml`, `docs/EXPERIMENTS.md` (final pre-registration), `results/studies/<study>/` layout, `tests/evaluation/test_stats.py`, `tests/evaluation/test_aggregate.py`.
- **IMPLEMENTATION TASKS:**
  1. `run_study.py`: expands (algos × seeds × levels) into jobs; runs them with a `--parallel k` process pool (each process with 1 torch thread); skips completed runs (idempotent); resumes interrupted ones; writes `study_manifest.json`.
  2. Final test evaluation for each completed run (plus Random with 5 sampling seeds and the heuristic reference).
  3. Aggregation → `results/studies/<study>/summary.json` (per-metric per-algo mean/std/CI, pairwise differences with Welch CI and Hedges' g, learning curves on a common env-step grid, hypothesis decisions per §4) → generate `docs/RESULTS.md` and figures.
  4. Evolution replay recording (0/10/50/100%).
  5. **Freeze and pre-register** in EXPERIMENTS.md: levels, budgets, seeds, `max_steps`, hyperparameters, primary metrics, decision rules, deviations policy.
  6. Dry run: `smoke` study (L1, 2 seeds, 200k steps) end-to-end.
- **TESTS:** Stats functions vs. scipy; aggregation on a synthetic study; idempotent skip/resume; study manifest integrity (config hashes).
- **VALIDATION GATE:** The smoke study runs end-to-end unattended and generates summary.json, RESULTS.md (marked SMOKE) and figures; EXPERIMENTS.md is frozen and committed (with user approval).
- **EXPECTED OUTPUT:** Study tooling; pre-registration; the exact commands for Phase 7.
- **KNOWN RISKS:** Process-pool deadlocks on macOS (use `spawn` start method); disk bloat; accidental test-set use before training completes.
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** Medium.
- **SKILLS/CONNECTORS:** `data:statistical-analysis` (to review the stats procedure), `dataviz` (figures).
- **OPUS ESCALATION:** if the statistical design itself needs changing (e.g., the unit of analysis or pairing cannot be implemented as specified).
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 7 — Controlled experiments (Random vs. IPPO vs. MAPPO)

- **OBJECTIVE:** Execute the pre-registered main study on L2 (and L1 as a secondary level), then evaluate, aggregate and assess the hypotheses honestly.
- **DEPENDENCIES:** Phase 6 (frozen pre-registration).
- **FILES/MODULES:** No new ML code expected. Outputs go in `results/studies/main_L2/`, `docs/RESULTS.md` and `docs/phase_reports/PHASE_7.md`.
- **IMPLEMENTATION TASKS:**
  1. Launch `run_study.py --study main_L2 --parallel <cores−1>` **as a background process from the terminal**. Claude MUST NOT keep long runs in its own context; it checks progress periodically (e.g., reading `study_manifest.json`).
  2. After completion: test evaluation → aggregation → figures.
  3. Fill the §22.4 "genuine learning" checklist, the §24.4 coordination criteria (partial; complete in Phase 8), and the hypothesis decisions (H1/H2 at minimum).
  4. **Sanity checklist** (any failure → STOP and escalate):
     - Test success ≤ validation success + noise.
     - 0% checkpoint ≈ Random.
     - No method completes faster than the heuristic's lower envelope by an implausible margin.
     - Reward components are consistent with the success trend.
     - Leak probe negative on L2.
     - Per-seed variance is plausible (no seed identical to another).
- **TESTS:** Re-run the unit test suite (it must still pass); reproducibility spot check: re-evaluate one checkpoint and get identical JSON.
- **VALIDATION GATE:** All planned runs completed (or failures documented with cause); summary with CIs generated; all sanity checks pass; hypothesis statements are worded exactly according to §4.2; no post-hoc changes to frozen settings (or deviations documented).
- **EXPECTED OUTPUT:** Results table, learning curves, dot plots, hypothesis status, an honest discussion (including null results).
- **KNOWN RISKS:** Long wall time; temptation to tune on test; overclaiming.
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** Medium (orchestration and reporting). Use High only for the written interpretation if results are mixed.
- **SKILLS/CONNECTORS:** `data:statistical-analysis`, `dataviz`; the terminal tool to launch the background study if available.
- **OPUS ESCALATION (STOP and recommend Opus High/Extra High):** any sanity check fails; reward hacking is observed; MAPPO is significantly *worse* than IPPO with a lower critic explained variance; learning succeeds on L1 but completely fails on L2; results are unstable across seeds (e.g., bimodal). **Never** tune blindly until results improve.
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 8 — Behaviour and coordination analysis

- **OBJECTIVE:** Implement §24 and §26 analyses, produce the evidence for H3 and the coordination claims, and generate the analysis JSON consumed by the UI.
- **DEPENDENCIES:** Phase 7 results.
- **FILES/MODULES:** `marl/analysis/{coordination,heatmaps}.py` (+ extensions to metrics), `scripts/analyze_study.py`, `results/studies/<study>/analysis/*.json`, figures, `tests/analysis/*`.
- **IMPLEMENTATION TASKS:** DER, DER@50, DER-vs-coverage curve, visitation overlap, redundant pursuit, allocation optimality, inter-robot distance, dispersion speed, workload Gini, idle; the isolation null (strongly recommended); input-ablation evaluation (zero ch 5 / ch 6); heatmaps per showcase map; discovery timelines; H3 decision; completion of the §24.4 criteria.
- **TESTS:** Each coordination metric against hand-built traces; isolation-null runner determinism; ablation hook tests (the channel is actually zeroed and only at evaluation).
- **VALIDATION GATE:** All tests pass; an analysis JSON exists for every algorithm × showcase map; H3 and the coordination claim are stated per the §4/§24.4 rules; RESULTS.md is regenerated.
- **EXPECTED OUTPUT:** A coordination evidence section; figures; data for the UI.
- **KNOWN RISKS:** Confounding coverage with coordination (use DER@50); overinterpreting small differences.
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** Medium (definitions are fully specified).
- **SKILLS/CONNECTORS:** `dataviz`, `data:statistical-analysis`.
- **OPUS ESCALATION:** coordination metrics contradict the performance metrics in a way that suggests a metric or environment bug that Sonnet cannot localize; or a claim hinges on a subtle methodological interpretation.
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 9 — Professional product/UI design (design authority phase)

- **OBJECTIVE:** Produce the approved design system, information architecture, component inventory, interaction model, responsive rules and visualization specification. This is a design phase; the only code is tokens plus one static prototype.
- **DEPENDENCIES:** Phases 1, 6–8 (to know which real data exists). It can start in parallel with Phase 8 on the user's instruction.
- **FILES/MODULES:** `docs/design/DESIGN_SYSTEM.md`, `docs/design/UI_SPEC.md` (per page: purpose, data sources/endpoints, layout, components, states incl. loading/empty/error, interactions, keyboard, responsive), `frontend/src/design/{tokens.css,bootstrap-theme.css,typography.css}`, `docs/design/screens/*` (Figma frames or PNG exports of the prototype), `frontend/src/pages/_prototype/SwarmLabPrototype.tsx` (flagged mock data, removed or promoted in Phase 11).
- **IMPLEMENTATION TASKS:** Start from §30 (binding baseline) and refine it with a critique of the reference image; validate the colour contrast and the CVD-safety of the robot palette; specify every component's props/states; specify each chart per §30.8; build the Swarm Lab static prototype using tokens and a **real replay file** from Phase 7 (preferred over mocks); run a design critique pass; present to the user for approval.
- **TESTS:** Token contrast check script (WCAG AA for text pairs); prototype renders at 1440/1024/768/375 widths (screenshots).
- **VALIDATION GATE:** DESIGN_SYSTEM.md and UI_SPEC.md complete; tokens implemented; prototype screenshots at 4 widths; accessibility contrast checks pass; **explicit user approval** recorded.
- **EXPECTED OUTPUT:** An approved design package.
- **KNOWN RISKS:** Generic dashboard look; decorative excess; the simulator losing dominance.
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** High.
- **SKILLS/CONNECTORS (verify availability first):** Design authority skills: `hallmark` (anti-generic design audits), `design:design-system`, `color-system`, `typography-scale`, `spacing-system`, `layout-grid`, `component-spec`, `data-visualization` / `dataviz`, `motion-system`, `micro-interaction-spec`, `accessibility-audit`, `critique-visual-hierarchy`, `critique-information-density`, `critique-color`. **Figma:** the official Figma MCP tools were listed in the Phase 0 environment, but the `plugin:figma:figma` server reported that it **requires authentication**. Use Figma (`figma:figma-generate-design`, `figma:figma-generate-library`, after loading `figma:figma-use`) only if the user has authorized it; otherwise produce HTML/React prototype screenshots. **Do not** use generic generators (e.g., `ui-ux-pro-max`, `jezweb-landing-page`, `frontend-design`-style "make it modern" prompts) as the design authority. They MAY be consulted for isolated sub-questions only.
- **OPUS ESCALATION:** none (design is not an Opus task).
- **STOP CONDITION:** User approval → report → STOP.

---

### PHASE 10 — FastAPI + WebSocket backend

- **OBJECTIVE:** Implement §28 as a thin adapter over `marl`, with full tests and parity with the evaluator.
- **DEPENDENCIES:** Phases 3–8 (policies, replays, results exist).
- **FILES/MODULES:** `backend/app/**`, `scripts/export_schemas.py`, `tests/backend/*`.
- **IMPLEMENTATION TASKS:** All REST endpoints; session manager with TTL/limits; WS sim stream and WS training stream (tailing JSONL); model registry; run index; replay store; job runner (subprocess); security (§37); OpenAPI + JSON Schema export for TS types.
- **TESTS:** TestClient for every endpoint (success, validation, 404, traversal); WS protocol tests; **backend ↔ evaluator parity**; job runner (launch the smoke config, cancel it, status transitions); resource limits.
- **VALIDATION GATE:** All backend tests pass; coverage ≥ 80%; the parity test passes; OpenAPI schema is exported; `uvicorn` serves real models and real results.
- **EXPECTED OUTPUT:** A working API; `docs/API.md` (generated from OpenAPI with short notes).
- **KNOWN RISKS:** Duplicating env logic in routes; blocking the event loop with inference (use a threadpool); WS backpressure.
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** Medium.
- **SKILLS/CONNECTORS:** `security-review` skill at the end of the phase; `code-review` (medium).
- **OPUS ESCALATION:** only if parity cannot be achieved for reasons that implicate env/policy design (not plumbing).
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 11 — React simulator (Swarm Lab + replay)

- **OBJECTIVE:** Implement the approved Swarm Lab: layered canvas renderer, live sessions over WebSocket, replay playback with timeline/scrubbing, robot focus with observation-window outline, layer toggles, scenario drawer, keyboard control.
- **DEPENDENCIES:** Phases 9 (approved design), 10 (API).
- **FILES/MODULES:** `frontend/src/features/{simulation,replay}/**`, `stores/simStore.ts`, `pages/SwarmLab.tsx`, `api/{ws.ts,endpoints/sessions.ts,endpoints/replays.ts}`, `api/types.gen.ts`, tests.
- **IMPLEMENTATION TASKS:** frameReducer (delta accumulation + keyframes); renderers per layer; interpolation; WS client with reconnect; replay controls; policy/map/level selectors from real endpoints; loading/empty/error states; reduced motion; responsive per §30.5.
- **TESTS:** frameReducer against a real replay fixture (final state equals the replay outcome); renderer smoke tests with a mock canvas context; store tests; component tests for controls; browser verification (screenshots at 1440/1024/768/375; interactions: play, pause, step, scrub, toggle layers).
- **VALIDATION GATE:** Tests pass; `npm run build` clean (mock guard passes); live session with a real MAPPO model runs smoothly (≥ 50 fps rendering at 1× on L2, measured via the Performance API); a replay of a real test episode plays and scrubs correctly; screenshots included.
- **EXPECTED OUTPUT:** A working flagship page.
- **KNOWN RISKS:** Re-render storms (keep the canvas outside React state); DPI scaling blur (handle `devicePixelRatio`); coordinate transposition.
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** High.
- **SKILLS/CONNECTORS:** built-in browser tools (`mcp__Claude_Browser__*`: preview_start, screenshot, read_console_messages, resize_window) for verification; `emil-animate` / `animation-principles` for the motion items in §30.7; `make-interfaces-feel-better` (optional).
- **OPUS ESCALATION:** none expected.
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 12 — Analytics + training UI

- **OBJECTIVE:** Implement Train Swarm, Algorithm Comparison, Swarm Analysis (including Swarm Evolution) and Experiment History, backed only by real data.
- **DEPENDENCIES:** Phases 10, 11.
- **FILES/MODULES:** `frontend/src/features/{training,comparison,analysis,experiments}/**`, `pages/*`, `lib/stats.ts`, tests.
- **IMPLEMENTATION TASKS:** Live training charts via WS with polling fallback; job launch form with schema-driven bounds; comparison tables, dot plots and learning curves from `/studies`; hypothesis status panel quoting pre-registered rules; heatmaps; coverage/DER charts; discovery timeline; workload/idle; Swarm Evolution 2×2 synchronized replays; experiment table and run detail; "View data" tables for charts.
- **TESTS:** Chart components with real fixture JSON; empty/error/loading state tests; formatting utilities; the training-form validation mirrors the backend schema.
- **VALIDATION GATE:** Every chart renders from real endpoints; no mock imports outside `mocks/`; the chart rules in §30.8 are verified by a checklist in the report; screenshots included.
- **EXPECTED OUTPUT:** The complete analytics UI.
- **KNOWN RISKS:** Chart clutter; misleading axes; heavy JSON on analysis pages (paginate/downsample on the server).
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** Medium.
- **SKILLS/CONNECTORS:** `dataviz` skill (mandatory read before writing chart code), `data-visualization`, browser tools for verification.
- **OPUS ESCALATION:** none expected.
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 13 — UI polish

- **OBJECTIVE:** Consistency, accessibility and micro-interaction polish without new features.
- **DEPENDENCIES:** Phases 11–12.
- **FILES/MODULES:** Styles and components only; `docs/design/POLISH_CHECKLIST.md`.
- **IMPLEMENTATION TASKS:** Token compliance audit (no raw hex/px outside tokens); spacing/typography consistency; focus states; reduced motion; micro-interaction timings per §30.7; empty/error copy; responsive fixes.
- **TESTS:** Grep check for raw colour/px literals outside `design/`; accessibility audit (axe via browser, or manual checklist); screenshots before and after.
- **VALIDATION GATE:** The token audit passes; there are no AA contrast failures; every page is keyboard navigable; the before/after screenshots are attached.
- **EXPECTED OUTPUT:** A polished UI.
- **KNOWN RISKS:** Scope creep into new features; over-animation.
- **RECOMMENDED MODEL:** **Haiku** for token/style replacements and repetitive fixes; **Sonnet** where interaction logic changes. **EFFORT:** Medium (Haiku) / Medium (Sonnet).
- **SKILLS/CONNECTORS:** `accessibility-audit`, `critique-typography`, `critique-composition`, `critique-color`, `jezweb-design-review`, `hallmark` (audit mode); browser tools.
- **OPUS ESCALATION:** never for polish.
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 14 — End-to-end integration

- **OBJECTIVE:** Prove that the whole system works together on a clean checkout, using only real data, and that the documentation is complete.
- **DEPENDENCIES:** All previous phases.
- **FILES/MODULES:** `tests/integration/*`, optional `frontend/e2e/*` (Playwright), `README.md` (quickstart, demo script), `docs/ARCHITECTURE.md` (final), `scripts/demo_setup.sh`.
- **IMPLEMENTATION TASKS:** Fresh-clone setup test (venv + npm install + `make test`); E2E flows (open Swarm Lab → live MAPPO session → replay → compare page → analysis → launch the smoke training job → watch live metrics → cancel); model → env → backend → frontend parity check (the frame shown equals the evaluator frame for the same seed); real-data audit (no mocks; every number traceable to a results file); performance check.
- **TESTS:** The full Python suite, including slow tests; Vitest; E2E smoke; `check_no_mocks`.
- **VALIDATION GATE:** Everything passes on a fresh clone following the README exactly; the real-data audit table is complete; the demo script runs in < 10 minutes.
- **EXPECTED OUTPUT:** A demo-ready system.
- **KNOWN RISKS:** Environment-specific setup issues (Python version, Node network stall); stale generated types.
- **RECOMMENDED MODEL:** Sonnet. **EFFORT:** High.
- **SKILLS/CONNECTORS:** `run` skill, built-in browser tools, `code-review` (high), `security-review`; GitHub via the `gh` CLI only if the user wants to push and it is authenticated (`plugin:engineering:github` reported **requires authentication** in Phase 0).
- **OPUS ESCALATION:** an integration discrepancy implicating model/env parity that Sonnet cannot localize.
- **STOP CONDITION:** Gate passes → report → STOP.

---

### PHASE 15 — Final MARL + system audit

- **OBJECTIVE:** An independent, adversarial audit of correctness, methodology, claims and engineering. **No new features.**
- **DEPENDENCIES:** Phase 14.
- **FILES/MODULES:** `docs/AUDIT.md` (findings with severity, evidence and fix status); fixes limited to defects.
- **AUDIT CHECKLIST:** environment correctness (re-derive the conflict rules, reward invariants); partial observability and actor information boundaries (re-run the counterfactual tests + leak probe on final models); CTDE; IPPO/MAPPO implementation vs. §15/§16 (re-check each §16.8 item); GAE/PPO math; reward design and hacking evidence (component logs, replays of top-return episodes); evaluation fairness (budgets, seeds, maps, tuning parity); seed reproducibility (re-train 1 short run and re-evaluate 1 checkpoint); generalization claims vs. evidence; metric definitions vs. code; checkpoint integrity; backend/model parity; frontend real-data usage; test coverage; architecture boundaries; documentation accuracy (every claim in RESULTS.md and README is traceable).
- **TESTS:** The full suite; targeted re-runs as above.
- **VALIDATION GATE:** No open Critical/High findings; all claims in the documentation match the evidence; AUDIT.md complete.
- **EXPECTED OUTPUT:** Audit report; final project status.
- **KNOWN RISKS:** Audit scope creep; "fixing" results rather than defects.
- **RECOMMENDED MODEL:** **Opus 5.5.** **EFFORT:** **Extra High (xhigh)** if supported, otherwise High.
- **SKILLS/CONNECTORS:** `code-review` (max), `security-review`, `data:statistical-analysis`.
- **OPUS ESCALATION:** n/a.
- **STOP CONDITION:** Audit complete → final report → STOP (project complete).

---

## 39. Model recommendation for every phase

| Phase | Model | Rationale |
|---|---|---|
| 0 Architecture | **Opus 5.5** | Methodology and architecture decisions |
| 1 Repo + tooling | Sonnet | Routine scaffolding with some config design |
| 2 Environment | Sonnet | Well-specified; test-heavy |
| 3 Validation + baselines | Sonnet | Well-specified; calibration procedure defined |
| 4 PPO core + IPPO | Sonnet | Standard PPO with explicit formulas and tests |
| 5 MAPPO + CTDE | **Opus 5.5** | Highest-risk correctness surface (information leakage, critic construction, parity) |
| 6 Experiment infra | Sonnet | Orchestration + stats with specified procedures |
| 7 Controlled experiments | Sonnet (→ Opus on red flags) | Mostly running and reporting; interpretation rules pre-written |
| 8 Coordination analysis | Sonnet | Metrics fully defined |
| 9 UI/UX design | Sonnet + design skills | Design judgement; not a reasoning-depth problem |
| 10 Backend | Sonnet | Routine FastAPI with parity tests |
| 11 React simulator | Sonnet | Substantial interaction/rendering logic |
| 12 Analytics UI | Sonnet | Charts/data plumbing |
| 13 Polish | **Haiku** (styles) / Sonnet (interaction) | Low-risk repetitive edits |
| 14 E2E integration | Sonnet | Cross-cutting debugging |
| 15 Final audit | **Opus 5.5** | Adversarial methodological review |

## 40. Effort recommendation for every phase

Claude Code effort levels are `low`, `medium`, `high`, `xhigh` (= "Extra High") and `max`.

| Phase | Effort | Why not lower / higher |
|---|---|---|
| 0 | High | — |
| 1 | Medium | Low risks sloppy pinning/config; High is unnecessary |
| 2 | High | Many interacting invariants; bugs here poison everything downstream |
| 3 | Medium | Clear procedures |
| 4 | High | GAE/truncation/masking subtleties |
| 5 | High (Opus) | xhigh is not needed: the design is already specified |
| 6 | Medium | Specified stats; mechanical orchestration |
| 7 | Medium | Running + reporting; interpretation is rule-based. Raise to High only for mixed results |
| 8 | Medium | Specified metrics |
| 9 | High | Design quality is a core deliverable |
| 10 | Medium | Routine |
| 11 | High | Rendering/state architecture |
| 12 | Medium | Mostly plumbing given the design spec |
| 13 | Medium (Haiku/Sonnet) | — |
| 14 | High | Cross-cutting failure analysis |
| 15 | Extra High (xhigh) if supported, else High (Opus) | Final correctness gate |

## 41. Skills/connectors for every phase

Observed in the Phase 0 environment (**re-verify each phase**; never claim use of an unavailable capability):

| Phase | Useful capabilities | Notes |
|---|---|---|
| 1 | Built-in browser preview tools | Verify the frontend shell |
| 2 | `engineering:testing-strategy` (optional) | — |
| 3 | `dataviz`, `data:statistical-analysis` (optional) | Calibration plots |
| 4 | `code-review` (low/medium), `dataviz` | Self-review of the PPO core |
| 5 | `code-review` (high) | MAPPO diff review |
| 6 | `data:statistical-analysis`, `dataviz` | Stats procedure, figures |
| 7 | `data:statistical-analysis`, `dataviz`, terminal tools | Background study launch |
| 8 | `dataviz`, `data:statistical-analysis` | — |
| 9 | `hallmark`, `design:design-system`, `color-system`, `typography-scale`, `spacing-system`, `layout-grid`, `component-spec`, `data-visualization`, `motion-system`, `micro-interaction-spec`, `accessibility-audit`, `critique-*`; **Figma MCP only if authorized** | Figma plugin server required authentication at Phase 0 |
| 10 | `security-review`, `code-review` | — |
| 11 | Built-in browser tools, `emil-animate`, `animation-principles` | Visual verification |
| 12 | `dataviz` (read before chart code), browser tools | — |
| 13 | `accessibility-audit`, `critique-*`, `jezweb-design-review`, `hallmark` | — |
| 14 | `run`, browser tools, `code-review` (high), `security-review`; `gh` CLI if authenticated | GitHub plugin required authentication at Phase 0 |
| 15 | `code-review` (max), `security-review`, `data:statistical-analysis` | — |

**Authorization notes (as observed on 2026-09-23):** `plugin:figma:figma` and `plugin:engineering:github` required authentication. They must be authorized via claude.ai connector settings or `claude mcp` in an interactive terminal before they can be used. Until then, work proceeds without them.

## 42. Opus escalation criteria

**Escalate (recommend switching to Opus 5.5, High or Extra High) when:**
1. The MARL methodology may be wrong (the spec is contradictory, or an invariant is provably unattainable).
2. MAPPO fails conceptually: it learns worse than IPPO **and** its critic's explained variance is lower, or the parity test cannot be made to pass.
3. The IPPO/MAPPO comparison appears invalid (budget, seed, map or tuning asymmetry discovered after runs).
4. Reward hacking is observed (return ↑ with success flat or ↓, invariant violations, exploitative replays).
5. CTDE may leak global information (any §12.4 test fails, or the leak probe is positive).
6. GAE/PPO mathematics appear wrong after unit tests have been checked against the spec formulas.
7. Policies collapse unexpectedly with no implementation cause found via the §36 checklist.
8. Generalization results are suspicious (test ≥ validation consistently, implausible retention).
9. The evaluation methodology may be invalid (contamination, pseudo-replication, selection on test).
10. A major architecture redesign is being considered.

**Do NOT escalate for:** a failing unit test with a clear cause, syntax/import/type errors, CSS/layout problems, component refactors, simple API bugs, slow code (profile first), or dependency installation problems.

**Escalation procedure:** the current model STOPS and writes the phase report with `VALIDATION GATE: FAIL`, including the evidence (logs, plots, failing tests, minimal repro) and the precise question for Opus. The user switches models and replies `CONTINUE`.

## 43. Validation gate for every phase (summary)

| Phase | Gate (all must hold) |
|---|---|
| 1 | `make lint test` ✓; `npm run build` ✓; health endpoint + shell ✓; no ML code |
| 2 | All env tests ✓; env coverage ≥ 90%; `parallel_api_test` ✓; fuzz ✓; leak tests ✓; SPS measured |
| 3 | Heuristic success ≥ 0.95 within `max_steps` on validation; random L2 success < 0.6 (or level adjusted + documented); metric/replay tests ✓; map sets frozen + hashed; Phase 4 thresholds pre-written |
| 4 | Unit tests ✓; smoke (a) ✓; L1 IPPO ≥ 2/3 seeds meet pre-written thresholds; 0% ckpt ≈ random; SPS recorded |
| 5 | Parity (bitwise) ✓; critic sanity ✓; §12.4 ✓; leak probe negative; L1 MAPPO ≥ 2/3 seeds meet thresholds; §16.8 checklist complete |
| 6 | Smoke study end-to-end ✓; summary/RESULTS/figures generated; EXPERIMENTS.md frozen |
| 7 | Planned runs complete/documented; CIs reported; sanity checklist ✓; claims per §4.2; no undocumented deviations |
| 8 | Coordination tests ✓; analysis JSON for all algo × showcase maps; H3 + §24.4 stated |
| 9 | Design docs + tokens + prototype screenshots (4 widths) + contrast ✓ + **user approval** |
| 10 | Backend tests ✓ (≥ 80% coverage); backend↔evaluator parity ✓; OpenAPI exported |
| 11 | Tests ✓; build + mock guard ✓; live real-model session ≥ 50 fps; real replay scrubs correctly; screenshots |
| 12 | All charts from real endpoints; no mock imports; §30.8 checklist ✓; screenshots |
| 13 | Token audit ✓; AA contrast ✓; keyboard nav ✓; before/after screenshots |
| 14 | Fresh-clone setup + full test suite ✓; E2E ✓; real-data audit ✓; demo < 10 min |
| 15 | No open Critical/High findings; claims ↔ evidence traceable; AUDIT.md complete |

## 44. Claude Code execution protocol

### 44.1 At the start of every phase

1. Read this specification, at minimum §1–5, the target phase in §38, and every section that phase references.
2. Read the previous phase report (`docs/phase_reports/PHASE_<n−1>.md`) and inspect the repository (`git status`, tree, test status) **before modifying anything**.
3. Inspect the available skills/connectors/tools relevant to the phase. Note which are available, which are unavailable and which need authorization.
4. State the phase scope in 3–6 bullets. Confirm it does not include next-phase work.
5. Use the recommended model/effort. If the current model is more expensive than needed, say so. If it is insufficient, say so.

### 44.2 During the phase

- Implement **only** this phase. Do not prepare next-phase files, stubs or "small parts".
- Follow the dependency rules (§5.2), the single code path (§5.3) and the real-data rule (§29.4).
- Fix ordinary bugs directly. For methodology problems, stop and escalate per §42.
- Never fabricate results, metrics or tool usage. Never tune on test maps. Never edit generated results by hand.
- Keep files focused (≲ 400 lines). No giant components or modules.
- Long computations (training studies) run as background processes. Do not stream them into the conversation.
- Commits: propose a commit at the end of each phase and commit only with the user's approval.

### 44.3 At the end of every phase

1. Run the relevant tests and validation checks.
2. Fix ordinary implementation failures.
3. Evaluate the validation gate honestly (PASS/FAIL).
4. Write the completion report (§44.4) to `docs/phase_reports/PHASE_<n>.md` **and** output it in the chat.
5. **STOP.**

### 44.4 Mandatory phase completion format

```
PHASE <NUMBER> COMPLETE

IMPLEMENTED:
- ...

FILES CREATED/CHANGED:
- ...

TESTS RUN:
- ...

RESULTS:
- ...

PROBLEMS FOUND:
- ...

PROBLEMS FIXED:
- ...

REMAINING CONCERNS:
- ...

VALIDATION GATE:
PASS / FAIL

CURRENT PROJECT STATUS:
...

NEXT PHASE:
...

RECOMMENDED MODEL:
Opus 5.5 / Sonnet / Haiku

RECOMMENDED EFFORT:
Low / Medium / High / Extra High

WHY:
...

USE OPUS ONLY IF:
...

RECOMMENDED SKILLS / CONNECTORS:
Only list capabilities actually available and useful.

ACTION REQUIRED:
Switch to the recommended model/effort if necessary and reply CONTINUE.

STOP HERE.
```

### 44.5 Absolute stop rule

After completing a phase: **STOP.** Do not start the next phase, prepare next-phase files, silently continue, use remaining context to get ahead, or implement "small parts" of the next phase. **The only authorization to begin the next phase is the exact user reply `CONTINUE`.** Approval of one phase does not carry over to the next.

---

## Appendix A — Default configuration files

### A.1 `configs/env/L2.yaml`

```yaml
name: L2
width: 10
height: 10
n_robots: 3
n_targets: 2
obstacle_density: 0.15
max_steps: 150            # provisional; calibrated in Phase 3, frozen in Phase 6
sensor_radius: 2.0
obs_radius: 5
comm_mode: shared_map     # none | shared_map | limited_range
comm_radius: 4.0
spawn: clustered
target_min_dist: 3
action_masking: true
sensing:
  occlusion: false
reward:
  explore_total: 1.0      # w_x = explore_total / |G|
  discover: 0.25
  rescue: 1.0
  complete: 2.0
  time_cost: 0.01
  collision: 0.05
  team_share: 1.0         # β; 1.0 = positive rewards fully shared
obs:
  agent_id_onehot: false
```

### A.2 `configs/algo/mappo.yaml` (IPPO identical except `critic.input: local`)

```yaml
name: mappo
critic:
  input: agent_specific_global   # ippo: local
actor:
  encoder: local_cnn_v1
  hidden: 256
ppo:
  n_envs: 16
  rollout_len: 128
  epochs: 5
  minibatches: 4
  clip: 0.2
  gamma: 0.99
  gae_lambda: 0.95
  lr_actor: 3.0e-4
  lr_critic: 3.0e-4
  adam_eps: 1.0e-5
  entropy_coef: 0.01
  max_grad_norm: 0.5
  value_norm: true
  value_clip: false
  target_kl: null
  lr_schedule: constant
eval:
  interval_updates: 20
  n_val_maps: 50
  stochastic: true
checkpoint:
  fractions: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
device: auto              # cpu on Apple Silicon unless benchmark says otherwise
deterministic: false
```

### A.3 `configs/experiments/main_L2.yaml`

```yaml
study: main_L2
levels: [L2]
algos: [ippo, mappo]
seeds: [0, 1, 2, 3, 4]
total_env_steps: 3000000
baselines:
  random: { eval_seeds: [0, 1, 2, 3, 4] }
  heuristic: { reference_only: true }
final_eval:
  map_sets: [test, structured]
  episodes_per_map: 2
  stochastic: true
  also_greedy: true
  checkpoint: final
```

## Appendix B — References

- Schulman et al. (2017). *Proximal Policy Optimization Algorithms.*
- Schulman et al. (2016). *High-Dimensional Continuous Control Using Generalized Advantage Estimation.*
- Yu et al. (2022). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (MAPPO).
- de Witt et al. (2020). *Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?* (IPPO).
- Lyu et al. (2021; extended 2023). *Contrasting Centralized and Decentralized Critics in Multi-Agent Reinforcement Learning.*
- Ng, Harada & Russell (1999). *Policy Invariance Under Reward Transformations* (potential-based shaping).
- Andrychowicz et al. (2021). *What Matters in On-Policy Reinforcement Learning?*
- Huang et al. (2022). *The 37 Implementation Details of Proximal Policy Optimization* (ICLR blog track).
- Agarwal et al. (2021). *Deep RL at the Edge of the Statistical Precipice* (rliable).
- Terry et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning.*
- Rashid et al. (2018). *QMIX: Monotonic Value Function Factorisation.*
- Okabe & Ito (2008). *Color Universal Design* (CVD-safe palette).

## Appendix C — Glossary

| Term | Meaning |
|---|---|
| Env step | One call to `step()` for one environment (all robots act once) |
| Agent step | One robot's action in one env step (= env steps × N) |
| Belief | What a robot is entitled to know under the communication mode |
| Team-known | Union of all robots' sensed cells (used by rewards/metrics; visible to actors only in mode B) |
| AS state | Agent-specific global state: shared global channels + agent's own position/belief channels + agent's own observation |
| CTDE | Centralized Training, Decentralized Execution |
| DER / DER@50 | Duplicate Exploration Ratio / at 50% coverage |
| Frontier | A known free cell adjacent to an unknown cell |
| No-follow rule | A robot may only move into a cell unoccupied at the start of the step |
| Truncation folding | Adding γ·V(final_obs) to the reward at a time-limit cut, then treating the step as done |
| Unit of analysis | The training seed (not the episode) |

---

**END OF SPECIFICATION — Phase 0 complete. Implementation MUST NOT begin until the user replies `CONTINUE`.**
