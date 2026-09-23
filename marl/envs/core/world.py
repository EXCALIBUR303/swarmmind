"""SwarmWorld: the environment core (§6). Orchestrates reset/step.

This is the SINGLE implementation of environment dynamics (§5.3). The
PettingZoo wrapper and the vectorized wrapper are both thin adapters over
this class — neither re-implements any of this logic.
"""

from __future__ import annotations

import numpy as np

from marl.envs.core.belief import sense_and_update
from marl.envs.core.config import EnvConfig
from marl.envs.core.dynamics import resolve_step
from marl.envs.core.global_state import (
    N_GLOBAL_CHANNELS,
    N_GLOBAL_VEC_FEATURES,
    agent_state_channels,
    full_global_grid,
    shared_global_grid,
    shared_global_vec,
)
from marl.envs.core.mapgen import generate_map
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.observations import N_OBS_CHANNELS, N_VEC_FEATURES, build_actor_observation
from marl.envs.core.rewards import RewardInputs, compute_rewards
from marl.envs.core.sensing import sensed_mask
from marl.envs.core.types import (
    DISCOVERED,
    HIDDEN,
    N_ACTIONS,
    OBSTACLE,
    RESCUED,
    RobotBelief,
    StepResult,
    WorldState,
)

__all__ = [
    "SwarmWorld",
    "N_OBS_CHANNELS",
    "N_VEC_FEATURES",
    "N_GLOBAL_CHANNELS",
    "N_GLOBAL_VEC_FEATURES",
    "N_ACTIONS",
]


class SwarmWorld:
    """The core multi-agent grid environment (§6)."""

    def __init__(self, config: EnvConfig) -> None:
        self.cfg = config
        self.state: WorldState | None = None
        self._n_reachable_free: int = 0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, map_spec: MapSpec | None = None) -> WorldState:
        """Reset the world, either to a given `map_spec` or a freshly
        procedurally generated one (deterministic given `seed`)."""
        if map_spec is None:
            if seed is None:
                raise ValueError("reset() requires either a seed or an explicit map_spec")
            map_spec = generate_map(self.cfg, seed)

        n_robots = len(map_spec.robot_starts)
        n_targets = len(map_spec.targets)
        if n_robots != self.cfg.n_robots:
            raise ValueError(
                f"map_spec has {n_robots} robot starts but config expects {self.cfg.n_robots}"
            )
        if n_targets != self.cfg.n_targets:
            raise ValueError(
                f"map_spec has {n_targets} targets but config expects {self.cfg.n_targets}"
            )

        terrain = map_spec.terrain_array()
        height, width = terrain.shape
        self._n_reachable_free = int(np.sum(terrain != OBSTACLE))

        robot_pos = np.array(map_spec.robot_starts, dtype=np.int32)
        target_pos = np.array(map_spec.targets, dtype=np.int32)
        target_status = np.full(n_targets, HIDDEN, dtype=np.int8)
        target_rescued_by = np.full(n_targets, -1, dtype=np.int32)
        target_t_discovered = np.full(n_targets, -1, dtype=np.int32)
        target_t_rescued = np.full(n_targets, -1, dtype=np.int32)

        beliefs = [
            RobotBelief(
                terrain_belief=np.full((height, width), -1, dtype=np.int8),
                target_known=np.zeros(n_targets, dtype=bool),
                target_rescued_known=np.zeros(n_targets, dtype=bool),
                visited_self=np.zeros((height, width), dtype=bool),
                visited_team_known=np.zeros((height, width), dtype=bool),
            )
            for _ in range(n_robots)
        ]

        state = WorldState(
            width=width,
            height=height,
            n_robots=n_robots,
            n_targets=n_targets,
            terrain=terrain,
            robot_pos=robot_pos,
            target_pos=target_pos,
            target_status=target_status,
            target_rescued_by=target_rescued_by,
            target_t_discovered=target_t_discovered,
            target_t_rescued=target_t_rescued,
            known_team=np.zeros((height, width), dtype=bool),
            visited_count=np.zeros((height, width), dtype=np.int32),
            beliefs=beliefs,
            t=0,
            map_id=map_spec.id,
            prev_action=np.full(n_robots, -1, dtype=np.int32),
            last_blocked=np.zeros(n_robots, dtype=bool),
        )
        self.state = state

        # Initial sensing pass so robots know their immediate surroundings
        # (and the action mask is meaningful) before any action is taken.
        sense_and_update(state, self.cfg.sensor_radius, self.cfg.comm_mode, self.cfg.comm_radius)
        for i in range(n_robots):
            state.visited_count[robot_pos[i, 0], robot_pos[i, 1]] += 1

        return state

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: np.ndarray) -> StepResult:
        """Advance the world by one joint step (§6.4)."""
        if self.state is None:
            raise RuntimeError("step() called before reset()")
        state = self.state

        actions = np.asarray(actions)
        if actions.shape != (state.n_robots,):
            raise ValueError(f"actions must have shape ({state.n_robots},), got {actions.shape}")
        if np.any(actions < 0) or np.any(actions >= N_ACTIONS):
            raise ValueError(f"all actions must be in [0, {N_ACTIONS}), got {actions}")
        actions = actions.astype(np.int32)

        known_before = int(np.sum(state.known_team & (state.terrain != OBSTACLE)))

        # Steps 2-5: propose, terrain check, conflict resolution, move.
        dyn = resolve_step(state.robot_pos, actions, state.terrain)
        state.robot_pos = dyn.new_pos
        for i in range(state.n_robots):
            state.visited_count[state.robot_pos[i, 0], state.robot_pos[i, 1]] += 1

        # Step 6: rescue.
        rescued: list[tuple[int, int]] = []
        occ: dict[tuple[int, int], list[int]] = {}
        for i in range(state.n_robots):
            occ.setdefault((int(state.robot_pos[i, 0]), int(state.robot_pos[i, 1])), []).append(i)
        for k in range(state.n_targets):
            if state.target_status[k] == RESCUED:
                continue
            ty, tx = int(state.target_pos[k, 0]), int(state.target_pos[k, 1])
            occupants = occ.get((ty, tx))
            if occupants:
                # Invariant (§6.4 step 6): a target cannot be rescued before
                # it is discovered, because sensor_radius >= 1 guarantees
                # every robot senses a destination cell at least one step
                # before it can move into it. Assert rather than silently
                # allow a HIDDEN target to be rescued.
                assert state.target_status[k] == DISCOVERED, (
                    f"target {k} rescued while still HIDDEN at t={state.t} — "
                    "sensor_radius invariant violated"
                )
                rescuer = occupants[0]  # deterministic: lowest robot index
                state.target_status[k] = RESCUED
                state.target_rescued_by[k] = rescuer
                state.target_t_rescued[k] = state.t
                rescued.append((k, rescuer))

        # Step 7: sense (updates known_team, beliefs, target discovery).
        newly_discovered, sensed_masks = sense_and_update(
            state, self.cfg.sensor_radius, self.cfg.comm_mode, self.cfg.comm_radius
        )

        known_after = int(np.sum(state.known_team & (state.terrain != OBSTACLE)))
        newly_known_count = known_after - known_before

        success = bool(np.all(state.target_status == RESCUED))
        n_rescued_total = int(np.sum(state.target_status == RESCUED))
        n_discovered_total = int(np.sum(state.target_t_discovered >= 0))

        # Step 8: rewards.
        blocked = dyn.blocked_by_robot | dyn.bumped_wall
        reward_inputs = RewardInputs(
            n_robots=state.n_robots,
            n_reachable_free=self._n_reachable_free,
            newly_known_count=max(newly_known_count, 0),
            n_newly_discovered=len(newly_discovered),
            n_rescued_this_step=len(rescued),
            success=success,
            known_count_after=known_after,
            blocked_by_robot=dyn.blocked_by_robot,
            bumped_wall=dyn.bumped_wall,
        )
        reward, components = compute_rewards(self.cfg.reward, reward_inputs)

        # Step 9: termination / truncation.
        terminated = success
        truncated = (state.t + 1 == self.cfg.max_steps) and not success
        # Success takes precedence at t+1 == max_steps (§6.4 step 9).

        coverage = known_after / max(self._n_reachable_free, 1)

        result = StepResult(
            reward=reward,
            reward_components=components,
            terminated=terminated,
            truncated=truncated,
            moved=dyn.moved,
            blocked_by_robot=dyn.blocked_by_robot,
            bumped_wall=dyn.bumped_wall,
            rescued=rescued,
            discovered=newly_discovered,
            newly_known_count=newly_known_count,
            t=state.t + 1,
            coverage=coverage,
            n_discovered=n_discovered_total,
            n_rescued=n_rescued_total,
            success=success,
        )

        # Step 10 bookkeeping: store prev_action/last_blocked for the NEXT
        # observation, then increment t.
        state.prev_action = actions
        state.last_blocked = blocked
        state.t += 1

        return result

    # ------------------------------------------------------------------
    # Observations (actor — LOCAL ONLY)
    # ------------------------------------------------------------------

    def _sensed_mask_for(self, robot_idx: int) -> np.ndarray:
        state = self.state
        assert state is not None
        pos = (int(state.robot_pos[robot_idx, 0]), int(state.robot_pos[robot_idx, 1]))
        return sensed_mask(pos, self.cfg.sensor_radius, state.height, state.width)

    def actor_obs(self, robot_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(grid[8,K,K], vec[10], mask[5]) for one robot (§8, §10.2)."""
        state = self.state
        if state is None:
            raise RuntimeError("actor_obs() called before reset()")
        sensed = self._sensed_mask_for(robot_idx)
        return build_actor_observation(
            state=state,
            robot_idx=robot_idx,
            obs_radius=self.cfg.obs_radius,
            sensed_mask_i=sensed,
            action_masking=self.cfg.action_masking,
            max_steps=self.cfg.max_steps,
            prev_action=int(state.prev_action[robot_idx]),
            blocked_last_step=bool(state.last_blocked[robot_idx]),
        )

    def all_actor_obs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Batched (grid[N,8,K,K], vec[N,10], mask[N,5]) for all robots."""
        state = self.state
        if state is None:
            raise RuntimeError("all_actor_obs() called before reset()")
        grids, vecs, masks = [], [], []
        for i in range(state.n_robots):
            g, v, m = self.actor_obs(i)
            grids.append(g)
            vecs.append(v)
            masks.append(m)
        return np.stack(grids), np.stack(vecs), np.stack(masks)

    # ------------------------------------------------------------------
    # Global state (critic — TRAINING ONLY)
    # ------------------------------------------------------------------

    def global_state(self) -> tuple[np.ndarray, np.ndarray]:
        """(shared_grid[7,H,W], vec[4]) — the team-shared portion of the
        global state (§9.1-9.2). Combine with `agent_state_channels(i)` for
        the full agent-specific critic input (§9.3)."""
        state = self.state
        if state is None:
            raise RuntimeError("global_state() called before reset()")
        return shared_global_grid(state), shared_global_vec(state, self.cfg.max_steps)

    def agent_state_channels(self, robot_idx: int) -> np.ndarray:
        state = self.state
        if state is None:
            raise RuntimeError("agent_state_channels() called before reset()")
        return agent_state_channels(state, robot_idx)

    def full_global_grid(self, robot_idx: int) -> np.ndarray:
        """Convenience: the full 9-channel agent-specific global grid."""
        state = self.state
        if state is None:
            raise RuntimeError("full_global_grid() called before reset()")
        return full_global_grid(state, robot_idx)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_ansi(self) -> str:
        """ASCII render (§7.2 render_mode='ansi'). Ground truth (debug use)."""
        state = self.state
        if state is None:
            raise RuntimeError("render_ansi() called before reset()")
        grid = [["#" if state.terrain[y, x] == OBSTACLE else "." for x in range(state.width)] for y in range(state.height)]
        for k in range(state.n_targets):
            if state.target_status[k] != RESCUED:
                y, x = state.target_pos[k]
                grid[y][x] = "T"
        for i in range(state.n_robots):
            y, x = state.robot_pos[i]
            grid[y][x] = str(i % 10)
        return "\n".join("".join(row) for row in grid)
