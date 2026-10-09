"""Live simulation sessions (§28.1, §28.4).

Each session is a `marl.evaluation.evaluator.EpisodeRunner` (the same
stepper `run_episode` uses) plus a `Policy`. The backend adds lifecycle
(TTL, limits) and serialisation only; it contains no environment or
policy logic."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.app.schemas.sessions import (
    FixedMap,
    LiveMetrics,
    PolicyChoice,
    SessionCreate,
    SessionInitial,
    SessionReset,
    SessionSnapshot,
    StepResponse,
)
from backend.app.services.errors import (
    CapacityError,
    ConflictError,
    NotFoundError,
    ValidationFailed,
)
from backend.app.services.map_service import MapService
from backend.app.services.model_registry import ModelRegistry
from marl.algorithms.baselines.frontier_heuristic import FrontierHeuristicPolicy
from marl.algorithms.baselines.random_policy import RandomPolicy
from marl.algorithms.common.policy import LearnedActorPolicy, Policy
from marl.analysis.metrics import compute_episode_metrics
from marl.envs.core.config import EnvConfig
from marl.envs.core.maps_io import MapSpec
from marl.envs.core.types import OBSTACLE
from marl.evaluation.evaluator import EpisodeRunner
from marl.evaluation.final_eval import episode_seed
from marl.replay.recorder import build_frame
from marl.replay.schema import ReplayFrame, ReplayMapInfo, ReplayPolicyInfo


def outcome_metrics(runner: EpisodeRunner) -> dict[str, Any]:
    """Same metric subset the replay recorder stores in `outcome.metrics`."""
    m = compute_episode_metrics(runner.trace, runner.n_reachable_free)
    return {
        "episode_return_mean": m.episode_return_mean,
        "coverage_final": m.coverage_final,
        "n_discovered": m.n_discovered,
        "n_rescued": m.n_rescued,
        "n_collisions": m.n_collisions,
    }


@dataclass
class Session:
    session_id: str
    cfg: EnvConfig
    policy_choice: PolicyChoice
    policy_info: ReplayPolicyInfo
    policy: Policy
    stochastic: bool
    seed: int
    runner: EpisodeRunner
    rng: np.random.Generator
    known_before: np.ndarray
    map_info: ReplayMapInfo
    initial: SessionInitial
    map_choice: Any
    owner: Any = None  # the live WebSocket driving this session, if any
    last_frame: ReplayFrame | None = None
    last_used: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)

    # ------------------------------------------------------------------
    def touch(self) -> None:
        self.last_used = time.monotonic()

    def live_metrics(self) -> LiveMetrics:
        tr, st = self.runner.trace, self.runner.state
        n_known_free = float(np.sum(st.known_team & (st.terrain != OBSTACLE)))
        collisions = int(np.sum(np.stack(tr.blocked_by_robot))) if tr.blocked_by_robot else 0
        ret = float(np.mean(np.sum(np.stack(tr.rewards), axis=0))) if tr.rewards else 0.0
        return LiveMetrics(
            t=int(st.t),
            coverage=n_known_free / max(self.runner.n_reachable_free, 1),
            n_discovered=int(np.sum(st.target_status >= 1)),
            n_rescued=int(np.sum(st.target_status == 2)),
            n_collisions=collisions,
            return_mean=ret,
            success=bool(tr.success),
            done=self.runner.done,
        )

    def final_metrics(self) -> dict[str, Any] | None:
        return outcome_metrics(self.runner) if self.runner.done else None

    def snapshot(self) -> SessionSnapshot:
        st = self.runner.state
        return SessionSnapshot(
            session_id=self.session_id,
            step_index=int(st.t),
            done=self.runner.done,
            stochastic=self.stochastic,
            seed=self.seed,
            env={"level": self.cfg.name, "config": self.cfg.model_dump()},
            map=self.map_info,
            policy=self.policy_info,
            initial=self.initial,
            current_pos=st.robot_pos.tolist(),
            current_known_team=np.argwhere(st.known_team).tolist(),
            target_status=st.target_status.tolist(),
            last_frame=self.last_frame,
            metrics=self.live_metrics(),
            final_metrics=self.final_metrics(),
        )

    def step_once(self) -> ReplayFrame:
        if self.runner.done:
            raise ConflictError("episode is finished; reset the session to start a new one")
        actions, result = self.runner.step(self.policy, self.rng)
        frame, self.known_before = build_frame(self.runner.state, actions, result, self.known_before)
        self.last_frame = frame
        self.touch()
        return frame

    def step(self, n: int) -> StepResponse:
        with self.lock:
            frames: list[ReplayFrame] = []
            for _ in range(n):
                if self.runner.done:
                    break
                frames.append(self.step_once())
            if not frames and self.runner.done:
                raise ConflictError("episode is finished; reset the session to start a new one")
            return StepResponse(
                session_id=self.session_id,
                frames=frames,
                step_index=int(self.runner.state.t),
                done=self.runner.done,
                metrics=self.live_metrics(),
                final_metrics=self.final_metrics(),
            )


class SessionManager:
    def __init__(
        self,
        maps: MapService,
        models: ModelRegistry,
        max_sessions: int,
        idle_ttl_seconds: float,
    ) -> None:
        self.maps = maps
        self.models = models
        self.max_sessions = max_sessions
        self.idle_ttl = idle_ttl_seconds
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # -- lifecycle ------------------------------------------------------
    def reap(self) -> int:
        now = time.monotonic()
        with self._lock:
            dead = [sid for sid, s in self._sessions.items() if now - s.last_used > self.idle_ttl]
            for sid in dead:
                del self._sessions[sid]
        return len(dead)

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def get(self, session_id: str) -> Session:
        self.reap()
        with self._lock:
            s = self._sessions.get(session_id)
        if s is None:
            raise NotFoundError(f"session '{session_id}' not found (closed or expired)")
        s.touch()
        return s

    def close(self, session_id: str) -> None:
        with self._lock:
            if self._sessions.pop(session_id, None) is None:
                raise NotFoundError(f"session '{session_id}' not found")

    def close_all(self) -> None:
        with self._lock:
            self._sessions.clear()

    # -- construction ---------------------------------------------------
    # Named levels are trusted; a client-supplied env_config is bounded so a
    # request cannot ask for an enormous observation window or episode.
    MAX_CUSTOM_STEPS = 1000
    MAX_CUSTOM_RADIUS = 8

    def _resolve_cfg(self, req: SessionCreate) -> EnvConfig:
        if req.env_config is not None:
            c = req.env_config
            if c.max_steps > self.MAX_CUSTOM_STEPS:
                raise ValidationFailed(f"max_steps must be <= {self.MAX_CUSTOM_STEPS} for a custom env_config")
            if c.obs_radius > self.MAX_CUSTOM_RADIUS or c.sensor_radius > self.MAX_CUSTOM_RADIUS:
                raise ValidationFailed(f"obs_radius and sensor_radius must be <= {self.MAX_CUSTOM_RADIUS} for a custom env_config")
            return c
        assert req.level is not None
        return self.maps.level_config(req.level)

    def _build_policy(self, choice: PolicyChoice, cfg: EnvConfig, stochastic: bool, seed: int) -> tuple[Policy, ReplayPolicyInfo]:
        if choice.kind == "random":
            return RandomPolicy(), ReplayPolicyInfo(kind="random", stochastic=stochastic, eval_seed=seed)
        if choice.kind == "heuristic":
            return FrontierHeuristicPolicy(), ReplayPolicyInfo(kind="heuristic", stochastic=stochastic, eval_seed=seed)
        assert choice.model_id is not None
        actor, meta, info = self.models.load_actor(choice.model_id)
        if info.get("algo") not in (None, choice.kind):
            raise ValidationFailed(f"model '{choice.model_id}' is a {info.get('algo')} model, not {choice.kind}")
        if info.get("level") not in (None, cfg.name):
            raise ValidationFailed(f"model '{choice.model_id}' was trained on {info.get('level')}, not {cfg.name}")
        if meta.get("grid_size", 11) != 2 * cfg.obs_radius + 1:
            raise ValidationFailed("model observation window does not match this environment")
        return (
            LearnedActorPolicy(actor, stochastic=stochastic),
            ReplayPolicyInfo(
                kind=choice.kind,
                run_id=info.get("run_id") or choice.model_id,
                checkpoint_step=info.get("trained_env_steps"),
                stochastic=stochastic,
                eval_seed=seed,
            ),
        )

    def _resolve_map(self, choice, cfg: EnvConfig, level_name: str, seed: int) -> tuple[int | None, MapSpec | None]:
        if isinstance(choice, FixedMap):
            m = self.maps.find_map(choice.set, level_name, choice.id)
            if (m.width, m.height) != (cfg.width, cfg.height) or len(m.robot_starts) != cfg.n_robots or len(m.targets) != cfg.n_targets:
                raise ValidationFailed("map does not match the environment configuration")
            return None, m
        return seed, None

    def _build_runner(self, cfg, map_choice, level_name, seed):
        gen_seed, map_spec = self._resolve_map(map_choice, cfg, level_name, seed)
        runner = EpisodeRunner(cfg, seed=gen_seed, map_spec=map_spec)
        st = runner.state
        played = MapSpec.from_terrain_array(
            id=st.map_id,
            terrain=st.terrain,
            robot_starts=[(int(p[0]), int(p[1])) for p in st.robot_pos],
            targets=[(int(p[0]), int(p[1])) for p in st.target_pos],
        )
        map_info = ReplayMapInfo(
            id=st.map_id,
            hash=played.content_hash,
            width=st.width,
            height=st.height,
            terrain=played.terrain_rows,
            robot_starts=[[int(p[0]), int(p[1])] for p in st.robot_pos],
            targets=[[int(p[0]), int(p[1])] for p in st.target_pos],
        )
        initial = SessionInitial(pos=st.robot_pos.tolist(), known_team=np.argwhere(st.known_team).tolist())
        # Same stream the evaluator uses for (map, eval_seed): parity (§28.4).
        rng = np.random.default_rng(episode_seed(st.map_id, seed))
        return runner, map_info, initial, rng

    def create(self, req: SessionCreate) -> Session:
        self.reap()
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise CapacityError(f"at most {self.max_sessions} live sessions; close one first")
        cfg = self._resolve_cfg(req)
        policy, policy_info = self._build_policy(req.policy, cfg, req.stochastic, req.seed)
        runner, map_info, initial, rng = self._build_runner(cfg, req.map, cfg.name, req.seed)
        session = Session(
            session_id=uuid.uuid4().hex,
            cfg=cfg,
            policy_choice=req.policy,
            policy_info=policy_info,
            policy=policy,
            stochastic=req.stochastic,
            seed=req.seed,
            runner=runner,
            rng=rng,
            known_before=runner.state.known_team.copy(),
            map_info=map_info,
            initial=initial,
            map_choice=req.map,
        )
        with self._lock:
            if len(self._sessions) >= self.max_sessions:
                raise CapacityError(f"at most {self.max_sessions} live sessions; close one first")
            self._sessions[session.session_id] = session
        return session

    def reset(self, session_id: str, req: SessionReset) -> Session:
        """Restart the episode. Without a body it replays the same
        (map, seed); a new `seed` regenerates a procedural map (a fixed map
        stays fixed); a new `map` switches maps."""
        s = self.get(session_id)
        with s.lock:
            seed = s.seed if req.seed is None else req.seed
            map_choice = req.map if req.map is not None else s.map_choice
            runner, map_info, initial, rng = self._build_runner(s.cfg, map_choice, s.cfg.name, seed)
            s.runner, s.map_info, s.initial, s.rng = runner, map_info, initial, rng
            s.map_choice, s.seed = map_choice, seed
            s.policy_info = s.policy_info.model_copy(update={"eval_seed": seed})
            s.known_before = runner.state.known_team.copy()
            s.last_frame = None
            s.touch()
        return s
