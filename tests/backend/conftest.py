"""Backend test fixtures.

`results_tree` builds a small results/ + models/ layout from REAL code paths
(an untrained-but-genuine exported actor, a recorded heuristic replay, real
JSON the way the pipeline writes it). These are TEST FIXTURES: nothing from
them is ever reported as a result. Tests that need the real study
artifacts (`real_*` fixtures) skip when results/ is absent (it is gitignored)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.main import create_app
from backend.app.settings import Settings
from marl.algorithms.baselines.frontier_heuristic import FrontierHeuristicPolicy
from marl.envs.core.config import LevelRegistry
from marl.networks.actor import ActorPolicy
from marl.replay.loader import save_replay
from marl.replay.recorder import record_episode
from marl.replay.schema import ReplayPolicyInfo
from marl.utils.paths import REPO_ROOT, RESULTS_DIR

RUN_ID = "s0-20260101T000000Z-aaaa1111"
REPLAY_ID = "L1-0000_heuristic_demo"
ORIGIN = "http://localhost:5173"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def make_run(runs_dir: Path, study: str, algo: str, level: str, run_id: str, status: str = "completed", actor: bool = True) -> Path:
    run = runs_dir / study / algo / level / run_id
    (run / "checkpoints").mkdir(parents=True)
    manifest = {
        "run_id": run_id, "study": study, "algo": algo, "level": level, "seed": 0,
        "status": status, "total_env_steps": 2048, "trained_env_steps": 2048, "updates": 3,
        "start_time": "2026-01-01T00:00:00+00:00", "end_time": "2026-01-01T00:01:00+00:00",
        "final_val_metrics": {"success_rate": 0.5},
        "checkpoints": [{"path": "/abs/secret/path/step_0.pt", "env_steps": 0}],
        "run_dir": str(REPO_ROOT / "results" / "runs" / study / algo / level / run_id),
        "home_note": str(Path.home() / "private" / "x"),
        "versions": {"python": "3.11"},
    }
    write_json(run / "manifest.json", manifest)
    rows = [{"update": i, "env_steps": 256 * (i + 1), "mean_reward": 0.1 * i} for i in range(3)]
    (run / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "eval_val.jsonl").write_text(json.dumps({"update": 2, "env_steps": 768, "success_rate": 0.5}) + "\n")
    write_json(
        run / "eval" / f"test_{level}.json",
        {"run_id": run_id, "modes": {"stochastic": {"aggregate": {"success_rate": 0.4}, "episodes": [{"x": 1}]},
                                     "greedy": {"aggregate": {"success_rate": 0.2}, "episodes": [{"x": 2}]}}},
    )
    if actor:
        torch.manual_seed(0)
        cfg = LevelRegistry().get(level)
        net = ActorPolicy(grid_size=2 * cfg.obs_radius + 1)
        torch.save(
            {"schema_version": 1, "actor_state_dict": net.state_dict(),
             "meta": {"grid_size": 2 * cfg.obs_radius + 1, "hidden": 256, "algo": algo, "level": level}},
            run / "actor.pt",
        )
    return run


@pytest.fixture
def results_tree(tmp_path: Path) -> Path:
    runs = tmp_path / "results" / "runs"
    make_run(runs, "demo", "ippo", "L1", RUN_ID)
    make_run(runs, "demo", "mappo", "L1", "s1-20260101T000000Z-bbbb2222", status="running", actor=False)
    corrupt = runs / "demo" / "ippo" / "L1" / "s2-corrupt"
    corrupt.mkdir(parents=True)
    (corrupt / "manifest.json").write_text("{not json")

    cfg = LevelRegistry().get("L1")
    replay = record_episode(
        cfg, FrontierHeuristicPolicy(), np.random.default_rng(0),
        ReplayPolicyInfo(kind="heuristic", eval_seed=0), REPLAY_ID, seed=3,
    )
    rdir = tmp_path / "results" / "replays" / "demo" / RUN_ID
    save_replay(replay, rdir / f"{REPLAY_ID}.json.gz")
    write_json(rdir / "index.json", [{"replay_id": REPLAY_ID, "map_id": replay.map.id, "algo": "ippo", "seed": 0,
                                       "fraction": 1.0, "checkpoint_step": 2048, "success": replay.outcome.success,
                                       "steps": replay.outcome.steps}])

    sdir = tmp_path / "results" / "studies" / "demo"
    write_json(sdir / "summary.json", {"study": "demo", "smoke": True, "generated": "x", "levels": {"L1": {}}, "primary": {}})
    write_json(sdir / "study_manifest.json", {"study": "demo", "jobs": [1, 2], "deviations": [], "git": {"secret": "x"}})
    write_json(sdir / "analysis" / "L1_ippo_L1-m1_heatmap.json",
               {"map_id": "L1-m1", "n_episodes": 2, "obstacles": [[False]], "visit_counts": [[3]], "collision_counts": [[1]]})
    (tmp_path / "models").mkdir()
    return tmp_path


@pytest.fixture
def make_client(results_tree: Path) -> Callable[..., TestClient]:
    clients: list[TestClient] = []

    def _make(**overrides) -> TestClient:
        s = Settings(
            results_dir=results_tree / "results",
            models_dir=results_tree / "models",
            allowed_hosts=["testserver", "localhost"],
            cors_origin=ORIGIN,
            _env_file=None,
            **overrides,
        )
        c = TestClient(create_app(s))
        c.__enter__()
        clients.append(c)
        return c

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()


# --- real study artifacts (gitignored; tests skip when absent) -------------

REAL_REPLAY_ID = "L2-struct-bottleneck_vertical_mappo_s0_step999424_eval0"


@pytest.fixture(scope="session")
def real_client():
    if not (RESULTS_DIR / "replays" / "main_L2").is_dir():
        pytest.skip("real study artifacts (results/) not present")
    app = create_app(Settings(allowed_hosts=["localhost", "testserver"], _env_file=None))
    with TestClient(app) as c:
        yield c
