"""Study runner: idempotent skip/resume, manifest integrity, frozen-input
detection and the held-out gate (§33, Phase 6 tests). Uses a tiny real L1
study run inline (parallel=0) into temporary directories."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from marl.evaluation.final_eval import HeldOutAccessError, episode_seed, load_heldout_maps
from marl.experiments.config import StudyConfig
from marl.experiments.runner import (
    TrainJob,
    check_frozen,
    classify_run,
    freeze_study,
    frozen_inputs,
    plan_train_jobs,
    resolved_run_config,
    run_study,
)
from marl.utils.hashing import config_hash

TINY = {
    "study": "tiny",
    "smoke": True,
    "levels": ["L1"],
    "algos": ["ippo", "mappo"],
    "seeds": [0],
    "total_env_steps": 32,  # 2 updates of 2 envs x 8 steps
    "algo_overrides": {"ppo.n_envs": 2, "ppo.rollout_len": 8, "eval.n_val_maps": 2, "eval.interval_updates": 1,
                       "checkpoint.fractions": [0.0, 0.5, 1.0]},
    "baselines": {"random": {"eval_seeds": [0, 1]}},
    "final_eval": {"map_sets": ["test", "structured"], "episodes_per_map": 2},
    "evolution_replays": None,
}


def _setup(tmp_path: Path, **changes) -> tuple[StudyConfig, dict]:
    raw = {**TINY, **changes}
    yml = tmp_path / "tiny.yaml"
    yml.write_text(yaml.safe_dump(raw))
    kw = dict(runs_root=tmp_path / "runs", studies_root=tmp_path / "studies", study_yaml=yml, max_eval_maps=2,
              access_log=tmp_path / "access.jsonl", frozen_file=tmp_path / "FROZEN.json", log=lambda _m: None)
    return StudyConfig.model_validate(raw), kw


def test_study_config_guards():
    with pytest.raises(ValueError, match="critic"):
        StudyConfig.model_validate({**TINY, "algo_overrides": {"critic.input": "local"}})
    with pytest.raises(ValueError, match="duplicate"):
        StudyConfig.model_validate({**TINY, "seeds": [0, 0]})
    with pytest.raises(ValueError, match="primary"):
        StudyConfig.model_validate({**TINY, "final_eval": {"stochastic": False}})
    with pytest.raises(ValueError):
        StudyConfig.model_validate({**TINY, "study": "../escape"})
    cfg = StudyConfig.model_validate(TINY)
    ippo, mappo = cfg.resolve_algo("ippo"), cfg.resolve_algo("mappo")
    # Overrides land identically on both algorithms; only critic.input differs.
    assert ippo.ppo == mappo.ppo and ippo.ppo.n_envs == 2
    assert ippo.model_dump(exclude={"name", "critic"}) == mappo.model_dump(exclude={"name", "critic"})


def test_plan_pairs_algorithms_per_seed():
    cfg = StudyConfig.model_validate({**TINY, "seeds": [0, 1]})
    assert [j.key for j in plan_train_jobs(cfg)] == [
        "train/ippo/L1/s0", "train/mappo/L1/s0", "train/ippo/L1/s1", "train/mappo/L1/s1"]


def test_run_study_end_to_end_is_idempotent(tmp_path):
    cfg, kw = _setup(tmp_path)
    m = run_study(cfg, parallel=0, **kw)
    jobs = m.data["jobs"]
    train_keys = [k for k in jobs if k.startswith("train/")]
    assert len(train_keys) == 2 and all(jobs[k]["status"] == "completed" for k in jobs), jobs

    # Manifest integrity: recorded config hashes are the hashes of the resolved configs.
    for job in plan_train_jobs(cfg):
        assert jobs[job.key]["config_hash"] == config_hash(resolved_run_config(cfg, job.algo, job.level, job.seed))
        assert Path(jobs[job.key]["run_dir"]).name.endswith(jobs[job.key]["config_hash"])
    assert m.data["inputs_sha256"] == frozen_inputs(cfg, kw["study_yaml"])

    # Held-out eval files: L1 has no structured set -> skipped, test written with both modes.
    run_dir = Path(jobs["train/ippo/L1/s0"]["run_dir"])
    ev = json.loads((run_dir / "eval" / "test_L1.json").read_text())
    assert set(ev["modes"]) == {"stochastic", "greedy"}
    assert ev["modes"]["stochastic"]["aggregate"]["n_episodes"] == 4  # 2 maps x eval seeds {0,1}
    assert not (run_dir / "eval" / "structured_L1.json").exists()
    assert (tmp_path / "studies" / "tiny" / "baselines" / "random" / "L1" / "s1" / "test.json").is_file()
    val = [json.loads(x) for x in (run_dir / "eval_val.jsonl").read_text().splitlines()]
    assert val[0]["env_steps"] == 0  # step-0 validation point

    # Every held-out access is logged, and only after training completed.
    log = [json.loads(x) for x in kw["access_log"].read_text().splitlines()]
    assert {e["who"] for e in log} == {"final_eval", "baseline"} and all(e["split"] == "test" for e in log)

    # Second run: nothing retrains, nothing re-evaluates.
    mtime = (run_dir / "eval" / "test_L1.json").stat().st_mtime_ns
    m2 = run_study(cfg, parallel=0, **kw)
    assert all(m2.data["jobs"][k]["action"] == "skip" for k in train_keys)
    assert (run_dir / "eval" / "test_L1.json").stat().st_mtime_ns == mtime
    assert all(e.get("written", []) == [] for k, e in m2.data["jobs"].items() if not k.startswith("train/"))


def test_interrupted_run_is_resumed_not_restarted(tmp_path):
    cfg, kw = _setup(tmp_path, algos=["ippo"])
    run_study(cfg, parallel=0, stages=("train",), **kw)
    job = plan_train_jobs(cfg)[0]
    h8 = config_hash(resolved_run_config(cfg, job.algo, job.level, job.seed))
    (run_dir,) = (tmp_path / "runs" / "tiny" / "ippo" / "L1").iterdir()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["status"] = "interrupted"
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    state = classify_run(job, h8, tmp_path / "runs")
    assert state.action == "resume" and state.run_dir == run_dir
    m = run_study(cfg, parallel=0, stages=("train",), **kw)
    assert m.data["jobs"][job.key]["action"] == "resume"
    assert len(list((tmp_path / "runs" / "tiny" / "ippo" / "L1").iterdir())) == 1
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "completed"


def test_budget_mismatch_is_refused(tmp_path):
    cfg, kw = _setup(tmp_path, algos=["ippo"])
    run_study(cfg, parallel=0, stages=("train",), **kw)
    job = TrainJob("tiny", "ippo", "L1", 0, 64)
    h8 = config_hash(resolved_run_config(cfg, "ippo", "L1", 0))
    with pytest.raises(RuntimeError, match="refusing to mix budgets"):
        classify_run(job, h8, tmp_path / "runs")


def test_frozen_study_detects_changed_inputs(tmp_path):
    cfg, kw = _setup(tmp_path, algos=["ippo"])
    freeze_study(cfg, kw["study_yaml"], kw["frozen_file"])
    assert check_frozen("tiny", frozen_inputs(cfg, kw["study_yaml"]), kw["frozen_file"]) == []
    kw["study_yaml"].write_text(kw["study_yaml"].read_text() + "\n# edited after freezing\n")
    dev = check_frozen("tiny", frozen_inputs(cfg, kw["study_yaml"]), kw["frozen_file"])
    assert len(dev) == 1 and "changed since freeze" in dev[0]
    with pytest.raises(RuntimeError, match="frozen"):
        run_study(cfg, parallel=0, stages=("baselines",), **kw)
    m = run_study(cfg, parallel=0, stages=(), allow_deviation=True, **kw)
    assert m.data["deviations"] == dev


def test_heldout_gate_refuses_unfinished_runs_and_logs_access(tmp_path):
    log = tmp_path / "access.jsonl"
    with pytest.raises(HeldOutAccessError):
        load_heldout_maps("L1", "test", {"who": "t"}, run_status="running", log_path=log)
    assert not log.exists()
    with pytest.raises(ValueError):
        load_heldout_maps("L1", "validation", {"who": "t"}, log_path=log)
    maps = load_heldout_maps("L2", "structured", {"who": "t"}, run_status="completed", log_path=log)
    assert len(maps) == 12 and json.loads(log.read_text())["split"] == "structured"


def test_common_random_numbers_seed_derivation():
    # Same (map, eval seed) -> same draws for every learned method; Random's
    # sampling seeds give distinct, reproducible streams.
    assert episode_seed("m1", 0) == episode_seed("m1", 0)
    assert len({episode_seed("m1", 0), episode_seed("m1", 1), episode_seed("m2", 0)}) == 3
    assert episode_seed("m1", 0, 3) != episode_seed("m1", 0, 4)


def test_evolution_replays_use_same_maps_across_checkpoints_and_resimulate(tmp_path):
    from marl.envs.core.config import LevelRegistry
    from marl.experiments.outputs import record_evolution_replays
    from marl.replay.loader import load_replay, resimulate_matches

    evo = {"fractions": [0.0, 0.5, 1.0], "n_maps": 1, "split": "test", "seeds": [0]}
    cfg, kw = _setup(tmp_path, algos=["ippo"], evolution_replays=evo)
    run_study(cfg, parallel=0, stages=("train",), **kw)
    written = record_evolution_replays(cfg, studies_root=kw["studies_root"], replays_root=tmp_path / "replays",
                                       access_log=kw["access_log"])
    assert len(written) == 3  # 3 checkpoints x 1 map
    replays = [load_replay(p) for p in written]
    assert len({r.map.hash for r in replays}) == 1  # same map at every checkpoint
    steps = [r.policy.checkpoint_step for r in replays]
    assert steps == sorted(steps) and steps[0] == 0
    env_cfg = LevelRegistry().get("L1")
    assert all(resimulate_matches(r, env_cfg) for r in replays)
    index = json.loads((Path(written[0]).parent / "index.json").read_text())
    assert [e["fraction"] for e in index] == [0.0, 0.5, 1.0]


def test_sigterm_marks_training_run_interrupted(tmp_path):
    """A SIGTERM'd training process must leave a resumable `interrupted`
    manifest, not a stale `running` one."""
    import signal
    import subprocess
    import sys
    import time

    script = tmp_path / "t.py"
    script.write_text(f"""
import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})
from pathlib import Path
from marl.experiments.workers import _raise_on_sigterm
from marl.algorithms.common.config import AlgoConfig
from marl.algorithms.ippo.agent import IPPOAgent
from marl.envs.core.config import LevelRegistry
from marl.training.trainer import train
_raise_on_sigterm()
cfg = AlgoConfig.model_validate({{"name": "ippo"}})
cfg.ppo.n_envs, cfg.ppo.rollout_len, cfg.eval.n_val_maps, cfg.eval.interval_updates = 2, 8, 1, 1000
run = Path({str(tmp_path / 'run')!r}); (run / "checkpoints").mkdir(parents=True)
train(LevelRegistry().get("L1"), cfg, IPPOAgent, 0, 10**9, run, "t", "ippo", "L1")
""")
    proc = subprocess.Popen([sys.executable, str(script)])
    latest = tmp_path / "run" / "checkpoints" / "latest.pt"
    deadline = time.time() + 120
    while not latest.exists() and time.time() < deadline:
        time.sleep(0.5)
    assert latest.exists()
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=60) == 143
    assert json.loads((tmp_path / "run" / "manifest.json").read_text())["status"] == "interrupted"
