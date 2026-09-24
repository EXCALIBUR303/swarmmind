"""Study worker functions. Each runs in its own spawned process (one torch
thread), returns a plain status record and never raises, so one failing
job cannot take the study down."""

from __future__ import annotations

import os
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from marl.envs.core.config import LevelRegistry
from marl.evaluation.eval_sets import eval_set_exists
from marl.experiments.config import StudyConfig
from marl.experiments.jobs import TrainJob, _read_json, classify_run, resolved_run_config
from marl.utils.hashing import config_hash


def _raise_on_sigterm() -> None:
    """Turn SIGTERM into SystemExit so the trainer's handler marks the run
    `interrupted` (resumable) instead of the process dying mid-write."""
    import signal
    import sys

    signal.signal(signal.SIGTERM, lambda _sig, _frame: sys.exit(143))


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)
    _raise_on_sigterm()


def _agent_builder(algo: str):
    if algo == "ippo":
        from marl.algorithms.ippo.agent import IPPOAgent

        return IPPOAgent
    if algo == "mappo":
        from marl.algorithms.mappo.agent import MAPPOAgent

        return MAPPOAgent
    raise ValueError(f"unknown algo {algo!r}")


def train_worker(cfg_dict: dict[str, Any], job_dict: dict[str, Any], runs_root: str) -> dict[str, Any]:
    """Train one (algo, level, seed); returns a status record. Never raises."""
    from marl.training.run_dir import make_run_dir
    from marl.training.trainer import train

    cfg = StudyConfig.model_validate(cfg_dict)
    job = TrainJob(**job_dict)
    root = Path(runs_root)
    record: dict[str, Any] = {"key": job.key, "status": "failed", "run_dir": None, "error": None}
    try:
        resolved = resolved_run_config(cfg, job.algo, job.level, job.seed)
        h8 = config_hash(resolved)
        record["config_hash"] = h8
        state = classify_run(job, h8, root)
        record["action"] = state.action
        if state.action in ("skip", "busy"):
            record.update(status="completed" if state.action == "skip" else "busy", run_dir=str(state.run_dir))
            return record
        run_dir = state.run_dir or make_run_dir(job.study, job.algo, job.level, job.seed, resolved, base_dir=root)
        record["run_dir"] = str(run_dir)
        (run_dir / ".lock").write_text(str(os.getpid()))
        try:
            result = train(
                env_cfg=LevelRegistry().get(job.level),
                algo_cfg=cfg.resolve_algo(job.algo),
                agent_builder=_agent_builder(job.algo),
                run_seed=job.seed,
                total_env_steps=job.total_env_steps,
                run_dir=run_dir,
                study=job.study,
                algo_name=job.algo,
                level=job.level,
                resume_from=run_dir / "checkpoints" / "latest.pt" if state.action == "resume" else None,
            )
        finally:
            (run_dir / ".lock").unlink(missing_ok=True)
        record["status"] = result.status
    except Exception:
        record["error"] = traceback.format_exc(limit=8)
    return record


def _eval_key(run_dir: Path) -> str:
    """eval/<algo>/<level>/<run name> — run dirs are <study>/<algo>/<level>/<name>."""
    return f"eval/{run_dir.parent.parent.name}/{run_dir.parent.name}/{run_dir.name}"


def eval_worker(
    cfg_dict: dict[str, Any], run_dir: str, max_maps: int | None = None, access_log: str | None = None
) -> dict[str, Any]:
    """Final held-out evaluation of one completed run (§22.1). Never raises."""
    from marl.algorithms.common.policy import LearnedActorPolicy, load_actor
    from marl.evaluation.final_eval import (
        eval_file_is_current,
        evaluate_policy,
        load_heldout_maps,
        protocol_fields,
        write_eval_file,
    )

    cfg = StudyConfig.model_validate(cfg_dict)
    rd = Path(run_dir)
    record: dict[str, Any] = {"key": _eval_key(rd), "run_dir": run_dir, "status": "failed", "error": None}
    try:
        m = _read_json(rd / "manifest.json") or {}
        level, fe = m["level"], cfg.final_eval
        record.update(written=[], skipped=[])
        actor = None
        for split in fe.map_sets:
            if not eval_set_exists(level, split):
                record["skipped"].append(f"{split}: no frozen set for {level}")
                continue
            out = rd / "eval" / f"{split}_{level}.json"
            proto = protocol_fields(level, split, fe.eval_seeds, fe.modes)
            proto["max_maps"] = max_maps
            if eval_file_is_current(out, proto):
                record["skipped"].append(f"{split}: current")
                continue
            maps = load_heldout_maps(
                level,
                split,
                requester={"who": "final_eval", "run_id": rd.name, "algo": m.get("algo"), "seed": m.get("seed")},
                run_status=m.get("status"),
                log_path=Path(access_log) if access_log else None,
            )[:max_maps]
            if actor is None:
                actor, _meta = load_actor(str(rd / "actor.pt"))
            env_cfg = LevelRegistry().get(level)
            payload: dict[str, Any] = {
                "protocol": proto,
                "run_id": rd.name,
                "algo": m.get("algo"),
                "seed": m.get("seed"),
                "checkpoint": "final",
                "modes": {},
            }
            for mode in fe.modes:
                stochastic = mode == "stochastic"

                def make_policy(actor: Any = actor, stochastic: bool = stochastic) -> LearnedActorPolicy:
                    return LearnedActorPolicy(actor, stochastic=stochastic)

                payload["modes"][mode] = evaluate_policy(env_cfg, make_policy, maps, fe.eval_seeds)
            write_eval_file(out, payload)
            record["written"].append(str(out))
        record["status"] = "completed"
    except Exception:
        record["error"] = traceback.format_exc(limit=8)
    return record


def baseline_worker(
    cfg_dict: dict[str, Any],
    kind: str,
    level: str,
    sampling_seed: int | None,
    studies_root: str,
    max_maps: int | None = None,
    access_log: str | None = None,
) -> dict[str, Any]:
    """Random (one evaluation sampling seed) or heuristic on the validation
    set (learning-curve reference) and every held-out split. Never raises."""
    from marl.algorithms.baselines.frontier_heuristic import FrontierHeuristicPolicy
    from marl.algorithms.baselines.random_policy import RandomPolicy
    from marl.evaluation.eval_sets import load_eval_set
    from marl.evaluation.final_eval import (
        eval_file_is_current,
        evaluate_policy,
        load_heldout_maps,
        protocol_fields,
        write_eval_file,
    )

    cfg = StudyConfig.model_validate(cfg_dict)
    tag = f"s{sampling_seed}" if sampling_seed is not None else "ref"
    key = f"baseline/{kind}/{level}/{tag}"
    record: dict[str, Any] = {"key": key, "status": "failed", "error": None, "written": []}
    try:
        make: Callable[[], Any] = RandomPolicy if kind == "random" else FrontierHeuristicPolicy
        env_cfg = LevelRegistry().get(level)
        fe = cfg.final_eval
        out_dir = Path(studies_root) / cfg.study / "baselines" / kind / level / tag
        for split in ["validation", *fe.map_sets]:
            if not eval_set_exists(level, split):
                continue
            out = out_dir / f"{split}.json"
            proto = protocol_fields(level, split, fe.eval_seeds, ["stochastic"])
            proto["max_maps"] = max_maps
            if eval_file_is_current(out, proto):
                continue
            if split == "validation":
                maps = load_eval_set(level, split)[:max_maps]
            else:
                maps = load_heldout_maps(
                    level, split, requester={"who": "baseline", "kind": kind, "sampling_seed": sampling_seed},
                    log_path=Path(access_log) if access_log else None,
                )[:max_maps]
            res = evaluate_policy(env_cfg, make, maps, fe.eval_seeds, sampling_seed=sampling_seed)
            write_eval_file(out, {"protocol": proto, "kind": kind, "sampling_seed": sampling_seed,
                                  "modes": {"stochastic": res}})
            record["written"].append(str(out))
        record["status"] = "completed"
    except Exception:
        record["error"] = traceback.format_exc(limit=8)
    return record
