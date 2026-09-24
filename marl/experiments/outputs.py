"""Study outputs: aggregation → summary.json + figures + RESULTS.md, and
Swarm Evolution replays (Phase 6 tasks 3-4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from marl.analysis.aggregate import aggregate_study
from marl.analysis.figures import make_study_figures
from marl.analysis.report import render_results_md
from marl.experiments.config import StudyConfig
from marl.experiments.runner import _atomic_json, _read_json, plan_train_jobs
from marl.utils.paths import DOCS_DIR, REPLAYS_DIR, STUDIES_DIR


def planned_runs(cfg: StudyConfig, studies_root: Path = STUDIES_DIR) -> list[dict[str, Any]]:
    """Every planned learned run with its run_dir from the study manifest
    (None if it never completed), so missing runs are reported, not hidden."""
    manifest = _read_json(studies_root / cfg.study / "study_manifest.json") or {"jobs": {}}
    out = []
    for job in plan_train_jobs(cfg):
        entry = manifest["jobs"].get(job.key, {})
        run_dir = entry.get("run_dir") if entry.get("status") == "completed" else None
        out.append({"algo": job.algo, "level": job.level, "seed": job.seed, "run_dir": run_dir})
    return out


def write_study_outputs(
    cfg: StudyConfig,
    studies_root: Path = STUDIES_DIR,
    results_md: Path | None = None,
    figures_dir: Path | None = None,
) -> dict[str, Any]:
    study_dir = studies_root / cfg.study
    summary = aggregate_study(study_dir, planned_runs(cfg, studies_root), cfg.model_dump())
    _atomic_json(study_dir / "summary.json", summary)
    fig_dir = figures_dir or (DOCS_DIR / "figures" / cfg.study)
    figures = make_study_figures(summary, fig_dir)
    md_path = results_md or (DOCS_DIR / "RESULTS.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_results_md(summary, figures, md_path))
    return {"summary": str(study_dir / "summary.json"), "figures": [str(f) for f in figures], "results_md": str(md_path)}


def _checkpoint_for_fraction(manifest: dict[str, Any], fraction: float) -> dict[str, Any] | None:
    for ck in manifest.get("checkpoints", []):
        if any(abs(f - fraction) < 1e-9 for f in ck.get("fractions", [])):
            return ck
    return None


def record_evolution_replays(
    cfg: StudyConfig,
    studies_root: Path = STUDIES_DIR,
    replays_root: Path = REPLAYS_DIR,
    access_log: Path | None = None,
) -> list[str]:
    """Swarm Evolution (§26, §27.3): the SAME maps and evaluation seed at
    checkpoints 0/10/50/100% of each selected run, so the only thing that
    changes between the replays is training progress."""
    import torch

    from marl.algorithms.common.policy import LearnedActorPolicy
    from marl.envs.core.config import LevelRegistry
    from marl.evaluation.eval_sets import eval_set_exists, load_eval_set
    from marl.evaluation.final_eval import episode_seed, load_heldout_maps
    from marl.networks.actor import ActorPolicy
    from marl.replay.loader import save_replay
    from marl.replay.recorder import record_episode
    from marl.replay.schema import ReplayPolicyInfo

    ev = cfg.evolution_replays
    if ev is None:
        return []
    written: list[str] = []
    runs = [r for r in planned_runs(cfg, studies_root) if r["run_dir"] and r["seed"] in ev.seeds]
    for r in runs:
        rd = Path(r["run_dir"])
        manifest = _read_json(rd / "manifest.json") or {}
        level = r["level"]
        if not eval_set_exists(level, ev.split):
            continue
        requester = {"who": "evolution_replays", "run_id": rd.name}
        if ev.split == "validation":
            maps = load_eval_set(level, ev.split)[: ev.n_maps]
        else:
            maps = load_heldout_maps(level, ev.split, requester, manifest.get("status"), access_log)[: ev.n_maps]
        env_cfg = LevelRegistry().get(level)
        out_dir = replays_root / cfg.study / rd.name
        index = []
        for frac in ev.fractions:
            ck = _checkpoint_for_fraction(manifest, frac)
            if ck is None:
                continue
            state = torch.load(ck["path"], map_location="cpu", weights_only=False)
            actor = ActorPolicy(grid_size=2 * env_cfg.obs_radius + 1)
            actor.load_state_dict(state["actor_state_dict"])
            actor.eval()
            for m in maps:
                rid = f"{m.id}_{r['algo']}_s{r['seed']}_step{ck['env_steps']}_eval0"
                replay = record_episode(
                    env_cfg,
                    LearnedActorPolicy(actor, stochastic=True),
                    np.random.default_rng(episode_seed(m.id, 0)),
                    ReplayPolicyInfo(kind=r["algo"], run_id=rd.name, checkpoint_step=ck["env_steps"], eval_seed=0),
                    replay_id=rid,
                    map_spec=m,
                )
                path = out_dir / f"{rid}.json.gz"
                save_replay(replay, path)
                written.append(str(path))
                index.append({"replay_id": rid, "map_id": m.id, "algo": r["algo"], "seed": r["seed"],
                              "fraction": frac, "checkpoint_step": ck["env_steps"],
                              "success": replay.outcome.success, "steps": replay.outcome.steps})
        if index:
            (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return written
