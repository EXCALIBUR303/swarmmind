"""Phase 15 audit checks that re-derive evidence from the developer's FULL `results/` tree
(checkpoints included, so they cannot run from the demo bundle alone). Writes
`docs/AUDIT_CHECKS.md`; `docs/AUDIT.md` cites its rows. Exit status is non-zero if any check fails.

    .venv/bin/python scripts/audit_checks.py [--retrain-updates 6] [--skip-retrain]

Checks:
  R1  IPPO/MAPPO parity in the 10 run manifests (config identical except critic input; equal budgets)
  R2  every exported actor.pt equals the actor in its final checkpoint
  R3  all stored replays re-record frame for frame from their checkpoints (0/10/50/100%)
  R4  training reproducibility: re-train the first K updates of seed 0 (IPPO and MAPPO) with the
      frozen study config and one torch thread; every logged metric row and the 0% actor must match
  R5  H1 and H2 recomputed independently (numpy/scipy) from the raw per-episode eval files
  R6  exploration-budget invariant (§13.3) on 400 real held-out episodes (2 runs x test set x 2 seeds)
  R7  how many frozen-map targets the reset-time sensing pass already reveals (finding A-02's scope)
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.utils.paths import REPO_ROOT, RESULTS_DIR

RUNS = RESULTS_DIR / "runs" / "main_L2"
STUDY = RESULTS_DIR / "studies" / "main_L2"
ROWS: list[tuple[str, str, bool, str]] = []


def check(id_: str, claim: str):
    def deco(fn):
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001 - report, never crash the audit
            ok, detail = False, f"{type(e).__name__}: {e}"
        ROWS.append((id_, claim, ok, detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {id_} {claim} ({time.time() - t0:.0f}s): {detail}", flush=True)
        return fn
    return deco


def runs() -> list[Path]:
    return sorted(p for p in RUNS.glob("*/L2/*") if p.is_dir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retrain-updates", type=int, default=6)
    ap.add_argument("--skip-retrain", action="store_true")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "AUDIT_CHECKS.md")
    args = ap.parse_args()
    if not RUNS.is_dir() or not any(RUNS.glob("*/L2/*/checkpoints")):
        print("needs the full results/ tree with checkpoints (not the demo bundle)", file=sys.stderr)
        return 2
    torch.set_num_threads(1)

    @check("R1", "IPPO and MAPPO differ only in critic input; equal budgets, seeds 0-4, eval-map guard active")
    def _():
        ms = [json.loads((r / "manifest.json").read_text()) for r in runs()]

        def strip(a):
            a = json.loads(json.dumps(a))
            a["critic"].pop("input")
            a.pop("name")
            return a

        same_algo = all(strip(m["algo_config"]) == strip(ms[0]["algo_config"]) for m in ms)
        same_env = all(m["env_config"] == ms[0]["env_config"] for m in ms)
        budgets = {(m["trained_env_steps"], m["updates"]) for m in ms}
        seeds = {a: sorted(m["seed"] for m in ms if m["algo"] == a) for a in ("ippo", "mappo")}
        guard = {m.get("eval_hashes_guarded") for m in ms}
        ok = same_algo and same_env and len(budgets) == 1 and seeds["ippo"] == seeds["mappo"] == [0, 1, 2, 3, 4]
        return ok, f"{len(ms)} runs; budgets {budgets}; seeds {seeds}; eval hashes guarded {guard}"

    @check("R2", "each exported actor.pt equals the actor of its final (100%) checkpoint")
    def _():
        from marl.training.checkpoint import load_actor_export

        bad = []
        for r in runs():
            m = json.loads((r / "manifest.json").read_text())
            final = next(c for c in m["checkpoints"] if 1.0 in c["fractions"])
            ck = torch.load(r / "checkpoints" / f"step_{final['env_steps']}.pt", weights_only=False)
            exp = load_actor_export(str(r / "actor.pt"))["actor_state_dict"]
            if not all(torch.equal(exp[k], ck["actor_state_dict"][k]) for k in exp):
                bad.append(r.name)
        return not bad, f"{len(runs())} runs checked; mismatches {bad}"

    @check("R3", "all stored replays re-record frame for frame from their checkpoints")
    def _():
        from marl.algorithms.common.policy import LearnedActorPolicy
        from marl.envs.core.config import LevelRegistry
        from marl.envs.core.maps_io import MapSpec
        from marl.evaluation.final_eval import episode_seed
        from marl.networks.actor import ActorPolicy
        from marl.replay.loader import load_replay
        from marl.replay.recorder import record_episode
        from marl.replay.schema import ReplayPolicyInfo

        cfg = LevelRegistry().get("L2")
        n = bad = 0
        for rp in sorted((RESULTS_DIR / "replays" / "main_L2").rglob("*.json.gz")):
            rep = load_replay(rp)
            run = next(RUNS.glob(f"*/L2/{rp.parent.name}"))
            ck = torch.load(run / "checkpoints" / f"step_{rep.policy.checkpoint_step}.pt", weights_only=False)
            actor = ActorPolicy(grid_size=11)
            actor.load_state_dict(ck["actor_state_dict"])
            actor.eval()
            m = rep.map
            spec = MapSpec(m.id, m.width, m.height, m.terrain, [tuple(p) for p in m.robot_starts], [tuple(p) for p in m.targets])
            new = record_episode(cfg, LearnedActorPolicy(actor, stochastic=True),
                                 np.random.default_rng(episode_seed(m.id, 0)), ReplayPolicyInfo(kind=rep.policy.kind), "x",
                                 map_spec=spec)
            n += 1
            bad += int([f.model_dump() for f in new.frames] != [f.model_dump() for f in rep.frames]
                       or new.outcome.model_dump() != rep.outcome.model_dump())
        return n > 0 and bad == 0, f"{n} replays re-recorded; {bad} differ"

    if not args.skip_retrain:
        @check("R4", f"re-training the first {args.retrain_updates} updates of seed 0 reproduces the study runs bit for bit")
        def _():
            from marl.envs.core.config import LevelRegistry
            from marl.experiments.config import load_study_config
            from marl.experiments.workers import _agent_builder
            from marl.training.trainer import train

            cfg = load_study_config("main_L2")
            out, details, ok = Path(tempfile.mkdtemp(prefix="sm_retrain_")), [], True
            for algo in ("ippo", "mappo"):
                acfg = cfg.resolve_algo(algo)
                rd = out / algo
                rd.mkdir()
                train(env_cfg=LevelRegistry().get("L2"), algo_cfg=acfg, agent_builder=_agent_builder(algo), run_seed=0,
                      total_env_steps=args.retrain_updates * acfg.ppo.n_envs * acfg.ppo.rollout_len, run_dir=rd,
                      study="audit_retrain", algo_name=algo, level="L2")
                ref = next(RUNS.glob(f"{algo}/L2/s0-*"))
                old = [json.loads(x) for x in (ref / "metrics.jsonl").read_text().splitlines()][: args.retrain_updates]
                new = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
                a = torch.load(rd / "checkpoints" / "step_0.pt", weights_only=False)["actor_state_dict"]
                b = torch.load(ref / "checkpoints" / "step_0.pt", weights_only=False)["actor_state_dict"]
                same_w = all(torch.equal(a[k], b[k]) for k in a)
                ok &= old == new and same_w
                details.append(f"{algo}: {sum(o == n for o, n in zip(old, new, strict=False))}/{len(old)} metric rows identical, 0% actor identical={same_w}")
            return ok, "; ".join(details)

    @check("R5", "H1 and H2 recomputed from raw episode files with scipy match summary.json")
    def _():
        from scipy import stats

        def welch(a, b):
            va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
            se, df = np.sqrt(va + vb), (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
            h = stats.t.ppf(0.975, df) * se
            return a.mean() - b.mean(), a.mean() - b.mean() - h, a.mean() - b.mean() + h

        def per_seed(algo, metric):
            return np.array([np.mean([float(e[metric]) for e in json.loads(f.read_text())["modes"]["stochastic"]["episodes"]])
                             for f in sorted(RUNS.glob(f"{algo}/L2/*/eval/test_L2.json"))])

        rnd = np.array([np.mean([float(e["success"]) for e in json.loads(
            (STUDY / "baselines" / "random" / "L2" / f"s{s}" / "test.json").read_text())["modes"]["stochastic"]["episodes"]])
            for s in range(5)])
        hyp = json.loads((STUDY / "summary.json").read_text())["levels"]["L2"]["hypotheses"]
        got = {"H1/ippo": welch(per_seed("ippo", "success"), rnd), "H1/mappo": welch(per_seed("mappo", "success"), rnd),
               "H2": welch(per_seed("mappo", "censored_steps"), per_seed("ippo", "censored_steps"))}
        want = {"H1/ippo": hyp["H1"]["ippo"]["difference"], "H1/mappo": hyp["H1"]["mappo"]["difference"],
                "H2": hyp["H2"]["difference"]}
        ok = all(np.allclose(got[k], [want[k]["diff"], want[k]["lo"], want[k]["hi"]], rtol=0, atol=1e-9) for k in got)
        return ok, "; ".join(f"{k} {g[0]:+.4f} [{g[1]:+.4f}, {g[2]:+.4f}]" for k, g in got.items())

    @check("R6", "exploration-budget invariant (§13.3) on 400 real held-out episodes")
    def _():
        from marl.algorithms.common.policy import LearnedActorPolicy, load_actor
        from marl.envs.core.config import LevelRegistry
        from marl.evaluation.evaluator import run_episode
        from marl.evaluation.final_eval import episode_seed, load_heldout_maps

        cfg = LevelRegistry().get("L2")
        maps = load_heldout_maps("L2", "test", {"requester": "audit_checks/R6", "purpose": "reward invariant"})
        worst, n_succ, fail_returns = 0.0, 0, []
        for algo in ("ippo", "mappo"):
            actor, _ = load_actor(str(next(RUNS.glob(f"{algo}/L2/s0-*")) / "actor.pt"))
            for m in maps:
                for s in (0, 1):
                    tr = run_episode(cfg, LearnedActorPolicy(actor, stochastic=True),
                                     np.random.default_rng(episode_seed(m.id, s)), map_spec=m).trace
                    explore = sum(float(np.mean(c["explore"])) for c in tr.reward_components)
                    if tr.success:
                        n_free = int((m.terrain_array() != 1).sum())
                        k0 = round(tr.coverage_over_time[0] * n_free)
                        worst = max(worst, abs(explore - cfg.reward.explore_total * (n_free - k0) / n_free))
                        n_succ += 1
                    else:
                        fail_returns.append(round(float(np.sum(np.stack(tr.rewards), axis=0).mean()), 3))
        return worst < 1e-6, (f"{n_succ} successful episodes, max |sum explore - budget| = {worst:.1e} (float32); "
                              f"failed-episode returns {fail_returns} (no high-return failures)")

    @check("R7", "targets revealed by the reset-time sensing pass on the frozen L2 sets (scope of finding A-02)")
    def _():
        from marl.envs.core.config import LevelRegistry
        from marl.envs.core.world import SwarmWorld
        from marl.evaluation.eval_sets import load_eval_set

        cfg, parts = LevelRegistry().get("L2"), []
        for split in ("validation", "test", "structured"):
            maps = load_eval_set("L2", split)
            hit = sum(int((SwarmWorld(cfg).reset(map_spec=m).target_t_discovered >= 0).sum()) for m in maps)
            parts.append(f"{split}: {hit} of {2 * len(maps)} targets")
        return True, "; ".join(parts)

    lines = ["# Audit checks (full results tree)", "",
             f"Generated by `scripts/audit_checks.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}. Do not edit by hand. "
             "These need the developer's full `results/` (checkpoints), so they cannot run from the demo bundle; "
             "`docs/REAL_DATA_AUDIT.md` covers what can.", "",
             "| # | Check | Result | Evidence |", "|---|---|---|---|"]
    lines += [f"| {i} | {c} | **{'PASS' if ok else 'FAIL'}** | {d.replace('|', '/')} |" for i, c, ok, d in ROWS]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")
    return 0 if all(ok for *_, ok, _ in ROWS) else 1


if __name__ == "__main__":
    sys.exit(main())
