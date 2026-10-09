"""Performance check (Phase 14): measured numbers for the claims the docs make.

Runs against the committed demo bundle in a temp `results/` and a real uvicorn process. Writes
`docs/PERFORMANCE.md`. Nothing is estimated: every figure is a measurement from this run on this
machine, with the machine named in the report so it is not mistaken for a general claim.

    .venv/bin/python scripts/perf_check.py [--out docs/PERFORMANCE.md]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import statistics as st
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np
import websockets

from marl.utils.paths import REPO_ROOT

LEVEL = "L2"


def ms(xs: list[float]) -> str:
    xs = sorted(xs)
    return f"median {st.median(xs):.1f} ms, p95 {xs[min(len(xs) - 1, int(len(xs) * 0.95))]:.1f} ms (n={len(xs)})"


def env_throughput() -> dict:
    """Same loop as scripts/bench_env.py: observation + action-mask construction + step (what a rollout does each tick)."""
    from marl.envs.core.config import LevelRegistry
    from marl.envs.core.world import SwarmWorld

    cfg = LevelRegistry().get(LEVEL)
    world = SwarmWorld(cfg)
    world.reset(seed=0)
    rng = np.random.default_rng(0)
    n_steps, t0 = 20_000, time.perf_counter()
    for _ in range(n_steps):
        _, _, masks = world.all_actor_obs()
        actions = np.array([rng.choice(np.where(masks[i] == 1)[0]) if np.any(masks[i] == 1) else 0 for i in range(cfg.n_robots)])
        result = world.step(actions)
        if result.terminated or result.truncated:
            world.reset(seed=int(rng.integers(0, 1_000_000)))
    dt = time.perf_counter() - t0
    return {"steps": n_steps, "seconds": dt, "steps_per_s": n_steps / dt}


def policy_latency(actor_path: Path) -> dict:
    """Wall time of one full episode with the learned actor through the evaluator (obs, masks, forward pass, env step)."""
    from marl.algorithms.common.policy import LearnedActorPolicy, load_actor
    from marl.envs.core.config import LevelRegistry
    from marl.evaluation.evaluator import run_episode

    actor, _ = load_actor(str(actor_path))
    cfg = LevelRegistry().get(LEVEL)
    per_step, total_steps = [], 0
    for seed in range(10):
        t0 = time.perf_counter()
        res = run_episode(cfg, LearnedActorPolicy(actor, stochastic=True), np.random.default_rng(seed), seed=seed)
        dt = time.perf_counter() - t0
        n = len(res.trace.actions)
        total_steps += n
        per_step.append(dt / n * 1000)
    return {"episodes": 10, "steps": total_steps, "ms_per_step": per_step}


async def ws_rate(server: str, session_id: str, speed: float) -> dict:
    stamps: list[float] = []
    async with websockets.connect(server.replace("http", "ws") + f"/ws/sessions/{session_id}", origin="http://localhost:5173") as ws:
        await ws.recv()
        await ws.send(json.dumps({"cmd": "speed", "value": speed}))
        await ws.recv()
        await ws.send(json.dumps({"cmd": "play"}))
        while True:
            m = json.loads(await asyncio.wait_for(ws.recv(), 60))
            if m["type"] == "frame":
                stamps.append(time.perf_counter())
            elif m["type"] == "episode_end":
                break
    gaps = [(b - a) * 1000 for a, b in zip(stamps, stamps[1:], strict=False)]
    return {"frames": len(stamps), "rate": (len(stamps) - 1) / (stamps[-1] - stamps[0]) if len(stamps) > 2 else None, "gaps": gaps}


def backend(results: Path) -> dict:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "SWARMMIND_RESULTS_DIR": str(results), "SWARMMIND_MODELS_DIR": str(results.parent / "models")}
    (results.parent / "models").mkdir(exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", str(port), "--ws-max-size", "65536"],
                            cwd=REPO_ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    out: dict = {}
    try:
        for _ in range(120):
            try:
                if httpx.get(f"{base}/api/health", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.25)
        c = httpx.Client(base_url=f"{base}/api", timeout=60)
        models = c.get("/models").json()
        mappo = next(m for m in models if m["algo"] == "mappo")["model_id"]

        def timed(fn, n=20):
            xs = []
            for i in range(n):
                t0 = time.perf_counter()
                fn(i)
                xs.append((time.perf_counter() - t0) * 1000)
            return xs

        for name, path in (("GET /health", "/health"), ("GET /models", "/models"), ("GET /experiments", "/experiments"),
                           ("GET /studies/main_L2 (summary + manifest)", "/studies/main_L2"), ("GET /replays (index)", "/replays")):
            out[name] = timed(lambda _i, p=path: c.get(p).raise_for_status())
        rid = c.get("/replays").json()[0]["replay_id"]
        r = c.get(f"/replays/{rid}")
        out[f"GET /replays/<id> ({len(r.content) / 1024:.0f} KiB)"] = timed(lambda _i: c.get(f"/replays/{rid}").raise_for_status())
        body = {"level": LEVEL, "policy": {"kind": "mappo", "model_id": mappo}, "stochastic": True}
        sids: list[str] = []

        def create(i):
            r = c.post("/sessions", json={**body, "seed": i})
            r.raise_for_status()
            sids.append(r.json()["session_id"])
            if len(sids) > 3:  # sessions are capped (4); free as we go
                c.delete(f"/sessions/{sids.pop(0)}")

        out["POST /sessions (MAPPO, procedural map)"] = timed(create)
        sid = sids[-1]
        steps = []
        while True:
            t0 = time.perf_counter()
            r = c.post(f"/sessions/{sid}/step", json={"n": 1}).json()
            steps.append((time.perf_counter() - t0) * 1000)
            if r["done"]:
                break
        out["POST /sessions/<id>/step n=1"] = steps
        for old in sids:  # the server caps live sessions at 4
            c.delete(f"/sessions/{old}")
        sid2 = c.post("/sessions", json={**body, "seed": 99}).json()["session_id"]
        out["ws8"] = asyncio.run(ws_rate(base, sid2, 8.0))
        c.delete(f"/sessions/{sid2}")
        sid3 = c.post("/sessions", json={**body, "seed": 5}).json()["session_id"]
        out["ws1"] = asyncio.run(ws_rate(base, sid3, 1.0))
        out["rss_mb"] = float(subprocess.run(["ps", "-o", "rss=", "-p", str(proc.pid)], capture_output=True, text=True).stdout.strip() or 0) / 1024
    finally:
        proc.terminate()
        proc.wait(10)
    return out


def bundle_sizes() -> dict:
    dist = REPO_ROOT / "frontend" / "dist"
    if not dist.is_dir():
        return {}
    import gzip

    js = [p for p in (dist / "assets").glob("*.js")]
    css = [p for p in (dist / "assets").glob("*.css")]
    gz = lambda ps: sum(len(gzip.compress(p.read_bytes())) for p in ps)  # noqa: E731
    return {"js": sum(p.stat().st_size for p in js), "js_gz": gz(js), "css": sum(p.stat().st_size for p in css), "css_gz": gz(css)}


def training_throughput() -> list[tuple[str, float, float]]:
    out = []
    runs = REPO_ROOT / "demo_data" / "results" / "runs" / "main_L2"
    for algo in ("ippo", "mappo"):
        rates = []
        for m in runs.glob(f"{algo}/L2/*/manifest.json"):
            d = json.loads(m.read_text())
            rates.append(d["trained_env_steps"] / d["wall_time_seconds"])
        out.append((algo, min(rates), max(rates)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "PERFORMANCE.md")
    args = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="sm_perf_"))
    res = tmp / "results"
    subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "demo_setup.py"), "--results", str(res)], check=True, capture_output=True)
    mappo_actor = next((res / "runs" / "main_L2" / "mappo").glob("L2/*/actor.pt"))
    print("env throughput ...", flush=True)
    env = env_throughput()
    print("policy latency ...", flush=True)
    pol = policy_latency(mappo_actor)
    print("backend ...", flush=True)
    be = backend(res)
    sizes = bundle_sizes()
    train = training_throughput()

    lines = ["# Performance", "",
             f"Generated by `scripts/perf_check.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}. Measured, not estimated; **one machine** "
             f"({platform.platform()}, {platform.machine()}, Python {platform.python_version()}, CPU only), single run. "
             "Treat these as order-of-magnitude figures for this laptop, not guarantees.", "",
             "## Simulation and policy", "",
             f"- Environment alone ({LEVEL}, random actions, single process): **{env['steps_per_s']:,.0f} env steps/s** ({env['steps']:,} steps in {env['seconds']:.1f} s).",
             f"- One full evaluator episode with the trained MAPPO actor (observations, masks, forward pass, env step): {ms(pol['ms_per_step'])} per environment step "
             f"({pol['steps']} steps over {pol['episodes']} episodes).",
             "- Training throughput recorded in the study manifests (end to end, including PPO updates and periodic validation, "
             "on this machine, 16 parallel envs): "
             + "; ".join(f"{a.upper()} {lo:.0f} env steps/s" if round(lo) == round(hi) else f"{a.upper()} {lo:.0f}–{hi:.0f} env steps/s" for a, lo, hi in train) + ".", "",
             "## Backend (real uvicorn process, demo bundle)", "", "| Endpoint | Latency |", "|---|---|"]  # /health shells out to git for the version block, hence its higher floor
    for k, v in be.items():
        if isinstance(v, list):
            lines.append(f"| {k} | {ms(v)} |")
    w8, w1 = be["ws8"], be["ws1"]
    lines += ["", f"- WebSocket frame delivery, 1× (nominal 4 steps/s): **{w1['rate']:.2f} frames/s** over {w1['frames']} frames.",
              f"- WebSocket frame delivery, 8× (requested 32 steps/s, server cap 30): **{w8['rate']:.1f} frames/s** over {w8['frames']} frames.",
              f"- Backend resident memory after the run: {be['rss_mb']:.0f} MB.", "",
              "## Frontend", ""]
    if sizes:
        lines.append(f"- Production bundle: JS {sizes['js'] / 1024:.0f} KiB ({sizes['js_gz'] / 1024:.0f} KiB gzipped), CSS {sizes['css'] / 1024:.0f} KiB ({sizes['css_gz'] / 1024:.0f} KiB gzipped).")
    e2e = REPO_ROOT / "docs" / "E2E_REPORT.md"
    line = next((ln for ln in e2e.read_text().splitlines() if "Rendering performance" in ln), None) if e2e.is_file() else None
    lines.append("- Canvas rendering, measured in the browser run (`frontend/e2e/run_e2e.py`, step 7): " + (line.split("|")[-2].strip() if line else "run the E2E to populate this line") + ".")
    lines += ["", "## What is *not* measured", "",
              "- No multi-user load test (the design is a single-user local app; at most 4 live sessions and 1 training job at a time).",
              "- No GPU measurements; no battery/thermal effects; Safari/Firefox rendering not measured."]
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
