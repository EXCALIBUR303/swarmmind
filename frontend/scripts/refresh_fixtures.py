"""Refresh the frontend test fixtures that are whole API responses, from a real backend.

The fixtures in `src/features/__fixtures__/` are captured API payloads (never hand-written);
`scripts/audit_real_data.py` (row E1) fails when one no longer matches the real data. This
script re-captures the study-scoped ones from a backend serving ONLY the committed demo bundle.
List fixtures that also hold data outside the bundle (`studies.json`) are merged: entries for
studies the bundle has are replaced, the others are kept as captured.

    .venv/bin/python frontend/scripts/refresh_fixtures.py
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
FX = REPO / "frontend" / "src" / "features" / "__fixtures__"
STUDY = "main_L2"


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="sm_fixtures_"))
    res = tmp / "results"
    subprocess.run([sys.executable, str(REPO / "scripts" / "demo_setup.py"), "--results", str(res)], check=True,
                   capture_output=True)
    (tmp / "models").mkdir()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "SWARMMIND_RESULTS_DIR": str(res), "SWARMMIND_MODELS_DIR": str(tmp / "models")}
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", str(port)],
                            cwd=REPO, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        api = httpx.Client(base_url=f"http://127.0.0.1:{port}/api", timeout=60)
        for _ in range(120):
            try:
                if api.get("/health").status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.25)
        run = json.loads((FX / "run_detail.json").read_text())["summary"]["run_id"]
        whole = {
            "study.json": f"/studies/{STUDY}",
            "run_eval.json": f"/experiments/{run}/eval",
            "analysis_L2_ippo_example_episode.json": f"/studies/{STUDY}/analysis/L2_ippo_example_episode",
            "analysis_L2_mappo_example_episode.json": f"/studies/{STUDY}/analysis/L2_mappo_example_episode",
        }
        for name, path in whole.items():
            (FX / name).write_text(json.dumps(api.get(path).raise_for_status().json()))
            print(f"refreshed {name} <- GET /api{path}")
        fresh = {s["study"]: s for s in api.get("/studies").raise_for_status().json()}
        old = json.loads((FX / "studies.json").read_text())
        (FX / "studies.json").write_text(json.dumps([fresh.get(s["study"], s) for s in old]))
        print(f"merged studies.json <- GET /api/studies ({', '.join(fresh)} replaced)")
    finally:
        proc.terminate()
        proc.wait(10)
    return 0


if __name__ == "__main__":
    sys.exit(main())
