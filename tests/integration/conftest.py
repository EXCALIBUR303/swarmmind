"""Integration fixtures: a REAL uvicorn server (real sockets, real Host/Origin
handling, real subprocess training jobs) over a results/ tree that contains
ONLY the committed demo bundle — i.e. what a clean clone has after
`scripts/demo_setup.sh`. Nothing here depends on the developer's own 6 GB
`results/` directory, so these tests mean the same thing on any machine."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from marl.utils.paths import REPO_ROOT

BUNDLE = REPO_ROOT / "demo_data"
ORIGIN = "http://localhost:5173"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture(scope="session")
def bundle_results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not (BUNDLE / "MANIFEST.json").is_file():
        pytest.fail("demo_data/ is missing: the integration suite must run on a full checkout")
    root = tmp_path_factory.mktemp("sm_integration")
    results = root / "results"
    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "demo_setup.py"), "--results", str(results)],
        capture_output=True, text=True, check=False,
    )
    assert out.returncode == 0, out.stderr
    (root / "models").mkdir()
    return results


@pytest.fixture(scope="session")
def server(bundle_results: Path) -> Iterator[str]:
    port = _free_port()
    env = {
        **os.environ,
        "SWARMMIND_RESULTS_DIR": str(bundle_results),
        "SWARMMIND_MODELS_DIR": str(bundle_results.parent / "models"),
        "SWARMMIND_PORT": str(port),
    }
    log = bundle_results.parent / "server.log"
    with log.open("w") as logf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1",
             "--port", str(port), "--ws-max-size", "65536"],
            cwd=REPO_ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            deadline = time.time() + 60
            while time.time() < deadline:
                if proc.poll() is not None:
                    pytest.fail(f"server exited early:\n{log.read_text()[-2000:]}")
                try:
                    if httpx.get(f"{base}/api/health", timeout=1).status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.25)
            else:
                pytest.fail("server did not become healthy in 60 s")
            yield base
        finally:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()


@pytest.fixture(scope="session")
def api(server: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=f"{server}/api", timeout=30) as c:
        yield c


def ws_url(server: str, path: str) -> str:
    return server.replace("http://", "ws://") + path
