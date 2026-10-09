"""Browser end-to-end run of the demo flow (Phase 14).

Starts, from scratch and on its own: a results/ tree holding ONLY the committed
demo bundle, the FastAPI backend, the PRODUCTION frontend build (`vite preview`)
and a fresh headless Chrome driven over the DevTools protocol (no Playwright:
it would add a second browser download; this needs only `websockets`, which the
backend already depends on). Every assertion compares what the page shows with
the real result file or the real stored replay it must come from.

Usage (from the repo root, after `make setup`, `scripts/demo_setup.sh`, `npm run build`):
    .venv/bin/python frontend/e2e/run_e2e.py [--out docs/E2E_REPORT.md] [--shots DIR]
Exit status 0 only if every step passes and the browser logged no errors.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import websockets

REPO = Path(__file__).resolve().parents[2]


def find_chrome() -> str:
    """Prefer `chrome-headless-shell` (puppeteer cache): installed Google Chrome ran its self-updater
    and was terminated ~35 s after launch on the development machine (observed, see docs/E2E_REPORT.md)."""
    if os.environ.get("CHROME_BIN"):
        return os.environ["CHROME_BIN"]
    shells = sorted(Path.home().glob(".cache/puppeteer/chrome-headless-shell/*/*/chrome-headless-shell"))
    return str(shells[-1]) if shells else "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


CHROME = find_chrome()
API, WEB, CDP_PORT = "http://127.0.0.1:8000", "http://localhost:5173", 9444
STUDY = "main_L2"


class Fail(AssertionError):
    pass


def check(cond: object, msg: str) -> None:
    if not cond:
        raise Fail(msg)


# ---------------------------------------------------------------- infrastructure
@dataclass
class Infra:
    root: Path
    procs: list[subprocess.Popen] = field(default_factory=list)

    @property
    def results(self) -> Path:
        return self.root / "results"

    def _spawn(self, argv: list[str], log: str, env: dict | None = None, cwd: Path = REPO) -> None:
        f = (self.root / log).open("w")
        self.procs.append(subprocess.Popen(argv, cwd=cwd, env={**os.environ, **(env or {})}, stdout=f, stderr=subprocess.STDOUT, start_new_session=True))

    def start(self) -> None:
        py = sys.executable
        r = subprocess.run([py, str(REPO / "scripts" / "demo_setup.py"), "--results", str(self.results)], capture_output=True, text=True)
        check(r.returncode == 0, f"demo_setup failed: {r.stderr}")
        (self.root / "models").mkdir(exist_ok=True)
        self._spawn([py, "-m", "uvicorn", "backend.app.main:app", "--port", "8000", "--ws-max-size", "65536"], "backend.log",
                    {"SWARMMIND_RESULTS_DIR": str(self.results), "SWARMMIND_MODELS_DIR": str(self.root / "models")})
        check((REPO / "frontend" / "dist" / "index.html").is_file(), "frontend/dist missing: run `npm run build` in frontend/")
        self._spawn(["npx", "vite", "preview", "--port", "5173", "--strictPort"], "web.log",
                    {"NODE_OPTIONS": "--no-network-family-autoselection"}, cwd=REPO / "frontend")
        headless = [] if "headless-shell" in CHROME else ["--headless=new"]
        self._spawn([CHROME, *headless, f"--remote-debugging-port={CDP_PORT}", f"--user-data-dir={self.root / 'chrome'}",
                     "--no-first-run", "--window-size=1440,900", "--force-device-scale-factor=1", "about:blank"], "chrome.log")
        for url in (f"{API}/api/health", f"{WEB}/", f"http://127.0.0.1:{CDP_PORT}/json/version"):
            wait_http(url)

    def stop(self) -> None:
        for p in self.procs:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(p.pid, 15)
        time.sleep(1)
        for p in self.procs:
            try:
                os.killpg(p.pid, 9)
            except (ProcessLookupError, PermissionError):
                with contextlib.suppress(ProcessLookupError):
                    p.kill()


def wait_http(url: str, timeout: float = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            return
        except Exception:
            time.sleep(0.3)
    raise Fail(f"{url} did not come up within {timeout}s")


def get_json(path: str):
    return json.loads(urllib.request.urlopen(f"{API}/api{path}", timeout=30).read())


# ---------------------------------------------------------------- browser driver
class Browser:
    def __init__(self, ws) -> None:
        self.ws, self._n = ws, 0
        self.errors: list[str] = []

    @classmethod
    async def connect(cls) -> Browser:
        tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{CDP_PORT}/json"))
        page = next(t for t in tabs if t["type"] == "page")
        b = cls(await websockets.connect(page["webSocketDebuggerUrl"], max_size=None, ping_interval=None))
        for d in ("Page", "Runtime", "Network", "Log"):
            await b.cmd(f"{d}.enable")
        await b.cmd("Emulation.setDeviceMetricsOverride", width=1440, height=900, deviceScaleFactor=1, mobile=False)
        return b

    def _event(self, m: dict) -> None:
        meth, p = m.get("method"), m.get("params", {})
        if meth == "Runtime.exceptionThrown":
            self.errors.append("exception: " + p["exceptionDetails"].get("exception", {}).get("description", p["exceptionDetails"]["text"])[:300])
        elif meth == "Runtime.consoleAPICalled" and p["type"] in ("error", "assert"):
            self.errors.append("console.error: " + " ".join(str(a.get("value", a.get("description", ""))) for a in p["args"])[:300])
        elif meth == "Log.entryAdded" and p["entry"]["level"] == "error":
            self.errors.append(f"log: {p['entry']['text']} {p['entry'].get('url', '')}"[:300])
        elif meth == "Network.responseReceived" and p["response"]["status"] >= 400:
            self.errors.append(f"http {p['response']['status']}: {p['response']['url']}")
        elif meth == "Network.loadingFailed" and not p.get("canceled"):
            self.errors.append(f"network failed: {p.get('errorText')} {p.get('requestId')}")

    async def cmd(self, method: str, **params):
        self._n += 1
        i = self._n
        await self.ws.send(json.dumps({"id": i, "method": method, "params": params}))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == i:
                if "error" in m:
                    raise Fail(f"{method}: {m['error']}")
                return m.get("result")
            self._event(m)

    async def js(self, expr: str):
        r = await self.cmd("Runtime.evaluate", expression=expr, awaitPromise=True, returnByValue=True)
        if r.get("exceptionDetails"):
            raise Fail("js error: " + json.dumps(r["exceptionDetails"])[:300])
        return r["result"].get("value")

    async def pump(self, seconds: float) -> None:
        """Drain pending protocol events (console/network errors) while waiting."""
        end = time.time() + seconds
        while time.time() < end:
            try:
                self._event(json.loads(await asyncio.wait_for(self.ws.recv(), max(0.01, end - time.time()))))
            except TimeoutError:
                return
            except websockets.ConnectionClosed as e:
                raise Fail(f"browser connection lost: {e}") from e

    async def wait(self, expr: str, timeout: float = 30, what: str = "") -> object:
        end = time.time() + timeout
        while time.time() < end:
            v = await self.js(f"!!({expr})")
            if v:
                return v
            await self.pump(0.15)
        raise Fail(f"timed out after {timeout}s waiting for {what or expr}")

    async def goto(self, path: str) -> None:
        await self.cmd("Page.navigate", url=WEB + path)
        await self.wait("document.readyState === 'complete' && !!document.querySelector('main, .sm-lab, #root > *')", 20, f"page {path}")
        await self.pump(0.4)

    async def click(self, selector: str, text: str | None = None) -> None:
        ok = await self.js(
            f"(() => {{ const els=[...document.querySelectorAll({json.dumps(selector)})]; "
            f"const e = {json.dumps(text)} ? els.find(x => x.innerText.trim() === {json.dumps(text)}) : els[0]; "
            "if(!e||e.disabled) return false; e.click(); return true; })()")
        check(ok, f"cannot click {selector!r} {text or ''}")

    async def select(self, label: str, value: str) -> None:
        ok = await self.js(
            f"(() => {{ const s=document.querySelector('select[aria-label={json.dumps(label)}]'); if(!s) return 'no select'; "
            f"if(![...s.options].some(o=>o.value==={json.dumps(value)})) return 'no option: '+[...s.options].map(o=>o.value).join('|'); "
            "Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype,'value').set.call(s,"
            f"{json.dumps(value)}); s.dispatchEvent(new Event('change',{{bubbles:true}})); return 'ok'; }})()")
        check(ok == "ok", f"select {label!r}={value!r}: {ok}")

    async def select_when_ready(self, label: str, value: str, timeout: float = 20) -> None:
        try:
            await self.wait(f"[...(document.querySelector('select[aria-label={json.dumps(label)}]')?.options||[])].some(o=>o.value==={json.dumps(value)})",
                            timeout, f"option {value!r} in {label!r}")
        except Fail as e:
            dump = await self.js("[...document.querySelectorAll('select')].map(s=>s.getAttribute('aria-label')+'='+s.value.slice(-26)+' ['+[...s.options].map(o=>o.value.slice(-30)).join(',').slice(0,160)+']').join(' ; ')")
            raise Fail(f"{e}; selects now: {dump}") from e
        await self.select(label, value)

    async def type_into(self, label_text: str, value: str) -> None:
        ok = await self.js(
            f"(() => {{ const l=[...document.querySelectorAll('label')].find(x=>x.querySelector('span')?.innerText.trim()==={json.dumps(label_text)}); "
            "const i=l?.querySelector('input'); if(!i) return false; "
            "Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(i,"
            f"{json.dumps(value)}); i.dispatchEvent(new Event('input',{{bubbles:true}})); return true; }})()")
        check(ok, f"no input labelled {label_text!r}")

    async def shot(self, path: Path) -> None:
        r = await self.cmd("Page.captureScreenshot", format="png")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(r["data"]))

    async def hud(self) -> dict:
        """The Swarm Lab mission strip as numbers."""
        txt = await self.js("document.querySelector('.sm-lab-mission')?.innerText || ''")
        m = re.search(r"t\s*(\d+)/(\d+).*?Rescued\s*(\d+)/(\d+).*?Coverage\s*(\d+)%.*?Collisions\s*(\d+)\s*(\S.*)?", txt, re.S)
        check(m, f"cannot parse HUD: {txt!r}")
        return {"t": int(m[1]), "max_t": int(m[2]), "rescued": int(m[3]), "targets": int(m[4]), "coverage_pct": int(m[5]),
                "collisions": int(m[6]), "status": (m[7] or "").strip()}


# ---------------------------------------------------------------- steps
@dataclass
class Result:
    name: str
    seconds: float
    detail: str
    ok: bool = True


def stored_replay(run_id: str, replay_id: str) -> dict:
    return json.load(gzip.open(next((REPO / "demo_data/results/replays" / STUDY / run_id).glob(f"{replay_id}.json.gz"))))


async def finish_playback(b: Browser, timeout: float = 90) -> dict:
    await b.select("Playback speed", "8")
    await b.wait("document.querySelector(\"button[aria-label^='Play']:not([disabled])\")", 20, "play button enabled")
    await b.click("button[aria-label^='Play']")
    await b.wait("/SUCCESS|FAILED/.test(document.querySelector('.sm-lab-mission')?.innerText||'')", timeout, "episode end")
    await b.pump(0.3)
    return await b.hud()


async def step_swarm_lab(b: Browser, shots: Path | None) -> str:
    mappo = next(r for r in get_json("/experiments") if r["algo"] == "mappo" and r["seed"] == 0)
    rid, map_id = mappo["run_id"], "L2-struct-bottleneck_vertical"
    replay_id = f"{map_id}_mappo_s0_step999424_eval0"
    ref = stored_replay(rid, replay_id)["outcome"]

    await b.goto("/")
    await b.wait("document.querySelector('canvas.sm-canvas')", 20, "simulation canvas")
    await b.wait("/t\\s*0\\/150/.test(document.querySelector('.sm-lab-mission')?.innerText||'') && document.querySelector('select[aria-label=\"Map\"]')?.options.length >= 5", 30, "lab bootstrapped")
    await b.select("Policy", rid)  # replay mode: run (the final checkpoint is selected by default)
    await b.wait(f"document.querySelector('select[aria-label=\"Policy\"]').value === {json.dumps(rid)} && document.querySelector('select[aria-label=\"Step\"]').value === '999424'", 20, "policy applied")
    await b.select_when_ready("Map", replay_id)
    await b.wait("/t\\s*0\\/150/.test(document.querySelector('.sm-lab-mission')?.innerText||'')", 20, "replay loaded")
    replay_hud = await finish_playback(b)
    if shots:
        await b.shot(shots / "e2e-1-replay.png")
    check(replay_hud["status"] == "SUCCESS" and ref["success"], f"replay status {replay_hud} vs outcome {ref}")
    check(replay_hud["t"] == ref["steps"], f"replay t={replay_hud['t']} but stored outcome steps={ref['steps']}")
    check(replay_hud["rescued"] == ref["metrics"]["n_rescued"], "replay rescued != stored n_rescued")
    check(replay_hud["collisions"] == ref["metrics"]["n_collisions"], "replay collisions != stored n_collisions")
    check(abs(replay_hud["coverage_pct"] - round(ref["metrics"]["coverage_final"] * 100)) <= 1, f"replay coverage {replay_hud['coverage_pct']}% vs {ref['metrics']['coverage_final']:.3f}")

    # Live: same model, same held-out map, same seed (0), stochastic -> must reach the same end state.
    await b.click("button", "Live")
    await b.wait("document.querySelector('select[aria-label=\"Policy\"]')?.options.length > 3", 15, "live policy list")
    await b.select("Policy", f"mappo:{rid}")
    await b.click("button[aria-label='Scenario settings']")
    await b.select("Map source", "structured")
    await b.click("button[type=submit]", "Apply and restart")
    await b.wait(f"[...(document.querySelector('select[aria-label=\"Map\"]')?.options||[])].some(o=>o.value==={json.dumps(map_id)})", 15, "structured map list")
    await b.select("Map", map_id)
    await b.wait("/t\\s*0\\/150/.test(document.querySelector('.sm-lab-mission')?.innerText||'')", 20, "live session created")
    live_hud = await finish_playback(b)
    if shots:
        await b.shot(shots / "e2e-2-live.png")
    for k in ("t", "rescued", "collisions", "coverage_pct", "status"):
        check(live_hud[k] == replay_hud[k], f"LIVE vs REPLAY mismatch on {k}: live={live_hud} replay={replay_hud}")
    return f"live MAPPO session on {map_id}: t={live_hud['t']}, rescued {live_hud['rescued']}/{live_hud['targets']}, coverage {live_hud['coverage_pct']}%, collisions {live_hud['collisions']} == stored replay == stored outcome"


def _num(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.\d+|-?\d+", s)]


async def step_compare(b: Browser, shots: Path | None) -> str:
    summary = get_json(f"/studies/{STUDY}")["summary"]
    on_disk = json.loads((REPO / "demo_data/results/studies" / STUDY / "summary.json").read_text())
    methods = on_disk["levels"]["L2"]["methods"]
    await b.goto("/compare")
    await b.wait("document.querySelectorAll('table tbody tr').length >= 4", 20, "comparison table")
    if shots:
        await b.shot(shots / "e2e-3-compare.png")
    rows = await b.js("[...document.querySelector('table').querySelectorAll('tbody tr')].map(r=>[...r.children].map(c=>c.innerText.trim()))")
    cols = [("success_rate", 3), ("censored_steps", 2), ("coverage", 3), ("return_mean", 2), ("rescue_rate", 3), ("collisions_per_100_agent_steps", 2)]
    n = 0
    for row in rows:
        key = {"Random": "random", "IPPO": "ippo", "MAPPO": "mappo"}.get(row[0])
        if key is None:
            continue
        st = methods[key]["splits"]["test"]["stochastic"]
        for (metric, digits), cell in zip(cols, row[2:], strict=True):
            iv = st[metric]["interval"]
            got = _num(cell)  # mean, sd, lo, hi
            exp = [iv["mean"], iv["std"], iv["lo"], iv["hi"]]
            tol = 0.5 * 10 ** (-digits) + 1e-9
            check(len(got) >= 4 and all(abs(g - e) <= tol for g, e in zip(got, exp, strict=False)), f"{row[0]} {metric}: page {cell!r} vs summary.json {exp}")
            n += 1
    check(n == 18, f"expected 18 learned/random cells, checked {n}")
    check(summary == get_json(f"/studies/{STUDY}")["summary"], "summary changed between reads")
    return f"{n} table cells (3 methods x 6 metrics, mean/SD/CI) equal results/studies/{STUDY}/summary.json"


async def step_analysis(b: Browser, shots: Path | None) -> str:
    runs = {(r["algo"], r["seed"]): r["run_id"] for r in get_json("/experiments")}
    idx = get_json("/replays")
    ex = json.loads((REPO / "demo_data/results/studies" / STUDY / "analysis/L2_mappo_example_episode.json").read_text())
    await b.goto("/analysis")
    await b.wait("document.querySelectorAll('svg').length >= 6", 20, "analysis charts")
    text = await b.js("document.querySelector('main').innerText")
    m = re.search(r"MAPPO on (\S+): (success|failure) after (\d+) steps", text)
    check(m, "illustrative-episode caption missing")
    steps = ex.get("steps") or ex.get("outcome", {}).get("steps")
    check(steps is None or int(m[3]) == int(steps), f"caption says {m[3]} steps; analysis file says {steps}")
    # Swarm Evolution: four checkpoints on one clock; the end states must equal the stored replays.
    await b.select("Playback speed", "4")
    await b.click("button[aria-label='Play all']")
    await b.wait("!/running/.test(document.querySelector('main').innerText.split('Swarm Evolution')[1]||'running')", 120, "evolution playback end")
    await b.pump(0.3)
    evo = await b.js("document.querySelector('main').innerText.split('Swarm Evolution')[1]")
    panels = re.findall(r"(\d+)% of training\n[^\n]*\nt (\d+)\nrescued (\d+)/(\d+)", evo)
    if shots:
        await b.shot(shots / "e2e-4-analysis.png")
    sel = await b.js("(()=>{const s=[...document.querySelectorAll('main select')].find(x=>[...x.options].some(o=>/seed 0/.test(o.text))); "
                     "const m=[...document.querySelectorAll('main select')].find(x=>[...x.options].some(o=>/bottleneck_horizontal/.test(o.text))); "
                     "return {run: s.options[s.selectedIndex].text, map: m.options[m.selectedIndex].text}})()")
    algo = "mappo" if "MAPPO" in sel["run"] else "ippo"
    run = runs[(algo, 0)]
    stored = {e["fraction"]: e for e in idx if e["run_id"] == run and e["map_id"] == "L2-struct-" + sel["map"]}
    check(len(panels) == 4 and len(stored) == 4, f"evolution panels={len(panels)} stored={len(stored)}")
    for (pct, t, rescued, _n), frac in zip(panels, (0.0, 0.1, 0.5, 1.0), strict=True):
        e = stored[frac]
        check(int(pct) == round(frac * 100) and int(t) == e["steps"], f"evolution @{frac}: page t={t} vs stored replay steps={e['steps']}")
        check((int(rescued) == 2) == e["success"], f"evolution @{frac}: rescued {rescued} vs stored success={e['success']}")
    return f"illustrative-episode caption '{m[0]}' matches the analysis file; 4 evolution panels end at the stored replays' step counts"


async def step_train_and_cancel(b: Browser, shots: Path | None) -> str:
    await b.goto("/train")
    await b.type_into("Total env steps", "400000")
    await b.type_into("Parallel envs", "4")
    await b.type_into("Rollout length", "64")
    await b.type_into("PPO epochs", "1")
    await b.click("button", "Start training")
    await b.wait("document.querySelector('section[aria-label=\"Job status\"]')", 20, "job card")
    await b.wait("document.querySelector('section[aria-label=\"Job status\"] a[href^=\"/experiments/\"]')", 90, "run id on job card")
    await b.wait("document.querySelector('main').innerText.includes('running')", 30, "running badge")
    # metrics arrive over the training WebSocket: wait for >= 3 distinct updates in the chart table or counter
    await b.wait("(document.querySelector('main').innerText.match(/update\\s*(\\d+)/i)||[0,0])[1] >= 2 || document.querySelectorAll('main svg path').length > 6", 90, "live metrics")
    await b.pump(1.0)
    if shots:
        await b.shot(shots / "e2e-5-train.png")
    run_id = await b.js("document.querySelector('section[aria-label=\"Job status\"] a[href^=\"/experiments/\"]').innerText.trim()")
    await b.click("button", "Cancel job")
    await b.wait("/cancelled/.test(document.querySelector('section[aria-label=\"Job status\"]')?.innerText||'')", 40, "cancelled badge")
    manifest = None
    for _ in range(40):  # the trainer writes `interrupted` on SIGTERM
        try:
            manifest = get_json(f"/experiments/{run_id}")["summary"]
        except Exception:
            manifest = None
        if manifest and manifest.get("status") == "interrupted":
            break
        time.sleep(0.5)
    check(manifest and manifest["status"] == "interrupted", f"run {run_id} status after cancel: {manifest and manifest.get('status')}")
    return f"launched a real training job from the UI (run {run_id}), watched live metrics, cancelled it; trainer manifest says interrupted"


async def step_experiments(b: Browser, shots: Path | None) -> str:
    runs = get_json("/experiments")
    await b.goto("/experiments")
    await b.wait(f"/{len(runs)} of {len(runs)} runs/.test(document.querySelector('main').innerText)", 20, "run list")
    rows = await b.js("[...document.querySelectorAll('main table tbody tr')].map(r=>[...r.children].map(c=>c.innerText.trim()))")
    by_id = {r[-1]: r for r in rows}
    study = [r for r in runs if r["study"] == STUDY]
    check(len(study) == 10, f"expected 10 study runs, API lists {len(study)}")
    for r in study:
        row = by_id.get(r["run_id"])
        check(row, f"run {r['run_id']} not listed")
        check(row[0].endswith("completed") and row[1].lower() == r["algo"] and int(row[3]) == r["seed"], f"row mismatch for {r['run_id']}: {row}")
        check(abs(float(row[5]) - r["final_val_metrics"]["success_rate"]) < 0.0005 + 1e-9, f"final val success: page {row[5]} vs manifest {r['final_val_metrics']['success_rate']}")
    ui = [r for r in runs if r["study"] == "ui"]
    check(all(by_id[r["run_id"]][0].endswith(r["status"]) for r in ui), "UI-launched run status not shown")
    return f"history lists {len(rows)} runs: the 10 study runs match their manifests (algo, seed, final validation success); the UI-launched run shows its real status ({', '.join(r['status'] for r in ui) or 'none'})"


async def step_settings(b: Browser, shots: Path | None) -> str:
    await b.goto("/settings")
    await b.wait("document.querySelector('h1')?.innerText.includes('Settings')", 10, "settings heading")
    return "settings page renders"


async def step_render_perf(b: Browser, shots: Path | None) -> str:
    """Phase 11 gate (>= 50 fps while animating at 1x on L2), re-measured on the production build."""
    mappo = next(r for r in get_json("/experiments") if r["algo"] == "mappo" and r["seed"] == 0)
    await b.goto("/")
    await b.wait("document.querySelector('canvas.sm-canvas')", 20, "canvas")
    await b.click("button", "Live")
    await b.wait("document.querySelector('select[aria-label=\"Policy\"]')?.options.length > 3", 15, "live policy list")
    await b.select("Policy", f"mappo:{mappo['run_id']}")
    await b.wait("/t\\s*0\\/150/.test(document.querySelector('.sm-lab-mission')?.innerText||'')", 20, "live session")
    await b.js("window.__swarmmindPerfReset && window.__swarmmindPerfReset()")
    await b.wait("document.querySelector(\"button[aria-label^='Play']:not([disabled])\")", 20, "play enabled (live socket open)")
    await b.select("Playback speed", "0.5")  # 2 steps/s: the longest animated stretch per episode
    await b.click("button[aria-label^='Play']")
    await b.pump(6.0)
    st = await b.js("typeof window.__swarmmindPerf === 'function' ? window.__swarmmindPerf() : null")
    check(st and st["samples"] >= 60, f"too few animated frames sampled: {st}")
    check(st["fps"] >= 50, f"animated fps {st['fps']:.1f} < 50 (gate)")
    return f"{st['samples']} animated frames: {st['fps']:.1f} fps (p95 frame {st['frameMsP95']:.1f} ms), draw mean {st['drawMsMean']:.2f} ms / max {st['drawMsMax']:.2f} ms, headless software rendering"


STEPS = [("1. Swarm Lab: replay and live MAPPO session agree with the stored result", step_swarm_lab),
         ("2. Algorithm Comparison equals summary.json", step_compare),
         ("3. Swarm Analysis and Evolution equal stored artifacts", step_analysis),
         ("4. Launch training, watch live metrics, cancel", step_train_and_cancel),
         ("5. Experiment History", step_experiments),
         ("6. Settings", step_settings),
         ("7. Rendering performance (animated fps, production build)", step_render_perf)]


async def run(shots: Path | None) -> tuple[list[Result], list[str]]:
    b = await Browser.connect()
    results: list[Result] = []
    t_all = time.time()
    for name, fn in STEPS:
        t0 = time.time()
        try:
            detail = await fn(b, shots)
            results.append(Result(name, time.time() - t0, detail))
        except Exception as e:  # noqa: BLE001 - report every failing step, then fail the run
            if shots:
                with contextlib.suppress(Exception):
                    await b.shot(shots / f"e2e-FAILED-{name[:1]}.png")
            results.append(Result(name, time.time() - t0, f"{type(e).__name__}: {e}", ok=False))
        print(f"[{'PASS' if results[-1].ok else 'FAIL'}] {name} ({results[-1].seconds:.1f}s) {results[-1].detail[:400]}", flush=True)
    await b.pump(0.5)
    results.append(Result("total", time.time() - t_all, ""))
    return results, b.errors


def report(results: list[Result], errors: list[str]) -> str:
    # The deliberately provoked 404 (`/api/experiments/<id>` polled before the run dir exists) is not expected: all must be empty.
    lines = ["# E2E report (browser, production build, demo bundle only)", "",
             f"Generated by `frontend/e2e/run_e2e.py` on {time.strftime('%Y-%m-%d %H:%M:%S')}. Do not edit by hand.", "",
             "| Step | Result | Seconds | What was verified |", "|---|---|---|---|"]
    for r in results[:-1]:
        lines.append(f"| {r.name} | {'PASS' if r.ok else '**FAIL**'} | {r.seconds:.1f} | {r.detail} |")
    lines += ["", f"Total wall time: **{results[-1].seconds:.0f} s**.", "", f"Browser console / network errors: **{len(errors)}**"]
    lines += [f"- `{e}`" for e in errors[:20]]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "E2E_REPORT.md")
    ap.add_argument("--shots", type=Path, default=None)
    ap.add_argument("--keep", action="store_true", help="keep the temp dir (server/chrome logs)")
    args = ap.parse_args()
    root = Path(tempfile.mkdtemp(prefix="sm_e2e_"))
    infra = Infra(root)
    try:
        infra.start()
        results, errors = asyncio.run(run(args.shots))
    finally:
        infra.stop()
        if args.keep:
            print(f"logs kept in {root}")
        else:
            shutil.rmtree(root, ignore_errors=True)
    text = report(results, errors)
    args.out.write_text(text)
    print(text)
    return 0 if all(r.ok for r in results) and not errors else 1


if __name__ == "__main__":
    sys.exit(main())
