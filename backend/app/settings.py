"""Backend settings (§28.1, §37 Security).

Binds to localhost by default and restricts CORS, Host and WebSocket
Origin to the configured local origins, per the local-first security
posture in the spec.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from marl.utils.paths import MODELS_DIR, REPO_ROOT, RESULTS_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWARMMIND_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origin: str = "http://localhost:5173"
    # Host-header allowlist (DNS-rebinding defence for a local-first server).
    allowed_hosts: list[str] = ["localhost", "127.0.0.1"]

    results_dir: Path = RESULTS_DIR
    models_dir: Path = MODELS_DIR
    train_script: Path = REPO_ROOT / "scripts" / "train.py"

    # Resource limits (§37).
    max_live_sessions: int = 4
    session_idle_ttl_seconds: int = 600
    max_steps_per_request: int = 50
    max_concurrent_training_jobs: int = 1
    max_ui_train_env_steps: int = 10_000_000
    ws_max_steps_per_second: int = 30
    ws_base_steps_per_second: float = 4.0  # at 1x speed
    ws_max_message_bytes: int = 4096

    @property
    def runs_dir(self) -> Path:
        return self.results_dir / "runs"

    @property
    def replays_dir(self) -> Path:
        return self.results_dir / "replays"

    @property
    def studies_dir(self) -> Path:
        return self.results_dir / "studies"

    @property
    def jobs_dir(self) -> Path:
        return self.results_dir / "jobs"


settings = Settings()
