"""Backend settings (§28.1, §37 Security).

Binds to localhost by default and restricts CORS to the configured dev
origin, per the local-first security posture in the spec.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict

from marl.utils.paths import MODELS_DIR, RESULTS_DIR


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SWARMMIND_", env_file=".env")

    host: str = "127.0.0.1"
    port: int = 8000
    cors_origin: str = "http://localhost:5173"

    results_dir: str = str(RESULTS_DIR)
    models_dir: str = str(MODELS_DIR)

    max_live_sessions: int = 4
    session_idle_ttl_seconds: int = 600
    max_steps_per_request: int = 50
    max_concurrent_training_jobs: int = 1


settings = Settings()
