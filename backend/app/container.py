"""Service container: built once per app, held on `app.state.services`."""

from __future__ import annotations

from dataclasses import dataclass

from backend.app.services.job_runner import JobRunner
from backend.app.services.map_service import MapService
from backend.app.services.model_registry import ModelRegistry
from backend.app.services.replay_store import ReplayStore
from backend.app.services.run_index import RunIndex
from backend.app.services.session_manager import SessionManager
from backend.app.services.study_service import StudyService
from backend.app.settings import Settings


@dataclass
class Services:
    settings: Settings
    runs: RunIndex
    models: ModelRegistry
    replays: ReplayStore
    studies: StudyService
    maps: MapService
    sessions: SessionManager
    jobs: JobRunner

    @classmethod
    def build(cls, settings: Settings) -> Services:
        runs = RunIndex(settings.runs_dir)
        models = ModelRegistry(runs, settings.models_dir)
        maps = MapService(settings.results_dir / "heldout_access_log.jsonl")
        return cls(
            settings=settings,
            runs=runs,
            models=models,
            replays=ReplayStore(settings.replays_dir),
            studies=StudyService(settings.studies_dir),
            maps=maps,
            sessions=SessionManager(
                maps, models, settings.max_live_sessions, settings.session_idle_ttl_seconds
            ),
            jobs=JobRunner(
                settings.train_script,
                settings.runs_dir,
                settings.jobs_dir,
                settings.max_concurrent_training_jobs,
                settings.max_ui_train_env_steps,
            ),
        )

    def refresh(self) -> None:
        self.runs.refresh()
        self.models.refresh()
        self.replays.refresh()
