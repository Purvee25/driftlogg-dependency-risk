"""Central configuration, loaded from environment or a local .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration.

    Values come from the environment, falling back to a local `.env` file.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="DRIFTLOGG_",
        extra="ignore",
    )

    github_token: str = ""
    """Personal access token. Raises the API limit from 60 to 5000 req/hour."""

    model_url: str = ""
    """Fallback: download the model from here if no local copy is found.

    Serverless deploys (Vercel) can't practically bundle the 1.3MB pickle
    through this project's manual file-upload path, so the deployed instance
    downloads it once from the public repo's raw GitHub URL and caches it to
    disk. Empty by default — nothing calls out over the network unless this is
    explicitly set (see vercel.json).
    """

    ecosystem: str = "npm"
    """Which package registry to study. Pick one and stay there."""

    data_dir: Path = PROJECT_ROOT / "data"
    request_timeout_seconds: float = 30.0
    max_retries: int = 5

    @property
    def raw_dir(self) -> Path:
        """Untouched API responses. Never edited, only appended to."""
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """Parsed but not yet feature-engineered."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Model-ready feature tables."""
        return self.data_dir / "processed"

    @property
    def released_model_path(self) -> Path:
        """The committed model, used when nothing has been trained locally.

        Lives outside `data/` precisely so it escapes the gitignore: CI has no
        training data, so without a checked-in model the GitHub Action would
        quietly fall back to the baseline and report far less than it should.
        """
        return PROJECT_ROOT / "models" / "model.pkl"

    @property
    def model_search_paths(self) -> tuple[Path, ...]:
        """Where to look for a model, most recent first.

        A freshly trained model wins over the released one so local iteration
        takes effect without a copy step.
        """
        return (self.processed_dir / "model.pkl", self.released_model_path)

    def ensure_dirs(self) -> None:
        """Create the data directories if they do not exist yet."""
        for directory in (self.raw_dir, self.interim_dir, self.processed_dir):
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
