from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    base_url: str = os.environ.get("ASSISTANT_BASE_URL", "http://127.0.0.1:8080/v1")
    model: str = os.environ.get("ASSISTANT_MODEL", "Qwen2.5-VL-3B-Instruct")
    camera_source: str | None = os.environ.get("ASSISTANT_CAMERA_SOURCE") or os.environ.get("ASSISTANT_CAMERA_URL")
    memory_db: Path = env_path("ASSISTANT_MEMORY_DB", str(PROJECT_ROOT / "data" / "memory.sqlite3"))
    captures_dir: Path = env_path("ASSISTANT_CAPTURES_DIR", str(PROJECT_ROOT / "data" / "captures"))
    stop_model_dir: Path = env_path("ASSISTANT_STOP_MODEL", str(PROJECT_ROOT / "models" / "stop_detector"))
    vosk_model_dir: Path = env_path(
        "ASSISTANT_VOSK_MODEL", str(PROJECT_ROOT / "models" / "vosk-model-small-ru-0.22")
    )
    tts_backend: str = os.environ.get("ASSISTANT_TTS", "system")
    language: str = os.environ.get("ASSISTANT_LANGUAGE", "ru")

    def ensure_dirs(self) -> None:
        self.memory_db.parent.mkdir(parents=True, exist_ok=True)
        self.captures_dir.mkdir(parents=True, exist_ok=True)
        self.stop_model_dir.mkdir(parents=True, exist_ok=True)


def load_dotenv(path: Path | None = None) -> None:
    """Tiny .env loader to avoid making python-dotenv a hard dependency."""
    env_file = path or PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
