from __future__ import annotations

import os
from pathlib import Path

_DEFAULT_ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env(env_path: str | os.PathLike[str] | None = None) -> None:
    resolved_env_path = Path(env_path) if env_path is not None else _DEFAULT_ENV_PATH

    try:
        from dotenv import load_dotenv
    except Exception:
        _load_dotenv_fallback(resolved_env_path)
        return

    load_dotenv(resolved_env_path)


def _load_dotenv_fallback(env_path: str | os.PathLike[str] = _DEFAULT_ENV_PATH) -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
