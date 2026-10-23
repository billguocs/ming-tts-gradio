"""Locate the Ming-3.0-tts checkout and the app data directory.

The workbench is a standalone repository: the model code, checkpoint and
reference assets live in a separate clone of
https://github.com/inclusionAI/Ming-3.0-tts. Point at it with ``--ming_root``
or the ``MING_TTS_ROOT`` environment variable; otherwise the usual sibling
layout is detected.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "webui_data"
ENV_VAR = "MING_TTS_ROOT"

_override: Path | None = None


def set_ming_root(value: str | os.PathLike | None) -> None:
    global _override
    _override = Path(value).expanduser().resolve() if value else None


def _candidates():
    if _override is not None:
        yield _override
    from_env = os.environ.get(ENV_VAR)
    if from_env:
        yield Path(from_env).expanduser()
    yield APP_DIR.parent / "Ming-3.0-tts"
    yield APP_DIR / "Ming-3.0-tts"
    yield Path.cwd()


def find_ming_root(required: bool = True) -> Path | None:
    """Return the model checkout root and put it on ``sys.path``."""
    for candidate in _candidates():
        try:
            root = Path(candidate).resolve()
        except OSError:
            continue
        if (root / "ming_tts.py").is_file() and (root / "tts_tasks.py").is_file():
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            return root
    if not required:
        return None
    raise RuntimeError(
        "找不到 Ming-3.0-tts 仓库。请先克隆 "
        "https://github.com/inclusionAI/Ming-3.0-tts，"
        f"然后用 --ming_root 或环境变量 {ENV_VAR} 指定其路径。")


def default_model_path() -> str:
    root = find_ming_root(required=False)
    if root is None:
        return "Ming-3.0-tts-0.5b"
    return str(root / "Ming-3.0-tts-0.5b")
