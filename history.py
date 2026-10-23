"""Generation history backed by a JSON index and WAV files."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path

import torch
import torchaudio

import paths

DATA_DIR = paths.DATA_DIR
HISTORY_DIR = DATA_DIR / "history"
INDEX_PATH = HISTORY_DIR / "index.json"

MAX_RECORDS = 200


def _ensure_dirs() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _read_index() -> dict:
    try:
        payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            return payload
    except (OSError, ValueError):
        pass
    return {"records": []}


def _write_index(payload: dict) -> None:
    _ensure_dirs()
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, INDEX_PATH)


def list_records() -> list[dict]:
    records = []
    for record in _read_index()["records"]:
        path = HISTORY_DIR / record.get("audio", "")
        if record.get("id") and path.exists():
            records.append(record)
    return records


def get_record(record_id: str) -> dict | None:
    for record in list_records():
        if record.get("id") == record_id:
            return record
    return None


def new_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


def audio_path(record_id: str) -> Path:
    _ensure_dirs()
    return HISTORY_DIR / f"{record_id}.wav"


def save_audio(waveform: torch.Tensor, sample_rate: int,
               record_id: str) -> Path:
    path = audio_path(record_id)
    waveform = waveform.detach().cpu().float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(str(path), waveform, sample_rate=sample_rate)
    return path


def copy_asset(source: str, record_id: str, suffix: str) -> str:
    """Copy a temporary upload next to the record so replay still works."""
    if not source or not os.path.isfile(source):
        return ""
    _ensure_dirs()
    target = HISTORY_DIR / f"{record_id}_{suffix}{Path(source).suffix or '.wav'}"
    shutil.copy2(source, target)
    return target.name


def add(record: dict) -> dict:
    payload = _read_index()
    payload["records"].insert(0, record)
    dropped = payload["records"][MAX_RECORDS:]
    payload["records"] = payload["records"][:MAX_RECORDS]
    _write_index(payload)
    for old in dropped:
        for field in ("audio", "ref_audio", "spk_emb"):
            name = old.get(field)
            if name:
                try:
                    (HISTORY_DIR / name).unlink()
                except OSError:
                    pass
    return record


def rows() -> list[list[str]]:
    table = []
    for record in list_records():
        text = record.get("text", "")
        table.append([
            record.get("time", ""),
            record.get("scenario_label", ""),
            text if len(text) <= 40 else text[:39] + "…",
            record.get("voice", ""),
            f"{record.get('duration', 0):.2f}",
        ])
    return table


def absolute(name: str) -> str:
    return str((HISTORY_DIR / name).resolve()) if name else ""
