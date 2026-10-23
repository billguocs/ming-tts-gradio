"""Reference voice presets and the built-in IP voice catalog.

The IP catalog is parsed from ``cookbooks/instructions.md`` at import time so
the 500+ entries stay in sync with the documentation instead of being copied
into code. User presets live in ``webui_data/voices``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import paths

DATA_DIR = paths.DATA_DIR
VOICES_DIR = DATA_DIR / "voices"
INDEX_PATH = VOICES_DIR / "index.json"

PREFERRED_IPS = ("灵小柔", "灵楷", "US_James", "US_Stella")


def _instructions_md() -> Path:
    return paths.find_ming_root() / "cookbooks" / "instructions.md"


def _builtin_prompts_json() -> Path:
    return paths.find_ming_root() / "data" / "voice_presets" / "voice_prompts.json"


@dataclass(frozen=True)
class VoicePreset:
    key: str
    label: str
    audio: str
    text: str
    language: str
    kind: str  # builtin | user


@dataclass(frozen=True)
class IpVoice:
    ip: str
    label: str
    language: str


# ---------------------------------------------------------------------------
# Built-in reference recordings
# ---------------------------------------------------------------------------

def _builtin_presets() -> list[VoicePreset]:
    presets: list[VoicePreset] = []
    try:
        root = paths.find_ming_root()
        prompts_json = _builtin_prompts_json()
        payload = json.loads(prompts_json.read_text(encoding="utf-8"))
    except (OSError, ValueError, RuntimeError):
        root, prompts_json, payload = None, None, {}
    for name, entry in payload.items():
        audio = (prompts_json.parent / entry["audio"]).resolve()
        presets.append(VoicePreset(
            key=f"builtin:{name}",
            label=f"内置示例 · {name}（含转写）",
            audio=str(audio),
            text=entry.get("text", ""),
            language="zh",
            kind="builtin",
        ))
    if root is None:
        return presets
    extra = (
        ("emotion_prompt", "情感示例", "data/wavs/emotion_prompt.wav"),
        ("yue_prompt", "粤语示例", "data/wavs/yue_prompt.wav"),
    )
    for name, label, relative in extra:
        path = (root / relative).resolve()
        if path.exists():
            presets.append(VoicePreset(
                key=f"builtin:{name}",
                label=f"内置示例 · {label}（无转写）",
                audio=str(path),
                text="",
                language="zh",
                kind="builtin",
            ))
    return presets


# ---------------------------------------------------------------------------
# IP catalog
# ---------------------------------------------------------------------------

def _table_rows(section: str) -> list[list[str]]:
    rows: list[list[str]] = []
    header_seen = False
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells and all(cell and set(cell) <= set("-: ") for cell in cells):
            continue  # separator row
        if not header_seen:
            header_seen = True
            continue
        rows.append(cells)
    return rows


def _section(text: str, start: str, end: str | None) -> str:
    index = text.find(start)
    if index < 0:
        return ""
    if end is None:
        return text[index:]
    stop = text.find(end, index + len(start))
    return text[index:] if stop < 0 else text[index:stop]


def _clean(cell: str) -> str:
    return cell.strip().strip("`").strip()


@lru_cache(maxsize=1)
def ip_catalog() -> tuple[IpVoice, ...]:
    try:
        text = _instructions_md().read_text(encoding="utf-8")
    except (OSError, RuntimeError):
        text = ""

    voices: list[IpVoice] = []

    plain = _section(
        text, "### 不带情感变体的中文音色", "### 带情感变体的中文音色")
    for row in _table_rows(plain):
        if len(row) < 2:
            continue
        ip = _clean(row[0])
        if not ip:
            continue
        sex = _clean(row[1])
        age = _clean(row[2]) if len(row) > 2 else ""
        detail = "，".join(part for part in (sex, age) if part)
        voices.append(IpVoice(
            ip=ip,
            label=f"{ip}（{detail}）" if detail else ip,
            language="zh",
        ))

    emotional = _section(
        text, "### 带情感变体的中文音色", "### 英文音色")
    for row in _table_rows(emotional):
        if len(row) < 2:
            continue
        base = _clean(row[0])
        variants = re.findall(r"`([^`]+)`", row[-1])
        if not variants:
            continue
        for part in row[-1].split("<br/>"):
            match = re.search(r"(`([^`]+)`)", part)
            if not match:
                continue
            emotion = part.split(":", 1)[0].strip().strip("：")
            voices.append(IpVoice(
                ip=match.group(2).strip(),
                label=f"{base}｜{emotion}" if emotion else base,
                language="zh",
            ))

    english = _section(text, "### 英文音色", "## <a id=\"voice-design\">")
    for row in _table_rows(english):
        ip = _clean(row[0]) if row else ""
        if not ip:
            continue
        accent = _clean(row[2]) if len(row) > 2 else ""
        voices.append(IpVoice(
            ip=ip,
            label=f"{ip}（{accent}）" if accent else ip,
            language="en",
        ))

    if not voices:
        # Keep the UI usable if the documentation layout ever changes.
        fallback = (
            ("US_James", "en"), ("US_Stella", "en"), ("US_Emma", "en"),
            ("UK_George", "en"), ("UK_Charlotte", "en"),
            ("灵小柔", "zh"), ("灵楷", "zh"),
        )
        return tuple(IpVoice(ip=ip, label=ip, language=lang)
                     for ip, lang in fallback)

    def sort_key(voice: IpVoice):
        if voice.ip in PREFERRED_IPS:
            return (0, PREFERRED_IPS.index(voice.ip))
        return (1 if voice.language == "zh" else 2, 0)

    return tuple(sorted(voices, key=sort_key))


def ip_choices() -> list[tuple[str, str]]:
    return [(voice.label, voice.ip) for voice in ip_catalog()]


def ip_language(ip: str) -> str:
    for voice in ip_catalog():
        if voice.ip == ip:
            return voice.language
    return "en" if str(ip).lower().startswith(("us_", "uk_")) else "zh"


# ---------------------------------------------------------------------------
# User voice library
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)


def _read_index() -> dict:
    try:
        payload = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("voices"), list):
            return payload
    except (OSError, ValueError):
        pass
    return {"voices": []}


def _write_index(payload: dict) -> None:
    _ensure_dirs()
    tmp = INDEX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    os.replace(tmp, INDEX_PATH)


def list_user_voices() -> list[dict]:
    entries = []
    for entry in _read_index()["voices"]:
        path = VOICES_DIR / entry.get("audio", "")
        if not entry.get("id") or not path.exists():
            continue
        entries.append(entry)
    return entries


def user_presets() -> list[VoicePreset]:
    presets = []
    for entry in list_user_voices():
        audio = (VOICES_DIR / entry["audio"]).resolve()
        presets.append(VoicePreset(
            key=f"user:{entry['id']}",
            label=f"我的音色 · {entry.get('name') or entry['id']}",
            audio=str(audio),
            text=entry.get("text", ""),
            language=entry.get("language", "") or "",
            kind="user",
        ))
    return presets


def preset_choices() -> list[tuple[str, str]]:
    return [(preset.label, preset.key)
            for preset in (_builtin_presets() + user_presets())]


def voice_by_key(key: str) -> VoicePreset | None:
    if not key:
        return None
    for preset in _builtin_presets() + user_presets():
        if preset.key == key:
            return preset
    return None


def save_user_voice(name: str, audio_path: str, text: str,
                    language: str = "") -> dict:
    name = (name or "").strip()
    language = (language or "").strip()
    if language == "auto":
        language = ""
    if not name:
        raise ValueError("请先填写音色名称")
    if not audio_path or not os.path.isfile(audio_path):
        raise ValueError("请先上传或录制一段参考音频")
    _ensure_dirs()
    voice_id = datetime.now().strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:6]
    suffix = Path(audio_path).suffix.lower() or ".wav"
    filename = f"{voice_id}{suffix}"
    shutil.copy2(audio_path, VOICES_DIR / filename)
    entry = {
        "id": voice_id,
        "name": name,
        "audio": filename,
        "text": (text or "").strip(),
        "language": (language or "").strip(),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    payload = _read_index()
    payload["voices"].append(entry)
    _write_index(payload)
    return entry


def delete_user_voice(voice_id: str) -> bool:
    payload = _read_index()
    remaining, removed = [], None
    for entry in payload["voices"]:
        if entry.get("id") == voice_id:
            removed = entry
        else:
            remaining.append(entry)
    if removed is None:
        return False
    payload["voices"] = remaining
    _write_index(payload)
    audio = VOICES_DIR / removed.get("audio", "")
    try:
        if audio.is_file():
            audio.unlink()
    except OSError:
        pass
    return True


def library_rows() -> list[list[str]]:
    rows = []
    for entry in list_user_voices():
        rows.append([
            entry.get("name", ""),
            entry.get("language", "") or "—",
            (entry.get("text", "") or "（无转写）")[:60],
            entry.get("created", ""),
            entry.get("id", ""),
        ])
    return rows


def voice_by_id(voice_id: str) -> VoicePreset | None:
    return voice_by_key(f"user:{voice_id}")
