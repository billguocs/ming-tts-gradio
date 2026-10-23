"""Model lifecycle and request validation for the workbench.

Everything the UI needs to turn a form dictionary into a ``MingTTS.synthesize``
call lives here, so the Gradio layer stays wiring.
"""

from __future__ import annotations

import importlib.util
import random
import threading

import numpy as np
import torch

import paths
import voices

NORMALIZATION_AVAILABLE = importlib.util.find_spec("pynini") is not None

MULTILINGUAL_IDS = {"ja", "ko", "de", "fr", "es"}


class InputError(ValueError):
    """A form problem that the user can fix."""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _optional(value) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def build_request(scenario, form: dict) -> dict:
    """Validate the form and return synthesize kwargs plus a seed."""
    text = (form.get("text") or "").strip()
    if not text:
        raise InputError("请输入要合成的文本。")

    task = scenario.task
    controls: dict = {}
    use_spk_emb = False
    use_zero_spk_emb = False
    prompt_wav = _optional(form.get("ref_audio"))
    spk_emb_path = _optional(form.get("spk_emb_file"))
    prompt_text = (form.get("prompt_text") or "").strip()

    if scenario.ref_mode == "clone":
        if prompt_wav:
            if not prompt_text:
                raise InputError(
                    "克隆需要与参考音频一致的参考文本；请填写参考文本，"
                    "或改用说话人嵌入文件。")
            use_spk_emb = bool(form.get("use_spk_emb"))
        elif spk_emb_path:
            use_spk_emb = True
            prompt_text = ""
        else:
            raise InputError("请上传参考音频，或在高级参数里提供说话人嵌入文件。")
    elif scenario.ref_mode == "speaker":
        if spk_emb_path:
            use_spk_emb = True
            prompt_wav = None
        elif prompt_wav:
            use_spk_emb = True
            prompt_text = ""  # these tasks do not take a transcript
        else:
            raise InputError("请上传参考音频，或在高级参数里提供说话人嵌入文件。")
    elif scenario.ref_mode == "optional":
        # Freeform: reference audio is optional and off by default.
        if form.get("use_ref"):
            if not prompt_wav:
                raise InputError("已勾选使用参考音频，请上传或选择一个音色。")
            if not prompt_text:
                raise InputError("使用参考音频时需要填写参考文本。")
            use_spk_emb = True
        else:
            prompt_wav = None
            prompt_text = ""
            spk_emb_path = None
            use_zero_spk_emb = True
    else:
        prompt_wav = None
        prompt_text = ""
        spk_emb_path = None
        use_zero_spk_emb = True

    # -- task controls ---------------------------------------------------
    if task == "BASIC":
        speed = form.get("speed") or "中速"
        pitch = form.get("pitch") or "中"
        volume = form.get("volume") or "中"
        controls = {"speed": speed, "pitch": pitch, "volume": volume}
    elif task == "EMOTION":
        emotion = form.get("emotion")
        if not emotion:
            raise InputError("请选择情感。")
        controls = {"emotion": emotion}
    elif task == "DIALECT":
        dialect = form.get("dialect")
        if not dialect:
            raise InputError("请选择方言。")
        controls = {"dialect": dialect}
    elif task == "IP":
        ip = form.get("ip")
        if not ip:
            raise InputError("请选择内置音色。")
        controls = {"ip": ip}
    elif task == "VOICE_DESIGN":
        style = (form.get("style") or "").strip()
        if not style:
            raise InputError("请填写音色设计描述。")
        controls = {"style": style}
    elif task == "FREEFORM":
        instruction = (form.get("instruction") or "").strip()
        if not instruction:
            raise InputError("请填写自由指令，例如：用温柔、舒缓的声音讲述睡前故事。")
    elif task == "TTS":
        if (form.get("instruction") or "").strip():
            raise InputError("克隆/多语言任务不接受自由指令，请清空指令框。")

    # -- language --------------------------------------------------------
    language = form.get("language") or "auto"
    if task == "MULTILINGUAL":
        if language not in MULTILINGUAL_IDS:
            raise InputError("多语言任务必须选择日语、韩语、德语、法语或西班牙语。")
    elif task == "DIALECT":
        language = None
    elif task == "IP":
        language = voices.ip_language(controls["ip"]) if language == "auto" else language
    elif language == "auto":
        language = None

    request = {
        "task": task,
        "text": text,
        "language_id": language,
        "instruction": (form.get("instruction") or "").strip() or None,
        "controls": controls or None,
        "prompt_wav_path": prompt_wav,
        "prompt_text": prompt_text or None,
        "spk_emb_path": spk_emb_path if use_spk_emb else None,
        "use_spk_emb": bool(use_spk_emb),
        "use_zero_spk_emb": bool(use_zero_spk_emb),
        "cfg": float(form.get("cfg", 2.0)),
        "sigma": float(form.get("sigma", 0.25)),
        "temperature": float(form.get("temperature", 0.0)),
        "max_length": int(form.get("max_length", 24)),
        "max_decode_steps": int(form.get("max_decode_steps", 1000)),
        "trim_silence": bool(form.get("trim_silence", True)),
        "raw_text": bool(form.get("raw_text", False)),
        "text_norm": bool(form.get("text_norm", False)),
        "seed": int(form.get("seed", 1895)),
    }
    return request


class Engine:
    """Single model instance shared by every request."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        self.model_path = model_path
        self.device = device
        self._model = None
        self._error: str = ""
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def error(self) -> str:
        return self._error

    def load(self):
        with self._lock:
            if self._model is not None:
                return self._model
            if str(self.device).startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA 不可用：请检查 PyTorch 安装或改用 --device cpu。")
            paths.find_ming_root()  # puts the model repository on sys.path
            from ming_tts import MingTTS

            self._model = MingTTS(
                self.model_path,
                device=self.device,
                text_normalization=NORMALIZATION_AVAILABLE,
            )
            self._error = ""
            return self._model

    def try_load(self) -> bool:
        try:
            self.load()
            return True
        except Exception as error:  # noqa: BLE001
            self._error = f"{type(error).__name__}: {error}"
            return False

    def status_markdown(self) -> str:
        if self._model is not None:
            detail = f"设备 `{self.device}`"
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / 1024 ** 3
                total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
                detail += f" · 显存 {used:.1f}/{total:.0f} GB"
            note = ("" if NORMALIZATION_AVAILABLE else
                    " · 未安装 `pynini`，数字/日期归一化不可用")
            return f"🟢 模型已加载 · `{self.model_path}` · {detail}{note}"
        if self._error:
            return f"🔴 模型加载失败：{self._error}"
        return "🟡 模型未加载"

    def synthesize(self, request: dict):
        request = dict(request)
        seed = int(request.pop("seed", 1895))
        text_norm = bool(request.pop("text_norm", False))
        model = self.load()
        # Single-user, serialized queue: mutating the flag is safe and avoids
        # building a second model just to toggle normalization.
        model.text_normalization = bool(text_norm and NORMALIZATION_AVAILABLE)
        seed_everything(seed)
        waveform = model.synthesize(**request)
        return waveform, model.sample_rate
