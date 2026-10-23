"""Generate one short utterance per workbench scenario.

Runs every task the 0.5b checkpoint exposes through the same code path the
Gradio UI uses, and checks that each result is audible.

    .venv/Scripts/python.exe smoke_test.py --ming_root D:/AI/Ming-3.0-tts
    .venv/Scripts/python.exe smoke_test.py --only clone emotion ip
"""

from __future__ import annotations

import argparse
import time

import torch
import torchaudio

import engine
import paths
import scenarios

VOICE_TEXT = "诶，你今天有空吗？我们能不能聊聊这件事啊？"

# Scenario inputs that are independent of the model checkout.
CASES = {
    "clone": dict(
        text="傍晚的风从山口吹过来，远处的云慢慢染成了金色。",
        prompt_text=VOICE_TEXT),
    "para": dict(
        text="[Sigh]雨还是没有停，[Cough]我们恐怕只能把活动改到室内了。",
        prompt_text=VOICE_TEXT),
    "multilingual": dict(
        text="週末、駅前の小さな喫茶店で、季節限定の抹茶パフェを食べました。",
        language="ja", prompt_text=VOICE_TEXT),
    "freeform": dict(
        text="The old radio crackled softly as the storm approached.",
        language="en",
        instruction="Speak in a warm, intimate storytelling voice, unhurried."),
    "design": dict(
        text="冰川在漫长岁月中缓缓移动，留下深刻的痕迹。",
        style="成熟沉稳的男性纪录片旁白，吐字清晰，节奏舒缓。"),
    "basic": dict(
        text="今晚的观测将在云层散开后开始。",
        speed="慢速", pitch="中", volume="中"),
    "emotion": dict(
        text="所有的测试都通过了，我们终于可以把这个好消息告诉大家！",
        emotion="高兴"),
    "dialect": dict(
        text="今日天气几好，我哋一齐去海边行下啦。",
        dialect="cantonese"),
    "ip": dict(
        text="感谢你的认可，我们会继续把这件事做好。",
        ip="灵小柔"),
}

BASE_FORM = {
    "text": "",
    "preset": None,
    "ref_audio": None,
    "prompt_text": "",
    "use_spk_emb": True,
    "language": "auto",
    "instruction": "",
    "style": "",
    "speed": "中速",
    "pitch": "中",
    "volume": "中",
    "emotion": "中性",
    "dialect": "cantonese",
    "ip": "灵小柔",
    "use_ref": False,
    "spk_emb_file": None,
    "cfg": 2.0,
    "sigma": 0.25,
    "temperature": 0.0,
    "max_length": 24,
    "max_decode_steps": 1000,
    "seed": 1895,
    "raw_text": False,
    "trim_silence": True,
    "text_norm": False,
}

# Which reference recording each scenario uses.
REFERENCE_ASSETS = {
    "clone": "data/voice_presets/voice_audio/lingguang.wav",
    "para": "data/voice_presets/voice_audio/lingguang.wav",
    "multilingual": "data/voice_presets/voice_audio/lingguang.wav",
    "basic": "data/wavs/emotion_prompt.wav",
    "emotion": "data/wavs/emotion_prompt.wav",
    "dialect": "data/wavs/yue_prompt.wav",
}


def form_for(scenario, root) -> dict:
    form = dict(BASE_FORM)
    form.update(CASES[scenario.key])
    if scenario.key in REFERENCE_ASSETS:
        form["ref_audio"] = str(root / REFERENCE_ASSETS[scenario.key])
    return form


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ming_root", default=None,
                        help=f"Ming-3.0-tts 仓库路径；也可用环境变量 {paths.ENV_VAR}")
    parser.add_argument("--model_path", default=None,
                        help="默认 <ming_root>/Ming-3.0-tts-0.5b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", nargs="+",
                        choices=[s.key for s in scenarios.SCENARIOS])
    args = parser.parse_args()

    paths.set_ming_root(args.ming_root)
    root = paths.find_ming_root()
    model_path = args.model_path or str(root / "Ming-3.0-tts-0.5b")

    model = engine.Engine(model_path, device=args.device)
    print(f"loading {model_path} ...")
    model.load()

    output_dir = paths.DATA_DIR / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [s for s in scenarios.SCENARIOS
                if not args.only or s.key in args.only]
    failures = []
    for scenario in selected:
        request = engine.build_request(scenario, form_for(scenario, root))
        started = time.time()
        waveform, sample_rate = model.synthesize(request)
        elapsed = time.time() - started
        duration = waveform.shape[-1] / sample_rate
        rms = float(waveform.pow(2).mean().sqrt())
        path = output_dir / f"{scenario.key}.wav"
        torchaudio.save(str(path), waveform.detach().cpu().float(), sample_rate)
        ok = duration >= 0.4 and rms >= 1e-3
        status = "ok  " if ok else "FAIL"
        print(f"[{status}] {scenario.key:<13} {duration:5.2f}s  rms={rms:.4f}  "
              f"{elapsed:5.1f}s  -> {path}")
        if not ok:
            failures.append(scenario.key)

    if failures:
        print(f"\nfailed: {', '.join(failures)}")
        return 1
    print("\nall scenarios produced audible audio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
