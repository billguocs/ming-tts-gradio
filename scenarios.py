"""Scenario table for the Ming-3.0-tts Gradio workbench.

One entry per task the 0.5b checkpoint exposes. The table drives field
visibility, validation and the ``task`` string passed to ``MingTTS``.
"""

from __future__ import annotations

from dataclasses import dataclass

# Languages accepted by the multilingual task, and by the optional
# target-language line elsewhere.
MULTILINGUAL_LANGUAGES = (
    ("日语", "ja"),
    ("韩语", "ko"),
    ("德语", "de"),
    ("法语", "fr"),
    ("西班牙语", "es"),
)

SIMPLE_LANGUAGES = (
    ("自动（不添加语言行）", "auto"),
    ("中文", "zh"),
    ("英文", "en"),
)

ALL_LANGUAGES = SIMPLE_LANGUAGES + MULTILINGUAL_LANGUAGES

EMOTIONS = ("高兴", "悲伤", "愤怒", "恐惧", "厌恶", "惊讶", "中性")

# Inline tags accepted by the TTS task. Mirrors frontend.toolkit in the
# Ming-3.0-tts repository; kept here so the workbench does not need the model
# code until the first synthesis.
PARALINGUISTIC_TAGS = (
    "[Breathing]",
    "[Cough]",
    "[Sigh]",
    "[Laughter]",
    "[Surprise-ah]",
    "[Surprise-oh]",
    "[Dissatisfaction-hnn]",
    "[Uhm]",
    "[Confirmation-en]",
    "[Question-ei]",
    "[Question-ah]",
    "[Question-en]",
    "[Question-oh]",
    "[Question-huh]",
)

DIALECTS = (
    ("粤语", "cantonese"),
    ("川渝话", "chuanyu dialect"),
)

SPEED_VALUES = ("慢速", "中速", "快速")
PITCH_VALUES = ("低", "中", "高")
VOLUME_VALUES = ("低", "中", "高")


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    task: str
    description: str
    # reference audio: "clone" needs wav+text, "speaker" needs wav only,
    # "optional" (freeform) allows either, "none" uses a built-in voice.
    ref_mode: str
    language_mode: str  # none | optional | multilingual | ip
    controls: tuple = ()
    zero_spk: bool = False


SCENARIOS = (
    Scenario(
        key="clone",
        label="克隆 TTS",
        task="TTS",
        description="用一段参考录音和它的转写克隆音色。转写必须与录音一致，"
                    "否则音色相似度会下降。",
        ref_mode="clone",
        language_mode="optional",
    ),
    Scenario(
        key="para",
        label="副语言",
        task="TTS",
        description="在文本里插入行内标签制造呼吸、咳嗽、叹气、笑声和疑问语气。"
                    "标签可以出现在句子的任意位置，中英文通用。",
        ref_mode="clone",
        language_mode="optional",
    ),
    Scenario(
        key="multilingual",
        label="多语言",
        task="MULTILINGUAL",
        description="用同一段参考音色说日语、韩语、德语、法语或西班牙语。"
                    "目标语言必填；非中文/英文文本不经过中文归一化。",
        ref_mode="clone",
        language_mode="multilingual",
    ),
    Scenario(
        key="freeform",
        label="自由指令",
        task="FREEFORM",
        description="用一句自然语言描述声音和表达方式，不需要参考音频。"
                    "注意：不要同时使用音色设计、情感、方言或属性控制。",
        ref_mode="optional",
        language_mode="optional_all",
        zero_spk=True,
    ),
    Scenario(
        key="design",
        label="音色设计",
        task="VOICE_DESIGN",
        description="用结构化属性、自由描述或角色提示词凭空设计一个音色，"
                    "不需要参考音频。支持中英文描述。",
        ref_mode="none",
        language_mode="none",
        controls=("style",),
        zero_spk=True,
    ),
    Scenario(
        key="basic",
        label="属性控制",
        task="BASIC",
        description="在参考音色上控制语速、基频和音量。三项都需要指定；"
                    "该任务只使用参考音频的音色，不使用参考转写。",
        ref_mode="speaker",
        language_mode="none",
        controls=("speed", "pitch", "volume"),
    ),
    Scenario(
        key="emotion",
        label="情感",
        task="EMOTION",
        description="在参考音色上指定情感。该任务只使用参考音频的音色，"
                    "不使用参考转写。",
        ref_mode="speaker",
        language_mode="none",
        controls=("emotion",),
    ),
    Scenario(
        key="dialect",
        label="方言",
        task="DIALECT",
        description="用参考音色说粤语或川渝话。该任务不接受目标语言参数。",
        ref_mode="speaker",
        language_mode="none",
        controls=("dialect",),
    ),
    Scenario(
        key="ip",
        label="内置音色",
        task="IP",
        description="从 100+ 中文和 5 个英文内置音色中选择一个，不需要参考音频。"
                    "中文音色含大量情感变体。",
        ref_mode="none",
        language_mode="ip",
        controls=("ip",),
        zero_spk=True,
    ),
)

SCENARIO_MAP = {scenario.key: scenario for scenario in SCENARIOS}
SCENARIO_CHOICES = [(scenario.label, scenario.key) for scenario in SCENARIOS]

LANGUAGE_CHOICES = {
    "none": (),
    "optional": SIMPLE_LANGUAGES,
    "optional_all": ALL_LANGUAGES,
    "multilingual": MULTILINGUAL_LANGUAGES,
    "ip": SIMPLE_LANGUAGES,
}

# Order of the form values stored in history and replayed by "回填".
FORM_KEYS = (
    "text",
    "preset",
    "ref_audio",
    "prompt_text",
    "use_spk_emb",
    "language",
    "instruction",
    "style",
    "speed",
    "pitch",
    "volume",
    "emotion",
    "dialect",
    "ip",
    "use_ref",
    "spk_emb_file",
    "cfg",
    "sigma",
    "temperature",
    "max_length",
    "max_decode_steps",
    "seed",
    "raw_text",
    "trim_silence",
    "text_norm",
)
