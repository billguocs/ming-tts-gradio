"""Gradio workbench for Ming-3.0-tts-0.5b.

    .venv/Scripts/python.exe app.py --ming_root D:/AI/Ming-3.0-tts
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from functools import partial

import gradio as gr

import engine
import history
import paths
import scenarios
import voices

ENGINE: engine.Engine | None = None

LANGUAGE_LABELS = {
    "auto": "自动",
    "zh": "中文",
    "en": "英文",
    "ja": "日语",
    "ko": "韩语",
    "de": "德语",
    "fr": "法语",
    "es": "西班牙语",
}

CSS = """
.gradio-container {max-width: 1280px !important;}
.tag-btn {min-width: 0 !important; padding: 2px 8px !important; font-size: 12px !important;}
#status-bar {font-size: 13px;}
"""


def _status() -> str:
    return ENGINE.status_markdown() if ENGINE else "🟡 模型未初始化"


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------

VIS_COMPONENTS = (
    "ref_group",
    "prompt_text",
    "spk_toggle",
    "language",
    "instruction",
    "style",
    "attr_group",
    "emotion",
    "dialect",
    "ip",
    "optional_ref",
    "spk_emb_row",
    "ref_hint",
)


def _visibility_updates(scenario, use_ref: bool = False) -> list:
    ref_visible = scenario.ref_mode in {"clone", "speaker"} or (
        scenario.ref_mode == "optional" and use_ref)
    prompt_text_visible = ref_visible and scenario.ref_mode != "speaker"
    spk_visible = ref_visible and scenario.ref_mode == "clone"

    if scenario.ref_mode == "clone":
        hint = "参考文本必须与录音逐字一致；填写错误会明显降低音色相似度。"
    elif scenario.ref_mode == "speaker":
        hint = "该场景只使用参考音频的音色，**不需要**参考文本。"
    elif scenario.ref_mode == "optional":
        hint = "默认不使用参考音频（零嵌入）；勾选上方选项后可克隆音色。"
    else:
        hint = ""

    language_label = ("目标语言（多语言必填：日/韩/德/法/西）"
                      if scenario.task == "MULTILINGUAL" else "目标语言")

    return [
        gr.update(visible=ref_visible),
        gr.update(visible=prompt_text_visible),
        gr.update(visible=spk_visible),
        gr.update(visible=scenario.language_mode != "none",
                  label=language_label),
        gr.update(visible=scenario.task == "FREEFORM"),
        gr.update(visible=scenario.task == "VOICE_DESIGN"),
        gr.update(visible=scenario.task == "BASIC"),
        gr.update(visible=scenario.task == "EMOTION"),
        gr.update(visible=scenario.task == "DIALECT"),
        gr.update(visible=scenario.task == "IP"),
        gr.update(visible=scenario.ref_mode == "optional"),
        gr.update(visible=scenario.ref_mode in {"clone", "speaker"}),
        gr.update(visible=ref_visible, value=hint),
    ]


def on_scenario_change(key):
    scenario = scenarios.SCENARIO_MAP.get(key or "clone", scenarios.SCENARIOS[0])
    return [scenario.description] + _visibility_updates(scenario)


def on_optional_ref(use_ref, key):
    scenario = scenarios.SCENARIO_MAP.get(key or "clone", scenarios.SCENARIOS[0])
    return _visibility_updates(scenario, bool(use_ref))


# ---------------------------------------------------------------------------
# Presets and tags
# ---------------------------------------------------------------------------

def on_preset(key):
    preset = voices.voice_by_key(key)
    if preset is None:
        return gr.update(), gr.update()
    return preset.audio, preset.text


def append_tag(tag: str, text: str) -> str:
    text = text or ""
    if text and not text.endswith((" ", "\n")):
        text += " "
    return text + tag


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _voice_label(form: dict) -> str:
    preset = voices.voice_by_key(form.get("preset") or "")
    audio = form.get("ref_audio") or ""
    if preset is not None and audio and Path(preset.audio).resolve() == Path(audio).resolve():
        return preset.label
    if form.get("spk_emb_file"):
        return "说话人嵌入文件"
    if form.get("ref_audio"):
        return "上传的参考音频"
    return "零嵌入 / 内置音色"


def _params_summary(form: dict, request: dict) -> str:
    parts = [
        f"cfg={request['cfg']:.2f}",
        f"sigma={request['sigma']:.2f}",
        f"temperature={request['temperature']:.2f}",
        f"seed={request['seed']}",
    ]
    if request["raw_text"]:
        parts.append("raw_text")
    if not request["trim_silence"]:
        parts.append("保留尾静音")
    if request["text_norm"]:
        parts.append("文本归一化")
    return " · ".join(parts)


def generate(*values, progress=gr.Progress()):
    key = values[0]
    form = dict(zip(scenarios.FORM_KEYS, values[1:]))
    scenario = scenarios.SCENARIO_MAP.get(key)
    if scenario is None:
        raise gr.Error("未知场景。")
    if ENGINE is None:
        raise gr.Error("模型未初始化。")

    try:
        request = engine.build_request(scenario, form)
    except engine.InputError as error:
        raise gr.Error(str(error)) from error

    try:
        progress(0.15, desc="加载模型 / 校验参数")
        waveform, sample_rate = ENGINE.synthesize(request)
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        raise gr.Error(f"生成失败：{type(error).__name__}: {error}") from error

    progress(0.9, desc="保存结果")
    record_id = history.new_id()
    audio_path = history.save_audio(waveform, sample_rate, record_id)
    ref_copy = history.copy_asset(form.get("ref_audio"), record_id, "ref")
    spk_copy = history.copy_asset(form.get("spk_emb_file"), record_id, "spk")
    duration = waveform.shape[-1] / sample_rate

    record_form = dict(form)
    record_form["ref_audio"] = history.absolute(ref_copy) if ref_copy else ""
    record_form["spk_emb_file"] = history.absolute(spk_copy) if spk_copy else ""

    record = {
        "id": record_id,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scenario": scenario.key,
        "scenario_label": scenario.label,
        "task": scenario.task,
        "text": form["text"],
        "voice": _voice_label(form),
        "duration": duration,
        "audio": audio_path.name,
        "ref_audio": ref_copy,
        "spk_emb": spk_copy,
        "form": record_form,
        "params": _params_summary(form, request),
    }
    history.add(record)

    meta = (
        f"**{scenario.label}** · 时长 {duration:.2f}s · 采样率 {sample_rate} Hz\n\n"
        f"{record['params']}"
    )
    return (
        str(audio_path),
        str(audio_path),
        meta,
        f"✅ 已生成并保存到历史（{record['time']}）",
        history.rows(),
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def select_history(evt: gr.SelectData):
    records = history.list_records()
    index = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if not isinstance(index, int) or index >= len(records):
        return "", None, None, "未选择记录。"
    record = records[index]
    details = (
        f"**{record.get('scenario_label', '')}** · {record.get('time', '')} · "
        f"{record.get('duration', 0):.2f}s\n\n"
        f"{record.get('params', '')}\n\n"
        f"音色：{record.get('voice', '')}\n\n"
        f"文本：{record.get('text', '')}"
    )
    path = history.absolute(record.get("audio", ""))
    return record.get("id", ""), path, path, details


def refill(record_id):
    record = history.get_record(record_id) if record_id else None
    if record is None:
        raise gr.Error("请先在历史表中选择一条记录。")
    form = record.get("form", {})
    values = []
    for key in scenarios.FORM_KEYS:
        value = form.get(key)
        if key in {"preset", "ref_audio", "spk_emb_file"} and not value:
            value = None
        values.append(value)
    return [record.get("scenario", "clone")] + values


def refresh_history():
    return history.rows()


# ---------------------------------------------------------------------------
# Voice library
# ---------------------------------------------------------------------------

def _library_choices():
    return [(entry.get("name") or entry["id"], entry["id"])
            for entry in voices.list_user_voices()]


def _save_voice(name, audio, text, language):
    try:
        entries = voices.list_user_voices()
        if any(entry.get("name") == (name or "").strip() for entry in entries):
            raise ValueError("已存在同名音色，请换一个名字。")
        voices.save_user_voice(name, audio, text, language)
    except (ValueError, OSError) as error:
        return (f"⚠️ {error}", gr.update(), voices.library_rows(),
                gr.update(), gr.update())
    message = f"✅ 已保存音色「{(name or '').strip()}」。"
    return (
        message,
        gr.update(choices=voices.preset_choices()),
        voices.library_rows(),
        gr.update(choices=_library_choices()),
        gr.update(value=""),
    )


def save_voice_from_main(name, audio, text, language):
    return _save_voice(name, audio, text, language)


def save_voice_from_library(name, audio, text, language):
    return _save_voice(name, audio, text, language)


def delete_voice(voice_id):
    if not voice_id:
        return ("⚠️ 请选择要删除的音色。", gr.update(), voices.library_rows(),
                gr.update())
    removed = voices.delete_user_voice(voice_id)
    message = "✅ 已删除。" if removed else "⚠️ 未找到该音色。"
    return (
        message,
        gr.update(choices=voices.preset_choices()),
        voices.library_rows(),
        gr.update(choices=_library_choices(), value=None),
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    initial = scenarios.SCENARIOS[0]
    with gr.Blocks(
        title="Ming-3.0-tts 语音工作台",
        theme=gr.themes.Soft(),
        css=CSS,
    ) as demo:
        gr.Markdown("# Ming-3.0-tts 语音工作台\n"
                    "0.5B 全场景推理界面：克隆、多语言、副语言、自由指令、"
                    "音色设计、属性、情感、方言与内置音色。")
        gr.Markdown(_status(), elem_id="status-bar")

        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                scenario = gr.Radio(
                    choices=scenarios.SCENARIO_CHOICES,
                    value=initial.key,
                    label="场景",
                )
                scenario_desc = gr.Markdown(initial.description)
            with gr.Column(scale=4):
                text = gr.Textbox(
                    label="文本",
                    lines=5,
                    placeholder="输入要合成的文本。副语言标签可以直接写在句子里。",
                )
                tag_buttons = []
                for chunk in (scenarios.PARALINGUISTIC_TAGS[:7],
                              scenarios.PARALINGUISTIC_TAGS[7:]):
                    with gr.Row():
                        for tag in chunk:
                            tag_buttons.append(gr.Button(
                                tag, size="sm", elem_classes=["tag-btn"],
                                variant="secondary"))
                gr.Markdown("_点击标签追加到文本末尾；标签在中文和英文里都有效。_")

                with gr.Group() as ref_group:
                    preset = gr.Dropdown(
                        choices=voices.preset_choices(),
                        value=None,
                        label="参考音色 / 内置示例",
                        info="选择后自动填入音频与转写；也可以直接在右侧上传。",
                        filterable=True,
                    )
                    with gr.Row():
                        ref_audio = gr.Audio(
                            label="参考音频",
                            sources=["upload", "microphone"],
                            type="filepath",
                            scale=2,
                        )
                        prompt_text = gr.Textbox(
                            label="参考文本",
                            lines=6,
                            scale=2,
                            placeholder="与录音逐字一致的转写。",
                        )
                    spk_toggle = gr.Checkbox(
                        label="同时使用说话人嵌入（推荐）",
                        value=True,
                    )
                    with gr.Row():
                        voice_name = gr.Textbox(
                            label="保存为我的音色",
                            placeholder="给这段参考音频起个名字",
                            scale=3,
                        )
                        save_main_btn = gr.Button("保存到音色库", scale=1)
                        main_save_status = gr.Markdown("")
                    ref_hint = gr.Markdown(
                        "参考文本必须与录音逐字一致；填写错误会明显降低音色相似度。")

                language = gr.Dropdown(
                    choices=[(label, value)
                             for value, label in LANGUAGE_LABELS.items()],
                    value="auto",
                    label="目标语言",
                )
                instruction = gr.Textbox(
                    label="自由指令",
                    lines=3,
                    visible=False,
                    placeholder="例如：用温柔、舒缓的声音讲述睡前故事，像在轻声安慰孩子。",
                )
                style = gr.Textbox(
                    label="音色设计描述",
                    lines=5,
                    visible=False,
                    placeholder="支持结构化属性、自然语言描述或角色扮演提示词。",
                )
                with gr.Group(visible=False) as attr_group:
                    with gr.Row():
                        speed = gr.Dropdown(
                            choices=list(scenarios.SPEED_VALUES),
                            value="中速", label="语速")
                        pitch = gr.Dropdown(
                            choices=list(scenarios.PITCH_VALUES),
                            value="中", label="基频")
                        volume = gr.Dropdown(
                            choices=list(scenarios.VOLUME_VALUES),
                            value="中", label="音量")
                emotion = gr.Dropdown(
                    choices=list(scenarios.EMOTIONS),
                    value="中性", label="情感", visible=False,
                )
                dialect = gr.Dropdown(
                    choices=[(label, value)
                             for label, value in scenarios.DIALECTS],
                    value="cantonese", label="方言", visible=False,
                )
                ip = gr.Dropdown(
                    choices=voices.ip_choices(),
                    value="灵小柔",
                    label="内置音色（可搜索）",
                    filterable=True,
                    visible=False,
                )
                optional_ref = gr.Checkbox(
                    label="使用参考音频（可选，默认零嵌入）",
                    value=False,
                    visible=False,
                )

                with gr.Accordion("高级参数", open=False):
                    with gr.Row() as spk_emb_row:
                        spk_emb_file = gr.File(
                            label="说话人嵌入文件（.pt，未提供参考音频时使用）",
                            file_types=[".pt"],
                            type="filepath",
                        )
                    with gr.Row():
                        cfg = gr.Slider(0.5, 5.0, value=2.0, step=0.1,
                                        label="cfg")
                        sigma = gr.Slider(0.0, 1.0, value=0.25, step=0.01,
                                          label="sigma")
                        temperature = gr.Slider(0.0, 2.0, value=0.0, step=0.05,
                                                label="temperature")
                    with gr.Row():
                        max_length = gr.Slider(8, 120, value=24, step=1,
                                               label="分段长度 max_length")
                        max_decode_steps = gr.Slider(
                            100, 3000, value=1000, step=50,
                            label="最大解码步数")
                        seed = gr.Number(value=1895, precision=0,
                                         label="随机种子")
                    with gr.Row():
                        raw_text = gr.Checkbox(
                            label="不切分、不归一化（raw_text）", value=False)
                        trim_silence = gr.Checkbox(
                            label="去除尾部静音", value=True)
                        text_norm = gr.Checkbox(
                            label="文本归一化（需要 pynini）",
                            value=engine.NORMALIZATION_AVAILABLE,
                            interactive=engine.NORMALIZATION_AVAILABLE,
                        )
                    if not engine.NORMALIZATION_AVAILABLE:
                        gr.Markdown(
                            "⚠️ 当前环境未安装 `pynini`，数字/日期/单位归一化不可用，"
                            "文本会原样送入模型。")

                generate_btn = gr.Button("生成语音", variant="primary", size="lg")
                generate_status = gr.Markdown("")
                out_audio = gr.Audio(label="生成结果", type="filepath",
                                     interactive=False)
                out_file = gr.File(label="下载 WAV")
                out_meta = gr.Markdown("")

        with gr.Tabs():
            with gr.Tab("生成历史"):
                hist_refresh = gr.Button("刷新", size="sm")
                hist_table = gr.Dataframe(
                    headers=["时间", "场景", "文本", "音色", "时长(s)"],
                    datatype=["str"] * 5,
                    type="array",
                    value=history.rows(),
                    interactive=False,
                    max_height=320,
                )
                with gr.Row():
                    hist_audio = gr.Audio(label="回听", type="filepath",
                                          interactive=False)
                    hist_file = gr.File(label="下载 WAV")
                hist_details = gr.Markdown("")
                hist_refill = gr.Button("回填到表单")
                history_state = gr.State("")
            with gr.Tab("我的音色库"):
                gr.Markdown("保存常用的参考音频与转写，之后在参考音色下拉框里直接选择。")
                with gr.Row():
                    lib_name = gr.Textbox(label="名称", scale=2)
                    lib_audio = gr.Audio(label="音频", sources=["upload", "microphone"],
                                         type="filepath", scale=2)
                    lib_text = gr.Textbox(label="转写文本", lines=4, scale=3)
                lib_lang = gr.Dropdown(
                    choices=[("未指定", "")] + [
                        (label, code) for code, label in LANGUAGE_LABELS.items()
                        if code != "auto"],
                    value="",
                    label="语言（可选）",
                )
                lib_save = gr.Button("保存音色", variant="primary")
                lib_status = gr.Markdown("")
                lib_table = gr.Dataframe(
                    headers=["名称", "语言", "转写", "创建时间", "ID"],
                    datatype=["str"] * 5,
                    type="array",
                    value=voices.library_rows(),
                    interactive=False,
                    max_height=320,
                )
                with gr.Row():
                    lib_del_choices = gr.Dropdown(
                        choices=_library_choices(), label="选择要删除的音色")
                    lib_del = gr.Button("删除", variant="stop")

        # -- wiring ------------------------------------------------------
        visibility_components = {
            "ref_group": ref_group,
            "prompt_text": prompt_text,
            "spk_toggle": spk_toggle,
            "language": language,
            "instruction": instruction,
            "style": style,
            "attr_group": attr_group,
            "emotion": emotion,
            "dialect": dialect,
            "ip": ip,
            "optional_ref": optional_ref,
            "spk_emb_row": spk_emb_row,
            "ref_hint": ref_hint,
        }
        vis_components = [visibility_components[name] for name in VIS_COMPONENTS]
        form_components = {
            "text": text,
            "preset": preset,
            "ref_audio": ref_audio,
            "prompt_text": prompt_text,
            "use_spk_emb": spk_toggle,
            "language": language,
            "instruction": instruction,
            "style": style,
            "speed": speed,
            "pitch": pitch,
            "volume": volume,
            "emotion": emotion,
            "dialect": dialect,
            "ip": ip,
            "use_ref": optional_ref,
            "spk_emb_file": spk_emb_file,
            "cfg": cfg,
            "sigma": sigma,
            "temperature": temperature,
            "max_length": max_length,
            "max_decode_steps": max_decode_steps,
            "seed": seed,
            "raw_text": raw_text,
            "trim_silence": trim_silence,
            "text_norm": text_norm,
        }
        form_inputs = [form_components[key] for key in scenarios.FORM_KEYS]

        scenario.change(
            fn=on_scenario_change,
            inputs=scenario,
            outputs=[scenario_desc] + vis_components,
        )
        optional_ref.change(
            fn=on_optional_ref,
            inputs=[optional_ref, scenario],
            outputs=vis_components,
        )
        preset.change(
            fn=on_preset,
            inputs=preset,
            outputs=[ref_audio, prompt_text],
        )
        for button, tag in zip(tag_buttons, scenarios.PARALINGUISTIC_TAGS):
            button.click(fn=partial(append_tag, tag), inputs=text, outputs=text)

        generate_btn.click(
            fn=generate,
            inputs=[scenario] + form_inputs,
            outputs=[out_audio, out_file, out_meta, generate_status, hist_table],
        )

        hist_refresh.click(fn=refresh_history, outputs=hist_table)
        hist_table.select(
            fn=select_history,
            outputs=[history_state, hist_audio, hist_file, hist_details],
        )
        hist_refill.click(
            fn=refill,
            inputs=history_state,
            outputs=[scenario] + form_inputs,
        ).then(
            fn=on_scenario_change,
            inputs=scenario,
            outputs=[scenario_desc] + vis_components,
        )

        save_main_btn.click(
            fn=save_voice_from_main,
            inputs=[voice_name, ref_audio, prompt_text, language],
            outputs=[main_save_status, preset, lib_table, lib_del_choices,
                     voice_name],
        )
        lib_save.click(
            fn=save_voice_from_library,
            inputs=[lib_name, lib_audio, lib_text, lib_lang],
            outputs=[lib_status, preset, lib_table, lib_del_choices, lib_name],
        )
        lib_del.click(
            fn=delete_voice,
            inputs=lib_del_choices,
            outputs=[lib_status, preset, lib_table, lib_del_choices],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ming_root", default=None,
                        help="Ming-3.0-tts 仓库路径；默认找同级目录或环境变量 "
                             f"{paths.ENV_VAR}")
    parser.add_argument("--model_path", default=None,
                        help="本地 checkpoint 目录或 Hub repo id；"
                             "默认 <ming_root>/Ming-3.0-tts-0.5b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--no-preload", action="store_true",
                        help="启动时不加载模型，首次生成再加载")
    return parser.parse_args()


def main() -> None:
    global ENGINE
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    args = parse_args()
    paths.set_ming_root(args.ming_root)
    paths.DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_path = args.model_path or paths.default_model_path()
    ENGINE = engine.Engine(model_path, device=args.device)
    if not args.no_preload:
        print("[webui] loading model, this can take a minute ...")
        ok = ENGINE.try_load()
        print(f"[webui] model loaded: {ok}")
        if not ok:
            print(f"[webui] {ENGINE.error}")

    demo = build_ui()
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
        show_error=True,
        allowed_paths=[str(paths.DATA_DIR)],
    )


if __name__ == "__main__":
    main()
