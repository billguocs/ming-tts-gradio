# Ming-3.0-tts-0.5b Gradio 工作台设计

日期：2026-10-23
状态：已确认

## 目标

为本机（Windows 11 / RTX 4060 Ti 16GB）上的 Ming-3.0-tts-0.5b 提供一个覆盖全部推理场景的 Gradio 网页界面，单用户使用，中文界面。

## 场景矩阵

| 界面场景 | task | 必填 | 语音条件 | 备注 |
| --- | --- | --- | --- | --- |
| 克隆 TTS | `TTS` | 文本、参考音频、参考文本 | 参考音频抽取说话人嵌入 | 可关闭 spk emb |
| 副语言 | `TTS` | 同上 | 同上 | 行内标签按钮，14 个标签 |
| 多语言 | `MULTILINGUAL` | 文本、语言、参考音频/文本 | 同上 | 语言：ja/ko/de/fr/es |
| 自由指令 | `FREEFORM` | 文本、instruction | 默认零嵌入 | 高级区可选参考音频 |
| 音色设计 | `VOICE_DESIGN` | 文本、style | 零嵌入 | |
| 属性控制 | `BASIC` | 文本、speed/pitch/volume、参考音频 | 仅说话人嵌入 | 不接受参考文本 |
| 情感 | `EMOTION` | 文本、emotion、参考音频 | 仅说话人嵌入 | 7 种情感 |
| 方言 | `DIALECT` | 文本、dialect、参考音频 | 仅说话人嵌入 | 不接受 language_id |
| 内置音色 | `IP` | 文本、ip | 零嵌入 | 100+ 中文、5 英文 |

所有场景共享：高级参数（cfg/sigma/temperature/max_length/max_decode_steps/seed/raw_text/去尾静音）、文本归一化开关（pynini 缺失时禁用）、说话人嵌入文件（`.pt`）作为参考音频的替代。

## 页面结构

单页 + 左侧场景切换。切换场景只改变字段可见性，不清空文本。

- 顶部：模型状态、设备、显存。
- 主区：文本、标签插入栏、参考音频（内置示例/我的音色下拉 + 上传/录音）、场景参数、高级参数折叠、生成按钮、播放器/时长/下载。
- 底部常驻标签页：生成历史（回听、下载、一键回填）、我的音色库（命名保存参考音频与转写、删除）。

## 代码结构

```
webui/
  app.py         # Gradio 布局与事件绑定
  engine.py      # 模型单例、请求校验、合成、错误转换
  scenarios.py   # 场景定义与字段可见性
  voices.py      # 内置音色目录解析、自定义音色库
  history.py     # 历史索引与音频归档
webui_data/      # 用户数据，加入 .gitignore
```

内置音色表从 `cookbooks/instructions.md` 的 Markdown 表格解析，避免复制 500+ 条目。

## 数据流

表单 → `engine.build_request()` 校验并生成 `MingTTS.synthesize` 参数 → 合成波形 → 写 `webui_data/history/<id>.wav` → 追加索引 → 刷新历史列表。

## 错误处理

- UI 层按场景校验必填项与互斥参数，给出中文提示。
- 引擎层依赖 `tts_tasks.build_task_generation_kwargs` 兜底。
- 模型异常转为 `gr.Error` 中文摘要，完整堆栈打印到控制台。
- 模型加载失败时界面仍可打开，生成时返回明确错误。

## 并发与资源

单模型实例，`queue(default_concurrency_limit=1)` 串行推理；历史保留最近 200 条，超出自动清理音频文件。

## 验证

1. 无 GPU 校验：场景定义、音色目录解析、历史读写单元检查。
2. GPU 冒烟：9 个场景各用短文本生成一次，确认音频非静音。
3. 启动验证：本地启动 Gradio，确认页面可访问。
