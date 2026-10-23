# Ming-3.0-tts Gradio 工作台

[Ming-3.0-tts-0.5b](https://github.com/inclusionAI/Ming-3.0-tts) 的独立网页界面，
单页覆盖 0.5b checkpoint 的全部推理场景：零样本克隆、副语言标签、多语言、
自由指令、音色设计、语速/基频/音量、情感、方言和 100+ 内置音色。

模型代码与权重不随本仓库分发；界面通过路径引用一个 Ming-3.0-tts 检出。

## 准备

1. 检出并安装模型仓库：

   ```bash
   git clone https://github.com/inclusionAI/Ming-3.0-tts.git
   cd Ming-3.0-tts
   pip install -r requirements.txt
   ```

   权重可从 ModelScope 下载（国内更快）：

   ```bash
   modelscope download --model inclusionAI/Ming-3.0-tts-0.5b \
     --local_dir Ming-3.0-tts-0.5b
   ```

2. 在本仓库安装界面依赖（建议复用模型仓库的虚拟环境）：

   ```bash
   pip install -r requirements.txt
   ```

   `gradio` 必须保持 5.x：gradio 6 要求 `huggingface-hub>=1.16`，会与
   Ming-3.0-tts 固定的 `transformers==4.46.3`（要求 `<1.0`）冲突。

## 启动

```bash
python app.py --ming_root /path/to/Ming-3.0-tts
```

`--ming_root` 也可以省略：程序会依次尝试环境变量 `MING_TTS_ROOT`、
同级目录 `../Ming-3.0-tts`、当前目录。模型默认从
`<ming_root>/Ming-3.0-tts-0.5b` 加载。

常用参数：

| 参数 | 说明 |
| --- | --- |
| `--ming_root` | Ming-3.0-tts 仓库路径 |
| `--model_path` | 指定其它 checkpoint 或 Hub repo id |
| `--device` | 默认 `cuda:0` |
| `--host` / `--port` | 默认 `127.0.0.1:7860` |
| `--no-browser` | 不自动打开浏览器 |
| `--no-preload` | 先启动界面，首次生成再加载模型 |
| `--share` | Gradio 公网分享链接 |

## 场景

| 场景 | 需要的输入 | 说明 |
| --- | --- | --- |
| 克隆 TTS | 文本、参考音频、参考文本 | 参考文本必须与录音一致 |
| 副语言 | 同上 | 14 个行内标签按钮，点击追加到文本 |
| 多语言 | 文本、参考音频/文本、目标语言 | 日语、韩语、德语、法语、西班牙语 |
| 自由指令 | 文本、自然语言指令 | 默认零嵌入；可勾选使用参考音频 |
| 音色设计 | 文本、风格描述 | 零嵌入 |
| 属性控制 | 文本、语速/基频/音量、参考音频 | 不需要参考文本 |
| 情感 | 文本、情感、参考音频 | 高兴/悲伤/愤怒/恐惧/厌恶/惊讶/中性 |
| 方言 | 文本、方言、参考音频 | 粤语 / 川渝话 |
| 内置音色 | 文本、音色 | 484 个中文音色（含情感变体）+ 5 个英文音色 |

其它能力：

- 参考条件三选一：参考音频、`.pt` 说话人嵌入、零嵌入；冲突组合在生成前报错。
- 高级参数：`cfg`、`sigma`、`temperature`、`max_length`、`max_decode_steps`、
  `seed`、`raw_text`、去尾静音、文本归一化。
- 生成历史：回听、下载 WAV、一键回填全部参数（保留最近 200 条）。
- 我的音色库：保存命名后的参考音频与转写，之后从下拉框直接选择。
- 内置音色表从 `<ming_root>/cookbooks/instructions.md` 解析，随模型仓库更新。

> Podcast（多说话人）是 `Ming-3.0-tts-tiny` 的能力，0.5b checkpoint 不支持，
> 因此本界面不包含该场景。

## 数据目录

`webui_data/`（已加入 `.gitignore`）：

- `history/`：生成的 WAV 与 `index.json`。
- `voices/`：自定义音色音频与索引。
- `smoke/`：冒烟测试输出。

## 环境注意事项

- 未安装 `pynini` 时文本归一化开关不可用，数字/日期/单位会原样送入模型。
- 未安装 `flash-attn` 时自动回退 PyTorch SDPA，结果不变、速度略慢。
- 麦克风录音需要 `ffmpeg` 在 `PATH` 中。
- 模型以 bfloat16 常驻显存；单模型实例，Gradio 队列串行推理。

## 验证

```bash
# 9 个场景各生成一段短音频，检查非静音
python smoke_test.py --ming_root /path/to/Ming-3.0-tts

# 只跑指定场景
python smoke_test.py --ming_root /path/to/Ming-3.0-tts --only clone ip
```

## License

Apache-2.0，见 [LICENSE](LICENSE)。
