# AIspeaker

基于 Qwen3-TTS + SenseVoice + LLM 的实时 AI 语音对话助手。浏览器录音，流式语音识别、大模型推理、语音合成，全链路流式响应，延迟极低。

## 功能特性

- **实时语音对话** — 浏览器端录音，WebSocket 流式传输，按键即说
- **流式 ASR** — 基于 FunASR / SenseVoiceSmall，自动语音识别 + VAD 端点检测
- **流式 LLM** — 兼容 OpenAI API，支持任意大模型后端（MiniMax、OpenAI 等）
- **流式 TTS** — 基于 Qwen3-TTS，支持声音克隆（参考音频提取 speaker embedding）
- **多语言** — 中文 / 英文一键切换
- **技能系统** — 内置默认助手、雅思口语考官等角色，可自定义扩展
- **对话管理** — 多会话支持，历史记录持久化，侧边栏切换
- **实时监控** — 前端显示 RAM / GPU 显存使用情况
- **流式播放** — TTS 音频逐帧回放，边生成边播放，无需等待整句

## 技术栈

| 模块 | 技术 |
|------|------|
| 后端 | Python, Flask, Flask-SocketIO |
| ASR | FunASR (SenseVoiceSmall + fsmn-vad) |
| TTS | Faster-Qwen3-TTS (Qwen3-TTS-12Hz-0.6B-Base) |
| LLM | OpenAI 兼容 API (MiniMax-M2.7 等) |
| 前端 | 原生 HTML/JS, Web Audio API, AudioWorklet, Socket.IO Client |

## 项目结构

```
AIspeaker/
├── run.py                  # 入口，Flask + SocketIO 服务启动
├── config.yaml             # 配置文件（TTS/ASR/LLM/Server）
├── speaker.pt              # 声音克隆 embedding
├── voice_app/
│   ├── asr_engine.py       # 语音识别引擎
│   ├── tts_engine.py       # 语音合成引擎（流式）
│   ├── llm_client.py       # LLM 流式客户端
│   ├── pipeline.py         # ASR→LLM→TTS 全链路编排
│   ├── socket_handlers.py  # WebSocket 事件处理
│   ├── sentence_splitter.py# 流式分句
│   ├── audio_utils.py      # 音频格式转换
│   ├── chat_history.py     # 对话历史持久化
│   ├── skill_manager.py    # 技能（角色）管理
│   └── config.py           # 配置加载
├── skills/
│   ├── default.json        # 默认助手
│   └── ielts_examiner.json # 雅思口语考官
├── static/
│   └── index.html          # 前端单页应用
└── tmp/                    # 对话历史存储
```

## 快速开始

### 1. 环境准备

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install flask flask-socketio pyyaml openai funasr torch faster-qwen3-tts numpy psutil
```

### 2. 配置

编辑 `config.yaml`，填入你的 LLM API 信息：

```yaml
llm:
  base_url: "https://api.example.com/v1"
  api_key: "your-api-key"
  model: "your-model-name"
```

如需声音克隆，提供参考音频路径：

```yaml
tts:
  ref_audio: "path/to/your/ref_audio.wav"
```

### 3. 启动

```bash
python run.py
```

浏览器访问 `http://localhost:3033`，按住录音按钮即可开始对话。

## 自定义技能

在 `skills/` 目录下新建 JSON 文件：

```json
{
  "name": "技能名称",
  "icon": "emoji",
  "description": "技能描述",
  "prompts": {
    "zh": "中文 system prompt",
    "en": "English system prompt"
  }
}
```

## License

[Apache License 2.0](LICENSE)
