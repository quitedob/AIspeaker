import logging
from flask import Flask, send_from_directory, jsonify
from flask_socketio import SocketIO
from voice_app.config import load_config
from voice_app.tts_engine import TTSEngine
from voice_app.asr_engine import ASREngine
from voice_app.llm_client import LLMClient
from voice_app.socket_handlers import VoiceNamespace
from voice_app.skill_manager import load_skills, get_skills_list

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    config = load_config()

    # 1. Load skills
    skills = load_skills()

    # 2. TTS engine + warmup
    tts_cfg = config["tts"]
    tts = TTSEngine(
        model_id=tts_cfg["model_id"],
        speaker_pt=tts_cfg["speaker_pt"],
        ref_audio=tts_cfg.get("ref_audio", ""),
        device=tts_cfg.get("device", "cuda"),
        dtype=tts_cfg.get("dtype", "bfloat16"),
        max_seq_len=tts_cfg.get("max_seq_len", 2048),
        chunk_size=tts_cfg.get("chunk_size", 8),
    )
    tts.warmup()

    # 3. ASR engine
    asr_cfg = config["asr"]
    asr = ASREngine(
        model=asr_cfg.get("model", "iic/SenseVoiceSmall"),
        vad_model=asr_cfg.get("vad_model", "fsmn-vad"),
    )

    # 4. LLM client
    llm_cfg = config["llm"]
    llm = LLMClient(
        base_url=llm_cfg["base_url"],
        api_key=llm_cfg["api_key"],
        model=llm_cfg.get("model", "gpt-4o-mini"),
        max_tokens=llm_cfg.get("max_tokens", 512),
        temperature=llm_cfg.get("temperature", 0.7),
        system_prompt_zh=llm_cfg.get("system_prompt_zh", ""),
        system_prompt_en=llm_cfg.get("system_prompt_en", ""),
    )

    # 5. Flask + SocketIO
    app = Flask(__name__, static_folder="static")
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                        max_http_buffer_size=16 * 1024 * 1024)

    @app.route("/")
    def index():
        return send_from_directory("static", "index.html")

    @app.route("/api/skills")
    def api_skills():
        return jsonify(get_skills_list(skills))

    @app.route("/api/stats")
    def api_stats():
        import psutil
        vm = psutil.virtual_memory()
        stats = {
            "ram_used_gb": round((vm.total - vm.available) / (1024**3), 1),
            "ram_total_gb": round(vm.total / (1024**3), 1),
            "ram_percent": vm.percent,
            "gpu": None,
        }
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                used = total - free
                stats["gpu"] = {
                    "name": torch.cuda.get_device_name(0),
                    "used_gb": round(used / (1024**3), 1),
                    "total_gb": round(total / (1024**3), 1),
                    "percent": round(used / total * 100, 1),
                }
        except Exception:
            pass
        return jsonify(stats)

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        return response

    namespace = VoiceNamespace(asr, llm, tts, socketio, skills)
    socketio.on_namespace(namespace)

    srv_cfg = config.get("server", {})
    host = srv_cfg.get("host", "0.0.0.0")
    port = srv_cfg.get("port", 5000)

    logger.info(f"Starting server on {host}:{port}")
    socketio.run(app, host=host, port=port, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
