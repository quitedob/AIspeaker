import threading
import logging
import os
import tempfile
import torch
import numpy as np
from faster_qwen3_tts import FasterQwen3TTS

logger = logging.getLogger(__name__)

REF_AUDIO_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav"

# faster-qwen3-tts expects full language names, not abbreviations
_LANG_MAP = {
    "zh": "Chinese",
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ja": "Japanese",
    "ko": "Korean",
    "auto": "Auto",
}


def _normalize_language(lang: str) -> str:
    return _LANG_MAP.get(lang.lower(), lang)


def _download_ref_audio(url: str) -> str:
    import requests
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    logger.info(f"Downloading reference audio: {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def _extract_speaker_embedding(model, ref_audio_path: str, device: str, dtype) -> torch.Tensor:
    prompt_items = model.model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text="",
        x_vector_only_mode=True,
    )
    spk_emb = prompt_items[0].ref_spk_embedding
    if spk_emb.dtype != dtype:
        spk_emb = spk_emb.to(dtype)
    return spk_emb


class TTSEngine:
    def __init__(self, model_id: str, speaker_pt: str, ref_audio: str = "",
                 device: str = "cuda", dtype: str = "bfloat16",
                 max_seq_len: int = 2048, chunk_size: int = 8):
        self.device = device
        self.chunk_size = chunk_size
        self.lock = threading.Lock()

        torch_dtype = getattr(torch, dtype)

        logger.info(f"Loading TTS model: {model_id}")
        self.model = FasterQwen3TTS.from_pretrained(
            model_id,
            device=device,
            dtype=torch_dtype,
            attn_implementation="sdpa",
            max_seq_len=max_seq_len,
        )

        # Load or create speaker embedding
        if os.path.exists(speaker_pt):
            logger.info(f"Loading speaker embedding: {speaker_pt}")
            spk_tensor = torch.load(speaker_pt, weights_only=True, map_location=device)
            if spk_tensor.dtype != torch_dtype:
                spk_tensor = spk_tensor.to(torch_dtype)
        else:
            # Use local ref_audio if provided, otherwise download default
            if ref_audio and os.path.exists(ref_audio):
                logger.info(f"Extracting speaker embedding from: {ref_audio}")
                spk_tensor = _extract_speaker_embedding(
                    self.model, ref_audio, device, torch_dtype)
            else:
                logger.info("speaker.pt not found, downloading reference audio...")
                tmp_path = _download_ref_audio(REF_AUDIO_URL)
                try:
                    spk_tensor = _extract_speaker_embedding(
                        self.model, tmp_path, device, torch_dtype)
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            torch.save(spk_tensor.cpu(), speaker_pt)
            logger.info(f"Speaker embedding saved to {speaker_pt}")

        self.voice_clone_prompt = {
            "ref_code": [None],
            "ref_spk_embedding": [spk_tensor],
            "x_vector_only_mode": [True],
            "icl_mode": [False],
        }
        logger.info("TTS engine ready")

    def warmup(self):
        logger.info("TTS warmup: generating dummy audio to capture CUDA graphs...")
        with self.lock:
            for _ in self.model.generate_voice_clone_streaming(
                text="你好。",
                language="Chinese",
                voice_clone_prompt=self.voice_clone_prompt,
                chunk_size=self.chunk_size,
            ):
                pass
        logger.info("TTS warmup complete")

    def synthesize_streaming(self, text: str, language: str, cancel_event=None):
        """Yield (np.float32_audio, sample_rate) chunks. Thread-safe.
        If cancel_event is provided and gets set, stops generation early.
        Uses polling lock acquisition so cancellation isn't blocked by a held lock.
        """
        lang = _normalize_language(language)

        # Acquire lock with cancel-awareness (don't block forever if another
        # pipeline is finishing up its TTS)
        while True:
            if cancel_event and cancel_event.is_set():
                logger.info("TTS cancelled while waiting for lock")
                return
            acquired = self.lock.acquire(timeout=0.1)
            if acquired:
                break

        try:
            for audio_chunk, sr, timing in self.model.generate_voice_clone_streaming(
                text=text,
                language=lang,
                voice_clone_prompt=self.voice_clone_prompt,
                chunk_size=self.chunk_size,
            ):
                if cancel_event and cancel_event.is_set():
                    logger.info("TTS generation cancelled mid-stream")
                    return
                yield audio_chunk, sr
        finally:
            self.lock.release()
