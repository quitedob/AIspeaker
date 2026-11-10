import logging
import re
import tempfile
import os
from typing import Any, List
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

logger = logging.getLogger(__name__)

# SenseVoice tags like <|zh|>, <|HAPPY|>, <|BGM|>, <|Speech|>, <|withitn|> etc.
_TAG_RE = re.compile(r'<\|[^|]*\|>')


def _parse_asr_result(raw_res: Any) -> str:
    """健壮地解析 FunASR / SenseVoice 返回结果，支持多种结构：
    - 列表形式: [ { 'text': '...' }, ... ]
    - 字典形式: { 'text': '...', 'segments': [...], 'sentence_info': [...] }
    - 字符串形式
    返回拼接后的文本。
    """
    text_parts: List[str] = []
    try:
        if isinstance(raw_res, list):
            for item in raw_res:
                if isinstance(item, dict):
                    t = item.get('text') or item.get('output') or item.get('sentence', '')
                    if isinstance(t, str) and t.strip():
                        text_parts.append(t.strip())
                elif isinstance(item, str) and item.strip():
                    text_parts.append(item.strip())
        elif isinstance(raw_res, dict):
            if 'text' in raw_res and isinstance(raw_res['text'], str):
                text_parts.append(raw_res['text'].strip())
            if 'segments' in raw_res and isinstance(raw_res['segments'], list):
                for seg in raw_res['segments']:
                    if isinstance(seg, dict):
                        t = seg.get('text') or seg.get('content') or ''
                        if isinstance(t, str) and t.strip():
                            text_parts.append(t.strip())
            if 'sentence_info' in raw_res and isinstance(raw_res['sentence_info'], list):
                for s in raw_res['sentence_info']:
                    if isinstance(s, dict):
                        t = s.get('text') or s.get('sentence')
                        if isinstance(t, str) and t.strip():
                            text_parts.append(t.strip())
        elif isinstance(raw_res, str) and raw_res.strip():
            text_parts.append(raw_res.strip())
    except Exception:
        logger.warning("解析 ASR 原始结果时报错，尝试直接转换为字符串")
        try:
            text_parts.append(str(raw_res))
        except Exception:
            pass

    joined = " ".join([p for p in text_parts if p])

    # rich_transcription_postprocess + tag cleanup
    try:
        if joined:
            joined = rich_transcription_postprocess(joined)
    except Exception:
        pass

    # Strip remaining SenseVoice tags
    joined = _TAG_RE.sub('', joined).strip()
    # Collapse extra whitespace
    joined = re.sub(r'\s+', ' ', joined).strip()
    return joined


class ASREngine:
    def __init__(self, model: str = "iic/SenseVoiceSmall",
                 vad_model: str = "fsmn-vad", device: str = "cuda:0"):
        logger.info(f"Loading ASR model: {model} on {device}")
        self.model = AutoModel(
            model=model,
            vad_model=vad_model,
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
        )
        logger.info("ASR engine ready")

    def transcribe(self, audio_path: str) -> str:
        """Transcribe audio file to text. Uses robust result parsing
        and falls back to non-VAD mode if primary attempt returns empty."""
        logger.info(f"ASR transcribe: input={audio_path}")
        try:
            file_size = os.path.getsize(audio_path)
            logger.info(f"ASR input file size: {file_size} bytes")

            # Primary attempt with VAD
            result = self.model.generate(
                input=audio_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )
            logger.info(f"ASR raw output type={type(result).__name__}, value={result}")

            text = _parse_asr_result(result)
            if text:
                logger.info(f"ASR final text: '{text}'")
                return text

            # Fallback: retry without merge_vad in case VAD missed the speech
            logger.warning("ASR primary attempt returned empty, retrying without merge_vad...")
            result2 = self.model.generate(
                input=audio_path,
                cache={},
                language="auto",
                use_itn=True,
                batch_size_s=60,
                merge_vad=False,
            )
            logger.info(f"ASR fallback output: {result2}")

            text = _parse_asr_result(result2)
            if text:
                logger.info(f"ASR fallback text: '{text}'")
                return text

            logger.warning("ASR returned empty result on both attempts")
            return ""
        except Exception as e:
            logger.exception(f"ASR transcribe error: {e}")
            return ""
