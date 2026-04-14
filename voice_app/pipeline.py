import logging
import re
import tempfile
import os
import threading
import numpy as np
from .asr_engine import ASREngine
from .llm_client import LLMClient
from .sentence_splitter import SentenceSplitter
from .tts_engine import TTSEngine
from .audio_utils import numpy_to_int16_pcm

logger = logging.getLogger(__name__)

# Fixed frame size for smooth playback: 40ms at 24kHz = 960 samples
FRAME_MS = 40
PLAYBACK_SR = 24000
FRAME_SAMPLES = int(PLAYBACK_SR * FRAME_MS / 1000)  # 960


def transcribe_audio(asr: ASREngine, audio_bytes: bytes) -> str:
    """Save browser WAV audio (16kHz 16bit mono PCM) to temp file, run ASR.
    Handles long audio by relying on SenseVoice's built-in VAD segmentation.
    """
    logger.info(f"transcribe_audio: received {len(audio_bytes)} bytes, "
                f"header: {audio_bytes[:4] if len(audio_bytes) >= 4 else b'?'}")

    if len(audio_bytes) < 1000:
        logger.error("Audio data too small")
        return ""

    duration_s = 0
    if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
        import struct
        channels = struct.unpack_from('<H', audio_bytes, 22)[0]
        sample_rate = struct.unpack_from('<I', audio_bytes, 24)[0]
        bits = struct.unpack_from('<H', audio_bytes, 34)[0]
        data_size = struct.unpack_from('<I', audio_bytes, 40)[0]
        duration_s = data_size / (sample_rate * channels * bits // 8)
        logger.info(f"WAV: {sample_rate}Hz, {channels}ch, {bits}bit, "
                     f"data={data_size}B, duration={duration_s:.2f}s")
        if duration_s < 0.3:
            logger.error(f"Audio too short: {duration_s:.2f}s")
            return ""
    else:
        logger.warning(f"Not a valid WAV header: {audio_bytes[:12]}")

    path_wav = tempfile.mktemp(suffix=".wav")
    try:
        with open(path_wav, "wb") as f:
            f.write(audio_bytes)

        # Verify written file
        written_size = os.path.getsize(path_wav)
        logger.info(f"Wrote {written_size} bytes to {path_wav} (expected {len(audio_bytes)})")
        if written_size != len(audio_bytes):
            logger.error("File size mismatch after write!")

        text = asr.transcribe(path_wav)
        logger.info(f"ASR result ({duration_s:.1f}s audio): '{text[:120] if text else ''}'")
        return text.strip()
    except Exception as e:
        logger.exception("transcribe_audio error")
        return ""
    finally:
        try:
            os.unlink(path_wav)
        except OSError:
            pass


class _ThinkFilter:
    """Streaming filter that strips <think>...</think> blocks from LLM output."""

    def __init__(self):
        self._inside_think = False
        self._pending = ""

    def feed(self, token: str) -> str:
        self._pending += token
        output = ""
        while self._pending:
            if self._inside_think:
                close_idx = self._pending.lower().find("</think>")
                if close_idx != -1:
                    self._inside_think = False
                    self._pending = self._pending[close_idx + 8:]
                else:
                    if len(self._pending) > 8:
                        self._pending = self._pending[-8:]
                    break
            else:
                open_idx = self._pending.lower().find("<think>")
                if open_idx != -1:
                    output += self._pending[:open_idx]
                    self._inside_think = True
                    self._pending = self._pending[open_idx + 7:]
                else:
                    safe_len = len(self._pending)
                    for i in range(1, min(7, len(self._pending) + 1)):
                        if self._pending[-i:].lower() == "<think>"[:i]:
                            safe_len = len(self._pending) - i
                            break
                    if safe_len > 0:
                        output += self._pending[:safe_len]
                        self._pending = self._pending[safe_len:]
                    break
        return output

    def flush(self) -> str:
        if self._inside_think:
            self._pending = ""
            return ""
        remaining = self._pending
        self._pending = ""
        return remaining


def _clean_for_tts(text: str) -> str:
    """Strip Markdown formatting, quotes, and other symbols that TTS should not read aloud.
    The original text is still shown to the user; this only affects what gets synthesized.
    """
    # Bold/italic: **text**, *text*, __text__, _text_
    s = re.sub(r'\*{1,3}', '', text)
    s = re.sub(r'_{1,3}', '', s)
    # Strikethrough: ~~text~~
    s = re.sub(r'~~', '', s)
    # Inline code: `code`
    s = re.sub(r'`', '', s)
    # Headers: # ## ### etc.
    s = re.sub(r'^#{1,6}\s*', '', s, flags=re.MULTILINE)
    # Bullet points: - item, * item, + item, 1. item
    s = re.sub(r'^\s*[-*+]\s+', '', s, flags=re.MULTILINE)
    s = re.sub(r'^\s*\d+\.\s+', '', s, flags=re.MULTILINE)
    # Links: [text](url) -> text
    s = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', s)
    # Images: ![alt](url) -> alt
    s = re.sub(r'!\[([^\]]*)\]\([^)]*\)', r'\1', s)
    # Em dash variants
    s = s.replace('—', '，').replace('--', '，')
    # Horizontal rules: ---, ***, ___
    s = re.sub(r'^[-*_]{3,}\s*$', '', s, flags=re.MULTILINE)
    # Quotation marks (all variants)
    s = re.sub(r'["""\'\'「」『』【】《》〈〉]', '', s)
    # Parentheses and brackets
    s = re.sub(r'[()（）\[\]{}]', '', s)
    # Other decorative symbols TTS shouldn't read
    s = re.sub(r'[~·•→←↑↓|/\\]', '', s)
    # Collapse multiple spaces into one
    s = re.sub(r'[ \t]+', ' ', s)
    # Collapse multiple newlines
    s = re.sub(r'\n{2,}', '\n', s)
    # Strip leading/trailing whitespace per line
    s = '\n'.join(line.strip() for line in s.split('\n'))
    return s.strip()


class _FrameEmitter:
    """Accumulates float32 PCM and emits fixed-size int16 frames."""

    def __init__(self, on_audio_chunk, frame_samples: int = FRAME_SAMPLES):
        self._buf = np.empty(0, dtype=np.float32)
        self._on_chunk = on_audio_chunk
        self._frame = frame_samples

    def feed(self, audio: np.ndarray):
        if audio.size == 0:
            return
        self._buf = np.concatenate([self._buf, audio.ravel()])
        self._emit(final=False)

    def flush(self):
        self._emit(final=True)

    def _emit(self, final: bool):
        while self._buf.size >= self._frame:
            frame = self._buf[:self._frame]
            self._buf = self._buf[self._frame:]
            self._on_chunk(numpy_to_int16_pcm(frame))
        if final and self._buf.size > 0:
            self._on_chunk(numpy_to_int16_pcm(self._buf))
            self._buf = np.empty(0, dtype=np.float32)


def run_conversation_pipeline(
    asr: ASREngine,
    llm: LLMClient,
    tts: TTSEngine,
    audio_bytes: bytes,
    history: list[dict],
    language: str,
    on_transcript,
    on_llm_chunk,
    on_audio_chunk,
    on_audio_done,
    on_error,
    on_stage,
    cancel_event: threading.Event,
    skill_prompt: str = "",
):
    """Full pipeline: ASR -> LLM stream -> TTS stream -> audio chunks.

    - on_stage(stage_name) notifies the client which step is active
    - cancel_event: set by caller to abort the pipeline mid-flight
    - skill_prompt: if set, overrides the default system prompt for LLM
    """
    try:
        # 1. ASR
        on_stage("asr")
        logger.info("Pipeline: starting ASR...")
        text = transcribe_audio(asr, audio_bytes)
        if cancel_event.is_set():
            logger.info("Pipeline cancelled during ASR")
            return
        if not text:
            on_error("无法识别语音，请说大声一点或说长一点再试。")
            return
        logger.info(f"Pipeline: ASR result: {text[:80]}")
        on_transcript(text)
        history.append({"role": "user", "content": text})

        # 2. LLM streaming
        on_stage("llm")
        think_filter = _ThinkFilter()
        splitter = SentenceSplitter()
        emitter = _FrameEmitter(on_audio_chunk)
        full_display = ""
        tts_started = False

        logger.info("Pipeline: starting LLM stream...")
        for token in llm.chat_stream(history, language=language,
                                      skill_prompt=skill_prompt):
            if cancel_event.is_set():
                logger.info("Pipeline cancelled during LLM")
                return

            visible = think_filter.feed(token)
            if visible:
                on_llm_chunk(visible)
                full_display += visible
                splitter.add(visible)

                for sentence in splitter.get_sentences():
                    if cancel_event.is_set():
                        return
                    if not tts_started:
                        on_stage("tts")
                        tts_started = True
                    tts_text = _clean_for_tts(sentence)
                    if not tts_text:
                        continue
                    logger.info(f"Pipeline: TTS sentence: {tts_text[:60]}")
                    for audio_np, sr in tts.synthesize_streaming(tts_text, language, cancel_event):
                        if cancel_event.is_set():
                            return
                        emitter.feed(audio_np)

        if cancel_event.is_set():
            return

        # Flush think filter
        remaining_visible = think_filter.flush()
        if remaining_visible:
            on_llm_chunk(remaining_visible)
            full_display += remaining_visible
            splitter.add(remaining_visible)

        # Flush remaining text through TTS
        remaining = splitter.flush()
        if remaining:
            if not tts_started:
                on_stage("tts")
            tts_text = _clean_for_tts(remaining)
            if tts_text:
                logger.info(f"Pipeline: TTS flush: {tts_text[:60]}")
                for audio_np, sr in tts.synthesize_streaming(tts_text, language, cancel_event):
                    if cancel_event.is_set():
                        return
                    emitter.feed(audio_np)

        emitter.flush()
        history.append({"role": "assistant", "content": full_display})
        logger.info("Pipeline: done")

    except Exception as e:
        if not cancel_event.is_set():
            logger.exception("Pipeline error")
            on_error(str(e))
    finally:
        on_audio_done()
