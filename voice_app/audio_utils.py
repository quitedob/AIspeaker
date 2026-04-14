import io
import wave
import numpy as np


def pcm_bytes_to_wav_bytes(pcm_data: bytes, sample_rate: int = 24000,
                           sample_width: int = 2, channels: int = 1) -> bytes:
    """Convert raw PCM bytes to WAV-format bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def numpy_to_int16_pcm(audio: np.ndarray) -> bytes:
    """Convert float32 numpy audio to int16 PCM bytes."""
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    return pcm.tobytes()
