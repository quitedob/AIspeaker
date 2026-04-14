import re

# Sentence-ending punctuation for Chinese and English
_BOUNDARY_RE = re.compile(r'([。！？.!?\n])')


class SentenceSplitter:
    """Accumulates LLM tokens and yields complete sentences."""

    def __init__(self):
        self._buffer = ""

    def add(self, token: str):
        self._buffer += token

    def get_sentences(self) -> list[str]:
        """Return any complete sentences found in the buffer, removing them."""
        sentences = []
        while True:
            m = _BOUNDARY_RE.search(self._buffer)
            if not m:
                break
            end = m.end()
            sentence = self._buffer[:end].strip()
            if sentence:
                sentences.append(sentence)
            self._buffer = self._buffer[end:]
        return sentences

    def flush(self) -> str:
        """Return any remaining text in the buffer."""
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining
