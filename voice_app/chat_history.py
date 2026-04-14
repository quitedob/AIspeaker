"""Persistent chat history — each conversation is a separate JSON file in tmp/.
Conversations are independent of skills.
"""

import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

HISTORY_DIR = Path(__file__).resolve().parent.parent / "tmp"


def _ensure_dir():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def _safe(s: str) -> str:
    return "".join(c for c in s if c.isalnum() or c in "-_")[:64] or "x"


def _conv_path(conv_id: str) -> Path:
    return HISTORY_DIR / f"{_safe(conv_id)}.json"


def new_conversation_id() -> str:
    """Generate a unique conversation ID."""
    return f"conv_{int(time.time())}_{id(object()) % 10000:04d}"


def load_conversation(conv_id: str) -> dict:
    """Load a conversation. Returns {messages, title, created_at, skill_id}."""
    _ensure_dir()
    path = _conv_path(conv_id)
    if not path.exists():
        return {"messages": [], "title": "", "skill_id": "default", "created_at": ""}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load conversation {conv_id}: {e}")
        return {"messages": [], "title": "", "skill_id": "default", "created_at": ""}


def save_conversation(conv_id: str, messages: list[dict],
                      title: str = "", skill_id: str = "default"):
    """Save a conversation to disk."""
    _ensure_dir()
    path = _conv_path(conv_id)
    # Preserve created_at if file exists
    created_at = ""
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                old = json.load(f)
            created_at = old.get("created_at", "")
        except Exception:
            pass
    if not created_at:
        created_at = time.strftime("%Y-%m-%d %H:%M:%S")

    # Auto-generate title from first user message if empty
    if not title and messages:
        for m in messages:
            if m.get("role") == "user" and m.get("content"):
                title = m["content"][:30]
                break

    data = {
        "conv_id": conv_id,
        "title": title,
        "skill_id": skill_id,
        "created_at": created_at,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "message_count": len(messages),
        "messages": messages,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save conversation {conv_id}: {e}")


def delete_conversation(conv_id: str):
    path = _conv_path(conv_id)
    try:
        if path.exists():
            path.unlink()
    except Exception as e:
        logger.warning(f"Failed to delete conversation {conv_id}: {e}")


def list_conversations() -> list[dict]:
    """List all conversations, newest first. Returns summary info only."""
    _ensure_dir()
    convs = []
    for f in HISTORY_DIR.glob("conv_*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            convs.append({
                "conv_id": data.get("conv_id", f.stem),
                "title": data.get("title", "未命名对话"),
                "skill_id": data.get("skill_id", "default"),
                "message_count": data.get("message_count", 0),
                "updated_at": data.get("updated_at", ""),
                "created_at": data.get("created_at", ""),
            })
        except Exception:
            pass
    convs.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    return convs
