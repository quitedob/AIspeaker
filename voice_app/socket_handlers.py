import threading
import logging
from flask_socketio import Namespace
from .pipeline import run_conversation_pipeline
from .chat_history import (
    new_conversation_id, load_conversation, save_conversation,
    delete_conversation, list_conversations,
)
from .skill_manager import get_skill_prompt

logger = logging.getLogger(__name__)


class VoiceNamespace(Namespace):
    def __init__(self, asr, llm, tts, socketio, skills: dict):
        super().__init__("/voice")
        self.asr = asr
        self.llm = llm
        self.tts = tts
        self.sio = socketio
        self.skills = skills
        self._sessions = {}

    def on_connect(self):
        from flask import request
        sid = request.sid
        self._sessions[sid] = {
            "conv_id": None,
            "history": [],
            "language": "zh",
            "skill_id": "default",
            "cancel_event": None,
        }

    def on_disconnect(self):
        from flask import request
        sid = request.sid
        session = self._sessions.pop(sid, None)
        if session:
            if session.get("cancel_event"):
                session["cancel_event"].set()
            self._save_current(session)

    def _save_current(self, session):
        conv_id = session.get("conv_id")
        if conv_id and session["history"]:
            save_conversation(conv_id, session["history"],
                              skill_id=session.get("skill_id", "default"))

    def _send_conv_list(self, sid):
        """Send the conversation list to the client."""
        convs = list_conversations()
        self.sio.emit("conv_list", {"conversations": convs},
                      namespace="/voice", room=sid)

    def on_init_session(self, data):
        """Client connects and optionally resumes a conversation."""
        from flask import request
        sid = request.sid
        session = self._sessions.get(sid)
        if not session:
            return
        conv_id = data.get("conv_id", "")
        if conv_id:
            conv = load_conversation(conv_id)
            session["conv_id"] = conv_id
            session["history"] = conv.get("messages", [])
            session["skill_id"] = conv.get("skill_id", "default")
            self.sio.emit("conv_loaded", {
                "conv_id": conv_id,
                "messages": session["history"],
                "skill_id": session["skill_id"],
                "title": conv.get("title", ""),
            }, namespace="/voice", room=sid)
        self._send_conv_list(sid)

    def on_new_conversation(self, data=None):
        """Create a new empty conversation."""
        from flask import request
        sid = request.sid
        session = self._sessions.get(sid)
        if not session:
            return
        # Save current conversation first
        self._save_current(session)
        # Create new
        conv_id = new_conversation_id()
        session["conv_id"] = conv_id
        session["history"] = []
        # Keep current skill
        self.sio.emit("conv_loaded", {
            "conv_id": conv_id,
            "messages": [],
            "skill_id": session.get("skill_id", "default"),
            "title": "",
        }, namespace="/voice", room=sid)
        self._send_conv_list(sid)

    def on_switch_conversation(self, data):
        """Switch to an existing conversation."""
        from flask import request
        sid = request.sid
        session = self._sessions.get(sid)
        if not session:
            return
        conv_id = data.get("conv_id", "")
        if not conv_id:
            return
        # Cancel running pipeline
        self._cancel_current(session)
        # Save current
        self._save_current(session)
        # Load target
        conv = load_conversation(conv_id)
        session["conv_id"] = conv_id
        session["history"] = conv.get("messages", [])
        session["skill_id"] = conv.get("skill_id", "default")
        self.sio.emit("conv_loaded", {
            "conv_id": conv_id,
            "messages": session["history"],
            "skill_id": session["skill_id"],
            "title": conv.get("title", ""),
        }, namespace="/voice", room=sid)

    def on_delete_conversation(self, data):
        """Delete a conversation."""
        from flask import request
        sid = request.sid
        session = self._sessions.get(sid)
        if not session:
            return
        conv_id = data.get("conv_id", "")
        if not conv_id:
            return
        delete_conversation(conv_id)
        # If we deleted the active conversation, start a new one
        if session.get("conv_id") == conv_id:
            session["conv_id"] = None
            session["history"] = []
        self._send_conv_list(sid)

    def on_set_language(self, data):
        from flask import request
        session = self._sessions.get(request.sid)
        if session:
            session["language"] = data.get("language", "zh")

    def on_set_skill(self, data):
        from flask import request
        session = self._sessions.get(request.sid)
        if session:
            session["skill_id"] = data.get("skill_id", "default")

    def on_clear_history(self, data):
        from flask import request
        sid = request.sid
        session = self._sessions.get(sid)
        if session:
            session["history"] = []
            conv_id = session.get("conv_id")
            if conv_id:
                delete_conversation(conv_id)
                session["conv_id"] = None
            self._send_conv_list(sid)

    def _cancel_current(self, session):
        ev = session.get("cancel_event")
        if ev is not None:
            ev.set()
        session["cancel_event"] = None

    def on_interrupt(self, data=None):
        from flask import request
        session = self._sessions.get(request.sid)
        if session:
            self._cancel_current(session)
            self.sio.emit("interrupted", {}, namespace="/voice", room=request.sid)

    def on_audio_data(self, data):
        from flask import request
        sid = request.sid
        session = self._sessions.get(sid)
        if not session:
            return

        self._cancel_current(session)

        if isinstance(data, bytearray):
            audio_bytes = bytes(data)
        elif isinstance(data, bytes):
            audio_bytes = data
        elif isinstance(data, dict) and "data" in data:
            audio_bytes = bytes(data["data"])
        else:
            self.sio.emit("error", {"message": "Invalid audio data"},
                          namespace="/voice", room=sid)
            return

        # Auto-create conversation if none active
        if not session.get("conv_id"):
            session["conv_id"] = new_conversation_id()

        cancel_event = threading.Event()
        session["cancel_event"] = cancel_event

        language = session["language"]
        history = session["history"]
        conv_id = session["conv_id"]
        skill_id = session.get("skill_id", "default")
        skill_prompt = get_skill_prompt(self.skills, skill_id, language)
        sio = self.sio

        def on_stage(stage):
            if not cancel_event.is_set():
                sio.emit("stage", {"stage": stage}, namespace="/voice", room=sid)

        def on_transcript(text):
            if not cancel_event.is_set():
                sio.emit("transcript", {"text": text}, namespace="/voice", room=sid)

        def on_llm_chunk(token):
            if not cancel_event.is_set():
                sio.emit("llm_chunk", {"text": token}, namespace="/voice", room=sid)

        def on_audio_chunk(pcm_bytes):
            if not cancel_event.is_set():
                sio.emit("audio_chunk", pcm_bytes, namespace="/voice", room=sid)

        def on_audio_done():
            if not cancel_event.is_set():
                sio.emit("audio_done", {}, namespace="/voice", room=sid)
            if history:
                save_conversation(conv_id, history, skill_id=skill_id)
                # Update conv list after first message
                sio.emit("conv_list", {"conversations": list_conversations()},
                         namespace="/voice", room=sid)
            if session.get("cancel_event") is cancel_event:
                session["cancel_event"] = None

        def on_error(msg):
            if not cancel_event.is_set():
                logger.error(f"Pipeline error for {sid}: {msg}")
                sio.emit("error", {"message": msg}, namespace="/voice", room=sid)

        thread = threading.Thread(
            target=run_conversation_pipeline,
            args=(self.asr, self.llm, self.tts, audio_bytes, history, language,
                  on_transcript, on_llm_chunk, on_audio_chunk, on_audio_done,
                  on_error, on_stage, cancel_event),
            kwargs={"skill_prompt": skill_prompt},
            daemon=True,
        )
        thread.start()
