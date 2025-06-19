import json
import os
import logging
from datetime import datetime
from ..config import Config

logger = logging.getLogger(__name__)

INTERACTION_LOG_FILE = getattr(Config, 'INTERACTION_LOG_FILE', os.path.join(Config.LOG_DIR, 'interactions.jsonl'))
os.makedirs(os.path.dirname(INTERACTION_LOG_FILE), exist_ok=True)


def log_conversation(conversation_id: str, user_id: str, role: str, message: str) -> None:
    """Append a single conversation entry to the JSONL log file."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "conversation_id": str(conversation_id) if conversation_id else None,
        "user_id": str(user_id) if user_id else None,
        "role": role,
        "message": message or "",
    }
    try:
        with open(INTERACTION_LOG_FILE, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        logger.exception(f"Failed to write interaction log entry: {e}")
