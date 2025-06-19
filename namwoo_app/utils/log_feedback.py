import json
import os
from datetime import datetime
from ..config import Config

FEEDBACK_LOG_FILE = getattr(Config, 'FEEDBACK_LOG_FILE', os.path.join(Config.LOG_DIR, 'feedback.jsonl'))
os.makedirs(os.path.dirname(FEEDBACK_LOG_FILE), exist_ok=True)


def log_feedback(conversation_id: str, user_id: str, message: str, product_requested: str, recommendations: list, exact_match: bool, location: str, location_match: bool) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "conversation_id": str(conversation_id),
        "user_id": str(user_id),
        "message": message,
        "product_requested": product_requested,
        "location_mentioned": location,
        "bot_recommendations": recommendations,
        "exact_match_found": exact_match,
        "location_match": location_match,
    }

    try:
        with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
    except Exception as e:
        import logging
        logging.getLogger("system").exception(f"Failed to write feedback log: {e}")

