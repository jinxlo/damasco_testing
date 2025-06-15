from typing import Dict, List, Optional
import os
import json
import logging

from . import db_utils
from ..models.conversation_city import ConversationCity

_STORE_LOCATIONS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "store_locations.json",
)

try:
    with open(_STORE_LOCATIONS_PATH, "r", encoding="utf-8") as f:
        _STORES_DATA = json.load(f)
except Exception:
    _STORES_DATA = []

CITY_TO_WAREHOUSES: Dict[str, List[str]] = {}
for store in _STORES_DATA:
    city = store.get("city")
    whs = store.get("whsName")
    if city and whs:
        CITY_TO_WAREHOUSES.setdefault(city, []).append(whs)

VALID_CITIES: List[str] = sorted(CITY_TO_WAREHOUSES.keys())


logger = logging.getLogger(__name__)


def set_conversation_city(conversation_id: str, city: str) -> None:
    """Persist the detected city for a conversation."""
    if not (conversation_id and city):
        return
    try:
        with db_utils.get_db_session() as session:
            if not session:
                return
            record = session.query(ConversationCity).filter_by(
                conversation_id=conversation_id
            ).first()
            if record:
                record.city = city
            else:
                session.add(ConversationCity(conversation_id=conversation_id, city=city))
    except Exception as e:
        logger.exception(f"Error saving city for conversation {conversation_id}: {e}")


def get_conversation_city(conversation_id: str) -> Optional[str]:
    """Retrieve the stored city for a conversation."""
    if not conversation_id:
        return None
    try:
        with db_utils.get_db_session() as session:
            if not session:
                return None
            record = session.query(ConversationCity).filter_by(
                conversation_id=conversation_id
            ).first()
            return record.city if record else None
    except Exception as e:
        logger.exception(f"Error fetching city for conversation {conversation_id}: {e}")
        return None


def get_warehouses_for_city(city: str) -> List[str]:
    if not city:
        return []
    return CITY_TO_WAREHOUSES.get(city, [])


def get_city_warehouses(conversation_id: str) -> List[str]:
    city = get_conversation_city(conversation_id)
    if not city:
        return []
    return get_warehouses_for_city(city)


def detect_city_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    for city in VALID_CITIES:
        if city.lower() in lower:
            return city
    return None
