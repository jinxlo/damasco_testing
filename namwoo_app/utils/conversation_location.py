from typing import Dict, List, Optional
from ..data.store_locations import CITY_TO_WAREHOUSES, VALID_CITIES

_conversation_city_map: Dict[str, str] = {}


def set_conversation_city(conversation_id: str, city: str) -> None:
    if conversation_id and city:
        _conversation_city_map[conversation_id] = city


def get_conversation_city(conversation_id: str) -> Optional[str]:
    return _conversation_city_map.get(conversation_id)


def get_city_warehouses(conversation_id: str) -> List[str]:
    city = _conversation_city_map.get(conversation_id)
    if not city:
        return []
    return CITY_TO_WAREHOUSES.get(city, [])


def detect_city_from_text(text: str) -> Optional[str]:
    if not text:
        return None
    lower = text.lower()
    for city in VALID_CITIES:
        if city.lower() in lower:
            return city
    return None
