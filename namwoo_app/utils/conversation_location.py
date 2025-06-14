from typing import Dict, List, Optional
import os
import json

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

_conversation_city_map: Dict[str, str] = {}


def set_conversation_city(conversation_id: str, city: str) -> None:
    if conversation_id and city:
        _conversation_city_map[conversation_id] = city


def get_conversation_city(conversation_id: str) -> Optional[str]:
    return _conversation_city_map.get(conversation_id)


def get_warehouses_for_city(city: str) -> List[str]:
    if not city:
        return []
    return CITY_TO_WAREHOUSES.get(city, [])


def get_city_warehouses(conversation_id: str) -> List[str]:
    city = _conversation_city_map.get(conversation_id)
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
