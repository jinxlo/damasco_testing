import logging
from typing import Dict, List, Optional
import os
import json
import redis
from redis.exceptions import RedisError
from ..config import Config

logger = logging.getLogger(__name__)

# --- Redis Client Setup for Shared State ---
_redis_client = None
try:
    # Use the same broker_url from the existing config for consistency
    redis_url = Config.broker_url
    if redis_url:
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        # Test connection
        _redis_client.ping()
        logger.info("Redis connection for conversation location cache established successfully.")
    else:
        logger.error("Redis URL (broker_url) not found in Config. Conversation location caching will not work.")
except RedisError as e:
    logger.exception(f"Failed to connect to Redis for conversation cache: {e}")
    _redis_client = None

# --- Constants and Static Data Loading ---
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

# TTL for conversation location data in Redis (24 hours)
_LOCATION_CACHE_TTL_SECONDS = 24 * 60 * 60

def _get_redis_key(conversation_id: str) -> str:
    """Generates a namespaced Redis key."""
    return f"namwoo:conversation:{conversation_id}:city"


def set_conversation_city(conversation_id: str, city: str) -> None:
    """Stores the conversation's city in the shared Redis cache."""
    if not _redis_client:
        logger.warning("Redis client not available. Cannot set conversation city.")
        return
    if not conversation_id or not city:
        return

    try:
        redis_key = _get_redis_key(conversation_id)
        _redis_client.setex(redis_key, _LOCATION_CACHE_TTL_SECONDS, city)
        logger.info(f"Set location '{city}' for conversation {conversation_id} in Redis cache.")
    except RedisError as e:
        logger.exception(f"Failed to set location for conversation {conversation_id} in Redis: {e}")


def get_conversation_city(conversation_id: str) -> Optional[str]:
    """Retrieves the conversation's city from the shared Redis cache."""
    if not _redis_client:
        logger.warning("Redis client not available. Cannot get conversation city.")
        return None
    if not conversation_id:
        return None
        
    try:
        redis_key = _get_redis_key(conversation_id)
        city = _redis_client.get(redis_key)
        if city:
            logger.info(f"Retrieved location '{city}' for conversation {conversation_id} from Redis cache.")
        return city
    except RedisError as e:
        logger.exception(f"Failed to get location for conversation {conversation_id} from Redis: {e}")
        return None


def get_warehouses_for_city(city: str) -> List[str]:
    """Looks up the warehouse names for a given city from static data."""
    if not city:
        return []
    return CITY_TO_WAREHOUSES.get(city, [])


def get_city_warehouses(conversation_id: str) -> List[str]:
    """A helper function to get warehouses directly from a conversation ID."""
    city = get_conversation_city(conversation_id)
    if not city:
        return []
    return get_warehouses_for_city(city)


def detect_city_from_text(text: str) -> Optional[str]:
    """Detects a valid city name from a text string."""
    if not text:
        return None
    lower = text.lower()
    for city in VALID_CITIES:
        if city.lower() in lower:
            return city
    return None