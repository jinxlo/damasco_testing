import json
import logging
from typing import Dict
import redis
from redis.exceptions import RedisError
from ..config import Config

logger = logging.getLogger(__name__)

_redis_client = None
try:
    redis_url = Config.broker_url
    if redis_url:
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        logger.info("Redis connection for conversation color map established.")
    else:
        logger.error("Redis URL (broker_url) not found. Color map caching disabled.")
except RedisError as e:
    logger.exception(f"Failed to connect to Redis for color map: {e}")
    _redis_client = None

_COLOR_MAP_TTL_SECONDS = 2 * 60 * 60  # 2 hours

def _key(conv_id: str) -> str:
    return f"namwoo:conversation:{conv_id}:color_map"


def set_color_map(conv_id: str, mapping: Dict[str, str]) -> None:
    """Store the color->SKU mapping for a conversation."""
    if not _redis_client or not conv_id or not mapping:
        return
    try:
        _redis_client.setex(_key(conv_id), _COLOR_MAP_TTL_SECONDS, json.dumps(mapping))
        logger.info("Saved color map for conversation %s", conv_id)
    except RedisError as e:
        logger.exception("Failed to save color map for %s: %s", conv_id, e)


def get_color_map(conv_id: str) -> Dict[str, str]:
    if not _redis_client or not conv_id:
        return {}
    try:
        raw = _redis_client.get(_key(conv_id))
        return json.loads(raw) if raw else {}
    except (RedisError, json.JSONDecodeError) as e:
        logger.exception("Failed to fetch color map for %s: %s", conv_id, e)
        return {}

