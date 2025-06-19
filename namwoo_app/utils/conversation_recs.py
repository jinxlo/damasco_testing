import logging
from typing import List, Dict, Tuple, Optional
import json
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
        logger.info("Redis connection for conversation recommendations established.")
    else:
        logger.error("Redis URL (broker_url) not found. Recommendation caching disabled.")
except RedisError as e:
    logger.exception(f"Failed to connect to Redis for recommendation cache: {e}")
    _redis_client = None

_RECS_TTL_SECONDS = 2 * 60 * 60  # 2 hours


def _recs_key(conversation_id: str) -> str:
    return f"namwoo:conversation:{conversation_id}:recs"


def save_recommendations(conversation_id: str, products: List[Dict], intent: str) -> None:
    if not _redis_client or not conversation_id or not products:
        return
    try:
        data = {"intent": intent, "products": products}
        _redis_client.setex(_recs_key(conversation_id), _RECS_TTL_SECONDS, json.dumps(data))
        logger.info(f"Saved {len(products)} recommendations for conversation {conversation_id}")
    except RedisError as e:
        logger.exception(f"Failed to save recommendations for {conversation_id}: {e}")


def get_recommendations(conversation_id: str) -> Tuple[List[Dict], Optional[str]]:
    if not _redis_client or not conversation_id:
        return [], None
    try:
        raw = _redis_client.get(_recs_key(conversation_id))
        if not raw:
            return [], None
        data = json.loads(raw)
        return data.get("products", []), data.get("intent")
    except (RedisError, json.JSONDecodeError) as e:
        logger.exception(f"Failed to fetch recommendations for {conversation_id}: {e}")
        return [], None
