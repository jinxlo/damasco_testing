import json
import logging
from typing import List, Dict, Any

from openai import OpenAI, APIError, RateLimitError, APITimeoutError, BadRequestError

from ..config import Config
from . import recommender_service

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "SYSTEM:\n"
    "You are a senior sales associate.\n"
    "• Input: JSON ➜ {\"intent\": {...}, \"candidates\": [ {sku, name, brand, price, similarity, description} …] } (max 12 items).\n"
    "• Think like a human salesperson: adapt to any product category, matching the user's needs, budget hints, and brand or feature preferences.\n"
    "• Return ONLY a JSON object:\n"
    "  { \"ordered_skus\": [\"SKU1\", \"SKU2\", \"SKU3\" ] }\n"
    "• List exactly three SKUs, best fit first.\n"
    "• Do NOT add explanations, extra keys or more than 3 items.\n"
)


def _call_llm(messages: List[Dict[str, str]], model: str) -> str:
    client = OpenAI(api_key=Config.OPENAI_API_KEY, timeout=3.0)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=64,
    )
    return response.choices[0].message.content if response.choices else ""


def get_ranked_products(intent: Dict[str, Any], items: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    if not items:
        return []

    try:
        model = Config.RECOMMENDER_LLM_MODEL
        if not Config.OPENAI_API_KEY or not model:
            raise ValueError("LLM configuration missing")

        limited = items[:12]
        payload = json.dumps({"intent": intent, "candidates": limited}, ensure_ascii=False)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ]
        content = _call_llm(messages, model)
        data = json.loads(content)
        ordered = data.get("ordered_skus", [])
        ordered = ordered[:3]
        sku_to_item = {it.get("item_code"): it for it in items}
        ranked: List[Dict[str, Any]] = []
        for sku in ordered:
            if sku in sku_to_item:
                ranked.append(sku_to_item.pop(sku))
        ranked.extend(sku_to_item.values())
        return ranked[:top_n]
    except Exception as e:
        logger.exception("LLM ranking failed, falling back to python ranker: %s", e)
        try:
            return recommender_service.rank_products(intent, items, top_n)
        except Exception:
            return items[:top_n]
