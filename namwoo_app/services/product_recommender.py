import json
import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI

from ..config import Config
from . import product_service
from ..utils import conversation_location

logger = logging.getLogger(__name__)

_llm_client: Optional[OpenAI] = None
if Config.OPENAI_API_KEY:
    _llm_client = OpenAI(api_key=Config.OPENAI_API_KEY, timeout=5.0)

_RANK_PROMPT = (
    "You are a senior sales associate for an electronics store. "
    "Rank the provided products from best to worst match for the customer's request. "
    "Return ONLY JSON like {\"ordered_skus\": [\"SKU1\", \"SKU2\", \"SKU3\"]}."
)


def rank_products(user_intent: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """Use the configured LLM to intelligently rank candidate products."""
    if not candidates:
        return []

    model = getattr(Config, "RECOMMENDER_LLM_MODEL", "gpt-4o-mini")
    mode = getattr(Config, "RECOMMENDER_MODE", "llm")
    if mode != "llm" or not _llm_client or not model:
        return candidates[:top_n]

    limited = [
        {
            "item_code": it.get("item_code"),
            "name": it.get("item_name"),
            "brand": it.get("brand"),
            "price": it.get("price"),
            "similarity": it.get("similarity"),
            "description": it.get("llm_formatted_description") or it.get("description"),
        }
        for it in candidates[:12]
    ]

    payload = json.dumps({"intent": user_intent, "candidates": limited}, ensure_ascii=False)
    messages = [{"role": "system", "content": _RANK_PROMPT}, {"role": "user", "content": payload}]

    try:
        resp = _llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=64,
        )
        content = resp.choices[0].message.content if resp.choices else "{}"
        data = json.loads(content)
        ordered = data.get("ordered_skus", [])[:top_n]
    except Exception as e:
        logger.exception("LLM ranking failed: %s", e)
        return candidates[:top_n]

    sku_to_item = {p.get("item_code"): p for p in candidates}
    ranked: List[Dict[str, Any]] = []
    for sku in ordered:
        if sku in sku_to_item:
            ranked.append(sku_to_item.pop(sku))
    ranked.extend(list(sku_to_item.values()))
    return ranked[:top_n]


def get_ranked_products(intent: Dict[str, Any], city: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Retrieve products from the catalog and rank them."""
    query = intent.get("query")
    if not query:
        return []

    warehouses = conversation_location.get_warehouses_for_city(city) if city else None
    results = product_service.search_local_products(
        query_text=query,
        limit=getattr(Config, "PRODUCT_SEARCH_LIMIT", 10),
        filter_stock=True,
        warehouse_names=warehouses,
        min_price=intent.get("budget_min"),
        max_price=intent.get("budget_max"),
        sort_by=intent.get("sort_by"),
    )
    if not results:
        return []

    return rank_products(intent.get("query", ""), results, top_n=top_n)
