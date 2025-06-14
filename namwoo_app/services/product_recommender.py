"""LLM-based product ranking service."""
from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Optional

from openai import OpenAI

from ..config import Config
from . import product_service
from ..utils import conversation_location

logger = logging.getLogger(__name__)

_llm_client: Optional[OpenAI] = None
if getattr(Config, "OPENAI_API_KEY", None):
    try:
        _llm_client = OpenAI(api_key=Config.OPENAI_API_KEY, timeout=5.0)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to init OpenAI client: %s", exc)
        _llm_client = None

_RANK_PROMPT = (
    "Eres un asesor de ventas senior en una tienda de electronicos. "
    "Debes ordenar los productos proporcionados del mejor al peor para la peticion del cliente. "
    "Responde SOLO con JSON en el formato {\"ordered_skus\": [\"SKU1\", ...]}."
)


def _prepare_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sku": item.get("item_code"),
        "name": item.get("item_name"),
        "brand": item.get("brand"),
        "price": item.get("price"),
        "specs": item.get("especificacion") or item.get("specifitacion"),
        "desc": item.get("llm_formatted_description") or item.get("description"),
    }


def rank_products(user_intent: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """Use the configured LLM to intelligently rank candidate products."""
    if not candidates:
        return []

    mode = getattr(Config, "RECOMMENDER_MODE", "llm")
    model = getattr(Config, "RECOMMENDER_LLM_MODEL", "gpt-4.1")
    if mode != "llm" or not _llm_client:
        return candidates[:top_n]

    formatted = [_prepare_candidate(c) for c in candidates[:12]]
    payload = json.dumps({"intent": user_intent, "products": formatted}, ensure_ascii=False)
    messages = [
        {"role": "system", "content": _RANK_PROMPT},
        {"role": "user", "content": payload},
    ]

    try:
        resp = _llm_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0,
            max_tokens=128,
        )
        content = resp.choices[0].message.content if resp.choices else "{}"
        data = json.loads(content)
        ordered = data.get("ordered_skus", [])
    except Exception as exc:  # pragma: no cover - network issues etc.
        logger.exception("LLM ranking failed: %s", exc)
        return candidates[:top_n]

    sku_map = {p.get("item_code"): p for p in candidates}
    ranked: List[Dict[str, Any]] = []
    for sku in ordered:
        if sku in sku_map:
            ranked.append(sku_map.pop(sku))
    ranked.extend(list(sku_map.values()))
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

    return rank_products(query, results, top_n=top_n)
