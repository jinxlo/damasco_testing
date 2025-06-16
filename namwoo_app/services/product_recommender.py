"""LLM-based product ranking service."""
from __future__ import annotations

import json
import logging
import re
from typing import List, Dict, Any, Optional

from openai import OpenAI

from ..config import Config
try:  # Allow tests to stub this before import
    from . import product_service
except Exception:  # pragma: no cover - allow missing deps in tests
    product_service = None
try:
    from ..utils import conversation_location
except Exception:  # pragma: no cover - allow missing deps in tests
    conversation_location = None

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
    """
    Intelligently ranks and selects products.
    - For initial relevance-based searches, it provides a low-mid-high price spread.
    - For explicit price-sorted searches (e.g., "cheapest"), it respects that order.
    """
    if not candidates:
        return []

    # If configured for LLM ranking and a client is available, try that first
    if getattr(Config, "RECOMMENDER_MODE", "llm") == "llm" and _llm_client is not None:
        try:
            cand_payload = json.dumps([_prepare_candidate(c) for c in candidates], ensure_ascii=False)
            user_prompt = f"{cand_payload}\nUsuario: {user_intent}"
            response = _llm_client.chat.completions.create(
                model=getattr(Config, "RECOMMENDER_LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": _RANK_PROMPT}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                max_tokens=100,
            )
            ordered = json.loads(response.choices[0].message.content).get("ordered_skus", [])
            ranked = [next((c for c in candidates if c.get("item_code") == sku), None) for sku in ordered]
            ranked = [r for r in ranked if r]
            if ranked:
                return ranked[:top_n]
        except Exception as exc:  # pragma: no cover - be resilient during tests
            logger.warning("LLM ranking failed: %s", exc)

    # Check if the results are already sorted by price from the search_local_products call
    # This happens when the user asks for "cheapest" or "most expensive".
    is_price_sorted = any(
        s in user_intent.lower() for s in ["cheapest", "most affordable", "más barato", "mas economico", "price_asc", "price_desc"]
    )

    if is_price_sorted:
        logger.info(f"Results are already price-sorted. Returning top {top_n} candidates.")
        return candidates[:top_n]

    # --- NEW: Low-Mid-High Price Spread Logic for initial/relevance searches ---
    logger.info(f"Applying low-mid-high price spread ranking for initial relevance search.")
    if len(candidates) < 3:
        # Not enough candidates for a spread, just return them sorted by price.
        return sorted(candidates, key=lambda p: p.get("price") or 0.0)

    # Sort candidates by price to easily find cheapest, mid, and most expensive
    sorted_by_price = sorted(candidates, key=lambda p: p.get("price") or 0.0)

    cheapest = sorted_by_price[0]
    most_expensive = sorted_by_price[-1]
    
    # Find the middle product. Avoid picking the same as cheapest or most expensive if possible.
    mid_index = len(sorted_by_price) // 2
    middle = sorted_by_price[mid_index]

    # Assemble the final list, ensuring no duplicates and respecting top_n
    ranked: List[Dict[str, Any]] = []
    seen_model_names = set()

    for product in [cheapest, middle, most_expensive]:
        base_model = product.get("base_model_name")
        if base_model not in seen_model_names:
            ranked.append(product)
            seen_model_names.add(base_model)

    # Ensure the final list is sorted by price for presentation
    ranked.sort(key=lambda p: p.get("price") or 0.0)

    return ranked[:top_n]


def get_ranked_products(intent: Dict[str, Any], city: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Retrieve products from the catalog and rank them."""
    query = intent.get("query")
    if not query:
        return []

    sort_by_param = intent.get("sort_by", "relevance")

    global conversation_location
    if conversation_location is None:
        try:
            from ..utils import conversation_location as cl
            conversation_location = cl
        except Exception:  # pragma: no cover - keep optional for tests
            conversation_location = None

    warehouses = conversation_location.get_warehouses_for_city(city) if (city and conversation_location) else None
    results = product_service.search_local_products(
        query_text=query,
        limit=getattr(Config, "PRODUCT_SEARCH_LIMIT", 10),
        filter_stock=True,
        warehouse_names=warehouses,
        min_price=intent.get("budget_min"),
        max_price=intent.get("budget_max"),
        sort_by=sort_by_param,
    )
    if not results:
        return []
    
    # Pass the original user intent text to the ranker to check for price-related keywords
    return rank_products(query, results, top_n=top_n)