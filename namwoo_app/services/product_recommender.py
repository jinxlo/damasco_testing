import json
import logging
from typing import List, Dict, Any

from openai import OpenAI

from ..config import Config
from . import product_service
from ..utils import conversation_location

logger = logging.getLogger(__name__)

_llm_client: OpenAI = OpenAI(api_key=Config.OPENAI_API_KEY, timeout=3.0)

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

_INTENT_PROMPT = (
    "You are a helpful assistant. Extract the user's intended product category, use case, keywords, and budget from the text. "
    "Return only JSON with keys: category, use_case, keywords (list), budget_max."
)


def extract_structured_intent(raw_query: str) -> Dict[str, Any]:
    if not raw_query:
        return {}
    try:
        messages = [
            {"role": "system", "content": _INTENT_PROMPT},
            {"role": "user", "content": raw_query},
        ]
        response = _llm_client.chat.completions.create(
            model=Config.RECOMMENDER_LLM_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=80,
        )
        content = response.choices[0].message.content if response.choices else "{}"
        return json.loads(content)
    except Exception:
        logger.exception("Intent extraction failed")
        return {}


def _call_llm(messages: List[Dict[str, str]], model: str) -> str:
    response = _llm_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=64,
    )
    return response.choices[0].message.content if response.choices else ""


def _rank_with_llm(intent: Dict[str, Any], items: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
    if not items:
        return []
    try:
        model = Config.RECOMMENDER_LLM_MODEL
        if not Config.OPENAI_API_KEY or not model:
            raise ValueError("LLM configuration missing")

        structured_intent = extract_structured_intent(intent.get("raw_query", ""))
        logger.info(f"[LLM RANKING] Intent extracted: {structured_intent}")
        logger.info(f"[LLM RANKING] Candidates received: {len(items)}")

        limited = [
            {
                "item_code": it.get("item_code"),
                "name": it.get("item_name"),
                "brand": it.get("brand"),
                "price": it.get("price"),
                "similarity": it.get("similarity"),
                "description": it.get("llm_formatted_description") or it.get("description"),
            }
            for it in items[:12]
        ]

        payload = json.dumps({"intent": structured_intent, "candidates": limited}, ensure_ascii=False)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ]
        content = _call_llm(messages, model)
        data = json.loads(content)
        ordered = data.get("ordered_skus", [])[:top_n]
        sku_to_item = {it.get("item_code"): it for it in items}
        ranked: List[Dict[str, Any]] = []
        for sku in ordered:
            if sku in sku_to_item:
                ranked.append(sku_to_item.pop(sku))
        ranked.extend(list(sku_to_item.values()))
        logger.info(f"[LLM RANKING] Top SKUs returned: {ordered}")
        return ranked[:top_n]
    except Exception:
        logger.exception("LLM ranking failed.")
        return items[:top_n]


def get_ranked_products(intent: Dict[str, Any], city: str, top_n: int = 5) -> List[Dict[str, Any]]:
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
    brand = intent.get("brand")
    if brand:
        filtered = [r for r in results if (r.get("brand") or "").upper() == brand.upper()]
        if filtered:
            results = filtered
    for r in results:
        code = r.get("item_code")
        if code:
            variants = product_service.get_color_variants_for_sku(code)
            if variants and len(variants) > 1:
                r["color_variants"] = variants
    results.sort(key=lambda x: x.get("stock", 0), reverse=True)
    ranked = _rank_with_llm({"raw_query": query}, results, top_n=top_n)
    return ranked
