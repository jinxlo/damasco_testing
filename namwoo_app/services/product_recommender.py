"""LLM-based product ranking service."""
from __future__ import annotations

import json
import logging
import re
from typing import List, Dict, Any, Optional
from decimal import Decimal

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
try: # Optional import for testing environments
    from ..utils import product_utils
except Exception: # pragma: no cover - allow missing deps in tests
    product_utils = None


logger = logging.getLogger(__name__)

_llm_client: Optional[OpenAI] = None
if getattr(Config, "OPENAI_API_KEY", None):
    try:
        _llm_client = OpenAI(api_key=Config.OPENAI_API_KEY, timeout=10.0)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to init OpenAI client: %s", exc)
        _llm_client = None

# New, more sophisticated prompt for the AI Sales-Associate
_SALES_ASSOCIATE_PROMPT = (
    "Eres un experto asesor de ventas de tecnología para la tienda Damasco. "
    "Tu tarea es analizar la consulta de un cliente y la siguiente lista de productos disponibles (en formato JSON). "
    "Selecciona los 3 MEJORES productos que respondan a la necesidad del cliente. "
    "Para cada producto seleccionado, escribe una recomendación breve y persuasiva (máximo 15 palabras) en la clave 'reason_for_recommendation'. "
    "Esta razón debe ser útil y relevante para un cliente, como 'Ideal para juegos por su procesador' o 'La mejor cámara en este rango de precio'.\n\n"
    "Considera un balance entre precio, características y relevancia para la consulta del cliente. "
    "No te limites a los más caros o los más baratos. Piensa como un verdadero vendedor ayudando a un cliente a decidir.\n\n"
    "Responde ÚNICAMENTE con un objeto JSON válido que contenga una clave 'recommendations'. "
    "El valor debe ser una lista de hasta 3 objetos, cada uno con las claves 'sku' y 'reason_for_recommendation'.\n"
    "Ejemplo de respuesta:\n"
    "{\n"
    "  \"recommendations\": [\n"
    "    {\"sku\": \"D0001234\", \"reason_for_recommendation\": \"Excelente balance entre precio y rendimiento para el día a día.\"},\n"
    "    {\"sku\": \"D0005678\", \"reason_for_recommendation\": \"La mejor opción si tu prioridad es la fotografía profesional.\"},\n"
    "    {\"sku\": \"D0009012\", \"reason_for_recommendation\": \"Perfecto para juegos y multimedia por su pantalla y procesador.\"}\n"
    "  ]\n"
    "}"
)


def _prepare_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    """Prepares a candidate product dictionary for the LLM, ensuring JSON-serializable types."""
    price = item.get("price")
    return {
        "sku": item.get("item_code"),
        "name": item.get("item_name"),
        "brand": item.get("brand"),
        "price": float(price) if isinstance(price, Decimal) else price,
        "specs": item.get("especificacion") or item.get("specifitacion"),
        "desc": item.get("llm_summarized_description") or item.get("description"),
    }


def rank_products(user_intent: str, candidates: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Intelligently ranks and selects products using an LLM sales associate persona.
    Falls back to simpler logic if LLM ranking fails.
    """
    if not candidates:
        return []

    if getattr(Config, "RECOMMENDER_MODE", "llm") == "llm" and _llm_client is not None:
        try:
            prepared_candidates = [_prepare_candidate(c) for c in candidates]
            cand_payload = json.dumps(prepared_candidates, ensure_ascii=False)
            user_prompt = f"Lista de Productos:\n{cand_payload}\n\nConsulta del Cliente:\n'{user_intent}'"

            response = _llm_client.chat.completions.create(
                model=getattr(Config, "RECOMMENDER_LLM_MODEL", "gpt-4o-mini"),
                messages=[{"role": "system", "content": _SALES_ASSOCIATE_PROMPT}, {"role": "user", "content": user_prompt}],
                temperature=0.1,
                max_tokens=400, # Increased to allow for reasons
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            recommendations = data.get("recommendations", [])
            if not recommendations and "ordered_skus" in data:
                ordered_products = [
                    next((c for c in candidates if c.get("item_code") == sku), None)
                    for sku in data["ordered_skus"]
                ]
                ordered_products = [p for p in ordered_products if p]
                if ordered_products:
                    return ordered_products[:top_n]

            enriched_candidates = []
            # Use the LLM's ordering and enrichment
            for rec in recommendations:
                sku = rec.get("sku")
                product = next((c for c in candidates if c.get("item_code") == sku), None)
                if product:
                    product['reason_for_recommendation'] = rec.get('reason_for_recommendation', 'Una excelente opción.')
                    enriched_candidates.append(product)

            if enriched_candidates:
                logger.info(f"LLM Sales-Associate successfully recommended {len(enriched_candidates)} products.")
                return enriched_candidates[:top_n]

        except Exception as exc:
            logger.warning(f"Smarter LLM ranking failed: {exc}. Falling back to simpler method.")
            # Fall through to the simpler ranking logic below

    # --- FALLBACK LOGIC ---
    is_price_sorted = any(
        s in user_intent.lower() for s in ["cheapest", "most affordable", "más barato", "mas economico", "price_asc", "price_desc"]
    )
    if is_price_sorted:
        logger.info(f"Fallback: Results are already price-sorted. Returning top {top_n} candidates.")
        return candidates[:top_n]

    logger.info(f"Fallback: Applying low-mid-high price spread ranking.")
    if len(candidates) < 3:
        return sorted(candidates, key=lambda p: p.get("price") or 0.0)

    sorted_by_price = sorted(candidates, key=lambda p: p.get("price") or 0.0)
    cheapest = sorted_by_price[0]
    most_expensive = sorted_by_price[-1]
    mid_index = len(sorted_by_price) // 2
    middle = sorted_by_price[mid_index]

    ranked: List[Dict[str, Any]] = []
    seen_model_names = set()
    
    # Use the now-robust utility to get the base model name for deduplication
    get_base_name = getattr(product_utils, 'get_base_model_name', lambda name: name)

    for product in [cheapest, middle, most_expensive]:
        base_model = get_base_name(product.get("item_name", ""))
        if base_model not in seen_model_names:
            ranked.append(product)
            seen_model_names.add(base_model)

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
    
    return rank_products(query, results, top_n=top_n)