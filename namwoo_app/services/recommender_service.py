# namwoo_app/services/recommender_service.py
"""Simple top-N product ranker."""
from typing import List, Dict, Any

WEIGHTS = {
    "price_fit": 0.35,
    "similarity": 0.35,
    "pref_match": 0.20,
    "diversity": 0.10,
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in text.split()] if text else []


def _pref_match(item: Dict[str, Any], tokens: List[str]) -> float:
    haystack = " ".join(
        [
            str(item.get("item_name", "")),
            str(item.get("brand", "")),
            str(item.get("category", "")),
            str(item.get("llm_formatted_description", "")),
        ]
    ).lower()
    if not tokens:
        return 0.0
    hits = sum(tok in haystack for tok in tokens)
    return hits / len(tokens)


def _price_fit(price: float, budget_min, budget_max) -> float:
    if price is None:
        return 0.0
    try:
        p = float(price)
    except Exception:
        return 0.0
    if budget_min is None and budget_max is None:
        return 1.0
    if budget_min is None:
        span = budget_max or p
        return max(0.0, min(1.0, 1 - abs(p - span) / span))
    if budget_max is None:
        span = budget_min or p
        return max(0.0, min(1.0, 1 - abs(p - span) / span))
    span = max(1.0, float(budget_max) - float(budget_min))
    midpoint = (float(budget_max) + float(budget_min)) / 2.0
    return max(0.0, min(1.0, 1 - abs(p - midpoint) / span))


def rank_products(intent: Dict[str, Any], items: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    tokens = _tokenize(intent.get("raw_query", ""))
    budget_min = intent.get("budget_min")
    budget_max = intent.get("budget_max")

    candidates = [i for i in items if not i.get("is_accessory")]
    scored = []
    for item in candidates:
        price_fit = _price_fit(item.get("price"), budget_min, budget_max)
        pref = _pref_match(item, tokens)
        sim = float(item.get("similarity", 0))
        score = (
            WEIGHTS["price_fit"] * price_fit
            + WEIGHTS["similarity"] * sim
            + WEIGHTS["pref_match"] * pref
            + WEIGHTS["diversity"] * 1.0
        )
        scored.append({"item": item, "score": score})

    picked: List[Dict[str, Any]] = []
    seen_brands = set()
    while scored and len(picked) < top_n:
        scored.sort(key=lambda d: d["score"], reverse=True)
        best = scored.pop(0)
        item = best["item"]
        picked.append(item)
        seen_brands.add(item.get("brand"))
        for entry in scored:
            item2 = entry["item"]
            diversity = 1.0 if item2.get("brand") not in seen_brands else 0.5
            price_fit = _price_fit(item2.get("price"), budget_min, budget_max)
            pref = _pref_match(item2, tokens)
            sim = float(item2.get("similarity", 0))
            entry["score"] = (
                WEIGHTS["price_fit"] * price_fit
                + WEIGHTS["similarity"] * sim
                + WEIGHTS["pref_match"] * pref
                + WEIGHTS["diversity"] * diversity
            )
    return picked[:top_n]
