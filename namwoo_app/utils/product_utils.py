# namwoo_app/utils/product_utils.py
import re
from typing import Optional, Any


def _normalize_string_for_id_part(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def generate_product_location_id(item_code_raw: Any, whs_name_raw: Any) -> Optional[str]:
    """Generates the composite product ID consistently."""
    item_code = _normalize_string_for_id_part(item_code_raw)
    whs_name = _normalize_string_for_id_part(whs_name_raw)
    if not item_code or not whs_name:
        return None
    sanitized_whs_name = re.sub(r'[^a-zA-Z0-9_-]', '_', whs_name)
    product_id = f"{item_code}_{sanitized_whs_name}"
    if len(product_id) > 512:
        product_id = product_id[:512]
    return product_id


# --- New helper functions for color detection ---
_COLOR_PATTERNS = {
    "ROJO": ["ROJO", "RED"],
    "AZUL": ["AZUL", "BLUE"],
    "NEGRO": ["NEGRO", "BLACK"],
    "BLANCO": ["BLANCO", "WHITE"],
    "VERDE": ["VERDE", "GREEN"],
    "GRIS": ["GRIS", "GRAY", "GREY"],
}


def strip_color_suffix(item_code: str) -> str:
    """Return item_code without any recognised color suffix."""
    if not item_code:
        return ""
    code = item_code.upper()
    for canonical, variants in _COLOR_PATTERNS.items():
        for var in variants:
            if code.endswith(f"-{var}") or code.endswith(f"_{var}"):
                return code[: -len(var) - 1]
    return code


def extract_color_from_name(name: str) -> tuple[Optional[str], str]:
    """Return detected color and the base model name without the color."""
    if not name:
        return None, ""
    cleaned = name.strip()
    upper = cleaned.upper()
    for canonical, variants in _COLOR_PATTERNS.items():
        for var in variants:
            if upper.endswith(f" {var}") or upper.endswith(f"-{var}") or upper.endswith(f"_{var}"):
                base = re.sub(rf"[\s_-]*{re.escape(var)}$", "", cleaned, flags=re.IGNORECASE).strip()
                return canonical.capitalize(), base
    return None, cleaned


def group_products_by_model(products: list[dict]) -> list[dict]:
    """Group product variants by base model name and collect available colors."""
    grouped: dict[str, dict] = {}
    for prod in products:
        name = prod.get("itemName") or prod.get("item_name") or ""
        color, base = extract_color_from_name(name)
        key = base.upper()
        if key not in grouped:
            grouped[key] = {
                "model": base,
                "colors": [],
                "rep": prod,
            }
        if color and color not in grouped[key]["colors"]:
            grouped[key]["colors"].append(color)

    result: list[dict] = []
    for info in grouped.values():
        base_item = dict(info["rep"])  # shallow copy of first item
        base_item["model"] = info["model"]
        base_item["colors"] = info["colors"]
        result.append(base_item)
    return result


def get_available_brands(products: list[dict]) -> list[str]:
    """Return sorted list of unique brands from product dictionaries."""
    return sorted({p.get("brand") for p in products if p.get("brand")})


def format_product_response(grouped_product: dict) -> str:
    """Return a human friendly message for a grouped product entry."""
    model = grouped_product.get("model") or grouped_product.get("itemName", "")
    price_usd = grouped_product.get("price")
    price_bs = grouped_product.get("priceBolivar")
    colors = ", ".join(grouped_product.get("colors", []))
    description = grouped_product.get("description") or grouped_product.get(
        "llm_formatted_description", ""
    )
    store = grouped_product.get("store") or grouped_product.get("branchName") or "una de nuestras tiendas"

    response = f"""📱 *{model}*  
💵 *Precio:* ${price_usd:.2f} (Bs. {price_bs:,.2f})  
🎨 *Colores disponibles:* {colors}  
✨ *Características destacadas:* {description}

Disponible en {store}. ¿Quieres que lo reservemos para ti o deseas ver otras opciones? 😊
"""
    return response


def format_brand_list(brands: list) -> str:
    lines = ["📱 Estas son las marcas disponibles en nuestras tiendas:\n"]
    lines += [f"🔹 {brand}" for brand in sorted(brands)]
    lines.append("\n¿Quieres ver los modelos de alguna marca en particular?")
    return "\n".join(lines)
