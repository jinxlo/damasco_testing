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
