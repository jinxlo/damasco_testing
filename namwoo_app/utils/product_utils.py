# namwoo_app/utils/product_utils.py
import re
from typing import Optional, Any, List, Dict, Tuple
from collections import defaultdict
import locale
from decimal import Decimal

# Set locale for currency formatting if not already set
try:
    locale.setlocale(locale.LC_ALL, 'es_VE.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
    except locale.Error:
        locale.setlocale(locale.LC_ALL, '') # Fallback to system default

# --- NEW: Moved from openai_service.py to break circular import ---
def user_is_asking_for_cheapest(message: str) -> bool:
    """Return True if the user clearly wants the cheapest option."""
    if not message:
        return False
    CHEAP_KEYWORDS = [
        "más barato", "mas barato", "más económico", "mas economico", "menor precio",
        "el más barato", "el mas barato", "el más económico", "el mas economico",
        "el más barato que tengas", "no tengo presupuesto", "lo más económico",
        "lo mas economico", "dame el más barato", "dame el mas barato",
        "el de menor precio", "el menos costoso", "el más bajo", "el mas bajo",
        "más bajo posible", "mas bajo posible",
    ]
    import unicodedata
    normalized = unicodedata.normalize("NFKD", message).encode("ascii", "ignore").decode().lower()
    return any(kw in normalized for kw in CHEAP_KEYWORDS)
# --- END NEW ---

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


# --- Robust helper functions for grouping and formatting ---

BASE_MODEL_CLEANUP_PAT = re.compile(
    r'\s*\+?\s*(\d+GB|\d+G|\d+gb|\d+g|\d{3,4}gb|\d{3,4}g)\s*'
    r'|\s*\+?\s*(\d+TB|\d+tb)\s*'
    r'|\s*\(?[Cc]on [Oo]bsequio\)?\s*'
    r'|\s*\+ [Oo]bsequio\s*'
    r'|(\s*5G\s*)'
    r'|\s+(NEGRO|BLANCO|AZUL|VERDE|ROJO|GRIS|DORADO|PLATEADO|PURPURA|MORADO|AMARILLO|NARANJA|PLATA|GRAFITO|ROSADO)\s*$',
    flags=re.IGNORECASE
)

COLOR_EXTRACT_PAT = re.compile(
    r'\b(NEGRO|BLANCO|AZUL|VERDE|ROJO|GRIS|DORADO|PLATEADO|PURPURA|MORADO|AMARILLO|NARANJA|PLATA|GRAFITO|ROSADO)$',
    flags=re.IGNORECASE
)


def _get_base_model_name(item_name: str) -> str:
    """Strips specs and colors to find a base model name for grouping."""
    if not item_name:
        return "Producto Desconocido"
    base_name = item_name
    for _ in range(5):
        new_name = BASE_MODEL_CLEANUP_PAT.sub('', base_name).strip()
        if new_name == base_name:
            break
        base_name = new_name
    return base_name if base_name else item_name


def _extract_color_from_name(item_name: str) -> Optional[str]:
    """Extracts a color from the end of a product name string."""
    if not item_name:
        return None
    match = COLOR_EXTRACT_PAT.search(item_name.upper())
    return match.group(1).capitalize() if match else None


def extract_color_from_name(item_name: str) -> Tuple[Optional[str], str]:
    """Return detected color and base model name for compatibility."""
    base = _get_base_model_name(item_name)
    return _extract_color_from_name(item_name), base


def group_products_by_model(products: List[Dict]) -> List[Dict]:
    """Groups product variants by a robustly identified base model name."""
    if not products:
        return []
        
    grouped = defaultdict(lambda: {
        'representative_product': None,
        'colors': set(),
        'available_in_stores': set()
    })

    for product in products:
        item_name = product.get("item_name") or product.get("itemName", "")
        base_model_name = _get_base_model_name(item_name)

        if not grouped[base_model_name]['representative_product']:
            grouped[base_model_name]['representative_product'] = product

        color = _extract_color_from_name(item_name)
        if color:
            grouped[base_model_name]['colors'].add(color)
        
        branch = product.get("branch_name")
        if branch:
            grouped[base_model_name]['available_in_stores'].add(branch)

    result = []
    for base_model, data in grouped.items():
        final_product = dict(data['representative_product'])
        final_product['base_model_name'] = base_model
        final_product['model'] = base_model
        colors_sorted = sorted(list(data['colors']))
        final_product['available_colors'] = colors_sorted
        final_product['colors'] = colors_sorted
        final_product['available_in_stores'] = sorted(list(data['available_in_stores']))
        result.append(final_product)
        
    return result


def _get_key_specs(product: Dict, user_query: Optional[str] = None) -> str:
    """
    Gets a concise, user-friendly spec list. Prioritizes the 'especificacion'
    field if it exists, otherwise falls back to the LLM summary.
    """
    # Prioritize the structured `especificacion` field, accounting for the typo.
    spec_str = product.get("especificacion") or product.get("specifitacion") or product.get("description")
    
    if spec_str:
        # Clean up the spec string: replace newlines with a standard delimiter.
        specs = re.sub(r'[\r\n]+', ', ', spec_str).strip()
        # Normalize whitespace
        return ' '.join(specs.split())

    # Fallback to LLM summary if no structured specs
    summary = product.get("llm_summarized_description", "Descripción no disponible.").strip()
    return summary


def format_product_response(grouped_product: Dict, user_query: Optional[str] = None) -> str:
    """Formats a single grouped product into the desired 'Product Card' string."""
    model = grouped_product.get("base_model_name") or grouped_product.get("model") or grouped_product.get("item_name", "Producto")
    price_usd = grouped_product.get("price")
    price_bs = grouped_product.get("price_bolivar") or grouped_product.get("priceBolivar")
    
    price_usd_str = f"${price_usd:,.2f}" if isinstance(price_usd, (int, float, Decimal)) else "Precio no disponible"
    # Format Bolivares with space as thousand separator and comma as decimal
    if isinstance(price_bs, (int, float, Decimal)):
        price_bs_str = f"Bs. {price_bs:,.2f}"
    else:
        price_bs_str = ""

    colors = grouped_product.get("available_colors") or grouped_product.get("colors", [])
    colors_str = ", ".join(colors) if colors else "No especificado"

    description = _get_key_specs(grouped_product, user_query)

    stores = grouped_product.get("available_in_stores")
    if not stores:
        single_store = grouped_product.get("store")
        stores = [single_store] if single_store else []
    stores_str = f"Disponible en {', '.join(stores)}." if stores else ""
    
    card_lines = [
        f"📱 *{model.strip()}*",
        f"💵 *Precio:* {price_usd_str} ({price_bs_str})",
        f"🎨 *Colores disponibles:* {colors_str}",
        f"✨ *Características destacadas:* {description}",
    ]
    if stores_str:
        card_lines.append("")
        card_lines.append(stores_str)

    return "\n".join(card_lines).strip()


def format_multiple_products_response(products: List[Dict], user_query: Optional[str] = None) -> str:
    """Formats a list of grouped products into a single, multi-card response string."""
    if not products:
        return ""
        
    product_cards = [format_product_response(p, user_query) for p in products[:3]]
    
    response = "¡Claro! Aquí tienes algunas opciones excelentes para ti:\n\n"
    response += "\n\n---\n\n".join(product_cards)
    response += "\n\n¿Quieres que verifiquemos uno de estos para ti o deseas ver más opciones? 😊"
    
    return response


def get_available_brands(products: List[Dict]) -> List[str]:
    """Return sorted list of unique brand names from product dicts."""
    return sorted({p.get("brand") for p in products if p.get("brand")})


def format_brand_list(brands: list) -> str:
    """Formats a list of brand names into a human-friendly, formatted string."""
    if not brands:
        return ""

    lines = ["📱 Estas son las marcas que tenemos disponibles:\n"]
    lines.extend(f"🔹 {brand}" for brand in sorted(brands))
    lines.append("\n¿Te gustaría ver los modelos de alguna en específico?")
    return "\n".join(lines)