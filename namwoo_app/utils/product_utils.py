# namwoo_app/utils/product_utils.py
import re
from typing import Optional, Any, List, Dict
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

def user_is_asking_for_list(message: str) -> bool:
    """Return True if the user is asking for a list of all items."""
    if not message:
        return False
    LIST_KEYWORDS = [
        "dame una lista", "que modelos tienes", "cuales modelos", "cuales tienes",
        "todos los que tienes", "todos los modelos", "lista de", "listado de"
    ]
    import unicodedata
    normalized = unicodedata.normalize("NFKD", message).encode("ascii", "ignore").decode().lower()
    return any(kw in normalized for kw in LIST_KEYWORDS)
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

# CORRECTED REGEX: More specific to avoid removing parts of the model name like 'C' from 'SPARK 30C'
BASE_MODEL_CLEANUP_PAT = re.compile(
    r'\s*\+\s*OBSEQUIO|\(CON OBSEQUIO\)|'  # Match "+ OBSEQUIO" or "(CON OBSEQUIO)"
    r'\s+5G(?=\s|$)|'  # Match " 5G " only if followed by space or end of line
    r'\s+(NEGRO|BLANCO|AZUL|VERDE|ROJO|GRIS|DORADO|PLATEADO|PURPURA|MORADO|AMARILLO|NARANJA|PLATA|GRAFITO|ROSADO|PERLADO)$',  # Match color at the end of the string
    flags=re.IGNORECASE
)

COLOR_EXTRACT_PAT = re.compile(
    r'\b(NEGRO|BLANCO|AZUL|VERDE|ROJO|GRIS|DORADO|PLATEADO|PURPURA|MORADO|AMARILLO|NARANJA|PLATA|GRAFITO|ROSADO|PERLADO)$',
    flags=re.IGNORECASE
)


def get_base_model_name(item_name: str) -> str:
    """Strips known suffixes and colors to find a base model name for grouping."""
    if not item_name:
        return "Producto Desconocido"
    # The new regex is more precise and a loop is less necessary, but kept for safety.
    base_name = item_name.strip()
    for _ in range(3): # Reduced loop count
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


def extract_color_from_name(item_name: str) -> tuple[Optional[str], str]:
    """Return the extracted color and base model name."""
    color = _extract_color_from_name(item_name)
    base = get_base_model_name(item_name)
    return color, base


def get_available_brands(products: List[Dict]) -> List[str]:
    """Return a list of unique brand names from the provided product dicts."""
    if not products:
        return []
    seen = {str(p.get("brand", "")).strip() for p in products if p.get("brand")}
    return sorted(seen)


def group_products_by_model(products: List[Dict]) -> List[Dict]:
    """Groups product variants by a robustly identified base model name."""
    if not products:
        return []
        
    grouped = defaultdict(list)
    for product in products:
        item_name = product.get("item_name") or product.get("itemName", "")
        base_model_name = get_base_model_name(item_name)
        grouped[base_model_name].append(product)

    result = []
    for base_model, variants in grouped.items():
        # Choose the variant with the lowest price as the representative
        representative_product = min(variants, key=lambda p: p.get('price', float('inf')))
        
        # Aggregate all unique colors and store locations from all variants of this model
        all_colors = set()
        all_stores = set()
        for variant in variants:
            color = _extract_color_from_name(variant.get("item_name", ""))
            if color:
                all_colors.add(color)
            branch = variant.get("branch_name")
            if branch:
                all_stores.add(branch)
        
        # Build the final grouped product dictionary using data from the representative,
        # but overriding with the aggregated color and store lists.
        final_product = dict(representative_product)
        final_product['base_model_name'] = base_model # Use the cleaned base name for display
        final_product['available_colors'] = sorted(list(all_colors))
        final_product['available_in_stores'] = sorted(list(all_stores))
        result.append(final_product)
        
    return result


def _get_key_specs(product: Dict, user_query: Optional[str] = None) -> str:
    """Return a short description of key specs for a product."""

    MAX_LEN = 200

    # Prioritize the structured `especificacion` field (and common typo)
    spec_str = product.get("especificacion") or product.get("specifitacion")
    if spec_str:
        first_line = spec_str.strip().splitlines()[0]
        clean = " ".join(first_line.split())
        return clean[:MAX_LEN].rstrip()

    # Fallback to the LLM summary
    summary = (
        product.get("llm_summarized_description")
        or product.get("description")
        or "Descripción no disponible."
    ).strip()
    if not summary:
        return ""
    first_sentence = summary.split(".")[0].strip()
    if first_sentence:
        first_sentence += "." if summary.startswith(first_sentence) and summary[len(first_sentence):].lstrip().startswith(".") else ""
    clean_sum = " ".join(first_sentence.split())
    return clean_sum[:MAX_LEN].rstrip()


def format_product_response(grouped_product: Dict, user_query: Optional[str] = None) -> str:
    """Formats a single grouped product into the desired 'Product Card' string."""
    # Use the cleaned base_model_name for the title to avoid showing a color in it.
    model = grouped_product.get("base_model_name", "Producto")
    
    price_usd = grouped_product.get("price")
    price_bs = grouped_product.get("price_bolivar") or grouped_product.get("priceBolivar")
    
    price_usd_str = f"${price_usd:,.2f}" if isinstance(price_usd, (int, float, Decimal)) else "Precio no disponible"
    # Format Bolivares with space as thousand separator and comma as decimal
    price_bs_str = f"Bs. {price_bs:,.2f}" if isinstance(price_bs, (int, float, Decimal)) else ""

    # Use the aggregated list of all available colors.
    colors = grouped_product.get("available_colors", [])
    colors_str = ", ".join(colors) if colors else "No especificado"

    description = _get_key_specs(grouped_product, user_query)

    # Use the aggregated list of all available stores.
    stores = grouped_product.get("available_in_stores", [])
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


def format_product_list_response(products: List[Dict]) -> str:
    """Formats a list of grouped products into a simple bulleted list."""
    if not products:
        return "No encontré modelos que coincidan con tu búsqueda en este momento."

    # Sort by price ascending for the list view
    sorted_products = sorted(products, key=lambda p: p.get("price") or float('inf'))
    
    lines = ["¡Claro! Aquí tienes una lista de los modelos que encontré:\n"]
    seen_models = set()
    for product in sorted_products:
        # Use the base_model_name for de-duplication to show each unique model only once.
        model_name = product.get("base_model_name", "Producto Desconocido").strip()
        if model_name in seen_models:
            continue
        seen_models.add(model_name)
        
        price = product.get("price")
        price_str = f"${price:,.2f}" if isinstance(price, (int, float, Decimal)) else "Precio no disponible"
        lines.append(f"🔹 {model_name} - {price_str}")
    
    # Check if any models were actually added to the list after de-duplication
    if len(seen_models) == 0:
        return "No encontré modelos que coincidan con tu búsqueda en este momento."

    lines.append("\n¿Te gustaría ver más detalles de alguno de estos modelos?")
    return "\n".join(lines)


def format_brand_list(brands: list) -> str:
    """Formats a list of brand names into a human-friendly, formatted string."""
    if not brands:
        return ""

    lines = ["📱 Estas son las marcas que tenemos disponibles:\n"]
    lines.extend(f"🔹 {brand}" for brand in sorted(brands))
    lines.append("\n¿Te gustaría ver los modelos de alguna en específico?")
    return "\n".join(lines)