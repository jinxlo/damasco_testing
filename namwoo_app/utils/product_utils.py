# namwoo_app/utils/product_utils.py
import re
import logging
from typing import Optional, Any, List, Dict
from collections import defaultdict
import locale
from decimal import Decimal
from openai import OpenAI
try:
    from ..config import Config
except Exception:  # pragma: no cover - allow standalone usage without package
    class Config:
        pass

# Set locale for currency formatting if not already set
try:
    locale.setlocale(locale.LC_ALL, 'es_VE.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
        except locale.Error:
            pass


# Initialize a local client for this module if needed
_llm_formatter_client: Optional[OpenAI] = None
if getattr(Config, "OPENAI_API_KEY", None):
    try:
        _llm_formatter_client = OpenAI(api_key=Config.OPENAI_API_KEY, timeout=10.0)
    except Exception as exc:
        _llm_formatter_client = None

logger = logging.getLogger(__name__)

# This constant is now defined here so it's available to functions in this module
KNOWN_BRANDS = {'SAMSUNG', 'TECNO', 'XIAOMI', 'INFINIX', 'DAMASCO'}

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
    if any(kw in normalized for kw in LIST_KEYWORDS):
        return True
    if "modelos" in normalized and (
        "tienes" in normalized or "tienen" in normalized or "disponible" in normalized or "disponibles" in normalized
    ):
        return True
    return False

def user_is_asking_for_price(message: str) -> bool:
    """Return True if the user clearly wants the price of a product."""
    if not message:
        return False
    PRICE_KEYWORDS = [
        "precio de",
        "precio del",
        "cual es el precio",
        "cuanto cuesta",
        "cuánto cuesta",
        "dime el precio",
        "price of",
        "how much is",
        "how much does",
        "cost of",
        "cost for",
    ]
    import unicodedata
    normalized = unicodedata.normalize("NFKD", message).encode("ascii", "ignore").decode().lower()
    return any(kw in normalized for kw in PRICE_KEYWORDS)

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

KNOWN_COLORS = {
    'NEGRO', 'BLANCO', 'AZUL', 'VERDE', 'ROJO', 'GRIS', 'DORADO', 'PLATEADO',
    'PURPURA', 'MORADO', 'AMARILLO', 'NARANJA', 'PLATA', 'GRAFITO', 'ROSADO', 'PERLADO'
}
WORDS_TO_REMOVE = {'5G', 'OBSEQUIO', '+', 'CON'}

def get_base_model_name(item_name: str) -> str:
    """
    Robustly strips colors, specs, and junk words to find a base model name.
    This is more reliable than a single complex regex.
    """
    if not item_name:
        return "Producto Desconocido"

    words = item_name.upper().split()
    # Filter out colors, junk words, and pure specifiers like '256+8'
    base_words = [
        word for word in words if
        word not in KNOWN_COLORS and
        word not in WORDS_TO_REMOVE and
        not re.fullmatch(r'\d+\+\d+', word) and
        not re.fullmatch(r'\d+GB', word)
    ]

    base_name = ' '.join(base_words).strip()
    return base_name if base_name else item_name

def _extract_color_from_name(item_name: str) -> Optional[str]:
    """Extracts a color from the product name string by checking against a known set."""
    if not item_name:
        return None
    words = item_name.upper().split()
    for word in reversed(words):
        if word in KNOWN_COLORS:
            return word.capitalize()
    return None

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
        representative_product = min(variants, key=lambda p: p.get('price', float('inf')))
        
        all_colors = set()
        all_stores = set()
        for variant in variants:
            color = _extract_color_from_name(variant.get("item_name") or variant.get("itemName", ""))
            if color:
                all_colors.add(color)
            branch = variant.get("branch_name")
            if branch:
                all_stores.add(branch)
        
        final_product = dict(representative_product)
        final_product['base_model_name'] = base_model
        final_product['available_colors'] = sorted(list(all_colors))
        final_product['available_in_stores'] = sorted(list(all_stores))
        final_product['model'] = base_model
        final_product['colors'] = final_product['available_colors']
        final_product['store'] = ", ".join(final_product['available_in_stores']) if final_product['available_in_stores'] else None
        result.append(final_product)
        
    return result

def _get_best_available_specs_text(product: Dict) -> str:
    """Gets the best available raw text for specs, prioritizing structured data."""
    spec_str = product.get("especificacion") or product.get("specifitacion")
    if spec_str and isinstance(spec_str, str) and ":" in spec_str and len(spec_str.splitlines()) > 1:
        logger.debug(f"Using structured 'especificacion' field for product {product.get('item_name')}")
        return spec_str.strip()

    logger.debug(f"Falling back to description for product {product.get('item_name')}")
    return (
        product.get("llm_summarized_description")
        or product.get("description")
        or "Descripción no disponible."
    ).strip()

def _get_llm_formatted_specs(product: Dict) -> str:
    """Uses an LLM to generate a professional, single-line summary of the top 4 specs."""
    global _llm_formatter_client
    if not _llm_formatter_client:
        logger.warning("LLM formatter client not available. Falling back to raw specs.")
        return product.get("description", "Detalles adicionales no disponibles.")
    
    raw_spec_text = _get_best_available_specs_text(product)
    
    if "no disponible" in raw_spec_text.lower():
        return raw_spec_text

    system_prompt = (
        "Eres un redactor de marketing para una tienda de tecnología. Tu única tarea es tomar las especificaciones de un producto y reescribirlas como una sola frase atractiva y fácil de leer. "
        "Extrae las 4 características más importantes (como pantalla, procesador, cámara, almacenamiento, batería) y combínalas en una oración natural y fluida. "
        "Ejemplo: Si recibes 'Pantalla: 6.78\" FHD+, 120Hz\\nMemoria: 256GB\\nCámara: 108MP\\nBatería: 5000mAh', "
        "responde: 'Gran pantalla FHD+ de 6.78\" a 120Hz, 256GB de almacenamiento, cámara de 108MP y una potente batería de 5000mAh.' "
        "No añadas introducciones ni despedidas. Solo devuelve la frase."
    )
    
    try:
        completion = _llm_formatter_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_spec_text}
            ],
            temperature=0.2,
            max_tokens=100,
        )
        formatted_specs = completion.choices[0].message.content.strip()
        return formatted_specs if formatted_specs else "Características destacadas no disponibles."
    except Exception as e:
        logger.error(f"Error getting LLM-formatted specs for {product.get('item_name')}: {e}")
        return (product.get("especificacion") or product.get("specifitacion") or "Detalles no disponibles.").splitlines()[0]

def format_ai_recommendations(products: List[Dict]) -> str:
    """Formats the output from the AI Sales-Associate recommender with the new professional format."""
    if not products:
        return "Lo siento, no pude encontrar una recomendación adecuada en este momento. ¿Podrías intentar con otra búsqueda?"

    grouped_products = group_products_by_model(products)
    
    recommendation_reasons = {}
    for prod in products:
        base_name = get_base_model_name(prod.get("item_name", ""))
        if 'reason_for_recommendation' in prod:
            recommendation_reasons[base_name] = prod['reason_for_recommendation']

    response_parts = ["¡Claro! Basado en tu búsqueda, aquí tienes mis mejores recomendaciones para ti:"]

    for product in grouped_products:
        model_name = product.get("base_model_name")
        
        price_usd = product.get("price")
        price_bs = product.get("price_bolivar")
        
        price_usd_str = f"${price_usd:,.2f}" if isinstance(price_usd, (int, float, Decimal)) else "Precio no disponible"
        price_bs_str = f"Bs. {locale.format_string('%.2f', price_bs, grouping=True)}" if isinstance(price_bs, (int, float, Decimal)) else ""
        full_price_str = f"{price_usd_str} ({price_bs_str})" if price_bs_str else price_usd_str

        colors = product.get("available_colors", [])
        colors_str = ", ".join(colors) if colors else "No especificado"

        features_str = _get_llm_formatted_specs(product)
        
        response_parts.append(
            f"\n\n---\n\n"
            f"📱 ***{model_name.strip()}***\n"
            f"💵 *Precio:* {full_price_str}\n"
            f"🎨 *Colores disponibles:* {colors_str}\n"
            f"✨ *Características destacadas:* {features_str}"
        )
        
        reason = recommendation_reasons.get(model_name)
        if reason:
            response_parts.append(f"⭐ *Por qué te lo recomiendo:* {reason}")

    response_parts.append("\n\n¿Te gustaría saber más detalles de alguno de estos modelos? 😊")
    return "".join(response_parts)

def format_multiple_products_response(products: List[Dict], user_query: Optional[str] = None) -> str:
    """Formats a list of grouped products into a single, multi-card response string."""
    if not products:
        return ""
        
    grouped_products = group_products_by_model(products)
    product_cards = [format_product_response(p, user_query) for p in grouped_products[:3]]
    
    response = "¡Claro! Aquí tienes algunas opciones excelentes para ti:\n\n"
    response += "\n\n---\n\n".join(product_cards)
    response += "\n\n¿Quieres que verifiquemos uno de estos para ti o deseas ver más opciones? 😊"
    
    return response

def format_product_response(grouped_product: Dict, user_query: Optional[str] = None) -> str:
    """Formats a single grouped product into the desired 'Product Card' string."""
    model = grouped_product.get("base_model_name") or grouped_product.get("model", "Producto")
    
    price_usd = grouped_product.get("price")
    price_bs = grouped_product.get("price_bolivar") or grouped_product.get("priceBolivar")
    
    price_usd_str = f"${price_usd:,.2f}" if isinstance(price_usd, (int, float, Decimal)) else "Precio no disponible"
    if isinstance(price_bs, (int, float, Decimal)):
        price_bs_str = f"Bs. {price_bs:,.2f}"
    else:
        price_bs_str = ""
    full_price_str = f"{price_usd_str} ({price_bs_str})" if price_bs_str else price_usd_str

    colors = grouped_product.get("available_colors") or grouped_product.get("colors", [])
    colors_str = ", ".join(colors) if colors else "No especificado"

    description = _get_llm_formatted_specs(grouped_product)

    stores = grouped_product.get("available_in_stores") or ([] if grouped_product.get("store") is None else [grouped_product.get("store")])
    stores_str = f"Disponible en {', '.join(stores)}." if stores else ""
    
    card_lines = [
        f"📱 *{model.strip()}*",
        f"💵 *Precio:* {full_price_str}",
        f"🎨 *Colores disponibles:* {colors_str}",
        f"✨ *Características destacadas:* {description}",
    ]
    if stores_str:
        card_lines.append("")
        card_lines.append(stores_str)

    return "\n".join(card_lines).strip()


def _get_key_specs(product: Dict) -> str:
    """Utility for tests: returns the first 200 chars of main spec text."""
    text = product.get("especificacion") or product.get("llm_summarized_description") or ""
    text = text.split("\n", 1)[0]
    return text[:200]

def format_model_list_with_colors(products: List[Dict]) -> str:
    """Formats a list of models with their available colors. The filtering is now done in the DB."""
    if not products:
        return "No encontré modelos que coincidan con tu búsqueda en este momento."

    grouped = group_products_by_model(products)
    if not grouped:
        return "No encontré modelos de esa marca que coincidan con tu búsqueda."

    lines = ["¡Claro! Aquí tienes los modelos disponibles:\n"]
    for product in grouped:
        model_name = product.get("base_model_name", "Producto").strip()
        colors = product.get("available_colors", [])
        color_str = ", ".join(colors) if colors else "No especificado"
        lines.append(f"🔹 {model_name}: {color_str}")

    return "\n".join(lines)


def format_brand_list(brands: list) -> str:
    """Formats a list of brand names into a human-friendly, formatted string."""
    if not brands:
        return ""

    lines = ["📱 Estas son las marcas que tenemos disponibles:\n"]
    lines.extend(f"🔹 {brand}" for brand in sorted(brands))
    lines.append("\n¿Te gustaría ver los modelos de alguna en específico?")
    return "\n".join(lines)