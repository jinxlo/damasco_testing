# NAMWOO/services/openai_service.py
# -*- coding: utf-8 -*-
import logging
import json
import time # Keep time if used by retry logic within embedding_utils
import os # For constructing file paths
from typing import List, Dict, Optional, Tuple, Union, Any
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, BadRequestError
from flask import current_app # For accessing app config like OPENAI_EMBEDDING_MODEL

# Import local services and utils
from . import product_service
from . import support_board_service
try:
    from .recommender_service import rank_products
except Exception:  # Fallback for isolated test imports
    def rank_products(intent, items, top_n=3):
        return items[:top_n]
from ..config import Config # For SYSTEM_PROMPT, MAX_HISTORY_MESSAGES etc.
from ..utils import embedding_utils
from ..utils import conversation_location
import re


logger = logging.getLogger(__name__)

# Path to the store locations JSON file
# Assuming this script (openai_service.py) is in namwoo_app/services/
# and store_locations.json is in namwoo_app/data/
_STORE_LOCATIONS_FILE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), # Moves up one level to namwoo_app/
    'data', 
    'store_locations.json'
)
_cached_store_data: Optional[List[Dict[str, str]]] = None

# ---------------------------------------------------------------------------
# Helper to redact store details from tool outputs
# ---------------------------------------------------------------------------
_REDACT_PATTERN = re.compile(
    r"(branchName\s*:?\s*\"[^\"]*\"|address\s*:?\s*\"[^\"]*\"|whsName\s*:?\s*\"[^\"]*\"|Almac[eé]n[^,\n]+|Direcci[oó]n\:[^\n]+)",
    re.IGNORECASE,
)

# Patterns to detect price-sensitive or generic queries
_CHEAP_QUERY_PAT = re.compile(
    r"\b(?:mas\s+barato|m[aá]s\s+barato|barat(?:o|a)s?|econ[oó]mic(?:o|a)s?|cheapest)\b",
    re.IGNORECASE,
)
_GENERIC_PRODUCT_PAT = re.compile(
    r"^(?:celular(?:es)?|telefono|phone|tablet|laptop|televisor|tv|nevera|refrigerador|lavadora|secadora|microondas|freidora)$",
    re.IGNORECASE,
)

def user_is_asking_for_cheapest(message: str) -> bool:
    """Return True if the user clearly wants the cheapest option."""
    if not message:
        return False
    CHEAP_KEYWORDS = [
        "más barato",
        "mas barato",
        "más económico",
        "mas economico",
        "menor precio",
        "el más barato",
        "el mas barato",
        "el más económico",
        "el mas economico",
        "el más barato que tengas",
        "no tengo presupuesto",
        "lo más económico",
        "lo mas economico",
        "dame el más barato",
        "dame el mas barato",
        "el de menor precio",
        "el menos costoso",
        "el más bajo",
        "el mas bajo",
        "más bajo posible",
        "mas bajo posible",
    ]
    import unicodedata
    normalized = unicodedata.normalize("NFKD", message).encode("ascii", "ignore").decode().lower()
    return any(kw in normalized for kw in CHEAP_KEYWORDS)

def _redact_store_details(text: str) -> str:
    if not text:
        return text
    return _REDACT_PATTERN.sub("<REDACTED>", text)

# ---------------------------------------------------------------------------
# Initialise OpenAI client for Chat Completions
# This client instance is primarily for chat. Embeddings can use a fresh call
# or this client if preferred, but embedding_utils handles its own client init.
_chat_client: Optional[OpenAI] = None
try:
    openai_api_key = Config.OPENAI_API_KEY
    if openai_api_key:
        # Use configured timeout if available, otherwise default to 60.0
        timeout_seconds = getattr(Config, 'OPENAI_REQUEST_TIMEOUT', 60.0)
        _chat_client = OpenAI(api_key=openai_api_key, timeout=timeout_seconds)
        logger.info(f"OpenAI client initialized for Chat Completions service with timeout: {timeout_seconds}s.")
    else:
        _chat_client = None
        logger.error(
            "OpenAI API key not configured during initial load. "
            "Chat functionality will fail."
        )
except Exception as e: 
    logger.exception(f"Failed to initialize OpenAI client for chat during initial load: {e}")
    _chat_client = None

# ---------------------------------------------------------------------------
# Constants
MAX_HISTORY_MESSAGES = Config.MAX_HISTORY_MESSAGES 
TOOL_CALL_RETRY_LIMIT = 2 
DEFAULT_OPENAI_MODEL = getattr(Config, "OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = getattr(Config, "OPENAI_MAX_TOKENS", 1024)
DEFAULT_OPENAI_TEMPERATURE = getattr(Config, "OPENAI_TEMPERATURE", 0.7)

# ---------------------------------------------------------------------------
# Embedding Generation Function
# ---------------------------------------------------------------------------
def generate_product_embedding(text_to_embed: str) -> Optional[List[float]]:
    """
    Generates an embedding for the given product text using the configured
    OpenAI embedding model via embedding_utils.
    """
    if not text_to_embed or not isinstance(text_to_embed, str):
        logger.warning("openai_service.generate_product_embedding: No valid text provided.")
        return None
    embedding_model_name = Config.OPENAI_EMBEDDING_MODEL
    if not embedding_model_name:
        logger.error("openai_service.generate_product_embedding: OPENAI_EMBEDDING_MODEL not configured in Config.")
        return None
    logger.debug(f"Requesting embedding for text: '{text_to_embed[:100]}...' using model: {embedding_model_name}")
    embedding_vector = embedding_utils.get_embedding(
        text=text_to_embed,
        model=embedding_model_name
    )
    if embedding_vector is None:
        logger.error(f"openai_service.generate_product_embedding: Failed to get embedding from embedding_utils for text: '{text_to_embed[:100]}...'")
        return None
    logger.info(f"Successfully generated embedding for text (first 100 chars): '{text_to_embed[:100]}...'")
    return embedding_vector

# ---------------------------------------------------------------------------
# FUNCTION FOR PRODUCT DESCRIPTION SUMMARIZATION (using OpenAI)
# ---------------------------------------------------------------------------
def get_openai_product_summary(
    plain_text_description: str,
    item_name: Optional[str] = None
) -> Optional[str]:
    """
    Generates a product summary using OpenAI's chat completion.
    """
    global _chat_client
    if not _chat_client:
        logger.error("OpenAI client for chat not initialized. Cannot summarize description with OpenAI.")
        return None
    if not plain_text_description or not plain_text_description.strip():
        logger.debug("OpenAI summarizer: No plain text description provided to summarize.")
        return None
    prompt_context_parts = []
    if item_name:
        prompt_context_parts.append(f"Nombre del Producto: {item_name}")
    prompt_context_parts.append(f"Descripción Original (texto plano):\n{plain_text_description}")
    prompt_context = "\n".join(prompt_context_parts)
    system_prompt = (
        "Eres un redactor experto en comercio electrónico. Resume la siguiente descripción de producto. "
        "El resumen debe ser conciso (objetivo: 50-75 palabras, 2-3 frases clave), resaltar los principales beneficios y características, y ser factual. "
        "Evita la jerga de marketing, la repetición y frases como 'este producto es'. "
        "La salida debe ser texto plano adecuado para una base de datos de productos y un asistente de IA. "
        "No incluyas etiquetas HTML."
    )
    max_input_chars_for_summary = 3000
    if len(prompt_context) > max_input_chars_for_summary:
        cutoff_point = prompt_context.rfind('.', 0, max_input_chars_for_summary)
        if cutoff_point == -1: cutoff_point = max_input_chars_for_summary
        prompt_context = prompt_context[:cutoff_point] + " [DESCRIPCIÓN TRUNCADA]"
        logger.warning(f"OpenAI summarizer: Description for '{item_name or 'Unknown'}' was truncated for prompt construction.")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Por favor, resume la siguiente información del producto:\n\n{prompt_context}"}
    ]
    try:
        summarization_model = getattr(Config, "OPENAI_SUMMARY_MODEL", DEFAULT_OPENAI_MODEL)
        logger.debug(f"Requesting summary from OpenAI model '{summarization_model}' for item '{item_name or 'Unknown'}'")
        completion = _chat_client.chat.completions.create(
            model=summarization_model,
            messages=messages,
            temperature=0.2,
            max_tokens=150,
            n=1,
            stop=None,
        )
        summary = completion.choices[0].message.content.strip() if completion.choices and completion.choices[0].message.content else None
        if summary:
            logger.info(f"OpenAI summary generated for '{item_name or 'Unknown'}'. Preview: '{summary[:100]}...'")
        else:
            logger.warning(f"OpenAI returned an empty or null summary for '{item_name or 'Unknown'}'. Original text length: {len(plain_text_description)}")
        return summary
    except APIError as e:
        logger.error(f"OpenAI APIError during description summarization for '{item_name or 'Unknown'}': {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error calling OpenAI for description summarization for '{item_name or 'Unknown'}': {e}", exc_info=True)
        return None


def extract_customer_info_via_llm(message_text: str) -> Optional[Dict[str, Any]]:
    """Extract structured customer info from a plain text message using OpenAI."""
    global _chat_client
    if not _chat_client:
        logger.error("OpenAI client for chat not initialized. Cannot extract customer info.")
        return None

    system_prompt = (
        "Extrae la siguiente información del mensaje del cliente. "
        "Devuelve solo JSON válido con las claves: full_name, cedula, telefono, "
        "correo, direccion, productos y total. Si falta algún campo, usa null. "
        "No incluyas explicaciones ni comentarios."
    )
    user_prompt = f"Mensaje del cliente:\n\"\"\"{message_text}\"\"\""

    try:
        response = _chat_client.chat.completions.create(
            model=current_app.config.get("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_MODEL),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=256,
        )
        content = response.choices[0].message.content if response.choices else None
        if not content:
            logger.error("OpenAI returned empty content when extracting customer info.")
            return None
        return json.loads(content)
    except json.JSONDecodeError as jde:
        logger.error(f"JSON decoding error extracting customer info via OpenAI: {jde}")
        return None
    except Exception as e:
        logger.exception(f"Error extracting customer info via OpenAI: {e}")
        return None

# ===========================================================================
# ========== LLM TOOL IMPLEMENTATION FUNCTIONS  ==========
# ===========================================================================

def _load_store_data() -> List[Dict[str, str]]:
    """Loads store data from the JSON file, caching it after the first load."""
    global _cached_store_data
    if _cached_store_data is not None:
        return _cached_store_data
    
    try:
        if not os.path.exists(_STORE_LOCATIONS_FILE_PATH):
            logger.error(f"Store locations file not found at: {_STORE_LOCATIONS_FILE_PATH}")
            _cached_store_data = []
            return []
            
        with open(_STORE_LOCATIONS_FILE_PATH, 'r', encoding='utf-8') as f:
            _cached_store_data = json.load(f)
        logger.info(f"Successfully loaded store data from {_STORE_LOCATIONS_FILE_PATH}")
        return _cached_store_data
    except FileNotFoundError:
        logger.error(f"Store locations file not found: {_STORE_LOCATIONS_FILE_PATH}")
        _cached_store_data = []
        return []
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from store locations file {_STORE_LOCATIONS_FILE_PATH}: {e}")
        _cached_store_data = []
        return []
    except Exception as e:
        logger.exception(f"Unexpected error loading store data from {_STORE_LOCATIONS_FILE_PATH}: {e}")
        _cached_store_data = []
        return []

def _tool_get_store_info(city_name: Optional[str] = None) -> str:
    """
    Retrieves store information, optionally filtered by city.
    Returns a JSON string with store details or a list of available cities.
    """
    all_stores = _load_store_data()
    if not all_stores:
        return json.dumps({"status": "error", "message": "No se pudo cargar la información de las tiendas."}, ensure_ascii=False)

    if city_name:
        # Normalize city name for comparison (e.g., lowercase, remove accents if necessary)
        normalized_city_name = city_name.strip().lower()
        # A more robust normalization might involve unidecode for accents:
        # from unidecode import unidecode
        # normalized_city_name = unidecode(city_name.strip().lower())

        found_stores = [
            store for store in all_stores 
            if store.get("city", "").strip().lower() == normalized_city_name
            # or unidecode(store.get("city","").strip().lower()) == normalized_city_name # if using unidecode
        ]
        
        if found_stores:
            # Return whsName, branchName, address for each store in the city
            # The system prompt will instruct the LLM to use whsName for product searches
            return json.dumps({"status": "success", "city": city_name, "stores": found_stores}, ensure_ascii=False, indent=2)
        else:
            available_cities = sorted(list(set(store.get("city") for store in all_stores if store.get("city"))))
            return json.dumps({
                "status": "city_not_found", 
                "message": f"No se encontraron tiendas en '{city_name}'. Ciudades disponibles: {', '.join(available_cities)}.",
                "available_cities": available_cities
            }, ensure_ascii=False, indent=2)
    else:
        # If no city is specified, return a list of all unique city names where stores are present.
        available_cities = sorted(list(set(store.get("city") for store in all_stores if store.get("city"))))
        if available_cities:
            return json.dumps({"status": "success", "available_cities": available_cities, "message": "Por favor, especifica una ciudad para obtener detalles de las tiendas."}, ensure_ascii=False, indent=2)
        else:
            return json.dumps({"status": "no_cities_found", "message": "No hay ciudades con tiendas configuradas en el sistema."}, ensure_ascii=False)


def _tool_get_color_variants(item_code: str) -> str:
    """Return color variants for a product SKU."""
    if not item_code:
        return json.dumps({"status": "error", "message": "El parámetro item_code es requerido."}, ensure_ascii=False)
    variants = product_service.get_color_variants_for_sku(item_code)
    if variants is None:
        return json.dumps({"status": "error", "message": "No se pudo buscar variantes."}, ensure_ascii=False)
    return json.dumps({"status": "success" if variants else "not_found", "variants": variants}, ensure_ascii=False, indent=2)


def sanitize_tool_args(args: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
    """Sanitize tool arguments from the LLM before execution."""
    if not str(args.get("conversation_id", "")).isdigit():
        logger.info(
            "Replacing invalid conversation_id '%s' with '%s'", args.get("conversation_id"), conversation_id
        )
        args["conversation_id"] = conversation_id
    return args


def _sanitize_template_variables(vars: list[str]) -> list[str]:
    """Ensure the address field uses a known store/branch name."""
    # El campo "dirección" en la plantilla de WhatsApp SIEMPRE debe ser la
    # sucursal donde está disponible el producto. Nunca la dirección del cliente.
    if not isinstance(vars, list) or len(vars) != 8:
        return vars
    direccion = vars[5]
    branch_names = [s.get("branchName") for s in _load_store_data() if s.get("branchName")]
    if direccion not in branch_names:
        logger.info("Overriding direccion '%s' with 'Sucursal por confirmar'", direccion)
        vars[5] = "Sucursal por confirmar"
    return vars



def _tool_send_whatsapp_order_summary_template(
    customer_platform_user_id: str,
    conversation_id: str,
    template_variables: list[str],
) -> Dict:
    """WARNING: This function signature must always match the tool schema
    exposed to the LLM for function calling. Any changes require updates in
    both places."""

    phone_number = customer_platform_user_id
    ok = support_board_service.send_whatsapp_template(
        to=phone_number,
        template_name="confirmacion_datos_cliente",
        template_languages="es_ES",
        parameters=template_variables,
        recipient_id=conversation_id,
    )
    return {"status": "success" if ok else "failed"}

# ===========================================================================

# ---------------------------------------------------------------------------
# Tool definitions for OpenAI
# ---------------------------------------------------------------------------
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_store_info",
            "description": "Obtiene información de las tiendas Damasco, como direcciones y nombres de almacén (whsName) para búsquedas de productos. Puede filtrar por ciudad o listar todas las ciudades con tiendas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "Opcional. El nombre de la ciudad para la cual obtener información de tiendas. Si se omite, se devuelve una lista de todas las ciudades con tiendas."
                    }
                },
                "required": [] 
            }
        }
    },
    { 
        "type": "function",
        "function": {
            "name": "search_local_products",
            "description": ( 
                "Busca en el catálogo de productos de la tienda Damasco usando una consulta en lenguaje natural. "
                "Ideal cuando el usuario pregunta por tipos de productos o características. "
                "Devuelve una lista de productos coincidentes con nombre, marca, precio (USD), precio en Bolívares (`priceBolivar`), y una descripción lista para el usuario (`llm_formatted_description`)."
            ),
            "parameters": { 
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": (
                            "Consulta del usuario describiendo el producto. "
                            "Ej: 'televisor inteligente de 55 pulgadas', 'neveras Samsung'."
                        ),
                    },
                    "filter_stock": {
                        "type": "boolean",
                        "description": (
                            "Opcional. Si es true (defecto), filtra solo productos con stock. "
                            "False si se quiere verificar si un producto existe en catálogo sin importar stock."
                        ),
                        "default": True,
                    },
                    "warehouse_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Opcional. Lista de nombres de almacén (`whsName`) para filtrar la búsqueda de productos. "
                            "Obtén estos nombres usando la herramienta `get_store_info` basada en la ciudad del usuario."
                        ),
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Opcional. El precio mínimo del producto en USD para filtrar la búsqueda."
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Opcional. El precio máximo del producto en USD para filtrar la búsqueda."
                    },
                },
                "required": ["query_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_product_details",
            "description": ( 
                "Obtiene información detallada y actualizada de un producto específico de Damasco, incluyendo precio (USD), precio en Bolívares (`priceBolivar`), y stock por sucursal. "
                "Usar cuando el usuario pregunta por un producto específico (por SKU/código) o después de `search_local_products` si quiere más detalles."
            ),
            "parameters": { 
                "type": "object",
                "properties": {
                    "product_identifier": {
                        "type": "string",
                        "description": "El código de item (SKU) del producto o el ID compuesto (itemCode_warehouseName).",
                    },
                    "identifier_type": {
                        "type": "string",
                        "enum": ["sku", "composite_id"], 
                        "description": "Especifica si 'product_identifier' es 'sku' (para todas las ubicaciones) o 'composite_id' (para una ubicación específica).",
                    },
                },
                "required": ["product_identifier", "identifier_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_color_variants",
            "description": "Obtiene los distintos códigos de item (SKU) que representan el mismo producto en otros colores, basándose en el SKU proporcionado.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item_code": {
                        "type": "string",
                        "description": "SKU base del producto para buscar variantes de color."
                    }
                },
                "required": ["item_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_order_summary_template",
            "description": "Envía la plantilla de resumen de pedido 'confirmacion_datos_cliente' por WhatsApp al cliente una vez que se hayan recolectado todos los datos necesarios (nombre, apellido, cédula, teléfono, correo, dirección, productos y total).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_platform_user_id": {
                        "type": "string",
                        "description": "ID del usuario en la plataforma de mensajería (generalmente el número de teléfono para WhatsApp, o el ID de usuario interno si es un cliente existente)."
                    },
                    "conversation_id": {
                        "type": "string",
                        "description": "ID de la conversación actual de Support Board."
                    },
                    "template_variables": {
                        "type": "array",
                        "description": "Lista de EXACTAMENTE 8 cadenas en el siguiente orden: 1. Nombre(s) del cliente, 2. Apellido(s) del cliente, 3. Cédula, 4. Teléfono de contacto, 5. Correo electrónico, 6. Dirección de envío/facturación, 7. Descripción de los productos (ej: 'Producto A x1, Producto B x2'), 8. Precio total (ej: '$125.50 USD').",
                        "items": {"type": "string"},
                        "minItems": 8,
                        "maxItems": 8
                    }
                },
                "required": ["customer_platform_user_id", "conversation_id", "template_variables"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Helper: format Support‑Board history for OpenAI
# ---------------------------------------------------------------------------
def _format_sb_history_for_openai(
    sb_messages: Optional[List[Dict[str, Any]]], 
) -> List[Dict[str, Any]]: 
    if not sb_messages:
        return []
    openai_messages: List[Dict[str, Any]] = []
    bot_user_id_str = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None
    if not bot_user_id_str:
        logger.error("Cannot format SB history: SUPPORT_BOARD_DM_BOT_USER_ID is not configured.")
        return []
    for msg in sb_messages:
        sender_id = msg.get("user_id")
        text_content = msg.get("message", "").strip()
        attachments = msg.get("attachments")
        image_urls: List[str] = []
        if attachments and isinstance(attachments, list):
            for att in attachments:
                if (isinstance(att, dict) and att.get("url") and 
                    (att.get("type", "").startswith("image") or 
                     any(att["url"].lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]))):
                    url = att["url"]
                    if url.startswith(("http://", "https://")):
                        image_urls.append(url)
                    else:
                        logger.warning("Skipping possible non‑public URL for attachment %s", url)
        if not text_content and not image_urls:
            continue
        if sender_id is None:
            continue
        role = "assistant" if str(sender_id) == bot_user_id_str else "user"
        content_list_for_openai: List[Dict[str, Any]] = []
        if text_content:
            content_list_for_openai.append({"type": "text", "text": text_content})
        current_openai_model = getattr(Config, "OPENAI_CHAT_MODEL", DEFAULT_OPENAI_MODEL)
        vision_capable_models = ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini"] 
        if image_urls and current_openai_model in vision_capable_models:
            for img_url in image_urls:
                content_list_for_openai.append({"type": "image_url", "image_url": {"url": img_url}})
        elif image_urls: 
            logger.warning(f"Image URLs found but current model {current_openai_model} may not support vision. Images not explicitly sent.")
            if not text_content:
                 content_list_for_openai.append({"type": "text", "text": "[Usuario envió una imagen]"})
        if content_list_for_openai:
            if len(content_list_for_openai) == 1 and content_list_for_openai[0]["type"] == "text":
                openai_messages.append({"role": role, "content": content_list_for_openai[0]["text"]})
            else:
                openai_messages.append({"role": role, "content": content_list_for_openai})
    return openai_messages

# ---------------------------------------------------------------------------
# Helper: format search results
# ---------------------------------------------------------------------------
def _format_search_results_for_llm(results: Optional[List[Dict[str, Any]]]) -> str:
    """
    Formats product search results for the LLM.
    """
    if results is None:
        return json.dumps({"status": "error", "message": "Lo siento, ocurrió un error interno al buscar en el catálogo. Por favor, intenta de nuevo más tarde."}, ensure_ascii=False)
    if not results:
        return json.dumps({"status": "not_found", "message": "Lo siento, no pude encontrar productos que coincidan con esa descripción en nuestro catálogo actual."}, ensure_ascii=False)
    try:
        return json.dumps({"status": "success", "products": results}, indent=2, ensure_ascii=False)
    except (TypeError, ValueError) as err:
        logger.error(f"JSON serialisation error for search results: {err}", exc_info=True)
        return json.dumps({"status": "error", "message": "Lo siento, hubo un problema al formatear los resultados de la búsqueda."}, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Helper: live‑detail formatter
# ---------------------------------------------------------------------------
def _format_live_details_for_llm(details: Optional[Dict[str, Any]], identifier_type: str = "ID") -> str: 
    """
    Formats live product details for the LLM.
    """
    if details is None: 
        return json.dumps({"status": "error", "message": f"Lo siento, no pude recuperar los detalles en tiempo real para ese producto ({identifier_type})."}, ensure_ascii=False)
    if not details: 
         return json.dumps({"status": "not_found", "message": f"No se encontraron detalles para el producto con el {identifier_type} proporcionado."}, ensure_ascii=False)
    
    product_info = {
        "name": details.get("item_name", "Producto Desconocido"),
        "item_code": details.get("item_code", "N/A"),
        "id": details.get("id"), 
        "description": details.get("llm_formatted_description") or \
                       details.get("llm_summarized_description") or \
                       details.get("plain_text_description_derived", "Descripción no disponible."),
        "brand": details.get("brand", "N/A"),
        "category": details.get("category", "N/A"),
        "price": details.get("price"), 
        "priceBolivar": details.get("priceBolivar"), 
        "stock": details.get("stock"),
        "warehouse_name": details.get("warehouse_name"),
        "branch_name": details.get("branch_name")
    }

    return json.dumps({"status": "success", "product": product_info}, indent=2, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Main processing entry‑point
# ---------------------------------------------------------------------------
def process_new_message(
    sb_conversation_id: str,
    new_user_message: Optional[str], 
    conversation_source: Optional[str],
    sender_user_id: str,
    customer_user_id: str,
    triggering_message_id: Optional[str],
) -> None:
    global _chat_client
    if not _chat_client:
        logger.error("OpenAI client for chat not initialized. Cannot process message.")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text="Disculpa, el servicio de IA no está disponible en este momento.",
            source=conversation_source, target_user_id=customer_user_id, conversation_details=None, triggering_message_id=triggering_message_id,
        )
        return

    logger.info(
        "Processing message for SB Conv %s (trigger_user=%s, customer=%s, source=%s, trig_msg_id=%s)",
        sb_conversation_id, sender_user_id, customer_user_id, conversation_source, triggering_message_id,
    )

    if new_user_message: # This existing logic is good for setting city from user message
        detected_city = conversation_location.detect_city_from_text(new_user_message)
        if detected_city:
            # Check if this city is valid according to our store list.
            # This validation can also happen within conversation_location or here.
            # For now, assuming conversation_location.set_conversation_city handles it or
            # the get_store_info tool will indirectly validate.
            logger.info(f"Detected city '{detected_city}' from user message. Saving to conversation context.")
            conversation_location.set_conversation_city(sb_conversation_id, detected_city)
        else:
            logger.info(f"No city detected in new user message for SB Conv {sb_conversation_id}.")
            # The LLM will be prompted to ask for the city if not already set.

        # Additional heuristics before engaging the LLM
        lower_msg = new_user_message.lower()
        if _GENERIC_PRODUCT_PAT.fullmatch(lower_msg.strip()):
            support_board_service.send_reply_to_channel(
                conversation_id=sb_conversation_id,
                message_text="¿Buscas alguna marca o modelo en particular?",
                source=conversation_source,
                target_user_id=customer_user_id,
                conversation_details=None,
                triggering_message_id=triggering_message_id,
            )
            return


    conversation_data = support_board_service.get_sb_conversation_data(sb_conversation_id)
    if conversation_data is None or not conversation_data.get("messages"): 
        logger.error(f"Failed to fetch conversation data or no messages found for SB Conv {sb_conversation_id}. Aborting.")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text="Lo siento, tuve problemas para acceder al historial de esta conversación. ¿Podrías intentarlo de nuevo?",
            source=conversation_source, target_user_id=customer_user_id, conversation_details=None, triggering_message_id=triggering_message_id
        )
        return

    sb_history_list = conversation_data.get("messages", []) 
    try:
        openai_history = _format_sb_history_for_openai(sb_history_list)
    except Exception as err:
        logger.exception(f"Error formatting SB history for Conv {sb_conversation_id}: {err}")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text="Lo siento, tuve problemas al procesar el historial de la conversación.",
            source=conversation_source, target_user_id=customer_user_id, conversation_details=conversation_data, triggering_message_id=triggering_message_id
        )
        return

    if not openai_history: 
        if new_user_message and not sb_history_list: 
            logger.info(f"Formatted OpenAI history is empty for Conv {sb_conversation_id}, using new_user_message as initial prompt.")
            openai_history = [{"role": "user", "content": new_user_message}]
        else:
            logger.error(f"Formatted OpenAI history is empty for Conv {sb_conversation_id}, and no new message or history. Aborting.")
            support_board_service.send_reply_to_channel(
                conversation_id=sb_conversation_id, message_text="Lo siento, no pude procesar los mensajes anteriores adecuadamente.",
                source=conversation_source, target_user_id=customer_user_id, conversation_details=conversation_data, triggering_message_id=triggering_message_id
            )
            return

    system_prompt_content = Config.SYSTEM_PROMPT # This will be the updated prompt later
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt_content}] + openai_history

    max_hist_current = getattr(Config, "MAX_HISTORY_MESSAGES", MAX_HISTORY_MESSAGES)
    if len(messages) > (max_hist_current + 1 ): 
        messages = [messages[0]] + messages[-(max_hist_current):]

    final_assistant_response: Optional[str] = None
    try:
        tool_call_count = 0
        while tool_call_count <= TOOL_CALL_RETRY_LIMIT:
            abort_tool_calls = False
            openai_model = current_app.config.get("OPENAI_CHAT_MODEL", DEFAULT_OPENAI_MODEL)
            max_tokens = current_app.config.get("OPENAI_MAX_TOKENS", DEFAULT_MAX_TOKENS)
            temperature = current_app.config.get("OPENAI_TEMPERATURE", DEFAULT_OPENAI_TEMPERATURE)

            call_params: Dict[str, Any] = {
                "model": openai_model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            }
            
            if tool_call_count < TOOL_CALL_RETRY_LIMIT and not (messages[-1].get("role") == "tool"):
                 call_params["tools"] = tools_schema 
                 call_params["tool_choice"] = "auto"
            else: 
                call_params.pop("tools", None)
                call_params.pop("tool_choice", None)

            logger.debug(f"OpenAI API call attempt {tool_call_count + 1} for Conv {sb_conversation_id}. Tools offered: {'tools' in call_params}")
            response = _chat_client.chat.completions.create(**call_params)
            response_message = response.choices[0].message 

            if response.usage:
                 logger.info(f"OpenAI Tokens (Conv {sb_conversation_id}, Attempt {tool_call_count+1}): Prompt={response.usage.prompt_tokens}, Completion={response.usage.completion_tokens}, Total={response.usage.total_tokens}")

            messages.append(response_message.model_dump(exclude_none=True))
            tool_calls = response_message.tool_calls

            if not tool_calls:
                final_assistant_response = response_message.content
                logger.info(f"OpenAI response (no tool call this turn) for Conv {sb_conversation_id}: '{str(final_assistant_response)[:200]}...'")
                break 

            tool_outputs_for_llm: List[Dict[str, str]] = [] 
            for tc in tool_calls:
                fn_name = tc.function.name
                tool_call_id = tc.id 
                args_str = "" 
                try:
                    args_str = tc.function.arguments
                    args = json.loads(args_str)
                    args = sanitize_tool_args(args, sb_conversation_id)
                except json.JSONDecodeError as json_err:
                    logger.error(f"JSONDecodeError for tool {fn_name} args (Conv {sb_conversation_id}): {args_str}. Error: {json_err}")
                    args = {} 
                    output_txt = json.dumps({"status": "error", "message": f"Error: Argumentos para {fn_name} no son JSON válido: {args_str}"}, ensure_ascii=False)
                else: 
                    output_txt = json.dumps({"status":"error", "message":f"Error: Falló la ejecución de la herramienta {fn_name}."}, ensure_ascii=False) 
                
                logger.info(f"OpenAI requested tool call: {fn_name} with args: {args} for Conv {sb_conversation_id}")
                
                try:
                    if fn_name == "get_store_info":
                        city_arg = args.get("city_name")
                        output_txt = _tool_get_store_info(city_name=city_arg)
                    
                    elif fn_name == "search_local_products":
                        query = args.get("query_text")
                        filter_stock_flag = args.get("filter_stock", True)
                        warehouse_names_arg = args.get("warehouse_names")
                        min_price_arg = args.get("min_price") # Extract min_price
                        max_price_arg = args.get("max_price") # Extract max_price
                        cheapest_intent = bool(query and user_is_asking_for_cheapest(query))
                        generic_intent = bool(query and _GENERIC_PRODUCT_PAT.fullmatch(query.strip()))

                        if generic_intent and not cheapest_intent and min_price_arg is None and max_price_arg is None:
                            final_assistant_response = "¿Buscas alguna marca o modelo en particular?"
                            abort_tool_calls = True
                            break

                        # If warehouse_names are not provided by LLM (e.g., first product search after getting city)
                        # try to get them from conversation_location context.
                        if not warehouse_names_arg:
                            user_city = conversation_location.get_conversation_city(sb_conversation_id)
                            if user_city:
                                logger.info(f"LLM did not provide warehouse_names for search. Fetching for city '{user_city}' from context for Conv {sb_conversation_id}.")
                                # Call _tool_get_store_info internally to get whsNames for the city
                                store_info_json_str = _tool_get_store_info(city_name=user_city)
                                store_info_data = json.loads(store_info_json_str)
                                if store_info_data.get("status") == "success" and store_info_data.get("stores"):
                                    warehouse_names_arg = [s.get("whsName") for s in store_info_data["stores"] if s.get("whsName")]
                                    logger.info(f"Using whsNames for city '{user_city}': {warehouse_names_arg}")
                                else:
                                    logger.warning(f"Could not retrieve whsNames for city '{user_city}' to aid search. Proceeding without specific whsNames.")
                            else:
                                logger.info(f"No city in context and no warehouse_names provided by LLM for search in Conv {sb_conversation_id}. Product search will be national or un-filtered by warehouse.")
                        
                        if query:
                            search_res = product_service.search_local_products(
                                query_text=query,
                                filter_stock=filter_stock_flag,
                                warehouse_names=warehouse_names_arg,
                                min_price=min_price_arg,
                                max_price=max_price_arg,
                                sort_by_price_asc=cheapest_intent,
                                exclude_accessories=generic_intent
                            )
                            intent_data = {
                                "raw_query": query,
                                "budget_min": min_price_arg,
                                "budget_max": max_price_arg,
                            }
                            ranked = (
                                rank_products(intent_data, search_res)
                                if Config.RECOMMENDER_ENABLED
                                else search_res
                            )
                            output_txt = _format_search_results_for_llm(ranked)
                        else:
                            output_txt = json.dumps({"status": "error", "message": "Error: 'query_text' es requerido para search_local_products."}, ensure_ascii=False)
                    
                    elif fn_name == "get_live_product_details":
                        ident = args.get("product_identifier")
                        id_type = args.get("identifier_type")
                        if ident and id_type:
                            details_result = None 
                            if id_type == "sku":
                                details_result = product_service.get_live_product_details_by_sku(item_code_query=ident)
                                output_txt = _format_live_details_for_llm(details_result, identifier_type="SKU")
                            elif id_type == "composite_id":
                                details_result = product_service.get_live_product_details_by_id(composite_id=ident)
                                output_txt = _format_live_details_for_llm(details_result, identifier_type="ID Compuesto")
                            else:
                                output_txt = json.dumps({"status": "error", "message": f"Error: Tipo de identificador '{id_type}' no soportado. Use 'sku' o 'composite_id'."}, ensure_ascii=False)
                        else:
                            output_txt = json.dumps({"status": "error", "message": "Error: Faltan 'product_identifier' o 'identifier_type' para get_live_product_details."}, ensure_ascii=False)


                    elif fn_name == "get_color_variants":
                        item_code_arg = args.get("item_code")
                        if item_code_arg:
                            variants = product_service.get_color_variants_for_sku(item_code_arg)
                            if variants is None:
                                output_txt = json.dumps({"status": "error", "message": "No se pudo buscar variantes."}, ensure_ascii=False)
                            else:
                                status = "success" if variants else "not_found"
                                output_txt = json.dumps({"status": status, "variants": variants}, ensure_ascii=False, indent=2)
                        else:
                            output_txt = json.dumps({"status": "error", "message": "Error: Falta 'item_code' para get_color_variants."}, ensure_ascii=False)


                    elif fn_name == "send_whatsapp_order_summary_template":
                        cust_id_arg = args.get("customer_platform_user_id") or customer_user_id 
                        conv_id_arg = args.get("conversation_id") or sb_conversation_id
                        template_vars_arg = _sanitize_template_variables(
                            args.get("template_variables")
                        )
                        
                        if not cust_id_arg:
                            logger.error(f"Missing 'customer_platform_user_id' for {fn_name} in Conv {sb_conversation_id}, and fallback customer_user_id is also missing/invalid.")
                            output_txt = json.dumps({"status": "error", "message": "Error: Falta el ID del cliente para enviar la plantilla."}, ensure_ascii=False)
                        elif not conv_id_arg:
                             logger.error(f"Missing 'conversation_id' for {fn_name} in Conv {sb_conversation_id}, and fallback sb_conversation_id is also missing/invalid.")
                             output_txt = json.dumps({"status": "error", "message": "Error: Falta el ID de la conversación para enviar la plantilla."}, ensure_ascii=False)
                        elif not template_vars_arg:
                            logger.error(f"Missing 'template_variables' for {fn_name} in Conv {sb_conversation_id}.")
                            output_txt = json.dumps({"status": "error", "message": "Error: Faltan las variables para la plantilla."}, ensure_ascii=False)
                        else:
                            output_txt = _tool_send_whatsapp_order_summary_template(
                                customer_platform_user_id=str(cust_id_arg), 
                                conversation_id=str(conv_id_arg),       
                                template_variables=template_vars_arg,
                            )
                    else:
                        output_txt = json.dumps({"status": "error", "message": f"Error: Herramienta desconocida '{fn_name}'."}, ensure_ascii=False)
                        logger.warning(f"LLM called unknown tool: {fn_name} in Conv {sb_conversation_id}")

                except Exception as tool_exec_err: 
                    logger.exception(f"Tool execution error for {fn_name} (Conv {sb_conversation_id}): {tool_exec_err}")
                    output_txt = json.dumps({"status": "error", "message": f"Error interno al ejecutar la herramienta {fn_name}: {str(tool_exec_err)}"}, ensure_ascii=False)
                
                sanitized_output = _redact_store_details(output_txt)
                tool_outputs_for_llm.append({
                    "tool_call_id": tool_call_id,
                    "role": "tool",
                    "name": fn_name,
                    "content": sanitized_output,
                })

            if abort_tool_calls:
                break

            messages.extend(tool_outputs_for_llm)
            tool_call_count += 1
            
            if tool_call_count > TOOL_CALL_RETRY_LIMIT and not final_assistant_response:
                logger.warning(f"Tool call retry limit ({TOOL_CALL_RETRY_LIMIT}) strictly exceeded for Conv {sb_conversation_id}. Breaking loop.")
                break

            if abort_tool_calls:
                break

    except RateLimitError:
        logger.warning(f"OpenAI RateLimitError for Conv {sb_conversation_id}")
        final_assistant_response = ("Estoy experimentando un alto volumen de solicitudes. "
                                    "Por favor, espera un momento y vuelve a intentarlo.")
    except APITimeoutError:
        logger.warning(f"OpenAI APITimeoutError for Conv {sb_conversation_id}")
        final_assistant_response = "No pude obtener respuesta del servicio de IA (OpenAI) a tiempo. Por favor, intenta más tarde."
    except BadRequestError as bre:
        logger.error(f"OpenAI BadRequestError for Conv {sb_conversation_id}: {bre}", exc_info=True)
        if "image_url" in str(bre).lower() and "invalid" in str(bre).lower():
            final_assistant_response = ("Parece que una de las imágenes en nuestra conversación no pudo ser procesada. "
                                        "¿Podrías intentarlo sin la imagen o con una diferente?")
        else: 
            error_code = getattr(bre, 'code', None)
            if error_code == 'invalid_request_error' and 'tools' in str(bre).lower():
                 final_assistant_response = ("Lo siento, hubo un problema con la forma en que intenté usar mis herramientas internas. "
                                            "Intentaré de nuevo o puedes reformular tu solicitud.")
                 logger.error(f"BadRequestError possibly related to tool usage: {bre.message}")
            else:
                final_assistant_response = ("Lo siento, hubo un problema con el formato de nuestra conversación. "
                                            "Por favor, revisa si enviaste alguna imagen que no sea válida o reformula tu pregunta.")
    except APIError as apie:
        logger.error(f"OpenAI APIError for Conv {sb_conversation_id} (Status: {apie.status_code}): {apie}", exc_info=True)
        final_assistant_response = (f"Hubo un error ({apie.status_code}) con el servicio de IA. Por favor, inténtalo más tarde.")
    except Exception as e:
        logger.exception(f"Unexpected OpenAI interaction error for Conv {sb_conversation_id}: {e}")
        final_assistant_response = ("Ocurrió un error inesperado al procesar tu solicitud. Por favor, intenta de nuevo.")

    if final_assistant_response:
        logger.info(f"Final assistant response for Conv {sb_conversation_id}: '{str(final_assistant_response)[:200]}...'")
        support_board_service.send_reply_to_channel( 
            conversation_id=sb_conversation_id, message_text=str(final_assistant_response),
            source=conversation_source, target_user_id=customer_user_id,
            conversation_details=conversation_data, triggering_message_id=triggering_message_id,
        )
    else: 
        logger.error("No final assistant response generated for Conv %s; sending generic fallback.", sb_conversation_id)
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text=("Lo siento, no pude generar una respuesta en este momento. Por favor, intenta de nuevo."),
            source=conversation_source, target_user_id=customer_user_id,
            conversation_details=conversation_data, triggering_message_id=triggering_message_id,
        )

# --- End of NAMWOO/services/openai_service.py ---