# NAMWOO/services/openai_service.py
# -*- coding: utf-8 -*-
import logging
import json
import time # Keep time if used by retry logic within embedding_utils
import os # For constructing file paths
from typing import List, Dict, Optional, Tuple, Union, Any
from decimal import Decimal
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, BadRequestError
try:
    from flask import current_app, g  # Access app config and request context when available
except Exception:  # pragma: no cover - allow running without Flask app context
    from flask import current_app

    class _DummyG:
        pass

    g = _DummyG()

# Import local services and utils
from . import product_service
from . import support_board_service
from . import product_recommender
from ..config import Config # For SYSTEM_PROMPT, MAX_HISTORY_MESSAGES etc.
from ..utils import embedding_utils
from ..utils import conversation_location
from ..utils import product_utils # Import our new formatting utils
try:  # Optional import for testing environments
    from ..utils.whs_utils import canonicalize_whs_name
except Exception:  # pragma: no cover - fallback for stripped test modules
    def canonicalize_whs_name(name):
        return name

# Re-export helper for backward compatibility in tests
user_is_asking_for_cheapest = getattr(
    product_utils, "user_is_asking_for_cheapest", lambda message: False
)
user_is_asking_for_list = getattr(
    product_utils, "user_is_asking_for_list", lambda message: False
)
user_is_asking_for_price = getattr(
    product_utils, "user_is_asking_for_price", lambda message: False
)
import re


logger = logging.getLogger(__name__)

# Path to the store locations JSON file
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
_SKU_PATTERN = re.compile(r'\b(D\d{4,})\b')

def _redact_store_details(text: str) -> str:
    if not text:
        return text
    return _REDACT_PATTERN.sub("<REDACTED>", text)

# ---------------------------------------------------------------------------
# Initialise OpenAI client for Chat Completions
# ---------------------------------------------------------------------------
_chat_client: Optional[OpenAI] = None
try:
    openai_api_key = Config.OPENAI_API_KEY
    if openai_api_key:
        timeout_seconds = getattr(Config, 'OPENAI_REQUEST_TIMEOUT', 60.0)
        _chat_client = OpenAI(api_key=openai_api_key, timeout=timeout_seconds)
        logger.info(f"OpenAI client initialized for Chat Completions service with timeout: {timeout_seconds}s.")
    else:
        _chat_client = None
        logger.error("OpenAI API key not configured. Chat functionality will fail.")
except Exception as e:
    logger.exception(f"Failed to initialize OpenAI client during initial load: {e}")
    _chat_client = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_HISTORY_MESSAGES = Config.MAX_HISTORY_MESSAGES
TOOL_CALL_RETRY_LIMIT = 2
DEFAULT_OPENAI_MODEL = getattr(Config, "OPENAI_CHAT_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = getattr(Config, "OPENAI_MAX_TOKENS", 1024)
DEFAULT_OPENAI_TEMPERATURE = getattr(Config, "OPENAI_TEMPERATURE", 0.7)

# ---------------------------------------------------------------------------
# Internal Helper for Specification Extraction
# ---------------------------------------------------------------------------
def _extract_specs_from_query(query_text: str) -> List[str]:
    """Uses an LLM to extract key technical specs from a user's query."""
    if not query_text or not _chat_client:
        return []

    system_prompt = (
        "Analiza la consulta de un usuario y extrae los requisitos técnicos clave (como RAM, almacenamiento, tamaño de pantalla, etc.). "
        "Normaliza los valores: combina el número y la unidad sin espacios (ej. '8 gb' -> '8gb'), y elimina palabras descriptivas como 'de ram'. "
        "Devuelve una lista JSON de strings en una clave 'specs'. Si no hay especificaciones, devuelve una lista vacía.\n"
        "Ejemplos:\n"
        "Input: 'celular de 8gb de ram y 256gb de almacenamiento'\n"
        "Output: {\"specs\": [\"8gb\", \"256gb\"]}\n"
        "Input: 'un teléfono con pantalla amoled de 120hz'\n"
        "Output: {\"specs\": [\"amoled\", \"120hz\"]}\n"
        "Input: 'el celular más barato'\n"
        "Output: {\"specs\": []}"
    )
    try:
        response = _chat_client.chat.completions.create(
            model=DEFAULT_OPENAI_MODEL, # Use a fast model for this simple task
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query_text}
            ],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content:
            data = json.loads(content)
            specs = data.get("specs", [])
            if isinstance(specs, list):
                logger.info(f"Extracted specifications from query '{query_text}': {specs}")
                return [str(s).lower().strip() for s in specs if s]
        return []
    except (json.JSONDecodeError, TypeError, IndexError, KeyError) as e:
        logger.warning(f"Could not extract specs from query '{query_text}'. Error: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error extracting specs: {e}", exc_info=True)
        return []


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
    all_stores = _load_store_data()
    if not all_stores:
        return json.dumps({"status": "error", "message": "No se pudo cargar la información de las tiendas."}, ensure_ascii=False)

    if city_name:
        normalized_city_name = city_name.strip().lower()
        found_stores = [
            store for store in all_stores
            if store.get("city", "").strip().lower() == normalized_city_name
        ]

        if found_stores:
            return _redact_store_details(json.dumps({"status": "success", "city": city_name, "stores": found_stores}, ensure_ascii=False, indent=2))
        else:
            available_cities = sorted(list(set(store.get("city") for store in all_stores if store.get("city"))))
            return json.dumps({
                "status": "city_not_found",
                "message": f"No se encontraron tiendas en '{city_name}'. Ciudades disponibles: {', '.join(available_cities)}.",
                "available_cities": available_cities
            }, ensure_ascii=False, indent=2)
    else:
        available_cities = sorted(list(set(store.get("city") for store in all_stores if store.get("city"))))
        if available_cities:
            return json.dumps({"status": "success", "available_cities": available_cities, "message": "Por favor, especifica una ciudad para obtener detalles de las tiendas."}, ensure_ascii=False, indent=2)
        else:
            return json.dumps({"status": "no_cities_found", "message": "No hay ciudades con tiendas configuradas en el sistema."}, ensure_ascii=False)


def _tool_get_color_variants(product_identifier: str) -> str:
    if not product_identifier:
        return json.dumps({"status": "error", "message": "El parámetro product_identifier es requerido."}, ensure_ascii=False)
    variants = product_service.get_color_variants(product_identifier)
    if variants is None:
        return json.dumps({"status": "error", "message": "No se pudo buscar variantes."}, ensure_ascii=False)
    
    # After getting the variant SKUs, we get the color names for a user-friendly response.
    color_names = set()
    for sku in variants:
        color, _ = product_utils.extract_color_from_name(sku)
        if color:
            color_names.add(color)

    # If SKUs were found but no colors could be extracted, return the SKUs themselves.
    final_list = sorted(list(color_names)) if color_names else variants

    return json.dumps({"status": "success" if final_list else "not_found", "variants": final_list}, ensure_ascii=False, indent=2)


def sanitize_tool_args(args: Dict[str, Any], conversation_id: str) -> Dict[str, Any]:
    if not str(args.get("conversation_id", "")).isdigit():
        logger.info(
            "Replacing invalid conversation_id '%s' with '%s'", args.get("conversation_id"), conversation_id
        )
        args["conversation_id"] = conversation_id
    return args


def _sanitize_template_variables(vars: list[str]) -> list[str]:
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
    """Send checkout confirmation template using live pricing and context."""

    # Prefer IDs from Flask context only when necessary
    context_customer = getattr(g, "customer_user_id", None)
    context_conv = getattr(g, "conversation_id", None)

    if not customer_platform_user_id and context_customer:
        customer_platform_user_id = context_customer
        logger.info(
            "Using context customer_user_id '%s' for WhatsApp template", customer_platform_user_id
        )
    else:
        logger.info(
            "Using provided customer identifier '%s' for WhatsApp template", customer_platform_user_id
        )

    if context_conv:
        conversation_id = context_conv

    logger.info(
        "Orchestrating direct template send for conv %s (user %s).",
        conversation_id,
        customer_platform_user_id,
    )

    # Inject live price based on SKU if present
    product_sku = template_variables[0] if template_variables else None
    get_details = getattr(product_service, "get_live_product_details", None)
    if product_sku and callable(get_details):
        details = get_details(product_sku)
        if details and details.get("price") is not None:
            template_variables[-1] = str(details["price"])

    if hasattr(support_board_service, "send_whatsapp_template_direct"):
        support_board_service.send_whatsapp_template_direct(
            user_id=customer_platform_user_id,
            conversation_id=conversation_id,
            template_variables=template_variables,
        )
    else:
        support_board_service.send_whatsapp_template(
            to=customer_platform_user_id,
            template_name="confirmacion_datos_cliente",
            template_languages="es_ES",
            parameters=template_variables,
            recipient_id=conversation_id,
        )

    return {"status": "success"}

def _tool_find_relevant_accessory(main_product_category: str, warehouse_names: List[str]) -> str:
    """Finds a relevant, in-stock accessory for a given product category and location."""
    accessory = product_service.find_relevant_accessory(main_product_category, warehouse_names)
    if accessory:
        return json.dumps({"status": "success", "accessory": accessory}, ensure_ascii=False)
    else:
        return json.dumps({"status": "not_found", "message": "No relevant accessory found."}, ensure_ascii=False)

def _tool_get_product_availability(
    product_identifier: str, conversation_id: str
) -> str:
    """Gets all available branch names for a given product SKU or model name, filtered by the conversation's city."""
    user_city = conversation_location.get_conversation_city(conversation_id)
    warehouses = None
    if user_city:
        warehouses = conversation_location.get_warehouses_for_city(user_city)
        logger.info(
            f"Availability check for '{product_identifier}' will be filtered by warehouses for city '{user_city}': {warehouses}"
        )
    else:
        logger.warning(
            f"No city context for conversation {conversation_id}. Availability check for '{product_identifier}' will be nationwide."
        )

    locations = product_service.get_product_availability(
        product_identifier, warehouse_names=warehouses
    )
    if locations:
        return json.dumps({"status": "success", "locations": locations}, ensure_ascii=False)
    else:
        return json.dumps(
            {"status": "not_found", "message": "Product not found or out of stock in this city."}, ensure_ascii=False
        )


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
                "Busca productos específicos por características, precio y uso previsto. "
                "Utilízala cuando el usuario solicite productos en particular. "
                "Para preguntas generales sobre marcas disponibles usa `get_available_brands`."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": (
                            "What the user is looking for, like 'phone for gaming' or 'cheap washing machine'. "
                            "Include all technical specifications like '8gb ram' in the query. "
                            "Avoid pricing words; use 'sort_by' instead."
                        ),
                    },
                    "exclude_accessories": {
                        "type": "boolean",
                        "description": (
                            "Set to false ONLY if the user is explicitly asking for accessories like cases or chargers. "
                            "Defaults to true to filter out accessories from primary product searches."
                        ),
                        "default": True,
                    },
                    "warehouse_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of `whsName` values returned from `get_store_info`. Do NOT use `branchName` or city names. Use values like 'Almacen Principal CCCT'.",
                    },
                    "min_price": {
                        "type": "number",
                        "description": "Use if user provides a price range or max budget."
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Use if user provides a price range or max budget."
                    },
                    "sort_by": {
                        "type": "string",
                        "description": "Use 'price_asc' for queries like 'cheapest', 'most affordable'. Use 'price_desc' for 'most expensive'. Otherwise use 'relevance'."
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Opcional. Número máximo de resultados a retornar."
                    },
                    "min_score": {
                        "type": "number",
                        "description": "Opcional. Umbral mínimo de similitud para aceptar un resultado. Valor por defecto 0.35.",
                    },
                    "exclude_skus": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Opcional. Una lista de SKUs de productos a excluir de los resultados de búsqueda para evitar repeticiones."
                    }
                },
                "required": ["query_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_brands",
            "description": "Returns a list of all available product brands. Use this when the user asks 'what brands do you have?' or similar questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Opcional. Categoría de producto para filtrar las marcas, por ejemplo 'Celular'."
                    }
                },
                "required": []
            }
        }
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
            "description": "Obtiene los distintos colores disponibles para un modelo de producto. Úsalo cuando el usuario pregunte '¿qué colores tienes para el [modelo]?'",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_identifier": {
                        "type": "string",
                        "description": "El nombre del modelo del producto (ej. 'TECNO SPARK 30C') o un SKU específico (ej. 'D0007783') para buscar sus variantes de color."
                    }
                },
                "required": ["product_identifier"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_relevant_accessory",
            "description": "Busca un accesorio relevante y en stock para un producto principal en las mismas ubicaciones donde el producto principal está disponible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "main_product_category": {
                        "type": "string",
                        "description": "La categoría del producto principal (ej. 'CELULAR') para encontrar un accesorio adecuado."
                    },
                    "warehouse_names": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Lista de nombres de almacén ('whsName') donde el producto principal está disponible."
                    }
                },
                "required": ["main_product_category", "warehouse_names"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_availability",
            "description": "Verifica todas las sucursales (branch_name) donde un producto específico, identificado por su SKU o nombre de modelo, está disponible con stock.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_identifier": {
                        "type": "string",
                        "description": "El SKU (item_code) o el nombre del modelo del producto para verificar su disponibilidad en las tiendas."
                    }
                },
                "required": ["product_identifier"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_whatsapp_order_summary_template",
            "description": "Envía la plantilla de resumen de pedido 'confirmacion_datos_cliente' por WhatsApp al cliente una vez que se hayan recolectado todos los datos necesarios (nombre, apellido, cédula, teléfono, correo, dirección de la sucursal, productos y total).",
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
                        "description": "Lista de EXACTAMENTE 8 cadenas en el siguiente orden: 1. Nombre(s) del cliente, 2. Apellido(s) del cliente, 3. Cédula, 4. Teléfono de contacto, 5. Correo electrónico, 6. Nombre de la sucursal de retiro, 7. Descripción de los productos (ej: 'Producto A x1, Producto B x2'), 8. Precio total (ej: '$125.50 USD').",
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

def _extract_skus_from_text(text: str) -> List[str]:
    """Extracts product SKUs (e.g., D0001234) from a string."""
    if not text or not isinstance(text, str):
        return []
    return _SKU_PATTERN.findall(text)

# ---------------------------------------------------------------------------
# Helper: format Support‑Board history for OpenAI and prune it
# ---------------------------------------------------------------------------
def _prune_and_format_sb_history(
    sb_messages: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not sb_messages:
        return [], []

    # 1. First pass: convert to OpenAI format
    openai_messages: List[Dict[str, Any]] = []
    bot_user_id_str = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None
    if not bot_user_id_str:
        logger.error("Cannot format SB history: SUPPORT_BOARD_DM_BOT_USER_ID is not configured.")
        return [], []

    for msg in sb_messages:
        sender_id = msg.get("user_id")
        text_content = msg.get("message", "").strip()
        
        if not text_content:
            continue
        if sender_id is None:
            continue
            
        role = "assistant" if str(sender_id) == bot_user_id_str else "user"
        openai_messages.append({"role": role, "content": text_content})
    
    # 2. Second pass: prune duplicates and extract SKUs
    pruned_messages: List[Dict[str, Any]] = []
    seen_signatures = set()
    all_shown_skus = set()

    for msg in reversed(openai_messages):
        role = msg.get('role')
        content = msg.get('content')
        
        if role == 'assistant' and isinstance(content, str):
            all_shown_skus.update(_extract_skus_from_text(content))
        
        # Create a signature to detect and remove exact duplicates
        signature = f"{role}:{content}"
        if signature not in seen_signatures:
            pruned_messages.append(msg)
            seen_signatures.add(signature)
    
    pruned_messages.reverse() # Restore chronological order
    
    logger.info(f"Original history had {len(openai_messages)} messages. Pruned to {len(pruned_messages)}.")
    
    return pruned_messages, list(all_shown_skus)


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

    try:
        g.customer_user_id = customer_user_id
        g.conversation_id = sb_conversation_id
    except Exception:
        pass

    logger.info(
        "Processing message for SB Conv %s (trigger_user=%s, customer=%s, source=%s, trig_msg_id=%s)",
        sb_conversation_id, sender_user_id, customer_user_id, conversation_source, triggering_message_id,
    )

    if new_user_message:
        detected_city = conversation_location.detect_city_from_text(new_user_message)
        if detected_city:
            logger.info(f"Detected city '{detected_city}' from user message. Saving to conversation context.")
            conversation_location.set_conversation_city(sb_conversation_id, detected_city)
        else:
            logger.info(f"No city detected in new user message for SB Conv {sb_conversation_id}.")

    conversation_data = support_board_service.get_sb_conversation_data(sb_conversation_id)
    if conversation_data is None or not conversation_data.get("messages"):
        logger.error(f"Failed to fetch conversation data or no messages found for SB Conv {sb_conversation_id}. Aborting.")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text="Lo siento, tuve problemas para acceder al historial de esta conversación. ¿Podrías intentarlo de nuevo?",
            source=conversation_source, target_user_id=customer_user_id, conversation_details=None, triggering_message_id=triggering_message_id
        )
        return

    if new_user_message and product_utils.user_is_asking_for_list(new_user_message):
        brand = product_utils.extract_brand_from_message(new_user_message)
        if brand:
            warehouses = conversation_location.get_city_warehouses(sb_conversation_id)
            products = product_service.search_local_products(
                query_text=brand,
                warehouse_names=warehouses,
                is_list_query=True,
                limit=getattr(Config, "PRODUCT_SEARCH_LIMIT", 20),
            )
            if products is not None:
                formatted = product_utils.format_model_list_with_colors(products)
                support_board_service.send_reply_to_channel(
                    conversation_id=sb_conversation_id,
                    message_text=formatted,
                    source=conversation_source,
                    target_user_id=customer_user_id,
                    conversation_details=conversation_data,
                    triggering_message_id=triggering_message_id,
                )
                return

    sb_history_list = conversation_data.get("messages", [])
    try:
        openai_history, previously_shown_skus = _prune_and_format_sb_history(sb_history_list)
        logger.info(f"Identified {len(previously_shown_skus)} previously shown SKUs in conversation {sb_conversation_id}: {previously_shown_skus}")
    except Exception as err:
        logger.exception(f"Error preparing and condensing SB history for Conv {sb_conversation_id}: {err}")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text="Lo siento, tuve problemas al procesar el historial de la conversación.",
            source=conversation_source, target_user_id=customer_user_id, conversation_details=conversation_data, triggering_message_id=triggering_message_id
        )
        return

    if not openai_history:
        logger.error(f"Formatted OpenAI history is empty for Conv {sb_conversation_id}. Aborting.")
        support_board_service.send_reply_to_channel(
            conversation_id=sb_conversation_id, message_text="Lo siento, no pude procesar los mensajes anteriores adecuadamente.",
            source=conversation_source, target_user_id=customer_user_id, conversation_details=conversation_data, triggering_message_id=triggering_message_id
        )
        return

    system_prompt_content = Config.SYSTEM_PROMPT
    
    # --- CONTEXT INJECTION FIX ---
    # Check for a saved location and inject it into the system prompt.
    saved_city = conversation_location.get_conversation_city(sb_conversation_id)
    if saved_city:
        contextual_note = (
            f"\n\n**NOTA DE CONTEXTO CRÍTICA:** La ubicación del usuario ha sido confirmada como **{saved_city}**. "
            "DEBES usar esta ubicación para todas las búsquedas de productos y no debes volver a preguntar por la ciudad, "
            "a menos que el usuario mencione explícitamente una nueva."
        )
        system_prompt_content += contextual_note
        logger.info(f"Injected location context '{saved_city}' into system prompt for conv {sb_conversation_id}.")
    # --- END CONTEXT INJECTION FIX ---

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt_content}] + openai_history

    max_hist_current = getattr(Config, "MAX_HISTORY_MESSAGES", MAX_HISTORY_MESSAGES)
    if len(messages) > (max_hist_current + 1 ):
        messages = [messages[0]] + messages[-(max_hist_current):]

    final_assistant_response: Optional[str] = None
    try:
        tool_call_count = 0
        while tool_call_count <= TOOL_CALL_RETRY_LIMIT:
            recommender_model = getattr(Config, "OPENAI_RECOMMENDER_MODEL", "gpt-4.1")
            current_model = recommender_model if Config.RECOMMENDER_MODE == 'llm' else DEFAULT_OPENAI_MODEL

            call_params = {
                "model": current_model,
                "messages": messages,
                "tools": tools_schema,
                "tool_choice": "auto",
            }
            logger.debug(f"OpenAI API call #{tool_call_count + 1} with model {current_model}...")
            response = _chat_client.chat.completions.create(**call_params)
            response_message = response.choices[0].message
            messages.append(response_message.model_dump(exclude_none=True))

            tool_calls = response_message.tool_calls
            if not tool_calls:
                final_assistant_response = response_message.content
                break

            tool_outputs = []
            for tc in tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                logger.info(f"LLM requested tool: {fn_name} with args: {args}")

                output_content_str = "{}"
                try:
                    if fn_name == "get_store_info":
                        output_content_str = _tool_get_store_info(**args)
                    elif fn_name == "get_available_brands":
                        brands = product_service.get_available_brands(**args)
                        formatted_brands = product_utils.format_brand_list(brands)
                        output_content_str = json.dumps({"status": "success" if brands else "not_found", "formatted_response": formatted_brands}, ensure_ascii=False)
                    elif fn_name == "search_local_products":
                        query_text = args.get("query_text", "")
                        
                        if 'warehouse_names' not in args or not args['warehouse_names']:
                            logger.info("LLM did not provide warehouse_names. Checking for saved location context...")
                            user_city = conversation_location.get_conversation_city(sb_conversation_id)
                            if user_city:
                                logger.info(f"Found saved city '{user_city}' for conv {sb_conversation_id}. Injecting warehouse names.")
                                args['warehouse_names'] = conversation_location.get_warehouses_for_city(user_city)
                            else:
                                logger.warning(f"No warehouse_names from LLM and no saved city context for conv {sb_conversation_id}. Search will be nationwide.")
                        
                        if "warehouse_names" in args and args["warehouse_names"]:
                            args["warehouse_names"] = [canonicalize_whs_name(n) for n in args["warehouse_names"]]
                        
                        triggering_user_message_content = ""
                        query_keywords = set(query_text.lower().split())
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                msg_content = msg.get("content", "").lower()
                                if any(kw in msg_content for kw in query_keywords):
                                    triggering_user_message_content = msg.get("content", "")
                                    break
                        if not triggering_user_message_content:
                            for msg in reversed(messages):
                                if msg.get("role") == "user":
                                    triggering_user_message_content = msg.get("content", "")
                                    break
                        
                        is_list_request = product_utils.user_is_asking_for_list(triggering_user_message_content)
                        is_price_request = product_utils.user_is_asking_for_price(triggering_user_message_content)
                        args['is_list_query'] = is_list_request
                        
                        required_specs = _extract_specs_from_query(query_text)
                        args['required_specs'] = required_specs
                        args['exclude_skus'] = previously_shown_skus
                        
                        candidate_products = product_service.search_local_products(**args)

                        if is_price_request:
                            logger.info(
                                f"Price query detected based on user message: '{triggering_user_message_content}'"
                            )
                            grouped = product_utils.group_products_by_model(candidate_products)
                            if grouped:
                                formatted_response = product_utils.format_product_response(
                                    grouped[0], query_text
                                )
                            else:
                                formatted_response = product_utils.format_ai_recommendations(candidate_products)
                        elif is_list_request:
                            logger.info(
                                f"List format requested based on user message: '{triggering_user_message_content}'"
                            )
                            formatted_response = product_utils.format_model_list_with_colors(candidate_products)
                        else:
                            logger.info("Recommendation format requested. Invoking AI Sales-Associate.")
                            ranked_products = product_recommender.rank_products(
                                user_intent=triggering_user_message_content,
                                candidates=candidate_products,
                                sort_by=args.get("sort_by")
                            )
                            formatted_response = product_utils.format_ai_recommendations(ranked_products)
                        
                        output_content_str = json.dumps({"status": "success" if candidate_products else "not_found", "formatted_response": formatted_response}, ensure_ascii=False)
                        
                    elif fn_name == "get_live_product_details":
                        ident = args.get("product_identifier")
                        id_type = args.get("identifier_type")
                        query_text = messages[-2].get('content', '') if len(messages) > 1 and messages[-2].get('role') == 'user' else ''
                        details_result = None
                        if ident and id_type:
                            if id_type == "sku":
                                details_list = product_service.get_live_product_details_by_sku(item_code_query=ident)
                                if details_list:
                                    grouped = product_utils.group_products_by_model(details_list)
                                    if grouped:
                                        details_result = product_utils.format_product_response(grouped[0], query_text)
                            elif id_type == "composite_id":
                                details_dict = product_service.get_live_product_details_by_id(composite_id=ident)
                                if details_dict:
                                    details_result = product_utils.format_product_response(details_dict, query_text)
                        output_content_str = json.dumps({"status": "success" if details_result else "not_found", "formatted_response": details_result}, ensure_ascii=False)
                    elif fn_name == "get_color_variants":
                        output_content_str = _tool_get_color_variants(**args)
                    elif fn_name == "find_relevant_accessory":
                        output_content_str = _tool_find_relevant_accessory(**args)
                    elif fn_name == "get_product_availability":
                        args['conversation_id'] = sb_conversation_id
                        output_content_str = _tool_get_product_availability(**args)
                    elif fn_name == "send_whatsapp_order_summary_template":
                        cust_id_arg = args.get("customer_platform_user_id") or customer_user_id
                        conv_id_arg = args.get("conversation_id") or sb_conversation_id
                        template_vars_arg = _sanitize_template_variables(args.get("template_variables"))
                        if not cust_id_arg or not conv_id_arg or not template_vars_arg:
                            output_content_str = json.dumps({"status": "error", "message": "Faltan datos para la plantilla."}, ensure_ascii=False)
                        else:
                            status_dict = _tool_send_whatsapp_order_summary_template(
                                customer_platform_user_id=str(cust_id_arg),
                                conversation_id=str(conv_id_arg),
                                template_variables=template_vars_arg,
                            )
                            output_content_str = json.dumps(status_dict, ensure_ascii=False)
                    else:
                        output_content_str = json.dumps({"status": "error", "message": f"Error: Herramienta desconocida '{fn_name}'."}, ensure_ascii=False)
                        logger.warning(f"LLM called unknown tool: {fn_name} in Conv {sb_conversation_id}")
                
                except Exception as tool_exec_err:
                    logger.exception(f"Error executing tool {fn_name}: {tool_exec_err}")
                    output_content_str = json.dumps({"status": "error", "message": str(tool_exec_err)}, ensure_ascii=False)

                tool_outputs.append({
                    "tool_call_id": tc.id,
                    "role": "tool",
                    "name": fn_name,
                    "content": output_content_str,
                })

            messages.extend(tool_outputs)
            tool_call_count += 1

    except RateLimitError:
        logger.warning(f"OpenAI RateLimitError for Conv {sb_conversation_id}")
        final_assistant_response = ("Estoy experimentando un alto volumen de solicitudes. "
                                    "Por favor, espera un momento y vuelve a intentarlo.")
    except APITimeoutError:
        logger.warning(f"OpenAI APITimeoutError for Conv {sb_conversation_id}")
        final_assistant_response = "No pude obtener respuesta del servicio de IA a tiempo. Por favor, intenta más tarde."
    except BadRequestError as bre:
        logger.error(f"OpenAI BadRequestError for Conv {sb_conversation_id}: {bre}", exc_info=True)
        final_assistant_response = ("Lo siento, hubo un problema con nuestra conversación. "
                                    "Por favor, reformula tu pregunta o intenta de nuevo.")
    except APIError as apie:
        logger.error(f"OpenAI APIError for Conv {sb_conversation_id} (Status: {apie.status_code}): {apie}", exc_info=True)
        final_assistant_response = (f"Hubo un error ({apie.status_code}) con el servicio de IA. Por favor, inténtalo más tarde.")
    except Exception as e:
        logger.exception(f"Unexpected OpenAI interaction error for Conv {sb_conversation_id}: {e}")
        final_assistant_response = ("Ocurrió un error inesperado al procesar tu solicitud. Por favor, intenta de nuevo.")

    if not final_assistant_response:
        last_message = messages[-1]
        if last_message.get("role") == "tool":
            try:
                tool_content = json.loads(last_message.get("content", "{}"))
                if "formatted_response" in tool_content and tool_content["formatted_response"]:
                    final_assistant_response = tool_content["formatted_response"]
            except (json.JSONDecodeError, TypeError):
                logger.warning("Could not parse final tool message content as JSON.")

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
    try:
        g.customer_user_id = None
        g.conversation_id = None
    except Exception:
        pass
