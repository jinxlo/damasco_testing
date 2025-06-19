# NAMWOO/services/google_service.py
# -*- coding: utf-8 -*-
#
# Description:
# This service module manages all interactions with Google's Gemini models
# via an OpenAI-compatible API wrapper. It is designed to be a feature-equivalent
# counterpart to `openai_service.py`, handling chat completions, tool use, and
# description summarization.
#
# Key Changes in This Version:
# - FIXED: Corrected a critical circular import error by changing the import for
#   `user_is_asking_for_cheapest` to point to its new location in `product_utils`.
# - ENHANCED: Re-structured the entire `process_new_message` function to mirror the
#   robust logic of `openai_service.py`, including a cleaner tool-calling loop.
# - FEATURE-PARITY: The tool schema (`tools_schema`) has been replaced with the
#   complete version from `openai_service.py`, adding support for get_store_info,
#   get_available_brands, get_color_variants, and send_whatsapp_order_summary_template.
# - ALIGNED: Tool handling logic now uses helper functions from `product_utils` for
#   consistent product formatting, just like the OpenAI service.
# - IMPROVED: Logging and error handling have been standardized for better debugging.

import logging
import json
import os
import re
from typing import List, Dict, Optional, Any, Union, Tuple

from openai import ( # This lib is used for the OpenAI-compatible Gemini endpoint
    OpenAI, APIError, RateLimitError, APITimeoutError, BadRequestError
)
from flask import current_app

# --- Local services and utilities ---
from . import product_service
from . import support_board_service
from . import product_recommender
from ..config import Config
from ..utils import conversation_location, embedding_utils
# CORRECTED IMPORT: The function now lives in product_utils
from ..utils import product_utils
try:
    from ..utils.whs_utils import canonicalize_whs_name
except ImportError:
    def canonicalize_whs_name(name): return name

logger = logging.getLogger(__name__)

# --- Initialise OpenAI-compatible client for Google Gemini ---
google_gemini_client: Optional[OpenAI] = None
try:
    google_api_key = Config.GOOGLE_API_KEY
    if google_api_key:
        # Using a generic base URL for Google's OpenAI-compatible endpoint
        google_base_url = "https://generativelanguage.googleapis.com/v1beta/"
        timeout_seconds = getattr(Config, 'GOOGLE_REQUEST_TIMEOUT', 60.0)
        google_gemini_client = OpenAI(
            api_key=google_api_key,
            base_url=google_base_url,
            timeout=timeout_seconds,
        )
        logger.info(f"Google Gemini client initialized (via OpenAI lib) with timeout {timeout_seconds}s.")
    else:
        logger.error("GOOGLE_API_KEY not configured. Gemini service is disabled.")
except Exception as e:
    logger.exception(f"Failed to initialise Google Gemini client: {e}")
    google_gemini_client = None

# --- Constants ---
MAX_HISTORY_MESSAGES = Config.MAX_HISTORY_MESSAGES
TOOL_CALL_RETRY_LIMIT = 2
DEFAULT_GOOGLE_MODEL = getattr(Config, "GOOGLE_GEMINI_MODEL", "gemini-1.5-flash-latest")
DEFAULT_MAX_TOKENS = getattr(Config, "GOOGLE_MAX_TOKENS", 1024)
DEFAULT_GOOGLE_TEMPERATURE = getattr(Config, "GOOGLE_TEMPERATURE", 0.7)

# --- LLM Helper: Product Description Summarization ---
def get_google_product_summary(
    plain_text_description: str,
    item_name: Optional[str] = None
) -> Optional[str]:
    """Generates a product summary using Google Gemini."""
    if not google_gemini_client:
        logger.error("Google Gemini client not available. Cannot summarize.")
        return None
    if not plain_text_description or not plain_text_description.strip():
        logger.debug("Google summarizer: No text provided to summarize.")
        return None

    system_prompt = (
        "Eres un redactor experto en comercio electrónico. Resume la siguiente descripción de producto. "
        "El resumen debe ser conciso (objetivo: 50-75 palabras, 2-3 frases clave), resaltar los principales beneficios y características, y ser factual. "
        "Evita la jerga de marketing, la repetición y frases como 'este producto es'. "
        "La salida debe ser texto plano adecuado para una base de datos y un asistente de IA. No incluyas etiquetas HTML."
    )
    user_prompt = f"Por favor, resume la siguiente información del producto:\n\nNombre: {item_name or 'N/A'}\nDescripción:\n{plain_text_description}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        summary_model = getattr(Config, "GOOGLE_SUMMARY_MODEL", DEFAULT_GOOGLE_MODEL)
        logger.debug(f"Requesting summary from Google model '{summary_model}' for item '{item_name or 'Unknown'}'")
        completion = google_gemini_client.chat.completions.create(
            model=summary_model,
            messages=messages,
            temperature=0.2,
            max_tokens=150,
        )
        summary = completion.choices[0].message.content.strip() if completion.choices else None
        if summary:
            logger.info(f"Google summary generated for '{item_name or 'Unknown'}'.")
        else:
            logger.warning(f"Google returned an empty summary for '{item_name or 'Unknown'}'.")
        return summary
    except Exception as e:
        logger.exception(f"Error calling Google for description summarization: {e}")
        return None

# --- Tool Implementations (Mirrored from openai_service.py) ---
# For consistency, these helpers are identical where possible.
_STORE_LOCATIONS_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'store_locations.json')
_cached_store_data: Optional[List[Dict[str, str]]] = None
_REDACT_PATTERN = re.compile(r"(branchName\s*:?\s*\"[^\"]*\"|address\s*:?\s*\"[^\"]*\"|whsName\s*:?\s*\"[^\"]*\"|Almac[eé]n[^,\n]+|Direcci[oó]n\:[^\n]+)", re.IGNORECASE)

def _redact_store_details(text: str) -> str:
    return _REDACT_PATTERN.sub("<REDACTED>", text) if text else text

def _load_store_data() -> List[Dict[str, str]]:
    global _cached_store_data
    if _cached_store_data is not None:
        return _cached_store_data
    try:
        with open(_STORE_LOCATIONS_FILE_PATH, 'r', encoding='utf-8') as f:
            _cached_store_data = json.load(f)
        logger.info("Successfully loaded store data for tool use.")
        return _cached_store_data
    except Exception as e:
        logger.exception(f"Error loading store data from {_STORE_LOCATIONS_FILE_PATH}: {e}")
        _cached_store_data = []
        return []

def _tool_get_store_info(city_name: Optional[str] = None) -> str:
    all_stores = _load_store_data()
    if not all_stores: return json.dumps({"status": "error", "message": "No se pudo cargar la información de las tiendas."}, ensure_ascii=False)
    if city_name:
        found_stores = [s for s in all_stores if s.get("city", "").strip().lower() == city_name.strip().lower()]
        if found_stores: return _redact_store_details(json.dumps({"status": "success", "stores": found_stores}, ensure_ascii=False, indent=2))
        else:
            cities = sorted(list(set(s.get("city") for s in all_stores if s.get("city"))))
            return json.dumps({"status": "city_not_found", "message": f"No hay tiendas en '{city_name}'. Ciudades disponibles: {', '.join(cities)}."}, ensure_ascii=False, indent=2)
    else:
        cities = sorted(list(set(s.get("city") for s in all_stores if s.get("city"))))
        return json.dumps({"status": "success", "available_cities": cities}, ensure_ascii=False, indent=2)

def _sanitize_template_variables(vars: list[str]) -> list[str]:
    if not isinstance(vars, list) or len(vars) != 8: return vars
    branch_names = [s.get("branchName") for s in _load_store_data() if s.get("branchName")]
    if vars[5] not in branch_names: vars[5] = "Sucursal por confirmar"
    return vars

def _tool_send_whatsapp_order_summary_template(customer_platform_user_id: str, conversation_id: str, template_variables: list[str]) -> Dict:
    ok = support_board_service.send_whatsapp_template(to=customer_platform_user_id, template_name="confirmacion_datos_cliente", template_languages="es_ES", parameters=template_variables, recipient_id=conversation_id)
    return {"status": "success" if ok else "failed"}

def _tool_get_color_variants(product_identifier: str) -> str:
    if not product_identifier:
        return json.dumps({"status": "error", "message": "El parámetro product_identifier es requerido."}, ensure_ascii=False)
    variants = product_service.get_color_variants(product_identifier)
    if variants is None:
        return json.dumps({"status": "error", "message": "No se pudo buscar variantes."}, ensure_ascii=False)

    color_names = set()
    for sku in variants:
        details_list = product_service.get_live_product_details_by_sku(sku)
        if details_list:
            item_name = details_list[0].get("item_name") or details_list[0].get("itemName", "")
            color, _ = product_utils.extract_color_from_name(item_name)
            if color:
                color_names.add(color)

    final_list = sorted(color_names) if color_names else variants
    return json.dumps({"status": "success" if final_list else "not_found", "variants": final_list}, ensure_ascii=False, indent=2)

# --- Tool Schema (Feature-Parity with openai_service.py) ---
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
            "description": "Busca productos específicos por características, precio y uso previsto. Utilízala cuando el usuario solicite productos en particular. Para preguntas generales sobre marcas disponibles usa `get_available_brands`.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "What the user is looking for, like 'phone for gaming' or 'cheap washing machine'. Include all technical specifications like '8gb ram' in the query. Avoid pricing words; use 'sort_by' instead."
                    },
                    "exclude_accessories": {
                        "type": "boolean",
                        "description": "Set to false ONLY if the user is explicitly asking for accessories like cases or chargers. Defaults to true to filter out accessories from primary product searches.",
                        "default": True
                    },
                    "warehouse_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of `whsName` values returned from `get_store_info`. Do NOT use `branchName` or city names. Use values like 'Almacen Principal CCCT'."
                    },
                    "min_price": {"type": "number", "description": "Use if user provides a price range or max budget."},
                    "max_price": {"type": "number", "description": "Use if user provides a price range or max budget."},
                    "sort_by": {"type": "string", "description": "Use 'price_asc' for queries like 'cheapest', 'most affordable'. Use 'price_desc' for 'most expensive'. Otherwise use 'relevance'."},
                    "limit": {"type": "integer", "description": "Opcional. Número máximo de resultados a retornar."},
                    "min_score": {"type": "number", "description": "Opcional. Umbral mínimo de similitud para aceptar un resultado. Valor por defecto 0.35."},
                    "exclude_skus": {"type": "array", "items": {"type": "string"}, "description": "Opcional. Una lista de SKUs de productos a excluir de los resultados de búsqueda para evitar repeticiones."}
                },
                "required": ["query_text"]
            }
        }
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
            "description": "Obtiene información detallada y actualizada de un producto específico, ya sea por su nombre de modelo o código SKU.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_identifier": {
                        "type": "string",
                        "description": "El nombre completo del modelo o el código SKU del producto. El sistema detectará el tipo automáticamente."
                    }
                },
                "required": ["product_identifier"]
            }
        }
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
            "name": "send_whatsapp_order_summary_template",
            "description": "Envía la plantilla de resumen de pedido 'confirmacion_datos_cliente' por WhatsApp al cliente una vez que se hayan recolectado todos los datos necesarios (nombre, apellido, cédula, teléfono, correo, dirección de la sucursal, productos y total).",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_platform_user_id": {"type": "string", "description": "ID del usuario en la plataforma de mensajería (generalmente el número de teléfono para WhatsApp, o el ID de usuario interno si es un cliente existente)."},
                    "conversation_id": {"type": "string", "description": "ID de la conversación actual de Support Board."},
                    "template_variables": {"type": "array", "description": "Lista de EXACTAMENTE 8 cadenas en el siguiente orden: 1. Nombre(s) del cliente, 2. Apellido(s) del cliente, 3. Cédula, 4. Teléfono de contacto, 5. Correo electrónico, 6. Nombre de la sucursal de retiro, 7. Descripción de los productos (ej: 'Producto A x1, Producto B x2'), 8. Precio total (ej: '$125.50 USD').", "items": {"type": "string"}, "minItems": 8, "maxItems": 8}
                },
                "required": ["customer_platform_user_id", "conversation_id", "template_variables"]
            }
        }
    }
]

_SKU_PATTERN = re.compile(r'\b(D\d{4,})\b')
def _extract_skus_from_text(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    return _SKU_PATTERN.findall(text)

# --- Helper: Format SB history for Gemini ---
def _prune_and_format_sb_history_for_gemini(sb_messages: Optional[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    if not sb_messages: return [], []
    
    openai_messages: List[Dict[str, Any]] = []
    bot_user_id_str = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None
    if not bot_user_id_str:
        logger.error("[Namwoo-Gemini] Cannot format SB history: SUPPORT_BOARD_DM_BOT_USER_ID is not configured.")
        return [], []

    for msg in sb_messages:
        sender_id = msg.get("user_id")
        text_content = msg.get("message", "").strip()
        if not text_content or sender_id is None: continue
        
        role = "assistant" if str(sender_id) == bot_user_id_str else "user"
        openai_messages.append({"role": role, "content": text_content})
    
    pruned_messages: List[Dict[str, Any]] = []
    seen_signatures = set()
    all_shown_skus = set()

    for msg in reversed(openai_messages):
        role, content = msg.get('role'), msg.get('content')
        if role == 'assistant' and isinstance(content, str):
            all_shown_skus.update(_extract_skus_from_text(content))
        
        signature = f"{role}:{content}"
        if signature not in seen_signatures:
            pruned_messages.append(msg)
            seen_signatures.add(signature)
    
    pruned_messages.reverse()
    logger.info(f"[Namwoo-Gemini] Original history had {len(openai_messages)} messages. Pruned to {len(pruned_messages)}.")
    return pruned_messages, list(all_shown_skus)

# --- Main Processing Entry Point ---
def process_new_message_gemini(
    sb_conversation_id: str,
    new_user_message: Optional[str],
    conversation_source: Optional[str],
    sender_user_id: str,
    customer_user_id: str,
    triggering_message_id: Optional[str],
):
    if not google_gemini_client:
        logger.error("[Namwoo-Gemini] Client not initialized. Cannot process message.")
        support_board_service.send_reply_to_channel(conversation_id=sb_conversation_id, message_text="Disculpa, el servicio de IA (Google) no está disponible.", source=conversation_source, target_user_id=customer_user_id, triggering_message_id=triggering_message_id)
        return

    logger.info(f"[Namwoo-Gemini] Processing message for SB Conv {sb_conversation_id}")
    
    if new_user_message:
        if city := conversation_location.detect_city_from_text(new_user_message):
            conversation_location.set_conversation_city(sb_conversation_id, city)

    conversation_data = support_board_service.get_sb_conversation_data(sb_conversation_id)
    if not conversation_data or not conversation_data.get("messages"):
        logger.error(f"[Namwoo-Gemini] Failed to fetch conversation for SB Conv {sb_conversation_id}.")
        support_board_service.send_reply_to_channel(conversation_id=sb_conversation_id, message_text="Lo siento, tuve problemas para acceder al historial de esta conversación.", source=conversation_source, target_user_id=customer_user_id, triggering_message_id=triggering_message_id)
        return

    gemini_history, previously_shown_skus = _prune_and_format_sb_history_for_gemini(conversation_data.get("messages", []))
    if not gemini_history:
        logger.error(f"[Namwoo-Gemini] Formatted history is empty for Conv {sb_conversation_id}. Aborting.")
        return

    messages: List[Dict[str, Any]] = [{"role": "system", "content": Config.SYSTEM_PROMPT}] + gemini_history
    if len(messages) > (MAX_HISTORY_MESSAGES + 1):
        messages = [messages[0]] + messages[-(MAX_HISTORY_MESSAGES):]

    final_assistant_response: Optional[str] = None
    try:
        tool_call_count = 0
        while tool_call_count <= TOOL_CALL_RETRY_LIMIT:
            logger.debug(f"[Namwoo-Gemini] API call #{tool_call_count + 1} with model {DEFAULT_GOOGLE_MODEL}...")
            response = google_gemini_client.chat.completions.create(model=DEFAULT_GOOGLE_MODEL, messages=messages, tools=tools_schema, tool_choice="auto")
            response_message = response.choices[0].message
            messages.append(response_message.model_dump(exclude_none=True))

            if not response_message.tool_calls:
                final_assistant_response = response_message.content
                break

            tool_outputs = []
            for tc in response_message.tool_calls:
                fn_name, args = tc.function.name, json.loads(tc.function.arguments)
                logger.info(f"[Namwoo-Gemini] LLM requested tool: {fn_name} with args: {args}")

                output_str = "{}"
                try:
                    if fn_name == "get_store_info":
                        output_str = _tool_get_store_info(**args)
                    elif fn_name == "get_available_brands":
                        brands = product_service.get_available_brands(**args)
                        output_str = json.dumps({"status": "success", "formatted_response": product_utils.format_brand_list(brands)}, ensure_ascii=False)
                    elif fn_name == "search_local_products":
                        if 'warehouse_names' not in args and (city := conversation_location.get_conversation_city(sb_conversation_id)):
                            args['warehouse_names'] = conversation_location.get_warehouses_for_city(city)
                        if "warehouse_names" in args and args["warehouse_names"]:
                            args["warehouse_names"] = [canonicalize_whs_name(n) for n in args["warehouse_names"]]
                        
                        args['exclude_skus'] = previously_shown_skus
                        
                        candidate_products = product_service.search_local_products(**args)
                        
                        last_user_message_content = ""
                        for msg in reversed(messages):
                            if msg.get("role") == "user":
                                last_user_message_content = msg.get("content", "")
                                break

                        is_list_request = product_utils.user_is_asking_for_list(last_user_message_content)
                        is_price_request = product_utils.user_is_asking_for_price(last_user_message_content)
                        if is_price_request:
                            grouped = product_utils.group_products_by_model(candidate_products)
                            if grouped:
                                formatted_response = product_utils.format_product_response(grouped[0], args.get("query_text", ""))
                            else:
                                formatted_response = product_utils.format_multiple_products_response(candidate_products, args.get("query_text", ""))
                        elif is_list_request:
                            formatted_response = product_utils.format_model_list_with_colors(candidate_products)
                        else:
                            ranked_products = product_recommender.rank_products(args.get("query_text", ""), candidate_products)
                            formatted_response = product_utils.format_multiple_products_response(ranked_products, args.get("query_text", ""))
                        
                        output_str = json.dumps({"status": "success" if candidate_products else "not_found", "formatted_response": formatted_response}, ensure_ascii=False)
                    elif fn_name == "get_live_product_details":
                        ident = args.get("product_identifier")
                        query_text = messages[-2].get('content', '') if len(messages) > 1 else ''
                        details_list = product_service.get_live_product_details(product_identifier=ident)
                        formatted = None
                        if details_list:
                            grouped = product_utils.group_products_by_model(details_list)
                            if grouped:
                                formatted = product_utils.format_product_response(grouped[0], query_text)
                        output_str = json.dumps({"status": "success" if formatted else "not_found", "formatted_response": formatted}, ensure_ascii=False)
                    elif fn_name == "get_color_variants":
                        output_str = _tool_get_color_variants(**args)
                    elif fn_name == "send_whatsapp_order_summary_template":
                        cust_id = args.get("customer_platform_user_id") or customer_user_id
                        conv_id = args.get("conversation_id") or sb_conversation_id
                        template_vars = _sanitize_template_variables(args.get("template_variables"))
                        status = _tool_send_whatsapp_order_summary_template(cust_id, conv_id, template_vars)
                        output_str = json.dumps(status, ensure_ascii=False)
                    else:
                        output_str = json.dumps({"status": "error", "message": f"Herramienta desconocida '{fn_name}'."})
                except Exception as e:
                    logger.exception(f"[Namwoo-Gemini] Error executing tool {fn_name}: {e}")
                    output_str = json.dumps({"status": "error", "message": str(e)})
                
                tool_outputs.append({"tool_call_id": tc.id, "role": "tool", "name": fn_name, "content": output_str})
            
            messages.extend(tool_outputs)
            tool_call_count += 1

    except (RateLimitError, APITimeoutError, BadRequestError, APIError) as e:
        logger.warning(f"[Namwoo-Gemini] API call failed for Conv {sb_conversation_id}: {type(e).__name__} - {e}")
        final_assistant_response = "Lo siento, estoy teniendo problemas de comunicación con el servicio de IA. Por favor, inténtalo de nuevo en un momento."
    except Exception as e:
        logger.exception(f"[Namwoo-Gemini] Unexpected interaction error for Conv {sb_conversation_id}: {e}")
        final_assistant_response = "Ocurrió un error inesperado. Por favor, intenta de nuevo."

    if not final_assistant_response:
        last_message = messages[-1]
        if last_message.get("role") == "tool":
            try:
                tool_content = json.loads(last_message.get("content", "{}"))
                if "formatted_response" in tool_content and tool_content["formatted_response"]:
                    final_assistant_response = tool_content["formatted_response"]
            except (json.JSONDecodeError, TypeError):
                logger.warning("[Namwoo-Gemini] Could not parse final tool message.")
    
    if final_assistant_response:
        logger.info(f"[Namwoo-Gemini] Final assistant response for Conv {sb_conversation_id}: '{str(final_assistant_response)[:200]}...'")
        support_board_service.send_reply_to_channel(conversation_id=sb_conversation_id, message_text=str(final_assistant_response), source=conversation_source, target_user_id=customer_user_id, conversation_details=conversation_data, triggering_message_id=triggering_message_id)
    else:
        logger.error("[Namwoo-Gemini] No final response generated for Conv %s; sending fallback.", sb_conversation_id)
        support_board_service.send_reply_to_channel(conversation_id=sb_conversation_id, message_text="Lo siento, no pude generar una respuesta en este momento. Por favor, intenta de nuevo.", source=conversation_source, target_user_id=customer_user_id, conversation_details=conversation_data, triggering_message_id=triggering_message_id)