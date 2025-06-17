# -*- coding: utf-8 -*-
import requests
import logging
import json
import re  # Import regex module for cleaning phone numbers
import time
from flask import current_app
from typing import Optional, List, Dict, Any

# Assuming Config is correctly imported and loads .env variables
from ..config import Config

logger = logging.getLogger(__name__)

# Use a shared session for all Support Board API calls. Some tests mock the
# requests module and may not provide Session, so fall back gracefully.
sb_api_session = requests.Session() if hasattr(requests, "Session") else requests

# --- PRIVATE HELPER: Make Support Board API Call ---
def _call_sb_api(payload: Dict) -> Optional[Any]:
    """Internal helper to make POST requests to the Support Board API."""
    api_url = current_app.config.get('SUPPORT_BOARD_API_URL')
    api_token = current_app.config.get('SUPPORT_BOARD_API_TOKEN')

    if not api_url or not api_token:
        logger.error("Support Board API URL or Token is not configured.")
        return None

    payload['token'] = api_token
    function_name = payload.get('function', 'N/A')
    logger.debug(f"Calling SB API URL: {api_url} with function: {function_name}")
    try:
        log_payload = payload.copy()
        if 'token' in log_payload:
            log_payload['token'] = '***' + log_payload['token'][-4:] if len(log_payload.get('token','')) > 4 else '***'
        log_payload_str = json.dumps(log_payload)
    except Exception:
        log_payload_str = str(payload)
    logger.debug(f"Payload for {function_name} (requests data param): {log_payload_str}")

    try:
        response = sb_api_session.post(api_url, data=payload, timeout=20)
        response.raise_for_status()
        response_json = response.json()
        try:
            log_response_str = json.dumps(response_json)
        except Exception:
            log_response_str = str(response_json)
        logger.debug(f"Raw SB API response for {function_name}: {log_response_str}")

        if response_json.get("success") is True:
             return response_json.get("response")
        else:
            error_detail = response_json.get("response", f"API call failed for function '{function_name}' with success=false or missing")
            logger.error(f"Support Board API reported failure for {function_name}: {error_detail}")
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP error calling Support Board API ({function_name}): {e}", exc_info=True)
        if e.response is not None:
            logger.error(f"Response body from failed request: {e.response.text[:500]}")
        return None
    except requests.exceptions.JSONDecodeError as e:
        raw_text = getattr(response, 'text', 'N/A')
        logger.error(f"Failed to decode JSON response from Support Board API ({function_name}): {e}. Response text: {raw_text[:500]}", exc_info=True)
        return None
    except Exception as e:
        logger.exception(f"Unexpected error calling SB API ({function_name}): {e}")
        return None

# --- Public Function: Get Conversation Data ---
def get_sb_conversation_data(conversation_id: str) -> Optional[Dict]:
    """Fetches the full conversation details from Support Board."""
    payload = {
        'function': 'get-conversation',
        'conversation_id': conversation_id
    }
    logger.info(f"Attempting to fetch full conversation data from Support Board API for ID: {conversation_id}")
    response_data = _call_sb_api(payload)
    if isinstance(response_data, dict):
        if "messages" in response_data and "details" in response_data:
            logger.info(f"Successfully fetched conversation data for SB conversation {conversation_id}")
            return response_data
        else:
            logger.warning(f"SB API get-conversation success reported, but response for {conversation_id} might be incomplete or malformed. Response: {response_data}")
            return response_data
    else:
        logger.error(f"Failed to fetch or parse valid conversation data dictionary for SB conversation {conversation_id}. Raw response from _call_sb_api call was not a valid dictionary: {response_data}")
        return None

# --- PRIVATE HELPER: Get User PSID (for FB/IG) ---
def _get_user_psid(user_id: str) -> Optional[str]:
    """Fetches user details and extracts the PSID (Facebook/Instagram ID)."""
    logger.info(f"Attempting to fetch user details for User ID: {user_id} to get PSID.")
    payload = {'function': 'get-user', 'user_id': user_id, 'extra': 'true'}
    user_data = _call_sb_api(payload)
    if user_data and isinstance(user_data, dict):
        details_list = user_data.get('details', [])
        expected_slug = 'facebook-id'
        if isinstance(details_list, list):
            for detail in details_list:
                if isinstance(detail, dict) and detail.get('slug') == expected_slug:
                    psid = detail.get('value')
                    if psid and isinstance(psid, str) and psid.strip():
                        logger.info(f"Found PSID (using slug '{expected_slug}') for User ID {user_id}")
                        return psid.strip()
                    else:
                        logger.warning(f"Found '{expected_slug}' detail slug for user {user_id} but its value is empty or invalid: '{psid}'")
            logger.warning(f"Could not find PSID using slug '{expected_slug}' in the details list for User ID: {user_id}. Details received: {details_list}")
            return None
        else:
             logger.warning(f"User details for {user_id} received, but 'details' key is not a list: {details_list}")
             return None
    else:
        logger.error(f"Failed to fetch or parse valid user details dictionary for User ID: {user_id} needed for PSID lookup.")
        return None

# --- CORRECTED PRIVATE HELPER: Get User WAID (for WhatsApp) ---
def _get_user_waid(user_id: str) -> Optional[str]:
    """
    Fetches user details from SB API using 'get-user' + 'extra=true'
    and extracts/formats the WAID. Ignores any pre-fetched data.
    Prioritizes 'phone' detail slug, falls back to 'first_name' if it looks like a number.
    Requires WHATSAPP_DEFAULT_COUNTRY_CODE in config for numbers missing the prefix.
    """
    logger.info(f"Attempting to get WAID for User ID: {user_id}. ALWAYS fetching user details via 'get-user' + 'extra=true'.")
    phone_number = None
    user_first_name = None
    user_details_data = None # Ensure we fetch fresh data

    # --- ALWAYS Fetch user details using get-user + extra=true ---
    logger.debug(f"Fetching user details for {user_id} using 'get-user' + 'extra=true'.")
    payload = {'function': 'get-user', 'user_id': str(user_id), 'extra': 'true'}
    user_details_data = _call_sb_api(payload)
    # --- End Fetch ---

    if user_details_data and isinstance(user_details_data, dict):
        # Extract first name for potential fallback
        user_first_name = user_details_data.get('first_name')
        details_list = user_details_data.get('details', []) # Expecting a list here from get-user

        if isinstance(details_list, list):
            # Try to find the phone number in the details list
            for detail in details_list:
                if isinstance(detail, dict) and detail.get('slug') == 'phone':
                    phone_value = detail.get('value')
                    if phone_value and isinstance(phone_value, str) and phone_value.strip():
                        phone_number = phone_value.strip()
                        logger.info(f"Found phone number for User ID {user_id} in 'phone' detail from get-user response.")
                        break # Found it, stop looking
                    else:
                        logger.warning(f"Found 'phone' detail slug for user {user_id} but its value is empty or invalid: '{phone_value}'")

            if not phone_number:
                logger.warning(f"Could not find valid 'phone' detail in the details list returned by get-user for User ID {user_id}. Details received: {details_list}. Will check first_name as fallback.")
        else:
            logger.warning(f"User details fetched via get-user for {user_id} received, but 'details' key is not a list or is missing. Response: {user_details_data}")
            # Continue to check first_name fallback
    else:
        # This error covers cases where the _call_sb_api call for get-user failed
        logger.error(f"Failed to fetch or parse valid user details dictionary via get-user for User ID: {user_id} needed for WAID lookup.")
        return None # Cannot proceed without user details

    # --- Fallback: Use first_name if phone number wasn't found in details ---
    if not phone_number:
        if user_first_name and isinstance(user_first_name, str):
            cleaned_first_name = user_first_name.strip()
            if re.fullmatch(r'[\d\s\-\(\)\+]+', cleaned_first_name) and len(re.sub(r'\D', '', cleaned_first_name)) >= 7:
                logger.info(f"Using 'first_name' field '{cleaned_first_name}' as fallback phone number for User ID {user_id}.")
                phone_number = cleaned_first_name
            else:
                 logger.warning(f"'first_name' for user {user_id} ('{cleaned_first_name}') does not appear to be a valid phone number. Cannot use as fallback.")
        else:
            logger.warning(f"Cannot fallback to first_name for user {user_id}: field is missing, not a string, or empty.")


    if not phone_number:
        logger.error(f"Could not determine phone number for User ID {user_id} from get-user details or first_name fallback.")
        return None

    # --- Format the phone number into WAID (Kept Unchanged) ---
    waid = re.sub(r'\D', '', phone_number) # Remove non-digits

    if not phone_number.lstrip().startswith('+') and not waid.startswith(Config.WHATSAPP_DEFAULT_COUNTRY_CODE or ''):
        default_cc = Config.WHATSAPP_DEFAULT_COUNTRY_CODE
        if default_cc:
            default_cc_digits = re.sub(r'\D', '', default_cc) # Clean the default CC itself
            if default_cc_digits: # Check if the cleaned default CC has digits
                logger.warning(f"Phone number '{phone_number}' for user {user_id} appears to be missing country code prefix. Prepending default: '{default_cc_digits}'.")
                waid = default_cc_digits + waid
            else:
                logger.error(f"Configured WHATSAPP_DEFAULT_COUNTRY_CODE ('{default_cc}') contains no digits. Cannot prepend.")
                return None # Cannot form valid WAID
        else:
            logger.error(f"Phone number '{phone_number}' for user {user_id} is missing country code prefix, and WHATSAPP_DEFAULT_COUNTRY_CODE is not set or is invalid. Cannot form valid WAID.")
            return None
    elif not phone_number.lstrip().startswith('+') and waid.startswith(Config.WHATSAPP_DEFAULT_COUNTRY_CODE or ''):
        logger.debug(f"Phone number '{phone_number}' for user {user_id} seemed to be missing '+' but already started with the default country code. Assuming it's correct.")


    logger.info(f"Successfully derived WAID '{waid}' for User ID {user_id}.")
    return waid


# --- PRIVATE HELPER: Send Messenger/Instagram Message (External Delivery via SB API) ---
def _send_messenger_message(
    psid: str,
    page_id: str,
    message_text: str,
    conversation_id: str,
    triggering_message_id: Optional[str]
) -> bool:
    """Sends a message via the SB messenger-send-message API."""
    logger.debug(f"[_send_messenger_message CALLED] Conv ID: {conversation_id}")
    bot_user_id = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None
    if not bot_user_id:
        logger.warning(f"SUPPORT_BOARD_DM_BOT_USER_ID not configured. Message attribution may be affected.")

    logger.info(f"Attempting to send Messenger/IG message via specific SB API for Conv ID {conversation_id} to PSID: ...{psid[-6:]} on Page ID: {page_id}")
    payload = { 'function': 'messenger-send-message', 'psid': psid, 'facebook_page_id': page_id, 'message': message_text }
    if triggering_message_id:
        payload['metadata'] = str(triggering_message_id)
    response_data = _call_sb_api(payload)
    if response_data:
        logger.info(f"Messenger/IG message acknowledged as successful by SB API for Conv ID {conversation_id}")
        return True
    logger.error(f"Failed to send Messenger/IG message via SB API for Conv ID {conversation_id}. Unexpected response structure.")
    return False

# --- PRIVATE HELPER: Add Message Internally to SB (Dashboard Visibility) ---
def _add_internal_sb_message(conversation_id: str, message_text: str, bot_user_id: str) -> bool:
    """Adds a message internally to the SB conversation using send-message."""
    if not bot_user_id:
        logger.error("Cannot add internal SB message: Bot User ID not provided or configured.")
        return False

    logger.info(f"Adding bot reply internally to SB conversation ID: {conversation_id} as User ID: {bot_user_id}")
    payload = {
        'function': 'send-message',
        'user_id': bot_user_id,
        'conversation_id': conversation_id,
        'message': message_text,
        'attachments': json.dumps([])
    }
    response_data = _call_sb_api(payload)
    if response_data:
        internal_msg_id = response_data.get('id', 'N/A')
        logger.info(f"Internal SB message added successfully (Internal Msg ID: {internal_msg_id}) to conversation {conversation_id}")
        return True
    else:
        logger.error(f"Failed to add internal SB message to conversation {conversation_id}. API response: {response_data}")
        return False


# --- NEW PUBLIC FUNCTION: Log Internal Message (for agent visibility) ---
def log_internal_message(conversation_id: str, message_text: str) -> bool:
    """Adds a message to the SB conversation for internal agent visibility only."""
    bot_user_id = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID)
    if not bot_user_id:
        logger.error("Cannot add internal SB message: SUPPORT_BOARD_DM_BOT_USER_ID not configured.")
        return False
    return _add_internal_sb_message(conversation_id, message_text, bot_user_id)


# --- PRIVATE HELPER: Send WhatsApp Message DIRECTLY via Meta Cloud API ---
def _send_whatsapp_cloud_api(recipient_waid: str, message_text: str) -> bool:
    token = Config.WHATSAPP_CLOUD_API_TOKEN
    phone_number_id = Config.WHATSAPP_PHONE_NUMBER_ID
    api_version = Config.WHATSAPP_API_VERSION

    if not token or not phone_number_id:
        logger.error("WhatsApp Cloud API Token or Phone Number ID not configured. Cannot send direct message.")
        return False
    api_url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }
    payload_dict = { "messaging_product": "whatsapp", "recipient_type": "individual", "to": recipient_waid, "type": "text", "text": { "preview_url": False, "body": message_text } }
    logger.info(f"Attempting to send direct WhatsApp message via Meta Cloud API to WAID: ...{recipient_waid[-6:]}")
    try:
        response = sb_api_session.post(api_url, headers=headers, json=payload_dict, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        if "messages" in response_json and len(response_json.get("messages", [])) > 0:
            logger.info(f"Direct WhatsApp API call successful.")
            return True
        else:
            logger.error(f"Direct WhatsApp API call returned unexpected success structure: {response_json}")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Direct WhatsApp API request error: {e}", exc_info=True)
        return False

# --- NEW PRIVATE HELPER: Send Telegram Message (External Delivery via SB API) ---
def _send_telegram_message(chat_id: str, message_text: str, conversation_id: Optional[str]) -> bool:
    logger.info(f"Attempting to send Telegram message via SB API to Chat ID: {chat_id} for SB Conv ID: {conversation_id or 'N/A'}")
    payload = { 'function': 'telegram-send-message', 'chat_id': chat_id, 'message': message_text, 'attachments': json.dumps([]) }
    if conversation_id: payload['conversation_id'] = conversation_id
    response_data = _call_sb_api(payload)
    if response_data and response_data.get("ok") is True:
        logger.info(f"Telegram message acknowledged as successful by SB API for Chat ID {chat_id}")
        return True
    else:
        logger.error(f"Failed to send Telegram message via SB API for Chat ID {chat_id}. Response: {response_data}")
        return False

# --- REVISED PUBLIC FUNCTION: send_reply_to_channel ---
def send_reply_to_channel(conversation_id: str, message_text: str, source: Optional[str], target_user_id: str, conversation_details: Optional[Dict], triggering_message_id: Optional[str]) -> bool:
    if not message_text or not message_text.strip():
        logger.warning(f"Attempted to send empty reply to conversation {conversation_id}. Skipping.")
        return False

    effective_source = source.strip().lower() if isinstance(source, str) and source.strip() else 'web'
    logger.info(f"Routing reply for conversation {conversation_id} to user {target_user_id} via channel '{effective_source}'")

    external_success = False
    dm_bot_user_id = str(Config.SUPPORT_BOARD_DM_BOT_USER_ID) if Config.SUPPORT_BOARD_DM_BOT_USER_ID else None

    if effective_source == 'wa':
        recipient_waid = _get_user_waid(target_user_id)
        if recipient_waid: external_success = _send_whatsapp_cloud_api(recipient_waid, message_text)
    elif effective_source in ['fb', 'ig']:
        conv_details = conversation_details or get_sb_conversation_data(conversation_id)
        if conv_details:
            psid = _get_user_psid(target_user_id)
            page_id = conv_details.get('details', {}).get('extra')
            if psid and page_id: external_success = _send_messenger_message(psid, str(page_id), message_text, conversation_id, triggering_message_id)
    elif effective_source == 'tg':
        conv_details = conversation_details or get_sb_conversation_data(conversation_id)
        if conv_details:
            chat_id = conv_details.get('details', {}).get('extra')
            if chat_id: external_success = _send_telegram_message(str(chat_id), message_text, conversation_id)

    if external_success and dm_bot_user_id:
        _add_internal_sb_message(conversation_id=conversation_id, message_text=message_text, bot_user_id=dm_bot_user_id)
    
    return external_success

# --- RETAINED FOR BACKWARD COMPATIBILITY OR OTHER USES ---
def send_whatsapp_template_direct(user_id: str, conversation_id: str, template_variables: list[str]) -> bool:
    """Sends the standard order confirmation template to a user's WhatsApp."""
    template_name = "confirmacion_datos_cliente"
    template_language = "es_ES"
    recipient_waid = _get_user_waid(user_id)
    if not recipient_waid:
        logger.error(
            f"Cannot send template '{template_name}': Failed to derive WAID for user_id '{user_id}'."
        )
        return False

    ok = send_whatsapp_template_to_phone(
        to_phone_number=recipient_waid,
        template_name=template_name,
        template_language=template_language,
        parameters=template_variables,
    )

    if ok:
        try:
            log_text = f"📝 Plantilla '{template_name}' enviada a {recipient_waid}"
            _add_internal_sb_message(conversation_id, log_text, str(Config.SUPPORT_BOARD_DM_BOT_USER_ID))
        except Exception:
            logger.exception("Failed to log internal template send message")
    return ok


def send_whatsapp_template(to: str, template_name: str, template_languages: str, parameters: list[str], recipient_id: str) -> bool:
    """Legacy wrapper used in tests. Sends template via Support Board API."""
    payload = {
        "function": "messaging-platforms-send-template",
        "to": to,
        "template_name": template_name,
        "template_languages": template_languages,
        "parameters": json.dumps(["", ",".join(parameters), ""]),
        "recipient_id": recipient_id,
    }
    return _call_sb_api(payload) is not None

# --- NEW RELIABLE FUNCTION: Send WhatsApp Template to a raw phone number ---
def send_whatsapp_template_to_phone(to_phone_number: str, template_name: str, template_language: str, parameters: list[str]) -> bool:
    """Sends a WhatsApp template message directly via the Meta Cloud API to a specific phone number."""
    token = Config.WHATSAPP_CLOUD_API_TOKEN
    phone_number_id = Config.WHATSAPP_PHONE_NUMBER_ID
    api_version = Config.WHATSAPP_API_VERSION

    if not all([token, phone_number_id, to_phone_number]):
        logger.error("WhatsApp Cloud API credentials or target phone number are missing. Cannot send template.")
        return False
    
    recipient_waid = re.sub(r'\D', '', to_phone_number)
    default_cc = Config.WHATSAPP_DEFAULT_COUNTRY_CODE
    if default_cc and not recipient_waid.startswith(default_cc):
        recipient_waid = default_cc + recipient_waid

    api_url = f"https://graph.facebook.com/{api_version}/{phone_number_id}/messages"
    headers = { "Authorization": f"Bearer {token}", "Content-Type": "application/json" }
    
    body_parameter_objects = [{"type": "text", "text": str(p)} for p in parameters]
    components = [{"type": "body", "parameters": body_parameter_objects}]
    
    payload_dict = {
        "messaging_product": "whatsapp", "to": recipient_waid, "type": "template",
        "template": { "name": template_name, "language": {"code": template_language}, "components": components, },
    }

    logger.info(f"Attempting to send direct WhatsApp template '{template_name}' to WAID: ...{recipient_waid[-6:]}")
    logger.debug(f"Direct WhatsApp Template API Payload: {json.dumps(payload_dict)}")

    try:
        response = sb_api_session.post(api_url, headers=headers, json=payload_dict, timeout=30)
        response.raise_for_status()
        response_json = response.json()
        
        if "messages" in response_json and isinstance(response_json.get("messages"), list) and len(response_json["messages"]) > 0:
            logger.info(f"Direct WhatsApp template API call successful.")
            return True
        else:
            logger.error(f"Direct WhatsApp template API call returned unexpected success structure: {response_json}")
            return False
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Direct WhatsApp template API request error: {e}", exc_info=True)
        return False

# --- PUBLIC FUNCTION: Route Conversation to Sales ---
def route_conversation_to_sales(conversation_id: str) -> None:
    """Assign conversation to the Sales department and disable the bot."""
    sales_department_id = Config.SUPPORT_BOARD_SALES_DEPARTMENT_ID
    if not sales_department_id:
        logger.error(f"Cannot route conversation {conversation_id}: SUPPORT_BOARD_SALES_DEPARTMENT_ID not configured.")
        return

    logger.info(f"Routing conversation {conversation_id} to Sales Department ID {sales_department_id} and pausing bot.")
    _call_sb_api({
        "function": "update-conversation-department",
        "conversation_id": conversation_id,
        "department": sales_department_id,
    })
    _call_sb_api({
        "function": "sb-human-takeover",
        "conversation_id": conversation_id,
    })


# --- Helper: Fetch recent messages for a user ---
def get_recent_messages(user_id: str, limit: int = 10) -> List[Dict]:
    payload = {
        "function": "get-messages",
        "user_id": user_id,
        "limit": limit,
    }
    result = _call_sb_api(payload)
    return result if isinstance(result, list) else []


def time_window_now_minus(minutes: int) -> float:
    return time.time() - (minutes * 60)


def recent_message_contains_checkout_template(user_id: str, window_minutes: int = 10) -> bool:
    messages = get_recent_messages(user_id, limit=10)
    for msg in messages:
        if (
            msg.get("direction") == "out"
            and msg.get("channel") == "whatsapp"
            and "hemos recibido tu pedido" in msg.get("message", "").lower()
            and msg.get("timestamp", 0) >= time_window_now_minus(window_minutes)
        ):
            return True
    return False


def route_to_sales(user_id: str):
    sales_department_id = Config.SUPPORT_BOARD_SALES_DEPARTMENT_ID
    if not sales_department_id:
        logger.error("SUPPORT_BOARD_SALES_DEPARTMENT_ID not configured")
        return
    _call_sb_api({
        "function": "route-to-department",
        "user_id": user_id,
        "department_id": sales_department_id,
    })