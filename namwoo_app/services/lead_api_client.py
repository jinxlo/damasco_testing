# namwoo_app/services/lead_api_client.py
import requests
import json
from flask import current_app # Assuming your Flask app context is available from NamDamasco_Conversational_Core
                               # and config variables are loaded into current_app.config

# --- Private Helper Functions ---

def _get_api_headers() -> dict | None:
    """
    Constructs the necessary headers for API calls, including the API key.
    Returns None if the API key is not configured.
    """
    api_key = current_app.config.get('LEAD_CAPTURE_API_KEY')
    if not api_key:
        current_app.logger.error("LEAD_CAPTURE_API_KEY is not configured for Lead API Client.")
        return None
    
    headers = { # Storing in a variable for potential logging before returning
        "Content-Type": "application/json",
        "X-API-KEY": api_key
    }
    # current_app.logger.debug(f"Lead API Headers prepared: {json.dumps(headers)}") # Optional: log headers being prepared
    return headers

def _get_api_base_url() -> str | None:
    """
    Retrieves the base URL for the Lead Capture API service.
    Returns None if the URL is not configured.
    """
    base_url = current_app.config.get('LEAD_CAPTURE_API_URL')
    if not base_url:
        current_app.logger.error("LEAD_CAPTURE_API_URL is not configured for Lead API Client.")
        return None
    return base_url.rstrip('/') # Ensure no trailing slash for robust endpoint joining

# --- Public Client Functions ---

def call_initiate_lead_intent(
    conversation_id: str,
    products_of_interest: list, # List of dicts: [{"sku": "...", "description": "...", "quantity": ...}]
    payment_method_preference: str, # e.g., "direct_payment"
    platform_user_id: str = None,
    source_channel: str = None
) -> dict:
    """
    Calls the namdamasco_api service to create an initial lead intent.

    Args:
        conversation_id (str): The ID of the current conversation.
        products_of_interest (list): A list of product dictionaries. 
                                     Expected format: [{"sku": str, "description": str, "quantity": int}]
        payment_method_preference (str): The payment method chosen by the user.
        platform_user_id (str, optional): The user's ID on the messaging platform.
        source_channel (str, optional): The source channel of the conversation.

    Returns:
        dict: A dictionary with {"success": True/False, "data": response_json or None, "error_message": "..."}
              If successful, "data" will contain the response from the API, including the new "lead_id".
    """
    base_url = _get_api_base_url()
    headers = _get_api_headers()

    if not base_url or not headers:
        return {"success": False, "data": None, "error_message": "Lead API client is not properly configured (URL or Key missing)."}

    endpoint = f"{base_url}/leads/intent" # Corresponds to POST /api/v1/leads/intent in namdamasco_api
    payload = {
        "conversation_id": conversation_id, # ENSURE THIS IS THE DYNAMIC, CORRECT ID FROM THE CALLING FUNCTION
        "platform_user_id": platform_user_id,
        "source_channel": source_channel,
        "payment_method_preference": payment_method_preference,
        "products_of_interest": products_of_interest
    }
    
    current_app.logger.info(f"Calling Lead API (Initiate Intent): POST {endpoint}")
    current_app.logger.debug(f"Payload for POST {endpoint}: {json.dumps(payload)}")
    current_app.logger.debug(f"Headers for POST {endpoint}: {json.dumps(headers)}") # Added for debugging

    response = None # Initialize response to None for broader scope in error handling
    try:
        response = requests.post(endpoint, headers=headers, json=payload, timeout=10) # 10 second timeout
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        
        response_data = response.json()
        lead_id = response_data.get("id") # The namdamasco_api returns the created lead, which has an 'id'
        current_app.logger.info(f"Lead API (Initiate Intent) successful. Lead ID from API: {lead_id}. Full response: {response_data}")
        return {"success": True, "data": response_data, "error_message": None} # Pass full data back
    
    except requests.exceptions.HTTPError as http_err:
        error_details = "No response body"
        status_code_for_error = "N/A"
        if response is not None: # Check if response object exists
            status_code_for_error = response.status_code
            try:
                error_details = response.text # Try to get text, might fail if response is weird
            except Exception:
                current_app.logger.warning("Could not get text from HTTPError response.")
        
        current_app.logger.error(f"HTTP error calling Lead API (Initiate Intent): {http_err} - Status: {status_code_for_error} - Response: {error_details}")
        return {"success": False, "data": None, "error_message": f"API Error ({status_code_for_error}): {error_details}"}
    except requests.exceptions.RequestException as req_err:
        current_app.logger.error(f"Request exception calling Lead API (Initiate Intent): {req_err}")
        return {"success": False, "data": None, "error_message": f"Connection Error: {req_err}"}
    except Exception as e:
        current_app.logger.error(f"Unexpected error in call_initiate_lead_intent: {e}", exc_info=True)
        return {"success": False, "data": None, "error_message": "An unexpected error occurred creating lead intent."}


def call_submit_customer_details(
    lead_id: str,
    customer_full_name: str,
    customer_email: str,
    customer_phone_number: str,
    customer_cedula: str | None = None,
    customer_address: str | None = None,
    is_iva_retention_agent: bool | None = None,
) -> dict:
    """
    Calls the namdamasco_api service to submit/update customer contact details for an existing lead.

    Args:
        lead_id (str): The UUID of the lead to update (obtained from initiate_lead_intent).
        customer_full_name (str): Customer's full name.
        customer_email (str): Customer's email.
        customer_phone_number (str): Customer's phone number.
        customer_cedula (str, optional): Documento de identificación del cliente.
        customer_address (str, optional): Dirección del cliente.
        is_iva_retention_agent (bool, optional): Indica si el cliente es agente de retención IVA.

    Returns:
        dict: A dictionary with {"success": True/False, "data": response_json or None, "error_message": "..."}
    """
    base_url = _get_api_base_url()
    headers = _get_api_headers()

    if not base_url or not headers:
        return {"success": False, "data": None, "error_message": "Lead API client not configured (URL or Key missing)."}
    if not lead_id:
        current_app.logger.error("call_submit_customer_details called without a lead_id.")
        return {"success": False, "data": None, "error_message": "Lead ID is required to submit customer details."}

    endpoint = f"{base_url}/leads/{lead_id}/customer-details" # Corresponds to PUT /api/v1/leads/{id}/customer-details
    payload = {
        "customer_full_name": customer_full_name,
        "customer_email": customer_email,
        "customer_phone_number": customer_phone_number,
        "customer_cedula": customer_cedula,
        "customer_address": customer_address,
        "is_iva_retention_agent": is_iva_retention_agent,
    }

    current_app.logger.info(f"Calling Lead API (Submit Details): PUT {endpoint}")
    current_app.logger.debug(f"Payload for PUT {endpoint}: {json.dumps(payload)}")
    current_app.logger.debug(f"Headers for PUT {endpoint}: {json.dumps(headers)}") # Added for debugging

    response = None # Initialize response to None
    try:
        response = requests.put(endpoint, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        response_data = response.json()
        current_app.logger.info(f"Lead API (Submit Details) successful for lead {lead_id}. Response: {response_data}")
        return {"success": True, "data": response_data, "error_message": None}

    except requests.exceptions.HTTPError as http_err:
        error_details = "No response body"
        status_code_for_error = "N/A"
        if response is not None:
            status_code_for_error = response.status_code
            try:
                error_details = response.text
            except Exception:
                current_app.logger.warning("Could not get text from HTTPError response.")

        current_app.logger.error(f"HTTP error calling Lead API (Submit Details) for lead {lead_id}: {http_err} - Status: {status_code_for_error} - Response: {error_details}")
        return {"success": False, "data": None, "error_message": f"API Error ({status_code_for_error}): {error_details}"}
    except requests.exceptions.RequestException as req_err:
        current_app.logger.error(f"Request exception calling Lead API (Submit Details) for lead {lead_id}: {req_err}")
        return {"success": False, "data": None, "error_message": f"Connection Error: {req_err}"}
    except Exception as e:
        current_app.logger.error(f"Unexpected error in call_submit_customer_details for lead {lead_id}: {e}", exc_info=True)
        return {"success": False, "data": None, "error_message": "An unexpected error occurred submitting customer details."}