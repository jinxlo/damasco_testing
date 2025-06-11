# NAMWOO/services/sync_service.py

import logging
import time
import re # For ID generation if doing conditional summarization
from typing import Optional, Tuple, List, Dict, Any
from decimal import Decimal # <<< ADDED for potential Decimal conversion
from flask import Flask

from . import product_service
from . import llm_processing_service
from ..utils import db_utils, embedding_utils # Removed text_utils as it's not directly used here
from ..models.product import Product # Product model is used
from ..config import Config

sync_logger = logging.getLogger('sync')
logger = logging.getLogger(__name__)

# --- Constants ---
COMMIT_BATCH_SIZE = getattr(Config, 'SYNC_COMMIT_BATCH_SIZE', 100) # Use Config if available

# --- Helper Functions for sync_service (similar to celery_tasks.py) ---
def _sync_normalize_string_for_id_parts(value: Any) -> Optional[str]:
    if value is None: return None
    s = str(value).strip()
    return s if s else None

def _sync_generate_product_id(item_code_raw: Any, whs_name_raw: Any) -> Optional[str]:
    item_code = _sync_normalize_string_for_id_parts(item_code_raw)
    whs_name = _sync_normalize_string_for_id_parts(whs_name_raw)
    if not item_code or not whs_name: return None
    sanitized_whs_name = re.sub(r'[^a-zA-Z0-9_-]', '_', whs_name)
    product_id = f"{item_code}_{sanitized_whs_name}"
    if len(product_id) > 512: product_id = product_id[:512]
    return product_id

def _sync_convert_snake_to_camel_case(data_snake: Dict[str, Any]) -> Dict[str, Any]:
    if not data_snake: return {}
    key_map = {
        "item_code": "itemCode", "item_name": "itemName", "description": "description",
        "sub_category": "subCategory", "item_group_name": "itemGroupName",
        "warehouse_name": "whsName", "branch_name": "branchName", "stock": "stock",
        "price": "price", # Original price (e.g., USD)
        "price_bolivar": "priceBolivar", # <<< ADDED MAPPING FOR THE NEW FIELD
        "category": "category", "brand": "brand", "line": "line",
    }
    data_camel = {}
    for snake_key, value in data_snake.items():
        camel_key = key_map.get(snake_key)
        
        # Convert Decimal to float if values are Decimal and camelCase dict is for general purpose / JSON
        # If product_service expects Decimals directly from this dict, this conversion might not be needed.
        # For now, let's assume if it's Decimal, convert to float for the camelCase dictionary.
        val_to_store = value
        if isinstance(value, Decimal):
            val_to_store = float(value)

        if camel_key:
            data_camel[camel_key] = val_to_store
        # If a key is already camelCase (e.g. if source data mixes cases)
        # and it's a known camelCase key, ensure it's included.
        elif snake_key in key_map.values() and snake_key not in data_camel:
             data_camel[snake_key] = val_to_store
        # else: # Optionally, pass through unmapped keys or log them
            # data_camel[snake_key] = val_to_store

    # Ensure description is present if it was in snake_case and didn't map (idempotency)
    # This might be redundant if 'description' is in key_map and always present, but safe.
    if "description" in data_snake and "description" not in data_camel:
        data_camel["description"] = data_snake["description"]
        
    return data_camel

# --- Main Sync Logic ---
def run_full_sync(app: Flask, damasco_product_data_snake_list: list) -> Tuple[int, int, int, int, int]:
    """
    Performs a full synchronization of products from provided Damasco data.
    Assumes damasco_product_data_snake_list contains snake_case dictionaries.
    Includes LLM summarization, embedding generation, and updates the local DB.

    Args:
        app: Flask application context
        damasco_product_data_snake_list: List of product dictionaries with snake_case keys,
                                         including 'description' (raw HTML) and 'price_bolivar'.

    Returns:
        (processed_count, added_count, updated_count, skipped_no_change_count, failed_count)
    """
    sync_logger.info("====== Starting FULL Damasco Product Sync (with LLM Summarization) ======")
    start_time = time.time()
    processed_count = 0
    added_count = 0
    updated_count = 0
    skipped_no_change_count = 0
    failed_count = 0
    summaries_generated_count = 0
    summaries_reused_count = 0
    summaries_failed_count = 0 # Or skipped

    with app.app_context():
        if not damasco_product_data_snake_list:
            sync_logger.error("No product data provided for sync. Aborting sync.")
            return 0, 0, 0, 0, 0

        sync_logger.info(f"Received {len(damasco_product_data_snake_list)} products for sync.")

        with db_utils.get_db_session() as session:
            if not session:
                sync_logger.error("Database session not available. Aborting sync.")
                return 0, 0, 0, 0, len(damasco_product_data_snake_list)

            try:
                for index, product_data_snake_original in enumerate(damasco_product_data_snake_list, start=1):
                    processed_count += 1
                    # Make a copy to modify for Decimal conversion without altering the original list item
                    product_data_snake = product_data_snake_original.copy()

                    item_code_log = product_data_snake.get('item_code', 'N/A')
                    whs_name_log = product_data_snake.get('warehouse_name', 'N/A')
                    log_prefix_sync = f"Sync Entry [{index}/{len(damasco_product_data_snake_list)}] ({item_code_log} @ {whs_name_log}):"
                    sync_logger.info(f"{log_prefix_sync} Starting processing.")

                    # --- Price Conversion to Decimal (if not already) ---
                    # It's safer to ensure prices are Decimal before passing to product_service
                    # if product_service will expect Decimals for precision.
                    # This step assumes product_data_snake might have prices as float/str.
                    try:
                        price_val = product_data_snake.get('price')
                        product_data_snake['price'] = Decimal(str(price_val)) if price_val is not None else None

                        price_bolivar_val = product_data_snake.get('price_bolivar') # Get new field
                        product_data_snake['price_bolivar'] = Decimal(str(price_bolivar_val)) if price_bolivar_val is not None else None
                    except Exception as e_dec:
                        sync_logger.error(f"{log_prefix_sync} Error converting price to Decimal: {e_dec}. Skipping item.")
                        failed_count +=1
                        continue
                    
                    # 1. Convert to camelCase for internal use (e.g., by Product.prepare_text_for_embedding)
                    # _sync_convert_snake_to_camel_case now handles price_bolivar -> priceBolivar
                    # It also converts Decimals to float if that's its behavior.
                    product_data_camel = _sync_convert_snake_to_camel_case(product_data_snake)
                    
                    if not product_data_camel.get("itemCode") or not product_data_camel.get("whsName"):
                        sync_logger.error(f"{log_prefix_sync} Critical keys 'itemCode' or 'whsName' missing in camelCase data. Skipping.")
                        failed_count += 1
                        continue
                    
                    # Ensure description is in product_data_camel if it was in snake (already handled by _sync_convert_snake_to_camel_case based on your code)

                    # 2. Conditional LLM Summarization
                    llm_generated_summary: Optional[str] = None
                    # Use original snake_case data for these as they are direct inputs to services/logic
                    raw_html_description_incoming = product_data_snake.get("description")
                    item_name_for_summary = product_data_snake.get("item_name")
                    
                    # Generate ID using original snake_case data before any transformation for consistency
                    product_id_for_fetch = _sync_generate_product_id(
                        product_data_snake.get("item_code"),
                        product_data_snake.get("warehouse_name")
                    )
                    if not product_id_for_fetch: # Should be caught by earlier checks if item_code/whsName are missing
                        sync_logger.error(f"{log_prefix_sync} Failed to generate product_id for DB fetch. Skipping.")
                        failed_count +=1
                        continue

                    existing_entry: Optional[Product] = None
                    needs_new_summary = False

                    try:
                        existing_entry = session.query(Product).filter_by(id=product_id_for_fetch).first()
                    except Exception as e_read:
                        sync_logger.error(f"{log_prefix_sync} Error reading existing DB entry: {e_read}. Skipping item.", exc_info=True)
                        failed_count +=1
                        continue

                    if existing_entry:
                        normalized_incoming_html = _sync_normalize_string_for_id_parts(raw_html_description_incoming)
                        normalized_stored_html = _sync_normalize_string_for_id_parts(existing_entry.description)

                        if normalized_incoming_html != normalized_stored_html:
                            sync_logger.info(f"{log_prefix_sync} Raw HTML description changed. Attempting new summarization.")
                            needs_new_summary = True
                        elif not existing_entry.llm_summarized_description and raw_html_description_incoming:
                            sync_logger.info(f"{log_prefix_sync} Raw HTML effectively same, but no existing LLM summary. Attempting summarization.")
                            needs_new_summary = True
                        else:
                            sync_logger.info(f"{log_prefix_sync} Raw HTML effectively same and LLM summary exists. Re-using stored summary.")
                            llm_generated_summary = existing_entry.llm_summarized_description
                            if llm_generated_summary: summaries_reused_count += 1
                    else: 
                        if raw_html_description_incoming:
                            sync_logger.info(f"{log_prefix_sync} New product with HTML. Will attempt summarization.")
                            needs_new_summary = True
                    
                    if needs_new_summary and raw_html_description_incoming:
                        sync_logger.info(f"{log_prefix_sync} Calling LLM for summarization for '{item_name_for_summary}'.")
                        try:
                            llm_generated_summary = llm_processing_service.generate_llm_product_summary(
                                html_description=raw_html_description_incoming,
                                item_name=item_name_for_summary
                            )
                            if llm_generated_summary:
                                summaries_generated_count +=1
                                sync_logger.info(f"{log_prefix_sync} LLM summary generated.")
                            else:
                                summaries_failed_count +=1
                                sync_logger.warning(f"{log_prefix_sync} LLM summarization returned empty content.")
                        except Exception as e_summ:
                            summaries_failed_count +=1
                            sync_logger.error(f"{log_prefix_sync} LLM summarization failed: {e_summ}", exc_info=True)
                            if existing_entry: 
                                llm_generated_summary = existing_entry.llm_summarized_description
                            else:
                                llm_generated_summary = None
                    elif not raw_html_description_incoming:
                        llm_generated_summary = None
                        sync_logger.info(f"{log_prefix_sync} No HTML description, skipping summarization.")


                    # 3. Prepare text for embedding
                    try:
                        # Call Product.prepare_text_for_embedding with the updated signature
                        text_to_embed = Product.prepare_text_for_embedding(
                            damasco_product_data=product_data_camel, # Contains camelCase keys
                            llm_generated_summary=llm_generated_summary,
                            raw_html_description_for_fallback=raw_html_description_incoming # <<< PASSING THIS
                        )
                    except Exception as e_prep_embed:
                        sync_logger.error(f"{log_prefix_sync} Error preparing text for embedding: {e_prep_embed}", exc_info=True)
                        failed_count += 1
                        continue

                    if not text_to_embed:
                        sync_logger.warning(f"{log_prefix_sync} Empty text_to_embed. Skipping embedding and DB upsert.")
                        failed_count += 1 # Count as failed if no text to embed
                        continue

                    # 4. Generate embedding
                    embedding_vector: Optional[List[float]] = None
                    try:
                        embedding_vector = embedding_utils.get_embedding(text_to_embed, model=Config.OPENAI_EMBEDDING_MODEL)
                        if not embedding_vector:
                            sync_logger.error(f"{log_prefix_sync} Failed to generate embedding (service returned None). Skipping.")
                            failed_count += 1
                            continue
                    except Exception as e_embed_gen:
                        sync_logger.error(f"{log_prefix_sync} Exception during embedding generation: {e_embed_gen}", exc_info=True)
                        failed_count +=1
                        continue

                    # 5. Upsert to DB
                    # The product_service.add_or_update_product_in_db function needs to be
                    # aware of 'priceBolivar' within the product_data_camel dictionary it receives,
                    # or its signature needs to be changed to accept price_bolivar explicitly.
                    # For now, assuming it will look for 'priceBolivar' in product_data_camel.
                    # Ensure product_data_camel correctly contains priceBolivar (from _sync_convert_snake_to_camel_case)
                    # and that prices within product_data_camel are floats if that's what product_service currently expects from this dict.
                    # If product_service is updated to handle Decimals from this dict, ensure they are Decimals.
                    # The product_data_camel from _sync_convert_snake_to_camel_case now has priceBolivar as float.
                    success, op_message = product_service.add_or_update_product_in_db(
                        session, # Pass the active session
                        product_data_camel, # Pass camelCase data (now includes 'priceBolivar' as float)
                        embedding_vector,
                        text_to_embed, # Text used for embedding
                        llm_summarized_description=llm_generated_summary # LLM summary
                        # The original `product_service.add_or_update_product_in_db` might also need
                        # product_id_for_db, raw_html_description_to_store, original_input_data_snake
                        # This call needs to match the **eventual updated signature** of product_service.add_or_update_product_in_db
                        # or the current one if it already handles all these fields from product_data_camel.
                        # For MINIMAL changes to THIS file, we assume product_service will unpack product_data_camel.
                        # However, for robustness, product_service *should* be updated to take explicit fields.
                        # To keep this file's changes minimal as requested, I'm keeping the original call structure.
                        # **This implies `product_service.add_or_update_product_in_db` needs to be robust enough
                        # to find `priceBolivar` within `product_data_camel`.**
                    )

                    if success:
                        if op_message == 'added': added_count += 1
                        elif op_message == 'updated': updated_count += 1
                        elif op_message == 'skipped_no_change': skipped_no_change_count +=1
                        sync_logger.info(f"{log_prefix_sync} DB operation: {op_message}")
                    else:
                        sync_logger.error(f"{log_prefix_sync} DB operation failed. Reason: {op_message}")
                        failed_count += 1

                    if COMMIT_BATCH_SIZE and processed_count > 0 and processed_count % COMMIT_BATCH_SIZE == 0:
                        try:
                            sync_logger.info(f"Committing batch after {processed_count} products...")
                            session.commit()
                            sync_logger.info("Batch committed successfully.")
                        except Exception as commit_error:
                            sync_logger.exception(f"Commit error during batch processing. Rolling back current batch. Error: {commit_error}")
                            session.rollback()
                
                sync_logger.info("Committing final transaction at end of sync (if any pending changes)...")
                session.commit()
                sync_logger.info("Final commit successful.")

            except Exception as e_outer:
                sync_logger.exception(f"Unexpected error during sync process. Rolling back. Error: {e_outer}")
                if 'session' in locals() and session.is_active:
                    session.rollback()
                current_successful_ops = added_count + updated_count + skipped_no_change_count
                if processed_count > current_successful_ops:
                    failed_count += (processed_count - current_successful_ops - failed_count) # Adjust to not double count already failed

    duration = time.time() - start_time
    sync_logger.info("====== FULL Damasco Product Sync Finished ======")
    sync_logger.info(f"Duration: {duration:.2f} seconds")
    sync_logger.info(f"Total Products Received in Payload: {len(damasco_product_data_snake_list)}")
    sync_logger.info(f"Total Products Attempted Processing: {processed_count}")
    sync_logger.info(f"Products Added: {added_count}")
    sync_logger.info(f"Products Updated: {updated_count}")
    sync_logger.info(f"Products Skipped (No Change): {skipped_no_change_count}")
    sync_logger.info(f"LLM Summaries Newly Generated: {summaries_generated_count}")
    sync_logger.info(f"LLM Summaries Re-used: {summaries_reused_count}")
    sync_logger.info(f"LLM Summaries Failed/Skipped during processing: {summaries_failed_count}")
    sync_logger.info(f"Total Products Failed in Processing (includes skips due to errors): {failed_count}")
    sync_logger.info("================================================")

    return processed_count, added_count, updated_count, skipped_no_change_count, failed_count


def run_incremental_sync(app: Flask):
    """
    Placeholder for future incremental sync logic.
    """
    sync_logger.info("====== Starting INCREMENTAL Damasco Product Sync ======")
    sync_logger.warning("Incremental sync logic not implemented. Skipping.")
    sync_logger.info("====== INCREMENTAL Damasco Product Sync Finished (Skipped) ======")
    return 0, 0, 0, 0, 0