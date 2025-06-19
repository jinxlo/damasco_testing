# NAMWOO/services/product_service.py

import logging
import re
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from decimal import InvalidOperation as InvalidDecimalOperation
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sqlalchemy import func, literal_column, distinct, or_, and_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import Config
from ..models.product import Product
from ..utils import db_utils, embedding_utils, product_utils
from ..utils.whs_utils import canonicalize_whs, canonicalize_whs_name

logger = logging.getLogger(__name__)

# --- Semantic Helpers ---
_ACCESSORY_PAT = re.compile(
    r"(base para|soporte|mount|stand|bracket|control(?: remoto)?|adaptador|"
    r"compresor|enfriador|deshumidificador|case|funda)",
    flags=re.I,
)
_MAIN_TYPE_PAT = re.compile(
    r"\b(tv|televisor|pantalla|nevera|refrigerador|aire acondicionado|"
    r"lavadora|secadora|freidora|microondas|horno)\b",
    flags=re.I,
)
# List of known brands for explicit filtering
KNOWN_BRANDS = {'SAMSUNG', 'TECNO', 'XIAOMI', 'INFINIX', 'DAMASCO'}


def _is_accessory(name: str) -> bool:
    if not name:
        return False
    return bool(_ACCESSORY_PAT.search(name))


def _extract_main_type(text: str) -> str:
    if not text:
        return ""
    m = _MAIN_TYPE_PAT.search(text)
    return m.group(0).lower() if m else ""

def _extract_color_from_name(item_name: str) -> Optional[str]:
    """Extract a color from a product name using the shared utility."""
    if not item_name:
        return None
    # Delegate to the robust implementation in product_utils for consistency.
    return product_utils._extract_color_from_name(item_name)


# --- Search Products ---
def search_local_products(
    query_text: str,
    limit: int = 20,
    filter_stock: bool = True,
    min_score: float = 0.35,
    warehouse_names: Optional[List[str]] = None,
    min_price: Optional[Union[float, int, str]] = None,
    max_price: Optional[Union[float, int, str]] = None,
    sort_by: Optional[str] = None,
    exclude_accessories: bool = True,
    required_specs: Optional[List[str]] = None,
    exclude_skus: Optional[List[str]] = None,
    is_list_query: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    if not query_text or not isinstance(query_text, str):
        logger.warning("Search query is empty or invalid.")
        return []

    log_message_parts = [
        f"Search initiated: '{query_text[:80]}…'",
        f"limit={limit}", f"is_list_query={is_list_query}",
    ]
    if warehouse_names:
        log_message_parts.append(f"warehouses={warehouse_names}")
    
    logger.info(", ".join(log_message_parts))

    # For list queries, we don't need a vector embedding, but create a dummy one to prevent errors.
    if is_list_query:
        query_emb = [0.0] * (Config.EMBEDDING_DIMENSION or 1536)
    else:
        embedding_model = (Config.OPENAI_EMBEDDING_MODEL if hasattr(Config, "OPENAI_EMBEDDING_MODEL") else "text-embedding-3-small")
        query_emb = embedding_utils.get_embedding(query_text, model=embedding_model)
        if not query_emb:
            logger.error("Query embedding generation failed – aborting search.")
            return None

    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for search.")
            return None
        try:
            base_q = session.query(
                Product,
                (Product.embedding.cosine_distance(query_emb) if not is_list_query else literal_column("'0'")).label("distance"),
            )

            if filter_stock:
                base_q = base_q.filter(Product.stock > 0)
            
            canonical_whs = [canonicalize_whs(w) for w in warehouse_names if w] if warehouse_names else None
            if canonical_whs:
                base_q = base_q.filter(Product.warehouse_name_canonical.in_(canonical_whs))

            if exclude_skus:
                base_q = base_q.filter(Product.item_code.notin_(exclude_skus))
            
            if exclude_accessories:
                base_q = base_q.filter(Product.sub_category.notilike('%ACCESORIO%'))
                base_q = base_q.filter(Product.category.notilike('%ACCESORIO%'))

            detected_brands = [brand for brand in KNOWN_BRANDS if brand.lower() in query_text.lower()]
            if detected_brands:
                brand_clauses = [Product.brand.ilike(f'%{b}%') for b in detected_brands]
                base_q = base_q.filter(or_(*brand_clauses))
                logger.info(f"Applying strict brand filter for: {detected_brands}")

            if min_price is not None:
                base_q = base_q.filter(Product.price >= min_price)
            if max_price is not None:
                base_q = base_q.filter(Product.price <= max_price)

            partition_order_by = (
                Product.price.asc() if is_list_query 
                else Product.embedding.cosine_distance(query_emb).asc()
            )
            
            ranked_subquery = base_q.add_column(
                func.row_number().over(
                    partition_by=Product.item_code,
                    order_by=partition_order_by
                ).label('rn')
            ).subquery('ranked_products')

            unique_candidates_q = session.query(ranked_subquery).filter(ranked_subquery.c.rn == 1)

            if is_list_query:
                unique_candidates_q = unique_candidates_q.order_by(ranked_subquery.c.item_name.asc())
            elif sort_by == "price_asc":
                unique_candidates_q = unique_candidates_q.order_by(ranked_subquery.c.price.asc())
            elif sort_by == "price_desc":
                unique_candidates_q = unique_candidates_q.order_by(ranked_subquery.c.price.desc())
            else:
                unique_candidates_q = unique_candidates_q.order_by(ranked_subquery.c.distance.asc())

            candidate_rows = unique_candidates_q.limit(limit * 3).all()
            logger.info(f"Stage 1 Search: Found {len(candidate_rows)} unique candidates.")

            if required_specs:
                filtered_candidates = []
                for row in candidate_rows:
                    authoritative_text = f"{row.item_name or ''} {row.especificacion or ''} {row.description or ''}".lower()
                    
                    all_specs_met = True
                    for spec in required_specs:
                        spec_lower = spec.lower()
                        spec_nums = re.findall(r'\d+', spec_lower)

                        if not spec_nums:
                            if spec_lower not in authoritative_text:
                                all_specs_met = False
                                break
                            continue

                        num_found_in_text = False
                        for num in spec_nums:
                            if any(token in spec_lower for token in ["gb", "ram", "memoria"]):
                                mem_pattern = re.compile(
                                    rf"(?:\b{num}\s*(?:gb|g|g\s*ram)?(?:\s*(?:ram|memoria))?|\b(?:ram|memoria)\s*:?\s*{num}(?:\s*gb)?|\b\d+\+{num}\b)",
                                    re.I,
                                )
                                if mem_pattern.search(authoritative_text):
                                    num_found_in_text = True
                                    break
                            else:
                                general_pattern = re.compile(rf"(?<!\d){num}(?!\d)")
                                if general_pattern.search(authoritative_text):
                                    num_found_in_text = True
                                    break

                        if not num_found_in_text:
                            all_specs_met = False
                            break
                    
                    if all_specs_met:
                        filtered_candidates.append(row)

                logger.info(f"Post-filtered candidates down to {len(filtered_candidates)} based on specs: {required_specs}")
                candidate_rows = filtered_candidates

            enriched_results = []
            for row in candidate_rows[:limit]: # Apply final limit after filtering
                representative_product_dict = row._asdict()
                similarity = 1 - float(representative_product_dict['distance'])
                
                if not is_list_query and similarity < min_score:
                    continue

                base_model_name = product_utils.get_base_model_name(representative_product_dict['item_name'])

                variants_and_locations_q = session.query(
                    Product.item_name, Product.branch_name
                ).filter(
                    Product.item_name.ilike(f"{base_model_name}%"),
                    Product.stock > 0
                ).distinct()

                if canonical_whs:
                    variants_and_locations_q = variants_and_locations_q.filter(Product.warehouse_name_canonical.in_(canonical_whs))

                available_variants_info = variants_and_locations_q.all()
                if not available_variants_info:
                    continue

                all_colors = set()
                all_locations = set()
                for variant_info in available_variants_info:
                    color = _extract_color_from_name(variant_info.item_name)
                    if color: all_colors.add(color)
                    if variant_info.branch_name: all_locations.add(variant_info.branch_name)

                final_product_dict = representative_product_dict
                final_product_dict['similarity'] = round(similarity, 4)
                final_product_dict['available_in_stores'] = sorted(list(all_locations))
                final_product_dict['available_colors'] = sorted(list(all_colors)) if all_colors else ["No especificado"]
                final_product_dict['base_model_name'] = base_model_name

                enriched_results.append(final_product_dict)

            logger.info(f"[Product Search] Final enriched results returned: {len(enriched_results)}")
            return enriched_results

        except SQLAlchemyError as db_exc:
            logger.exception("Database error during product search: %s", db_exc)
            return None
        except Exception as exc:
            logger.exception("Unexpected error during product search: %s", exc)
            return None


def _normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s if s else None


def get_product_by_id_from_db(
    db_session: Session, product_id: str
) -> Optional[Product]:
    """Helper to fetch a product by its composite ID."""
    if not product_id:
        return None
    return db_session.query(Product).filter(Product.id == product_id).first()


def add_or_update_product_in_db(
    session: Session,
    damasco_product_data_camel: Dict[str, Any],
    embedding_vector: Optional[Any],
    text_used_for_embedding: Optional[str],
    llm_summarized_description_to_store: Optional[str],
) -> Tuple[bool, str]:

    item_code = _normalize_string(damasco_product_data_camel.get("itemCode"))
    whs_name_raw = _normalize_string(damasco_product_data_camel.get("whsName"))
    if not item_code or not whs_name_raw:
        logger.error("Missing itemCode or whsName; cannot upsert product.")
        return False, "missing_item_or_whs"
    canonical_whs_name = canonicalize_whs_name(whs_name_raw) or whs_name_raw
    whs_canonical = canonicalize_whs(canonical_whs_name)

    if not damasco_product_data_camel or not isinstance(
        damasco_product_data_camel, dict
    ):
        return False, "Missing or invalid Damasco product data (camelCase)."

    embedding_vector_for_db: Optional[List[float]] = None
    if embedding_vector is not None:
        if isinstance(embedding_vector, np.ndarray):
            embedding_vector_for_db = embedding_vector.tolist()
        elif isinstance(embedding_vector, list):
            embedding_vector_for_db = embedding_vector
        else:
            logger.error(
                f"Product upsert {item_code}@{whs_canonical}: Unexpected embedding vector type ({type(embedding_vector)})."
            )
            return False, "invalid_embedding_type"

        expected_dim = (
            Config.EMBEDDING_DIMENSION
            if hasattr(Config, "EMBEDDING_DIMENSION") and Config.EMBEDDING_DIMENSION
            else None
        )
        if (
            expected_dim
            and embedding_vector_for_db
            and len(embedding_vector_for_db) != expected_dim
        ):
            return False, (
                f"Embedding dimension mismatch for {item_code}@{whs_canonical} (expected {expected_dim}, got {len(embedding_vector_for_db)})."
            )

    if embedding_vector_for_db is not None and (
        not text_used_for_embedding or not isinstance(text_used_for_embedding, str)
    ):
        logger.warning(
            f"Product {item_code}@{whs_canonical}: Embedding vector present, but text_used_for_embedding is missing or invalid."
        )

    log_prefix = f"ProductService DB Upsert ({item_code}@{whs_canonical}):"

    raw_html_description_to_store = damasco_product_data_camel.get("description")
    item_name = _normalize_string(damasco_product_data_camel.get("itemName"))

    price_from_damasco = damasco_product_data_camel.get("price")
    normalized_price_for_db: Optional[Decimal] = None
    if price_from_damasco is not None:
        try:
            normalized_price_for_db = Decimal(str(price_from_damasco)).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        except InvalidDecimalOperation:
            logger.warning(
                f"{log_prefix} Invalid price value '{price_from_damasco}' for USD price, treating as None."
            )

    price_bolivar_from_damasco = damasco_product_data_camel.get("priceBolivar")
    normalized_price_bolivar_for_db: Optional[Decimal] = None
    if price_bolivar_from_damasco is not None:
        try:
            normalized_price_bolivar_for_db = Decimal(
                str(price_bolivar_from_damasco)
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        except InvalidDecimalOperation:
            logger.warning(
                f"{log_prefix} Invalid priceBolivar value '{price_bolivar_from_damasco}', treating as None."
            )

    stock_from_damasco = damasco_product_data_camel.get("stock")
    normalized_stock_for_db = 0
    if stock_from_damasco is not None:
        try:
            normalized_stock_for_db = int(stock_from_damasco)
            if normalized_stock_for_db < 0:
                logger.warning(
                    f"{log_prefix} Negative stock value '{stock_from_damasco}' received, setting to 0."
                )
                normalized_stock_for_db = 0
        except (ValueError, TypeError):
            logger.warning(
                f"{log_prefix} Invalid stock value '{stock_from_damasco}', defaulting to 0."
            )

    norm_raw_html = _normalize_string(raw_html_description_to_store)
    norm_llm_summary = _normalize_string(llm_summarized_description_to_store)
    norm_searchable_text = _normalize_string(text_used_for_embedding)
    
    especificacion_value = (
        damasco_product_data_camel.get("specifitacion")
        or damasco_product_data_camel.get("especificacion")
    )

    new_values_map = {
        "item_code": item_code,
        "item_name": item_name,
        "description": norm_raw_html,
        "llm_summarized_description": norm_llm_summary,
        "category": _normalize_string(damasco_product_data_camel.get("category")),
        "sub_category": _normalize_string(
            damasco_product_data_camel.get("subCategory")
        ),
        "brand": _normalize_string(damasco_product_data_camel.get("brand")),
        "line": _normalize_string(damasco_product_data_camel.get("line")),
        "item_group_name": _normalize_string(
            damasco_product_data_camel.get("itemGroupName")
        ),
        "especificacion": _normalize_string(especificacion_value),
        "warehouse_name": canonical_whs_name,
        "branch_name": _normalize_string(damasco_product_data_camel.get("branchName")),
        "price": normalized_price_for_db,
        "price_bolivar": normalized_price_bolivar_for_db,
        "stock": normalized_stock_for_db,
        "searchable_text_content": norm_searchable_text,
        "embedding": embedding_vector_for_db,
        "source_data_json": damasco_product_data_camel,
    }
    product_id = product_utils.generate_product_location_id(item_code, canonical_whs_name)
    if not product_id:
        logger.error(f"{log_prefix} Failed to generate product ID.")
        return False, "invalid_generated_id"

    new_values_map.update({
        "id": product_id,
        "warehouse_name_canonical": whs_canonical,
    })

    try:
        insert_stmt = insert(Product).values(**new_values_map)

        update_set = {
            "item_name": insert_stmt.excluded.item_name,
            "description": insert_stmt.excluded.description,
            "llm_summarized_description": insert_stmt.excluded.llm_summarized_description,
            "category": insert_stmt.excluded.category,
            "sub_category": insert_stmt.excluded.sub_category,
            "brand": insert_stmt.excluded.brand,
            "line": insert_stmt.excluded.line,
            "item_group_name": insert_stmt.excluded.item_group_name,
            "especificacion": insert_stmt.excluded.especificacion,
            "warehouse_name": insert_stmt.excluded.warehouse_name,
            "branch_name": insert_stmt.excluded.branch_name,
            "price": insert_stmt.excluded.price,
            "price_bolivar": insert_stmt.excluded.price_bolivar,
            "stock": insert_stmt.excluded.stock,
            "searchable_text_content": insert_stmt.excluded.searchable_text_content,
            "embedding": insert_stmt.excluded.embedding,
            "source_data_json": insert_stmt.excluded.source_data_json,
            "updated_at": func.now(),
        }

        upsert_stmt = (
            insert_stmt.on_conflict_do_update(
                constraint="uq_item_code_per_whs_canonical",
                set_=update_set,
            )
            .returning(literal_column("xmax"))
        )

        result = session.execute(upsert_stmt)
        session.commit()

        row = result.fetchone()
        operation = "updated" if row and row[0] != 0 else "inserted"
        logger.info(f"{log_prefix} Upsert completed. Operation: {operation}.")
        return True, operation

    except SQLAlchemyError as db_exc:
        session.rollback()
        logger.error(
            f"{log_prefix} DB error during add/update: {db_exc}", exc_info=True
        )
        err_str = str(db_exc).lower()
        if "violates unique constraint" in err_str:
            return False, f"db_constraint_violation: {str(db_exc)[:200]}"
        return False, f"db_sqlalchemy_error: {str(db_exc)[:200]}"
    except Exception as exc:
        session.rollback()
        logger.exception(f"{log_prefix} Unexpected error processing: {exc}")
        return False, f"db_unexpected_error: {str(exc)[:200]}"


# --- Getter functions ---
def get_live_product_details_by_sku(
    item_code_query: str,
    warehouse_names: Optional[List[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    if not item_code_query:
        logger.error(
            "get_live_product_details_by_sku: Missing item_code_query argument."
        )
        return None
    normalized_item_code = _normalize_string(item_code_query)
    if not normalized_item_code:
        logger.warning(
            f"get_live_product_details_by_sku: item_code_query '{item_code_query}' normalized to empty/None."
        )
        return []
    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_live_product_details_by_sku.")
            return None
        try:
            query = session.query(Product).filter(Product.item_code.ilike(normalized_item_code))
            if warehouse_names:
                canonical_whs = [canonicalize_whs(w) for w in warehouse_names if w]
                if canonical_whs:
                    query = query.filter(Product.warehouse_name_canonical.in_(canonical_whs))
                    logger.info(
                        f"Filtering live details for SKU {normalized_item_code} to warehouses: {canonical_whs}"
                    )
            product_entries = query.all()
            if not product_entries:
                logger.info(
                    f"No product entries found with item_code: {normalized_item_code}"
                )
                return []
            results = [entry.to_dict(include_source=True) for entry in product_entries]
            logger.info(
                f"Found {len(results)} locations for item_code: {normalized_item_code}"
            )
            return results
        except SQLAlchemyError as db_exc:
            logger.exception(
                f"DB error fetching product by item_code: {normalized_item_code}, Error: {db_exc}"
            )
            return None
        except Exception as exc:
            logger.exception(
                f"Unexpected error fetching product by item_code: {normalized_item_code}, Error: {exc}"
            )
            return None


def get_live_product_details_by_id(composite_id: str) -> Optional[Dict[str, Any]]:
    if not composite_id:
        logger.error("get_live_product_details_by_id: Missing composite_id argument.")
        return None
    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_live_product_details_by_id.")
            return None
        try:
            product_entry = session.query(Product).filter_by(id=composite_id).first()
            if not product_entry:
                logger.info(f"No product entry found with composite_id: {composite_id}")
                return None
            return product_entry.to_dict(include_source=True)
        except SQLAlchemyError as db_exc:
            logger.exception(
                f"DB error fetching product by composite_id: {composite_id}, Error: {db_exc}"
            )
            return None
        except Exception as exc:
            logger.exception(
                f"Unexpected error fetching product by composite_id: {composite_id}, Error: {exc}"
            )
            return None

def get_color_variants(product_identifier: str) -> Optional[List[str]]:
    """
    Gets color variants for a product, accepting either a specific SKU
    or a general model name.
    """
    if not product_identifier:
        logger.error("get_color_variants: Missing product_identifier argument.")
        return None

    identifier = str(product_identifier).strip()
    # A simple heuristic to differentiate SKU from model name.
    is_sku = bool(re.match(r'^D\d+$', identifier, re.IGNORECASE))

    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_color_variants.")
            return None
        try:
            if is_sku:
                logger.info(f"Identifier '{identifier}' treated as SKU. Finding variants based on its base model.")
                base_code = product_utils.get_base_model_name(identifier)
                like_pattern = f"{base_code}%"
            else:
                logger.info(f"Identifier '{identifier}' treated as model name. Finding variants via ILIKE.")
                like_pattern = f"%{product_utils.get_base_model_name(identifier)}%"

            rows = (
                session.query(Product.item_code)
                .filter(Product.item_name.ilike(like_pattern))
                .distinct()
                .all()
            )
            variants = sorted({r[0] for r in rows if r[0]})
            logger.info(
                "Found %d potential color variants for identifier '%s'", len(variants), identifier
            )
            return variants
        except SQLAlchemyError as db_exc:
            logger.exception(
                "DB error fetching color variants for %s: %s", identifier, db_exc
            )
            return None
        except Exception as exc:
            logger.exception(
                "Unexpected error fetching color variants for %s: %s", identifier, exc
            )
            return None


def get_available_brands(category: Optional[str] = None) -> List[str]:
    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_available_brands.")
            return []
        try:
            query = session.query(distinct(Product.brand)).filter(Product.brand.isnot(None))
            if category:
                like_pattern = f"%{category}%"
                query = query.filter(
                    or_(
                        Product.category.ilike(like_pattern),
                        Product.sub_category.ilike(like_pattern)
                    )
                )
            results = query.order_by(Product.brand).all()
            brands = [row[0] for row in results if row[0]]
            logger.info(f"Found {len(brands)} brands for category '{category}': {brands}")
            return brands
        except SQLAlchemyError as e:
            logger.exception(f"Database error in get_available_brands for category '{category}': {e}")
            return []

# --- NEW FUNCTIONS FOR CONVERSATIONAL FLOW ---
def find_relevant_accessory(main_product_category: str, warehouse_names: List[str]) -> Optional[Dict[str, Any]]:
    """
    Finds a relevant, in-stock accessory for a given product.
    Example: For a 'CELULAR', it will look for 'cargador' or 'protector'.
    """
    if not main_product_category or not warehouse_names:
        return None

    accessory_keywords = {
        'CELULAR': ['cargador', 'audifono', 'forro', 'protector de pantalla', 'vidrio templado'],
        'TECNOLOGÍA': ['cargador', 'audifono', 'forro', 'protector', 'cable'], # Broader category
        'TELEVISOR': ['base', 'control', 'cable hdmi', 'protector de voltaje'],
        'LAVADORA': ['protector de voltaje'],
        'NEVERA': ['protector de voltaje'],
        'AIRE ACONDICIONADO': ['protector de voltaje', 'base para split', 'control']
    }
    
    # Use the specific category, fallback to a general one, then a default
    main_cat_upper = main_product_category.upper()
    keywords_to_search = accessory_keywords.get(main_cat_upper, ['protector de voltaje'])

    with db_utils.get_db_session() as session:
        if not session: return None
        try:
            canonical_whs_names = [canonicalize_whs(w) for w in warehouse_names if w]
            
            accessory = session.query(Product).filter(
                Product.stock > 0,
                Product.warehouse_name_canonical.in_(canonical_whs_names),
                or_(*[Product.item_name.ilike(f'%{key}%') for key in keywords_to_search]),
                or_(Product.category.ilike('%ACCESORIO%'), Product.sub_category.ilike('%ACCESORIO%'))
            ).order_by(Product.price.asc()).first()

            if accessory:
                logger.info(f"Found relevant accessory '{accessory.item_name}' for category '{main_product_category}'.")
                return accessory.to_dict()
            else:
                logger.info(f"No relevant in-stock accessory found for category '{main_product_category}' in given warehouses.")
                return None
        except Exception as e:
            logger.exception(f"Error finding relevant accessory: {e}")
            return None

def get_product_availability(
    product_identifier: str, warehouse_names: Optional[List[str]] = None
) -> Optional[List[str]]:
    """
    Returns a list of unique branch names where the product is in stock,
    optionally filtered by the warehouses of a specific city.
    Accepts either a SKU or a model name.
    """
    if not product_identifier:
        return None
    
    identifier = str(product_identifier).strip()
    is_sku = bool(re.match(r'^D\d+$', identifier, re.IGNORECASE))
    
    with db_utils.get_db_session() as session:
        if not session:
            return None
        try:
            query = session.query(distinct(Product.branch_name)).filter(
                Product.stock > 0,
                Product.branch_name.isnot(None)
            )

            # Filter by location if specified
            if warehouse_names:
                canonical_whs_names = [canonicalize_whs(w) for w in warehouse_names if w]
                if canonical_whs_names:
                    query = query.filter(Product.warehouse_name_canonical.in_(canonical_whs_names))
                    logger.info(f"Filtering availability search to warehouses: {canonical_whs_names}")

            if is_sku:
                logger.info(f"Getting availability for specific SKU: '{identifier}'")
                query = query.filter(Product.item_code.ilike(identifier))
            else:
                logger.info(f"Getting availability for model name: '{identifier}'")
                base_model_name = product_utils.get_base_model_name(identifier)
                query = query.filter(Product.item_name.ilike(f"{base_model_name}%"))

            results = query.order_by(Product.branch_name).all()
            
            locations = [row[0] for row in results if row[0]]
            logger.info(f"Found {len(locations)} in-stock locations for identifier '{identifier}': {locations}")
            return locations
        except Exception as e:
            logger.exception(f"Error getting availability for identifier '{identifier}': {e}")
            return None