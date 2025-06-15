# NAMWOO/services/product_service.py

import logging
import re
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from decimal import InvalidOperation as InvalidDecimalOperation
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sqlalchemy import func, literal_column, distinct, or_
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import Config
from ..models.product import Product
from ..utils import db_utils, embedding_utils, product_utils
from ..utils.whs_utils import canonicalize_whs

logger = logging.getLogger(__name__)

# --- Semantic Helpers ---
_ACCESSORY_PAT = re.compile(
    r"(base para|soporte|mount|stand|bracket|control(?: remoto)?|adaptador|"
    r"compresor|enfriador|deshumidificador)",
    flags=re.I,
)
_MAIN_TYPE_PAT = re.compile(
    r"\b(tv|televisor|pantalla|nevera|refrigerador|aire acondicionado|"
    r"lavadora|secadora|freidora|microondas|horno)\b",
    flags=re.I,
)


def _is_accessory(name: str) -> bool:
    if not name:
        return False
    return bool(_ACCESSORY_PAT.search(name))


def _extract_main_type(text: str) -> str:
    if not text:
        return ""
    m = _MAIN_TYPE_PAT.search(text)
    return m.group(0).lower() if m else ""


# --- Search Products ---
def search_local_products(
    query_text: str,
    limit: int = 30,
    filter_stock: bool = True,
    min_score: float = 0.35,
    warehouse_names: Optional[List[str]] = None,
    min_price: Optional[Union[float, int, str]] = None,
    max_price: Optional[Union[float, int, str]] = None,
    sort_by: Optional[str] = None,
    exclude_accessories: bool = False,
    required_specs: Optional[List[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    if not query_text or not isinstance(query_text, str):
        logger.warning("Search query is empty or invalid.")
        return []

    log_message_parts = [
        f"Vector search initiated: '{query_text[:80]}…'",
        f"limit={limit}",
        f"stock_filter={filter_stock}",
        f"min_score={min_score:.2f}",
    ]
    if warehouse_names:
        warehouse_names = [canonicalize_whs(w) for w in warehouse_names if w]
        log_message_parts.append(f"warehouses={warehouse_names}")
    if required_specs:
        log_message_parts.append(f"required_specs={required_specs}")
    if min_price is not None:
        log_message_parts.append(f"min_price={min_price}")
    if max_price is not None:
        log_message_parts.append(f"max_price={max_price}")
    if sort_by:
        log_message_parts.append(f"sort_by={sort_by}")
    if exclude_accessories:
        log_message_parts.append("exclude_accessories=True")

    logger.info(", ".join(log_message_parts))
    logger.debug(
        "search_local_products args - query_text=%s, limit=%s, filter_stock=%s, min_score=%s, warehouses=%s, min_price=%s, max_price=%s, sort_by=%s, exclude_accessories=%s, required_specs=%s",
        query_text,
        limit,
        filter_stock,
        min_score,
        warehouse_names,
        min_price,
        max_price,
        sort_by,
        exclude_accessories,
        required_specs,
    )

    embedding_model = (
        Config.OPENAI_EMBEDDING_MODEL
        if hasattr(Config, "OPENAI_EMBEDDING_MODEL")
        else "text-embedding-3-small"
    )
    query_emb = embedding_utils.get_embedding(query_text, model=embedding_model)
    if not query_emb:
        logger.error("Query embedding generation failed – aborting search.")
        return None
    logger.debug(
        "Generated embedding length: %d using model %s", len(query_emb), embedding_model
    )

    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for search.")
            return None
        try:
            q = session.query(
                Product,
                (1 - Product.embedding.cosine_distance(query_emb)).label("similarity"),
            )
            applied_filters = ["cosine_distance"]
            if filter_stock:
                q = q.filter(Product.stock > 0)
                applied_filters.append("stock>0")

            if warehouse_names:
                q = q.filter(Product.warehouse_name_canonical.in_(warehouse_names))
                applied_filters.append(f"warehouses={warehouse_names}")
            
            if required_specs:
                for spec in required_specs:
                    normalized_spec = spec.lower().replace(" ", "")
                    like_pattern = f"%{normalized_spec}%"
                    
                    q = q.filter(
                        or_(
                            func.lower(func.replace(Product.item_name, ' ', '')).like(like_pattern),
                            func.lower(func.replace(Product.description, ' ', '')).like(like_pattern),
                            func.lower(func.replace(Product.especificacion, ' ', '')).like(like_pattern),
                            func.lower(func.replace(Product.searchable_text_content, ' ', '')).like(like_pattern)
                        )
                    )
                    applied_filters.append(f"spec:{spec} (normalized: {normalized_spec})")
                logger.info(f"Applied robust hard filters for specs: {required_specs}")

            if min_price is not None:
                try:
                    min_price_decimal = Decimal(str(min_price)).quantize(
                        Decimal("0.01")
                    )
                    q = q.filter(Product.price >= min_price_decimal)
                    applied_filters.append(f"price>={min_price_decimal}")
                except InvalidDecimalOperation:
                    logger.warning(
                        f"Invalid min_price value '{min_price}' provided. Ignoring min_price filter."
                    )

            if max_price is not None:
                try:
                    max_price_decimal = Decimal(str(max_price)).quantize(
                        Decimal("0.01")
                    )
                    q = q.filter(Product.price <= max_price_decimal)
                    applied_filters.append(f"price<={max_price_decimal}")
                except InvalidDecimalOperation:
                    logger.warning(
                        f"Invalid max_price value '{max_price}' provided. Ignoring max_price filter."
                    )

            diagnostic_q = q.order_by(
                Product.embedding.cosine_distance(query_emb)
            ).limit(limit)
            diagnostic_rows: List[Tuple[Product, float]] = diagnostic_q.all()
            logger.debug("Applied filters for diagnostic query: %s", applied_filters)
            logger.debug(
                "Diagnostic - Top %d products before min_score filter:",
                len(diagnostic_rows),
            )
            for prod_entry_diag, sim_score_diag in diagnostic_rows:
                logger.info(
                    f"  - Product: {prod_entry_diag.item_name}, ID: {prod_entry_diag.id}, "
                    f"Similarity: {sim_score_diag:.4f}, Price: {prod_entry_diag.price}, "
                    f"Stock: {prod_entry_diag.stock}, Warehouse: {prod_entry_diag.warehouse_name}"
                )

            logger.info(
                f"[VECTOR SEARCH] Top similarity score: {diagnostic_rows[0][1]:.4f}"
                if diagnostic_rows
                else "[VECTOR SEARCH] Top similarity score: No results"
            )

            q_final = q.filter(
                (1 - Product.embedding.cosine_distance(query_emb)) >= min_score
            )
            if sort_by == "price_asc":
                q_final = q_final.order_by(Product.price.asc()).limit(limit)
            elif sort_by == "price_desc":
                q_final = q_final.order_by(Product.price.desc()).limit(limit)
            else:
                q_final = q_final.order_by(
                    Product.embedding.cosine_distance(query_emb)
                ).limit(limit)

            rows: List[Tuple[Product, float]] = q_final.all()
            logger.debug("DB returned %d rows after final filters", len(rows))

            final_rows = rows
            if not final_rows and diagnostic_rows:
                logger.warning(
                    f"Top similarity {diagnostic_rows[0][1]:.4f} was below threshold {min_score} or no results passed hard spec filters. Returning fallback result."
                )
                final_rows = diagnostic_rows[:1]
            results: List[Dict[str, Any]] = []
            for prod_location_entry, sim_score in final_rows:
                item_dict = prod_location_entry.to_dict(include_source=True)
                item_dict.update(
                    {
                        "similarity": round(float(sim_score), 4),
                        "is_accessory": _is_accessory(
                            prod_location_entry.item_name or ""
                        ),
                        "main_type": _extract_main_type(
                            prod_location_entry.item_name or ""
                        ),
                        "llm_formatted_description": prod_location_entry.format_for_llm(),
                    }
                )
                results.append(item_dict)
            if exclude_accessories:
                results = [r for r in results if not r.get("is_accessory")]

            if not results and filter_stock:
                logger.info(
                    "No products found with stock filter applied. Retrying without stock filter for reference."
                )
                return search_local_products(
                    query_text,
                    limit=limit,
                    filter_stock=False,
                    min_score=min_score,
                    warehouse_names=warehouse_names,
                    min_price=min_price,
                    max_price=max_price,
                    sort_by=sort_by,
                    exclude_accessories=exclude_accessories,
                    required_specs=required_specs,
                )

            if not results:
                logger.info(
                    "Vector search completed but no products matched the criteria."
                )
                return []

            logger.info(f"[VECTOR SEARCH] Results returned: {len(results)}")

            return results
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
    whs_canonical = canonicalize_whs(whs_name_raw)

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
        "warehouse_name": whs_name_raw,
        "branch_name": _normalize_string(damasco_product_data_camel.get("branchName")),
        "price": normalized_price_for_db,
        "price_bolivar": normalized_price_bolivar_for_db,
        "stock": normalized_stock_for_db,
        "searchable_text_content": norm_searchable_text,
        "embedding": embedding_vector_for_db,
        "source_data_json": damasco_product_data_camel,
    }
    product_id = product_utils.generate_product_location_id(item_code, whs_name_raw)
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
            product_entries = (
                session.query(Product).filter_by(item_code=normalized_item_code).all()
            )
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

def get_color_variants_for_sku(item_code_query: str) -> Optional[List[str]]:
    if not item_code_query:
        logger.error("get_color_variants_for_sku: Missing item_code_query argument.")
        return None
    base_code = product_utils.strip_color_suffix(str(item_code_query).strip())
    if not base_code:
        return []
    with db_utils.get_db_session() as session:
        if not session:
            logger.error("DB session unavailable for get_color_variants_for_sku.")
            return None
        try:
            like_pattern = f"{base_code}%"
            rows = (
                session.query(Product.item_code)
                .filter(Product.item_code.ilike(like_pattern))
                .distinct()
                .all()
            )
            variants = sorted({r[0] for r in rows if r[0]})
            logger.info(
                "Found %d potential color variants for base code %s", len(variants), base_code
            )
            return variants
        except SQLAlchemyError as db_exc:
            logger.exception(
                "DB error fetching color variants for %s: %s", base_code, db_exc
            )
            return None
        except Exception as exc:
            logger.exception(
                "Unexpected error fetching color variants for %s: %s", base_code, exc
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

def get_product_availability_by_sku(item_code: str) -> Optional[List[str]]:
    """Returns a list of unique branch names where the product is in stock."""
    if not item_code: return None
    with db_utils.get_db_session() as session:
        if not session: return None
        try:
            results = session.query(distinct(Product.branch_name)).filter(
                Product.item_code.ilike(item_code),
                Product.stock > 0,
                Product.branch_name.isnot(None)
            ).order_by(Product.branch_name).all()
            
            locations = [row[0] for row in results if row[0]]
            logger.info(f"Found {len(locations)} in-stock locations for SKU '{item_code}': {locations}")
            return locations
        except Exception as e:
            logger.exception(f"Error getting availability for SKU '{item_code}': {e}")
            return None