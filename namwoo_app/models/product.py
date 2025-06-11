# namwoo_app/models/product.py
import logging
import re # For whitespace normalization in prepare_text_for_embedding
from sqlalchemy import (
    Column, String, Text, TIMESTAMP, func, UniqueConstraint, Integer, NUMERIC
)
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from typing import Dict, Optional, List, Any # Added List, Any

from . import Base # Assuming Base is defined in models/__init__.py
from ..config import Config
from ..utils.text_utils import strip_html_to_text # Ensure this utility exists and works

logger = logging.getLogger(__name__)

class Product(Base):
    __tablename__ = 'products'

    # Composite Primary Key
    id = Column(String(512), primary_key=True, index=True, comment="Composite ID: item_code + sanitized warehouse_name")

    # Core product details
    item_code = Column(String(64), nullable=False, index=True, comment="Original item code from Damasco")
    item_name = Column(Text, nullable=False, comment="Product's full name or title")
    
    # Descriptions
    description = Column(Text, nullable=True, comment="Raw HTML product description from Damasco")
    llm_summarized_description = Column(Text, nullable=True, comment="LLM-generated summary of the description")

    # Descriptive attributes
    category = Column(String(128), index=True, nullable=True) # Ensuring all text fields can be nullable
    sub_category = Column(String(128), index=True, nullable=True)
    brand = Column(String(128), index=True, nullable=True)
    line = Column(String(128), nullable=True, comment="Product line, if available")
    item_group_name = Column(String(128), index=True, nullable=True, comment="Broader group name")

    # Location-specific attributes
    warehouse_name = Column(String(255), nullable=False, index=True, comment="Warehouse name") # Part of PK logic, so must be non-null
    branch_name = Column(String(255), index=True, nullable=True, comment="Branch name")
    
    # Financial and stock
    price = Column(NUMERIC(12, 2), nullable=True, comment="Price, typically in primary currency (e.g., USD)")
    price_bolivar = Column(NUMERIC(12, 2), nullable=True, comment="Price in Bolívares (Bs.)") # <<< NEW FIELD ADDED
    stock = Column(Integer, default=0, nullable=False) # Stock should have a default and be non-null
    
    # Embedding related fields
    searchable_text_content = Column(Text, nullable=True, comment="PLAIN TEXT content used to generate the embedding")
    embedding = Column(Vector(Config.EMBEDDING_DIMENSION if hasattr(Config, 'EMBEDDING_DIMENSION') and Config.EMBEDDING_DIMENSION else 1536), nullable=True, comment="pgvector embedding") 
    
    # Auditing
    source_data_json = Column(JSONB, nullable=True, comment="Original JSON data for this entry from Damasco") # Changed default from {} to nullable=True
    
    # Timestamps
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        UniqueConstraint('item_code', 'warehouse_name', name='uq_item_code_per_warehouse'),
    )

    def __repr__(self):
        return (f"<Product(id='{self.id}', item_name='{self.item_name[:30]}...', "
                f"warehouse='{self.warehouse_name}', stock={self.stock})>")

    def to_dict(self) -> Dict[str, Any]: # Explicitly typing return dict
        """Returns a dictionary representation of the product-location entry."""
        return {
            "id": self.id,
            "item_code": self.item_code,
            "item_name": self.item_name,
            "description": self.description, 
            "llm_summarized_description": self.llm_summarized_description,
            "plain_text_description_derived": strip_html_to_text(self.description or ""),
            "category": self.category,
            "sub_category": self.sub_category,
            "brand": self.brand,
            "line": self.line,
            "item_group_name": self.item_group_name,
            "warehouse_name": self.warehouse_name,
            "branch_name": self.branch_name,
            "price": float(self.price) if self.price is not None else None,
            "price_bolivar": float(self.price_bolivar) if self.price_bolivar is not None else None, # <<< NEW FIELD ADDED
            "stock": self.stock,
            "searchable_text_content": self.searchable_text_content,
            # "source_data_json": self.source_data_json, # Often too verbose for general dicts unless specifically requested
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def format_for_llm(self, include_stock_location: bool = True) -> str:
        """Formats product information for presentation by an LLM, prioritizing LLM summary."""
        price_str = f"${float(self.price):.2f}" if self.price is not None else "Precio no disponible"
        # <<< OPTIONAL: Add price_bolivar display here >>>
        price_bolivar_str = f" (Bs. {float(self.price_bolivar):.2f})" if self.price_bolivar is not None else ""
        
        current_description_text = ""
        if self.llm_summarized_description and self.llm_summarized_description.strip(): # Check if not empty
            current_description_text = self.llm_summarized_description.strip()
        elif self.description: 
            stripped_desc = strip_html_to_text(self.description)
            if stripped_desc and stripped_desc.strip(): # Check if not empty after stripping
                 current_description_text = stripped_desc.strip()
        
        desc_str_for_llm = f"Descripción: {current_description_text}" if current_description_text else "Descripción no disponible."
        
        base_info = (f"{self.item_name or 'Producto sin nombre'} "
                     f"(Marca: {self.brand or 'N/A'}, "
                     f"Categoría: {self.category or 'N/A'}). "
                     f"{price_str}{price_bolivar_str}. {desc_str_for_llm}") # <<< MODIFIED to include price_bolivar_str
        
        if include_stock_location:
            stock_str = f"Stock: {self.stock if self.stock is not None else 'N/A'}"
            location_str = f"Disponible en {self.warehouse_name or 'ubicación desconocida'}"
            if self.branch_name and self.branch_name != self.warehouse_name:
                location_str += f" (Sucursal: {self.branch_name})"
            return f"{base_info} {location_str}. {stock_str}."
        return base_info.strip()

    @classmethod # Or @staticmethod if cls is not used internally by this method
    def prepare_text_for_embedding(
        cls, # Added cls for classmethod convention
        damasco_product_data: Dict[str, Any], 
        llm_generated_summary: Optional[str],
        raw_html_description_for_fallback: Optional[str] # <<< MODIFIED: ADDED THIS ARGUMENT
    ) -> Optional[str]:
        """
        Constructs and cleans the text string for semantic embeddings.
        Prioritizes LLM-generated summary for the description part.
        If LLM summary is not available or empty, it uses the stripped text from raw_html_description_for_fallback.
        Expects damasco_product_data keys to be camelCase.
        """
        description_content_for_embedding = ""
        
        # Priority 1: Use provided LLM summary if it's not None and not empty after stripping
        if llm_generated_summary and llm_generated_summary.strip():
            description_content_for_embedding = llm_generated_summary.lower().strip()
        # Priority 2: Fallback to stripping the provided raw HTML description
        elif raw_html_description_for_fallback: # Check if fallback HTML is provided
            plain_html_text = strip_html_to_text(raw_html_description_for_fallback)
            if plain_html_text and plain_html_text.strip(): # Check if not empty after stripping
                description_content_for_embedding = plain_html_text.lower().strip()
        # (Implicit Priority 3: If damasco_product_data.get("description") was the old fallback)
        # The new logic in celery_tasks.py already passes the correct raw_html_incoming
        # so we don't need to look into damasco_product_data["description"] here for the fallback.
        
        parts_to_join: List[str] = []

        # Helper to add parts if they are valid strings
        def add_part(text: Any):
            if text and isinstance(text, str):
                cleaned = text.lower().strip()
                if cleaned:
                    parts_to_join.append(cleaned)
            elif text: # Attempt to convert other non-None types to string
                try:
                    cleaned = str(text).lower().strip()
                    if cleaned:
                        parts_to_join.append(cleaned)
                except Exception:
                    pass # Skip if conversion fails

        add_part(damasco_product_data.get("brand"))
        add_part(damasco_product_data.get("itemName")) # Item name is crucial
        
        if description_content_for_embedding: # Add only if it's not empty
            add_part(description_content_for_embedding)
            
        add_part(damasco_product_data.get("category"))
        add_part(damasco_product_data.get("subCategory"))
        add_part(damasco_product_data.get("itemGroupName"))
        add_part(damasco_product_data.get("line"))
        
        # Note: Numerical price fields like 'price' or 'priceBolivar' are generally NOT included
        # in the text for semantic embedding, as their exact values change frequently and
        # embeddings are more about the 'what' than precise, volatile numerical data.
        # The current logic correctly omits them.

        if not parts_to_join:
            logger.warning(f"No text parts found to build embedding string for itemCode: {damasco_product_data.get('itemCode')}")
            return None 

        final_text = " ".join(parts_to_join)
        # Normalize multiple spaces to single, and strip leading/trailing whitespace
        final_text = re.sub(r'\s+', ' ', final_text).strip() 
        
        return final_text if final_text else None

# --- End of NAMWOO/models/product.py ---