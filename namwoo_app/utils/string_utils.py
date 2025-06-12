"""
Lightweight helpers for common string normalisation.
"""
import unicodedata
from typing import Optional


def canonicalize_whs_name(name: str | None) -> Optional[str]:
    """Convert a warehouse name to its canonical internal form."""
    if not name:
        return None
    cleaned = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    cleaned = cleaned.strip().upper()
    return cleaned or None

