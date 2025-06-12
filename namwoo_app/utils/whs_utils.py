import re
import unicodedata


def canonicalize_whs(raw: str) -> str:
    """Return canonical warehouse name as ID-friendly string."""
    if raw is None:
        return ""
    s = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).upper().strip("_")
    return s
