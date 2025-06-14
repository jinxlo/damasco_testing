import json
import os
import re
import unicodedata

_BRANCH_TO_WHS: dict[str, str] | None = None


def _load_branch_map() -> dict[str, str]:
    """Load mapping of branchName -> whsName from store_locations.json."""
    global _BRANCH_TO_WHS
    if _BRANCH_TO_WHS is not None:
        return _BRANCH_TO_WHS

    _BRANCH_TO_WHS = {}
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "store_locations.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            stores = json.load(f)
        for store in stores:
            branch = store.get("branchName")
            whs = store.get("whsName")
            if branch and whs:
                key = canonicalize_whs(branch)
                _BRANCH_TO_WHS[key] = whs
                # also map the whs name itself
                _BRANCH_TO_WHS.setdefault(canonicalize_whs(whs), whs)
    except Exception:
        _BRANCH_TO_WHS = {}
    return _BRANCH_TO_WHS


def canonicalize_whs(raw: str) -> str:
    """Return canonical warehouse name as ID-friendly string."""
    if raw is None:
        return ""
    s = unicodedata.normalize("NFKD", raw).encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).upper().strip("_")
    return s


def canonicalize_whs_name(name: str | None) -> str | None:
    """Return the canonical `whsName` for a given branch or warehouse name."""
    if not name:
        return None
    mapping = _load_branch_map()
    key = canonicalize_whs(name)
    return mapping.get(key, name)
