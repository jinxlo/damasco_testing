import os
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SYSTEM_PROMPT_PATH = Path(os.path.dirname(__file__)).parent / "namwoo_app" / "data" / "system_prompt.txt"


def test_prompt_contains_snippet_and_no_leak_paragraphs():
    text = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
    snippet = "¡Entendido! 😊\n\n¿Qué tipo de producto estás buscando hoy?"
    assert snippet in text
    assert "Ahora tengo la información" not in text
    assert "En Caracas tenemos varias sucursales" not in text


def test_redaction_of_store_details():
    sample_json = (
        '{"city": "Caracas", "whsName": "Almacén Principal",'
        ' "branchName": "SABANA", "address": "Av. Principal"}'
    )
    pattern = re.compile(
        r"(branchName\s*:?\s*\"[^\"]*\"|address\s*:?\s*\"[^\"]*\"|whsName\s*:?\s*\"[^\"]*\"|Almac[eé]n[^,\n]+|Direcci[oó]n\:[^\n]+)",
        re.IGNORECASE,
    )
    redacted = pattern.sub("<REDACTED>", sample_json)
    assert "<REDACTED>" in redacted
    assert "address" not in redacted.lower()
