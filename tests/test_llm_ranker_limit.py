import importlib.util
import sys
import types
from pathlib import Path

"""Unit tests for the sales-associate LLM recommender."""

# Dummy openai module
DUMMY_RESPONSE = '{"ordered_skus": ["B", "A", "C", "D"]}'

dummy_openai = types.ModuleType('openai')
class DummyClient:
    def __init__(self, *a, **k):
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=lambda **kw: types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=DUMMY_RESPONSE))]
                )
            )
        )

dummy_openai.OpenAI = DummyClient
for name in ['APIError', 'RateLimitError', 'APITimeoutError', 'BadRequestError']:
    setattr(dummy_openai, name, Exception)
sys.modules['openai'] = dummy_openai

# Minimal config
config_pkg = types.ModuleType('namwoo_app.config')
config_mod = types.ModuleType('namwoo_app.config.config')
class DummyConfig:
    OPENAI_API_KEY = 'test'
    RECOMMENDER_LLM_MODEL = 'gpt-4o-mini'
config_mod.Config = DummyConfig
sys.modules['namwoo_app.config.config'] = config_mod
sys.modules['namwoo_app.config'] = config_pkg
config_pkg.Config = DummyConfig
sys.modules.setdefault('namwoo_app.services', types.ModuleType('namwoo_app.services'))
sys.modules.setdefault('namwoo_app.services.product_recommender', types.ModuleType('pr'))
conv_mod = sys.modules.setdefault('namwoo_app.utils.conversation_location', types.SimpleNamespace())
setattr(conv_mod, 'get_warehouses_for_city', lambda city: [])

MODULE_PATH = Path(__file__).resolve().parents[1] / 'namwoo_app' / 'services' / 'product_recommender.py'
spec = importlib.util.spec_from_file_location('product_recommender', MODULE_PATH)
recommender = importlib.util.module_from_spec(spec)
recommender.__package__ = 'namwoo_app.services'
spec.loader.exec_module(recommender)
get_ranked_products = recommender.get_ranked_products


def test_sales_associate_recommender_order(monkeypatch):
    items = [
        {"item_code": "A"},
        {"item_code": "B"},
        {"item_code": "C"},
        {"item_code": "D"},
    ]
    dummy_ps = types.SimpleNamespace(
        search_local_products=lambda **kw: items,
        get_color_variants_for_sku=lambda code: []
    )
    monkeypatch.setattr(recommender, "product_service", dummy_ps)
    ranked = get_ranked_products({"query": "phone"}, city="Caracas")
    assert [p["item_code"] for p in ranked[:3]] == ["B", "A", "C"]
    assert len(ranked[:3]) == 3

