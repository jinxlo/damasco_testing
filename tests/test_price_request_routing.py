import importlib.util
import sys
import types
from pathlib import Path

# Minimal openai and flask stubs
dummy_openai = types.ModuleType('openai')
dummy_openai.OpenAI = object
for n in ['APIError', 'RateLimitError', 'APITimeoutError', 'BadRequestError']:
    setattr(dummy_openai, n, Exception)
sys.modules.setdefault('openai', dummy_openai)

flask_mod = types.ModuleType('flask')
flask_mod.Flask = lambda *a, **k: None
flask_mod.current_app = types.SimpleNamespace(config={})
sys.modules.setdefault('flask', flask_mod)

dotenv_mod = types.ModuleType('dotenv')
dotenv_mod.load_dotenv = lambda *a, **k: None
sys.modules.setdefault('dotenv', dotenv_mod)

pkg = types.ModuleType('namwoo_app')
pkg.__path__ = []
services_pkg = types.ModuleType('namwoo_app.services')
services_pkg.__path__ = []
pkg.services = services_pkg
sys.modules.setdefault('namwoo_app', pkg)
sys.modules.setdefault('namwoo_app.services', services_pkg)

sys.modules.setdefault('namwoo_app.services.support_board_service', types.ModuleType('sb'))
sys.modules.setdefault('namwoo_app.services.product_service', types.ModuleType('ps'))
pr_mod = types.ModuleType('namwoo_app.services.product_recommender')
pr_mod.rank_products = lambda user_intent=None, candidates=None, sort_by=None: candidates or []
sys.modules.setdefault('namwoo_app.services.product_recommender', pr_mod)
sys.modules.setdefault('namwoo_app.utils.embedding_utils', types.ModuleType('eu'))
sys.modules.setdefault('namwoo_app.utils.conversation_location', types.ModuleType('cl'))

config_pkg = types.ModuleType('namwoo_app.config')
config_mod = types.ModuleType('namwoo_app.config.config')
class DummyConfig:
    OPENAI_API_KEY = None
    MAX_HISTORY_MESSAGES = 16
    OPENAI_CHAT_MODEL = 'gpt-4o-mini'
    OPENAI_MAX_TOKENS = 1024
    OPENAI_TEMPERATURE = 0.7
config_mod.Config = DummyConfig
sys.modules['namwoo_app.config.config'] = config_mod
sys.modules['namwoo_app.config'] = config_pkg
config_pkg.Config = DummyConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

spec = importlib.util.spec_from_file_location(
    'namwoo_app.services.openai_service',
    Path(__file__).resolve().parents[1] / 'namwoo_app' / 'services' / 'openai_service.py'
)
openai_service = importlib.util.module_from_spec(spec)
openai_service.__package__ = 'namwoo_app.services'
sys.modules[spec.name] = openai_service
spec.loader.exec_module(openai_service)


def test_price_query_returns_single_card(monkeypatch):
    items = [
        {"item_name": "SAMSUNG A16 NEGRO", "price": 150, "priceBolivar": 15000},
        {"item_name": "SAMSUNG A26 NEGRO", "price": 170, "priceBolivar": 17000},
    ]
    # simple grouping and formatting
    def fake_group(products):
        p = products[0]
        return [{"model": p["item_name"], "price": p["price"], "priceBolivar": p["priceBolivar"], "colors": [], "description": "", "store": "CCS"}]

    formatted = []
    def fake_format(group, query):
        formatted.append(group["model"])
        return f"CARD {group['model']}"

    monkeypatch.setattr(openai_service.product_utils, 'group_products_by_model', fake_group, raising=False)
    monkeypatch.setattr(openai_service.product_utils, 'format_product_response', fake_format, raising=False)

    result = openai_service._format_search_results(
        items,
        'samsung a16',
        'cuanto cuesta el a16',
        is_price_request=True,
        conversation_id='1'
    )
    assert result == 'CARD SAMSUNG A16 NEGRO'
    assert formatted == ['SAMSUNG A16 NEGRO']
