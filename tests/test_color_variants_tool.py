import importlib.util
import sys
import types
import os
import json
from pathlib import Path

# Minimal environment for importing openai_service
dummy_openai = types.ModuleType('openai')
dummy_openai.OpenAI = object
dummy_openai.APIError = Exception
dummy_openai.RateLimitError = Exception
dummy_openai.APITimeoutError = Exception
dummy_openai.BadRequestError = Exception
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

ps_mod = types.ModuleType('ps')
sys.modules.setdefault('namwoo_app.services.product_service', ps_mod)
sys.modules.setdefault('namwoo_app.services.support_board_service', types.ModuleType('sb'))
sys.modules.setdefault('namwoo_app.services.product_recommender', types.ModuleType('pr'))
utils_pkg = types.ModuleType('namwoo_app.utils')
utils_pkg.__path__ = []
sys.modules.setdefault('namwoo_app.utils', utils_pkg)
sys.modules.setdefault('namwoo_app.utils.embedding_utils', types.ModuleType('eu'))
sys.modules.setdefault('namwoo_app.utils.conversation_location', types.ModuleType('cl'))
pu_mod = types.ModuleType('pu')
pu_mod.extract_color_from_name = lambda name: (name.split()[-1].capitalize(), '')
sys.modules.setdefault('namwoo_app.utils.product_utils', pu_mod)

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
    os.path.join('namwoo_app', 'services', 'openai_service.py')
)
openai_service = importlib.util.module_from_spec(spec)
openai_service.__package__ = 'namwoo_app.services'
sys.modules[spec.name] = openai_service
spec.loader.exec_module(openai_service)

openai_service.product_utils = types.SimpleNamespace(
    extract_color_from_name=lambda name: (name.split()[-1].capitalize(), '')
)


def test_get_color_variants_maps_skus(monkeypatch):
    monkeypatch.setattr(openai_service.product_service, 'get_color_variants', lambda ident: ['A1', 'A2'], raising=False)

    def fake_details(sku=None, **kw):
        code = sku or kw.get('item_code_query')
        if code == 'A1':
            return [{'item_name': 'TECNO CAMON 40 PRO BLANCO'}]
        return [{'item_name': 'TECNO CAMON 40 PRO NEGRO'}]

    monkeypatch.setattr(openai_service.product_service, 'get_live_product_details_by_sku', fake_details, raising=False)

    out = openai_service._tool_get_color_variants('TECNO CAMON 40 PRO')
    data = json.loads(out)
    assert data['status'] == 'success'
    assert sorted(data['variants']) == ['Blanco', 'Negro']
