import importlib.util
import os
import sys
import types
from pathlib import Path
import pytest

dummy_openai = types.ModuleType('openai')
dummy_openai.OpenAI = object
dummy_openai.APIError = Exception
dummy_openai.RateLimitError = Exception
dummy_openai.APITimeoutError = Exception
dummy_openai.BadRequestError = Exception
sys.modules.setdefault('openai', dummy_openai)
sys.modules.setdefault('requests', types.ModuleType('requests'))
sys.modules.setdefault('celery', types.ModuleType('celery'))
celery_exc_mod = types.ModuleType('celery.exceptions')
celery_exc_mod.Ignore = type('Ignore', (), {})
celery_exc_mod.MaxRetriesExceededError = type('MaxRetriesExceededError', (), {})
sys.modules.setdefault('celery.exceptions', celery_exc_mod)
flask_mod = types.ModuleType('flask')
class DummyFlask:
    def __init__(self, *a, **k):
        pass

flask_mod.Flask = DummyFlask
flask_mod.current_app = types.SimpleNamespace(config={})
sys.modules.setdefault('flask', flask_mod)
dotenv_mod = types.ModuleType('dotenv')
dotenv_mod.load_dotenv = lambda *a, **k: None
sys.modules.setdefault('dotenv', dotenv_mod)
sys.modules.setdefault('numpy', types.ModuleType('numpy'))
pydantic_mod = types.ModuleType('pydantic')
pydantic_mod.BaseModel = object
pydantic_mod.ValidationError = Exception
pydantic_mod.validator = lambda *a, **k: (lambda f: f)
sys.modules.setdefault('pydantic', pydantic_mod)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pkg = types.ModuleType('namwoo_app')
pkg.__path__ = []
services_pkg = types.ModuleType('namwoo_app.services')
services_pkg.__path__ = []
pkg.services = services_pkg
utils_pkg = types.ModuleType('namwoo_app.utils')
utils_pkg.__path__ = []
pkg.utils = utils_pkg
sys.modules.setdefault('namwoo_app', pkg)
sys.modules.setdefault('namwoo_app.services', services_pkg)
sys.modules.setdefault('namwoo_app.utils', utils_pkg)
sys.modules.setdefault('namwoo_app.services.product_service', types.ModuleType('ps'))
sys.modules.setdefault('namwoo_app.services.support_board_service', types.ModuleType('sb'))
sys.modules.setdefault('namwoo_app.utils.embedding_utils', types.ModuleType('eu'))
sys.modules.setdefault('namwoo_app.utils.conversation_location', types.ModuleType('cl'))
pu_mod = types.ModuleType('pu')
pu_mod.user_is_asking_for_cheapest = lambda msg: any(k in msg.lower() for k in ['barato','economico','menor precio','menos costoso','mas bajo']) if msg else False
sys.modules.setdefault('namwoo_app.utils.product_utils', pu_mod)
sys.modules.setdefault('namwoo_app.services.product_recommender', types.ModuleType('pr'))

config_pkg = types.ModuleType('namwoo_app.config')
config_pkg.__path__ = []
config_mod = types.ModuleType('namwoo_app.config.config')
class DummyConfig:
    OPENAI_API_KEY = None
    MAX_HISTORY_MESSAGES = 16
    OPENAI_CHAT_MODEL = "gpt-4o-mini"
    OPENAI_MAX_TOKENS = 1024
    OPENAI_TEMPERATURE = 0.7

config_mod.Config = DummyConfig
sys.modules['namwoo_app.config.config'] = config_mod
sys.modules['namwoo_app.config'] = config_pkg
config_pkg.Config = DummyConfig

UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "namwoo_app", "services", "openai_service.py"))
spec = importlib.util.spec_from_file_location(
    "namwoo_app.services.openai_service", UTILS_PATH
)
openai_service = importlib.util.module_from_spec(spec)
openai_service.__package__ = "namwoo_app.services"
sys.modules[spec.name] = openai_service
spec.loader.exec_module(openai_service)

user_is_asking_for_cheapest = openai_service.user_is_asking_for_cheapest

@pytest.mark.parametrize("msg", [
    "Quiero el mas barato",
    "No tengo presupuesto, busco lo mas economico",
    "Dame el de menor precio",
    "Cual es el menos costoso?",
    "El mas bajo posible",
])
def test_detect_cheapest_phrases(msg):
    assert user_is_asking_for_cheapest(msg) is True

@pytest.mark.parametrize("msg", [
    "Quiero un celular bueno",
    "Busco una licuadora",
])
def test_detect_cheapest_negative(msg):
    assert user_is_asking_for_cheapest(msg) is False

