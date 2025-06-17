import importlib.util
import sys
import types
from pathlib import Path

dummy_openai = types.ModuleType('openai')
dummy_openai.OpenAI = object
dummy_openai.APIError = Exception
dummy_openai.RateLimitError = Exception
dummy_openai.APITimeoutError = Exception
dummy_openai.BadRequestError = Exception
sys.modules.setdefault('openai', dummy_openai)

sys.modules.setdefault('requests', types.ModuleType('requests'))
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
sys.modules.setdefault('pydantic', types.ModuleType('pydantic'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pkg = types.ModuleType('namwoo_app')
pkg.__path__ = []
services_pkg = types.ModuleType('namwoo_app.services')
services_pkg.__path__ = []
pkg.services = services_pkg
sys.modules.setdefault('namwoo_app', pkg)
sys.modules.setdefault('namwoo_app.services', services_pkg)
sys.modules.setdefault('namwoo_app.services.product_service', types.ModuleType('ps'))
sys.modules.setdefault('namwoo_app.services.support_board_service', types.ModuleType('sb'))
utils_pkg = types.ModuleType('namwoo_app.utils')
utils_pkg.__path__ = []
sys.modules.setdefault('namwoo_app.utils', utils_pkg)
sys.modules.setdefault('namwoo_app.utils.embedding_utils', types.ModuleType('eu'))
sys.modules.setdefault('namwoo_app.utils.conversation_location', types.ModuleType('cl'))
sys.modules.setdefault('namwoo_app.services.product_recommender', types.ModuleType('pr'))
pu_mod = types.ModuleType('pu')
def cheap(msg):
    if not msg:
        return False
    words = [
        "mas barato", "más barato", "mas economico", "más económico",
        "menor precio", "menos costoso", "mas bajo", "más bajo",
    ]
    msg = msg.lower()
    return any(w in msg for w in words)
pu_mod.user_is_asking_for_cheapest = cheap
sys.modules.setdefault('namwoo_app.utils.product_utils', pu_mod)
config_pkg = types.ModuleType('namwoo_app.config')
config_mod = types.ModuleType('namwoo_app.config.config')
class DummyConfig:
    WHATSAPP_TEMPLATE_LANGUAGES = "es_ES"
    OPENAI_API_KEY = None
    MAX_HISTORY_MESSAGES = 16
    OPENAI_CHAT_MODEL = "gpt-4o-mini"
    OPENAI_MAX_TOKENS = 1024
    OPENAI_TEMPERATURE = 0.7
config_mod.Config = DummyConfig
sys.modules['namwoo_app.config.config'] = config_mod
sys.modules['namwoo_app.config'] = config_pkg
config_pkg.Config = DummyConfig

spec = importlib.util.spec_from_file_location(
    "namwoo_app.services.openai_service",
    "namwoo_app/services/openai_service.py",
)
openai_service = importlib.util.module_from_spec(spec)
openai_service.__package__ = "namwoo_app.services"
sys.modules[spec.name] = openai_service
spec.loader.exec_module(openai_service)


def test_end_to_end_checkout(monkeypatch):
    calls = []

    def fake_send(**kwargs):
        calls.append(kwargs.get("parameters"))
        return True

    monkeypatch.setattr(openai_service.support_board_service, "send_whatsapp_template", fake_send, raising=False)

    res = openai_service._tool_send_whatsapp_order_summary_template(
        customer_platform_user_id="+1234567",
        conversation_id="999",
        template_variables=[str(i) for i in range(8)],
    )
    assert res == {"status": "success"}
    assert len(calls) == 1
    assert len(calls[0]) == 8
    names = [t["function"]["name"] for t in openai_service.tools_schema]
    assert "initiate_customer_information_collection" not in names


def test_direct_send_uses_provided_phone(monkeypatch):
    captured = {}

    def fake_direct(user_id=None, conversation_id=None, template_variables=None):
        captured['user_id'] = user_id
        captured['conversation_id'] = conversation_id
        captured['template_variables'] = template_variables
        return True

    monkeypatch.setattr(
        openai_service.support_board_service,
        "send_whatsapp_template_direct",
        fake_direct,
        raising=False,
    )

    openai_service.g.customer_user_id = "context_id"
    openai_service.g.conversation_id = "ctx_conv"

    res = openai_service._tool_send_whatsapp_order_summary_template(
        customer_platform_user_id="+5551111",
        conversation_id="orig_conv",
        template_variables=[str(i) for i in range(8)],
    )

    assert res == {"status": "success"}
    assert captured["user_id"] == "+5551111"
    assert captured["conversation_id"] == "ctx_conv"
