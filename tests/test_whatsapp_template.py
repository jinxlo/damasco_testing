import importlib.util
import json
import sys
import types
from pathlib import Path

requests_mod = types.ModuleType('requests')
sys.modules.setdefault('requests', requests_mod)
flask_mod = types.ModuleType('flask')
class DummyFlask:
    def __init__(self, *a, **k):
        pass

flask_mod.Flask = DummyFlask
flask_mod.current_app = types.SimpleNamespace(config={})
sys.modules.setdefault('flask', flask_mod)
dotenv_mod = types.ModuleType('dotenv')
def dummy_ld(*a, **k):
    return None
dotenv_mod.load_dotenv = dummy_ld
sys.modules.setdefault('dotenv', dotenv_mod)
sys.modules.setdefault('numpy', types.ModuleType('numpy'))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pkg = types.ModuleType('namwoo_app')
pkg.__path__ = []
services_pkg = types.ModuleType('namwoo_app.services')
services_pkg.__path__ = []
pkg.services = services_pkg
sys.modules.setdefault('namwoo_app', pkg)
sys.modules.setdefault('namwoo_app.services', services_pkg)
config_pkg = types.ModuleType('namwoo_app.config')
config_pkg.__path__ = []
config_mod = types.ModuleType('namwoo_app.config.config')
class DummyConfig:
    WHATSAPP_TEMPLATE_LANGUAGES = "es_ES"
    OPENAI_API_KEY = None
    MAX_HISTORY_MESSAGES = 16
config_mod.Config = DummyConfig
sys.modules.setdefault('namwoo_app.config', config_pkg)
sys.modules.setdefault('namwoo_app.config.config', config_mod)
config_pkg.Config = DummyConfig

spec = importlib.util.spec_from_file_location(
    "namwoo_app.services.support_board_service",
    "namwoo_app/services/support_board_service.py",
)
sbs = importlib.util.module_from_spec(spec)
sbs.__package__ = "namwoo_app.services"
sys.modules[spec.name] = sbs
spec.loader.exec_module(sbs)


def test_send_whatsapp_template_success(monkeypatch):
    captured = {}

    def fake_call(payload):
        captured.update(payload)
        return {"messages": [{"id": "1"}]}

    monkeypatch.setattr(sbs, "_call_sb_api", fake_call)

    ok = sbs._send_whatsapp_template(
        "+1234567890",
        "foo_template",
        "es_ES",
        ["a", "b"],
        phone_id="1"
    )
    assert ok is True
    assert json.loads(captured["parameters"]) == ["a", "b"]
