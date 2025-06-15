import importlib.util
import sys
import types
from pathlib import Path
from contextlib import contextmanager
import json

# Dummy external dependencies
dummy_openai = types.ModuleType('openai')
class DummyChatClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                # If last message was a tool output, return final text
                msgs = kwargs.get('messages', [])
                if msgs and isinstance(msgs[-1], dict) and msgs[-1].get('role') == 'tool':
                    message = types.SimpleNamespace(content='done', tool_calls=None, model_dump=lambda exclude_none=True: {'role': 'assistant', 'content': 'done'})
                else:
                    tc = types.SimpleNamespace(id='1', function=types.SimpleNamespace(name='search_local_products', arguments=json.dumps({'query_text': 'foo'})))
                    message = types.SimpleNamespace(content=None, tool_calls=[tc], model_dump=lambda exclude_none=True: {'role': 'assistant', 'tool_calls': [tc], 'content': None})
                return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

dummy_openai.OpenAI = lambda *a, **k: DummyChatClient()
for name in ['APIError', 'RateLimitError', 'APITimeoutError', 'BadRequestError']:
    setattr(dummy_openai, name, Exception)
sys.modules['openai'] = dummy_openai
sys.modules.setdefault('requests', types.ModuleType('requests'))
celery_mod = types.ModuleType('celery')
celery_mod.exceptions = types.SimpleNamespace(Ignore=type('I',(),{}), MaxRetriesExceededError=type('M',(),{}))
sys.modules.setdefault('celery', celery_mod)
sys.modules.setdefault('celery.exceptions', celery_mod.exceptions)
flask_mod = types.ModuleType('flask')
flask_mod.Flask = lambda *a, **k: None
flask_mod.current_app = types.SimpleNamespace(config={})
sys.modules.setdefault('flask', flask_mod)
dotenv_mod = types.ModuleType('dotenv')
dotenv_mod.load_dotenv = lambda *a, **k: None
sys.modules.setdefault('dotenv', dotenv_mod)
# Minimal packages
pkg = types.ModuleType('namwoo_app')
services_pkg = types.ModuleType('namwoo_app.services')
utils_pkg = types.ModuleType('namwoo_app.utils')
pkg.__path__ = []
services_pkg.__path__ = []
utils_pkg.__path__ = []
pkg.services = services_pkg
pkg.utils = utils_pkg
sys.modules.setdefault('namwoo_app', pkg)
sys.modules.setdefault('namwoo_app.services', services_pkg)
sys.modules.setdefault('namwoo_app.utils', utils_pkg)
models_mod = types.ModuleType('namwoo_app.models')
models_mod.Base = object
sys.modules['namwoo_app.models'] = models_mod
cc_mod = types.ModuleType('namwoo_app.models.conversation_city')
cc_mod.ConversationCity = type('CC', (), {})
sys.modules['namwoo_app.models.conversation_city'] = cc_mod
sys.modules.setdefault('namwoo_app.celery_tasks', types.ModuleType('ct'))
dbu_mod = types.ModuleType('db_utils')
@contextmanager
def dummy_ctx():
    yield None
dbu_mod.get_db_session = dummy_ctx
sys.modules.setdefault('namwoo_app.utils.db_utils', dbu_mod)
sys.modules.setdefault('namwoo_app.services.product_recommender', types.ModuleType('pr'))

# Load conversation_location module and patch DB access
CONV_PATH = Path(__file__).resolve().parents[1] / 'namwoo_app' / 'utils' / 'conversation_location.py'
spec = importlib.util.spec_from_file_location('namwoo_app.utils.conversation_location', CONV_PATH)
conv_mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = conv_mod
spec.loader.exec_module(conv_mod)

_fake_db = {}
class FakeModel:
    def __init__(self, conversation_id, city):
        self.conversation_id = conversation_id
        self.city = city
class FakeSession:
    def query(self, model):
        self.model = model
        return self
    def filter_by(self, conversation_id=None):
        self.cid = conversation_id
        return self
    def first(self):
        city = _fake_db.get(self.cid)
        return FakeModel(self.cid, city) if city else None
    def add(self, obj):
        _fake_db[obj.conversation_id] = obj.city
    def commit(self):
        pass
@contextmanager
def fake_get_db_session():
    yield FakeSession()

conv_mod.db_utils = types.SimpleNamespace(get_db_session=fake_get_db_session)
conv_mod.ConversationCity = FakeModel

# Provide module for Config before loading openai_service
config_pkg = types.ModuleType('namwoo_app.config')
config_mod = types.ModuleType('namwoo_app.config.config')
class DummyConfig:
    OPENAI_API_KEY = 'x'
    OPENAI_CHAT_MODEL = 'gpt-4o-mini'
    OPENAI_MAX_TOKENS = 1024
    OPENAI_TEMPERATURE = 0.7
    OPENAI_RECOMMENDER_MODEL = 'gpt-4.1'
    MAX_HISTORY_MESSAGES = 16
    SYSTEM_PROMPT = 'hi'
    RECOMMENDER_MODE = 'python'
    SUPPORT_BOARD_DM_BOT_USER_ID = '1'
config_mod.Config = DummyConfig
config_pkg.Config = DummyConfig
sys.modules['namwoo_app.config.config'] = config_mod
sys.modules['namwoo_app.config'] = config_pkg

# Stub dependent service modules
ps_mod = types.ModuleType('ps')
search_calls = []
def fake_search(**kw):
    search_calls.append(kw)
    return []
ps_mod.search_local_products = fake_search
sys.modules['namwoo_app.services.product_service'] = ps_mod
sb_mod = types.ModuleType('sb')
sb_mod.get_sb_conversation_data = lambda cid: {'messages': [{'user_id': '2', 'message': 'hola'}]}
sb_mod.send_reply_to_channel = lambda **kw: None
sys.modules['namwoo_app.services.support_board_service'] = sb_mod
pu_mod = types.ModuleType('pu')
pu_mod.group_products_by_model = lambda x: x
pu_mod.format_multiple_products_response = lambda products, q: 'ok'
sys.modules['namwoo_app.utils.product_utils'] = pu_mod
sys.modules['namwoo_app.utils.embedding_utils'] = types.ModuleType('eu')
sys.modules['namwoo_app.utils.whs_utils'] = types.ModuleType('wu')
wu_mod = sys.modules['namwoo_app.utils.whs_utils']
wu_mod.canonicalize_whs_name = lambda n: n

# Load openai_service using the real file
OAI_PATH = Path(__file__).resolve().parents[1] / 'namwoo_app' / 'services' / 'openai_service.py'
spec2 = importlib.util.spec_from_file_location('namwoo_app.services.openai_service', OAI_PATH)
openai_service = importlib.util.module_from_spec(spec2)
openai_service.__package__ = 'namwoo_app.services'
sys.modules[spec2.name] = openai_service
spec2.loader.exec_module(openai_service)

# Patch loaded modules
openai_service.product_service = ps_mod
openai_service.support_board_service = sb_mod
openai_service.product_utils = pu_mod
openai_service.canonicalize_whs_name = lambda n: n
openai_service.product_recommender = types.SimpleNamespace(rank_products=lambda q, c: c)
openai_service.conversation_location = conv_mod
openai_service._chat_client = DummyChatClient()


def test_city_persistence_across_workers(monkeypatch):
    global openai_service
    # First message with city detected
    openai_service.process_new_message('42', 'Hola estoy en Caracas', 'wa', 'u1', 'c1', None)
    assert _fake_db['42'] == 'Caracas'

    # Reload conversation_location to simulate a new worker
    spec_conv = importlib.util.spec_from_file_location('namwoo_app.utils.conversation_location', CONV_PATH)
    conv_new = importlib.util.module_from_spec(spec_conv)
    sys.modules[spec_conv.name] = conv_new
    spec_conv.loader.exec_module(conv_new)
    conv_new.db_utils = types.SimpleNamespace(get_db_session=fake_get_db_session)
    conv_new.ConversationCity = FakeModel
    openai_service.conversation_location = conv_new
    openai_service.process_new_message('42', 'Busco un telefono', 'wa', 'u1', 'c1', None)
    assert search_calls[-1].get('warehouse_names') == conv_new.get_warehouses_for_city('Caracas')
