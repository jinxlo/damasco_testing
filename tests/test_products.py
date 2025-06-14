import importlib.util
import os

import pytest

UTILS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "namwoo_app", "utils", "whs_utils.py")
)
spec = importlib.util.spec_from_file_location("whs_utils", UTILS_PATH)
whs_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(whs_utils)
canonicalize_whs = whs_utils.canonicalize_whs


class IntegrityError(Exception):
    pass


class FakeSession:
    def __init__(self):
        self._data = {}

    def add(self, product):
        key = (product["item_code"], product["warehouse_name_canonical"])
        if key in self._data:
            raise IntegrityError("duplicate key")
        self._data[key] = product

    def commit(self):
        pass

    def search(self, warehouse_names):
        canon = [canonicalize_whs(w) for w in warehouse_names]
        return [
            p for p in self._data.values() if p["warehouse_name_canonical"] in canon
        ]


@pytest.fixture
def session():
    return FakeSession()


def add_prod(sess, item_code, whs_name, **extra):
    prod = {
        "item_code": item_code,
        "warehouse_name": whs_name,
        "warehouse_name_canonical": canonicalize_whs(whs_name),
        "item_name": extra.get("item_name", "Foo"),
        "category": extra.get("category", "BAR"),
        "price": extra.get("price", 1),
        "stock": extra.get("stock", 0),
    }
    sess.add(prod)
    sess.commit()


def test_canonical_sku_uniqueness(session):
    add_prod(session, "TEST123", "Almacén Principal Sabana Grande")
    with pytest.raises(IntegrityError):
        add_prod(session, "TEST123", "ALMACEN PRINCIPAL SABANA_GRANDE")


def test_search_accent_insensitive(session):
    add_prod(session, "SKU1", "Sabana Grande", item_name="licuadora X")
    res = session.search(["Sabána Grande"])
    assert res, "Should find rows regardless of accent / case"


def test_product_to_dict_includes_source_data():
    PRODUCT_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "namwoo_app", "models", "product.py")
    )
    spec = importlib.util.spec_from_file_location("namwoo_app.models.product", PRODUCT_PATH)
    product_mod = importlib.util.module_from_spec(spec)
    product_mod.__package__ = "namwoo_app.models"
    # Stub pgvector.sqlalchemy.Vector to avoid dependency
    import types, sys
    fake_pgvector_sqlalchemy = types.ModuleType("pgvector.sqlalchemy")
    fake_pgvector_sqlalchemy.Vector = lambda *a, **k: None
    sys.modules.setdefault("pgvector.sqlalchemy", fake_pgvector_sqlalchemy)

    fake_sqlalchemy = types.ModuleType("sqlalchemy")
    fake_sqlalchemy.NUMERIC = fake_sqlalchemy.TIMESTAMP = lambda *a, **k: None
    fake_sqlalchemy.Integer = lambda *a, **k: None
    fake_sqlalchemy.String = lambda *a, **k: None
    fake_sqlalchemy.Text = lambda *a, **k: None
    fake_sqlalchemy.UniqueConstraint = lambda *a, **k: None
    fake_sqlalchemy.Column = lambda *a, **k: None
    fake_sqlalchemy.func = types.SimpleNamespace(now=lambda: None)
    sys.modules.setdefault("sqlalchemy", fake_sqlalchemy)
    fake_pg = types.ModuleType("sqlalchemy.dialects.postgresql")
    fake_pg.JSONB = object
    sys.modules.setdefault("sqlalchemy.dialects", types.ModuleType("sqlalchemy.dialects"))
    sys.modules.setdefault("sqlalchemy.dialects.postgresql", fake_pg)
    # Minimal config package to satisfy relative imports
    config_pkg = types.ModuleType("namwoo_app.config")
    config_mod = types.ModuleType("namwoo_app.config.config")
    class DummyConfig:
        pass
    config_mod.Config = DummyConfig
    sys.modules.setdefault("namwoo_app", types.ModuleType("namwoo_app"))
    sys.modules.setdefault("namwoo_app.config", config_pkg)
    sys.modules.setdefault("namwoo_app.config.config", config_mod)
    config_pkg.Config = DummyConfig
    utils_pkg = types.ModuleType("namwoo_app.utils")
    text_utils_mod = types.ModuleType("namwoo_app.utils.text_utils")
    text_utils_mod.strip_html_to_text = lambda x: x
    sys.modules.setdefault("namwoo_app.utils", utils_pkg)
    sys.modules.setdefault("namwoo_app.utils.text_utils", text_utils_mod)
    models_pkg = types.ModuleType("namwoo_app.models")
    models_pkg.Base = object
    sys.modules.setdefault("namwoo_app.models", models_pkg)

    spec.loader.exec_module(product_mod)
    Product = product_mod.Product

    p = Product()
    p.id = "abc@main"
    p.item_code = "ABC"
    p.item_name = "Test Product"
    p.warehouse_name = "Main"
    p.warehouse_name_canonical = "main"
    p.stock = 1
    p.source_data_json = {"foo": "bar"}

    default_dict = p.to_dict()
    assert "source_data_json" not in default_dict

    with_source = p.to_dict(include_source=True)
    assert with_source["source_data_json"] == {"foo": "bar"}
