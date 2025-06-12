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
