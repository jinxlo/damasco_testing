import importlib.util
import os
from pathlib import Path

MODULE_PATH = os.path.join(Path(__file__).resolve().parents[1],
                           'namwoo_app', 'services', 'recommender_service.py')
spec = importlib.util.spec_from_file_location('recommender_service', MODULE_PATH)
recommender = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recommender)
rank_products = recommender.rank_products


def test_rank_prefers_budget_fit():
    items = [
        {"item_code": "A", "price": 1000, "similarity": 0.9, "item_name": "Pro Phone", "brand": "BrandX", "category": "phones"},
        {"item_code": "B", "price": 350, "similarity": 0.85, "item_name": "Budget Phone", "brand": "BrandY", "category": "phones"},
    ]
    ranked = rank_products({"raw_query": "cheap flagship", "budget_max": 400}, items)
    assert ranked[0]["item_code"] == "B"


def test_rank_handles_empty_list():
    assert rank_products({"raw_query": "phone"}, []) == []


def test_rank_filters_accessories():
    items = [
        {"item_code": "A", "price": 10, "similarity": 0.9, "item_name": "Charger", "brand": "X", "category": "accessories", "is_accessory": True},
        {"item_code": "B", "price": 20, "similarity": 0.8, "item_name": "Cable", "brand": "X", "category": "accessories", "is_accessory": True},
    ]
    assert rank_products({"raw_query": "charger"}, items) == []
