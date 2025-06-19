import os
import importlib.util
import pytest

UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "namwoo_app", "utils", "product_utils.py"))
spec = importlib.util.spec_from_file_location("product_utils", UTILS_PATH)
product_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(product_utils)
generate_product_location_id = product_utils.generate_product_location_id
extract_color_from_name = product_utils.extract_color_from_name
group_products_by_model = product_utils.group_products_by_model
get_available_brands = product_utils.get_available_brands
format_product_response = product_utils.format_product_response
format_brand_list = product_utils.format_brand_list

WH_UTILS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "namwoo_app", "utils", "whs_utils.py"))
wh_spec = importlib.util.spec_from_file_location("whs_utils", WH_UTILS_PATH)
whs_utils = importlib.util.module_from_spec(wh_spec)
wh_spec.loader.exec_module(whs_utils)
canonicalize_whs_name = whs_utils.canonicalize_whs_name
product_utils.canonicalize_whs_name = canonicalize_whs_name


def test_generate_product_location_id_basic():
    assert generate_product_location_id("ABC123", "Main Warehouse") == "ABC123_Main_Warehouse"


def test_generate_product_location_id_none_or_blank():
    assert generate_product_location_id(None, "Main") is None
    assert generate_product_location_id("Item", "") is None
    assert generate_product_location_id("Item", "   ") is None


def test_generate_product_location_id_sanitizes():
    assert generate_product_location_id("A1", "Warehouse/Location-1") == "A1_Warehouse_Location-1"


def test_generate_product_location_id_truncates():
    item_code = "A" * 500
    whs_name = "B" * 20
    result = generate_product_location_id(item_code, whs_name)
    expected = (f"{item_code}_{whs_name}")[:512]
    assert result == expected


def test_extract_color_from_name():
    color, base = extract_color_from_name("TECNO CAMON 30 GRIS")
    assert color == "Gris"
    assert base == "TECNO CAMON 30"

    # English colour names should also be recognised
    color, base = extract_color_from_name("INFINIX HOT 50 BLUE")
    assert color == "Blue"
    assert base == "INFINIX HOT 50"

    # Spanish feminine and accented names
    color, base = extract_color_from_name("LICUADORA BLANCA")
    assert color == "Blanca"
    assert base == "LICUADORA"

    color, base = extract_color_from_name("PARLANTE PÚRPURA")
    assert color in ("Púrpura", "Purpura")
    assert base == "PARLANTE"


def test_extract_color_with_extra_words():
    text = "TECNO CAMON 40 256+8 BLANCO + OBSEQUIO"
    color, base = extract_color_from_name(text)
    assert color == "Blanco"
    assert base == "TECNO CAMON 40"


def test_group_products_by_model_and_brands():
    items = [
        {"itemName": "INFINIX HOT 50 NEGRO", "brand": "INFINIX", "price": 240},
        {"itemName": "INFINIX HOT 50 VERDE", "brand": "INFINIX", "price": 240},
        {"itemName": "TECNO SPARK 20 AZUL", "brand": "TECNO", "price": 200},
        {"itemName": "INFINIX HOT 50 BLUE", "brand": "INFINIX", "price": 250},
    ]
    grouped = group_products_by_model(items)
    assert len(grouped) == 2
    hot50 = next(g for g in grouped if g["model"] == "INFINIX HOT 50")
    assert sorted(hot50["colors"]) == ["Blue", "Negro", "Verde"]
    brands = get_available_brands(items)
    assert set(brands) == {"INFINIX", "TECNO"}


def test_format_product_response_and_brand_list():
    grouped_product = {
        "model": "TECNO CAMON 30",
        "price": 240,
        "priceBolivar": 24082,
        "colors": ["Negro", "Gris"],
        "description": "Pantalla FHD+",
        "store": "Caracas (CCCT)",
    }
    msg = format_product_response(grouped_product)
    assert "TECNO CAMON 30" in msg
    assert "$240.00" in msg
    assert "Bs. 24,082.00" in msg
    assert "Negro, Gris" in msg
    assert "Pantalla FHD+" in msg
    assert "Caracas (CCCT)" in msg

    brand_message = format_brand_list(["XIAOMI", "TECNO", "SAMSUNG"])
    assert brand_message.startswith("📱 Estas son las marcas")
    assert "🔹 SAMSUNG" in brand_message


def test_get_key_specs_truncates_especificacion():
    long_spec = "a" * 250 + "\nsecond line should be ignored"
    product = {"especificacion": long_spec}
    result = product_utils._get_key_specs(product)
    assert len(result) == 200
    assert "second line" not in result


def test_get_key_specs_truncates_llm_summary():
    long_summary = "Esta es una oracion muy larga " + ("b" * 240) + ". Otra oracion que no debe aparecer."
    product = {"llm_summarized_description": long_summary}
    result = product_utils._get_key_specs(product)
    assert len(result) == 200
    assert "Otra oracion" not in result


def test_user_is_asking_for_price_detection():
    price_msgs = [
        "¿Cuál es el precio del TECNO SPARK?",
        "precio de D0001234",
        "How much is the iPhone 15?",
        "que precio tienes el samsung a36 blanco?",
        "Cuánto cuesta el A16?",
        "En cuanto sale el Samsung A16",
        "uanto cuesta el samsun a16",
    ]
    for msg in price_msgs:
        assert product_utils.user_is_asking_for_price(msg)

    non_price_msgs = [
        "Tienes modelos disponibles?",
        "busco el mas barato",
    ]
    for msg in non_price_msgs:
        assert not product_utils.user_is_asking_for_price(msg)


def test_user_is_asking_for_list_detection():
    list_msgs = [
        "me puedes decir los modelos samsung que tienes disponible?",
        "¿qué modelos de xiaomi tienen?",
        "muéstrame todos los modelos que tienes"
    ]
    for msg in list_msgs:
        assert product_utils.user_is_asking_for_list(msg)

    non_list_msgs = [
        "tienes el modelo a26?",
        "quiero saber el precio del samsung a26"
    ]
    for msg in non_list_msgs:
        assert not product_utils.user_is_asking_for_list(msg)


def test_user_is_asking_for_best_detection():
    best_msgs = [
        "y de estos 3 cual seria el mejor?",
        "cual es el mejor de estos?",
        "which one is best?",
    ]
    for msg in best_msgs:
        assert product_utils.user_is_asking_for_best(msg)

    non_best_msgs = [
        "me recomiendas uno", 
        "dame detalles del a26",
    ]
    for msg in non_best_msgs:
        assert not product_utils.user_is_asking_for_best(msg)


def test_generate_product_location_id_canonicalizes_branch():
    result = generate_product_location_id("SKU1", "CCCT")
    assert result == "SKU1_Almacen_Principal_CCCT"


def test_canonicalize_whs_name_maps_branch():
    assert canonicalize_whs_name("CCCT") == "Almacen Principal CCCT"


def test_extract_brand_from_message():
    assert product_utils.extract_brand_from_message("que modelos samsung tienes") == "SAMSUNG"
    assert product_utils.extract_brand_from_message("tienes celulares xiaomi?") == "XIAOMI"
    assert product_utils.extract_brand_from_message("busco huawei") is None

