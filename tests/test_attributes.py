from backend.core.processor import build_attributes_string


def test_build_attributes_string_person():
    object_data = {
        "class": "person",
        "attributes": {
            "shirt_color": "Blue",
            "pant_color": "Black",
            "gender": "Male",
        },
    }
    assert build_attributes_string(object_data) == "Upper: Blue, Lower: Black, Gender: Male"


def test_build_attributes_string_vehicle_with_plate():
    object_data = {
        "class": "car",
        "attributes": {
            "vehicle_color": "White",
            "number_plate": "TS09AB1234",
            "body_type": "sedan",
        },
    }
    assert build_attributes_string(object_data) == "White, Plate: TS09AB1234, Category: sedan"


def test_build_attributes_string_unknown_defaults():
    object_data = {"class": "unknown", "attributes": {}}
    assert build_attributes_string(object_data) == "Unknown"
