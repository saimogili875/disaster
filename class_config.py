CANONICAL_CLASSES = {
    "person": ["person"],
    "vehicle_car": ["car", "vehicle_car"],
    "vehicle_truck": ["truck", "vehicle_truck"],
    "vehicle_bus": ["bus", "vehicle_bus"],
    "vehicle_motorcycle": ["motorcycle", "vehicle_motorcycle"],
    "vehicle_bicycle": ["bicycle", "vehicle_bicycle"],
    "fire": ["fire"],
    "smoke": ["smoke"],
    "flood_water": ["flood_water", "flood", "water"],
    "collapsed_building": ["collapsed_building", "building-total-destruction"],
    "damaged_building": ["building-major-damage", "building-minor-damage", "damaged_building"],
    "debris": ["debris"],
    "fallen_tree": ["fallen_tree", "tree"],
    "fallen_power_pole": ["fallen_power_pole", "power_pole"],
    "power_line": ["power_line"],
    "gas_cylinder": ["gas_cylinder"],
    "dog": ["dog"],
    "cat": ["cat"],
    "cow": ["cow"],
    "horse": ["horse"],
}

# Pre-compute reverse mapping from lowercased aliases to canonical class names
_ALIAS_LOOKUP = {}
for canonical_name, aliases in CANONICAL_CLASSES.items():
    _ALIAS_LOOKUP[canonical_name.lower()] = canonical_name
    for alias in aliases:
        _ALIAS_LOOKUP[alias.lower()] = canonical_name

def normalize_class_name(raw_name):
    """
    Look up raw_name against CANONICAL_CLASSES alias lists and return the canonical name.
    If no match is found, print a warning and return raw_name unchanged.
    """
    if not isinstance(raw_name, str):
        return raw_name

    key = raw_name.strip().lower()
    if key in _ALIAS_LOOKUP:
        return _ALIAS_LOOKUP[key]

    print(f"Warning: Class name '{raw_name}' not found in CANONICAL_CLASSES. Returning unchanged.")
    return raw_name
