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


CLASS_COLORS = {
    'person': (0, 0, 255),
    'fire': (0, 0, 255),
    'flood_water': (255, 0, 0),
    'collapsed_building': (0, 165, 255),
    'debris': (0, 255, 255),
    'vehicle_car': (0, 255, 0),
    'vehicle_bicycle': (255, 255, 0),
    'vehicle_motorcycle': (255, 0, 255),
    'vehicle_bus': (128, 0, 128),
    'vehicle_truck': (128, 128, 0),
    'smoke': (180, 180, 180),
    'fallen_tree': (0, 128, 0),
    'fallen_power_pole': (0, 128, 128),
    'power_line': (128, 128, 128),
    'gas_cylinder': (0, 0, 128),
}


def get_class_color(label):
    """Return a distinct BGR color tuple for a given class label."""
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    h = hash(label)
    return ((h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF))


def check_and_warn_unfinetuned_model(model):
    """Check if the loaded YOLO model is an un-fine-tuned COCO model (80 classes)."""
    names = getattr(model, 'names', {})
    is_coco = (len(names) == 80 and names.get(0) == 'person' and names.get(79) == 'toothbrush')
    if is_coco:
        print(
            "WARNING: Using ground-level COCO-trained model on aerial/drone imagery — vehicle and building "
            "classifications may be unreliable. Fine-tune on aerial datasets for accurate results."
        )
    return is_coco
