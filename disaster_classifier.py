"""
Disaster Type Classifier — auto-labels zones based on detection mix.

Analyzes the objects detected in a zone and classifies the disaster scenario:
  - FIRE EVENT: fire + smoke dominant
  - FLOOD EVENT: flood_water dominant
  - EARTHQUAKE: collapsed/damaged buildings + debris dominant
  - STORM: fallen trees + power poles + debris, no fire/flood
  - INDUSTRIAL: gas cylinders + fire/smoke
  - MULTI-HAZARD: significant presence of 2+ disaster types
  - LOW RISK: mostly vehicles/people, no hazards

Each zone gets a primary classification and a confidence level.
"""


DISASTER_SIGNATURES = {
    "FIRE EVENT": {
        "required": {"fire", "smoke"},
        "supporting": {"gas_cylinder", "debris"},
        "weight_fields": {"fire": 3.0, "smoke": 2.0, "gas_cylinder": 1.0},
        "min_required_match": 1,
        "color": "#ef4444",
        "icon": "F",
    },
    "FLOOD EVENT": {
        "required": {"flood_water"},
        "supporting": {"fallen_tree", "debris", "vehicle_car", "vehicle_truck"},
        "weight_fields": {"flood_water": 4.0, "fallen_tree": 1.0},
        "min_required_match": 1,
        "color": "#3b82f6",
        "icon": "W",
    },
    "EARTHQUAKE": {
        "required": {"collapsed_building", "damaged_building"},
        "supporting": {"debris", "fallen_power_pole", "power_line", "person"},
        "weight_fields": {"collapsed_building": 4.0, "damaged_building": 2.5, "debris": 1.5, "fallen_power_pole": 1.0},
        "min_required_match": 1,
        "color": "#a855f7",
        "icon": "E",
    },
    "STORM": {
        "required": {"fallen_tree", "fallen_power_pole"},
        "supporting": {"debris", "power_line", "damaged_building"},
        "weight_fields": {"fallen_tree": 3.0, "fallen_power_pole": 2.5, "debris": 1.0, "power_line": 1.0},
        "min_required_match": 1,
        "color": "#6366f1",
        "icon": "S",
    },
    "INDUSTRIAL HAZARD": {
        "required": {"gas_cylinder"},
        "supporting": {"fire", "smoke"},
        "weight_fields": {"gas_cylinder": 4.0, "fire": 2.0, "smoke": 1.5},
        "min_required_match": 1,
        "color": "#f59e0b",
        "icon": "I",
    },
}


def classify_zone(zone):
    """
    Classify a zone's disaster type based on its detection mix.

    Args:
        zone: Zone model instance

    Returns:
        dict with:
            disaster_type: str (e.g. "FIRE EVENT", "EARTHQUAKE")
            confidence: str ("high", "moderate", "low")
            confidence_pct: int (0-100)
            color: str (hex color)
            icon: str (single char)
            scores: dict of disaster_type -> score
            secondary_type: str or None (if multi-hazard)
    """
    from detection.models import Detection

    dets = Detection.objects.filter(zone=zone)
    if not dets.exists():
        return {
            "disaster_type": "NO DATA",
            "confidence": "none",
            "confidence_pct": 0,
            "color": "#64748b",
            "icon": "?",
            "scores": {},
            "secondary_type": None,
        }

    class_counts = {}
    for det in dets.values("object_class").distinct():
        cls = det["object_class"]
        class_counts[cls] = dets.filter(object_class=cls).count()

    total_dets = sum(class_counts.values())
    scores = {}

    for disaster_type, sig in DISASTER_SIGNATURES.items():
        score = 0.0
        required_matched = 0

        for cls in sig["required"]:
            if cls in class_counts and class_counts[cls] > 0:
                required_matched += 1
                weight = sig["weight_fields"].get(cls, 1.0)
                score += class_counts[cls] * weight

        if required_matched < sig["min_required_match"]:
            scores[disaster_type] = 0.0
            continue

        for cls in sig["supporting"]:
            if cls in class_counts and class_counts[cls] > 0:
                weight = sig["weight_fields"].get(cls, 0.5)
                score += class_counts[cls] * weight

        if total_dets > 0:
            score = score / total_dets

        scores[disaster_type] = round(score, 3)

    if not scores or max(scores.values()) == 0:
        hazard_classes = {"fire", "smoke", "flood_water", "collapsed_building",
                          "damaged_building", "debris", "fallen_tree",
                          "fallen_power_pole", "power_line", "gas_cylinder"}
        hazard_count = sum(class_counts.get(c, 0) for c in hazard_classes)

        if hazard_count == 0:
            return {
                "disaster_type": "LOW RISK",
                "confidence": "high",
                "confidence_pct": 85,
                "color": "#22c55e",
                "icon": "L",
                "scores": scores,
                "secondary_type": None,
            }

    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top_type, top_score = sorted_types[0] if sorted_types else ("UNKNOWN", 0)

    secondary_type = None
    if len(sorted_types) >= 2 and sorted_types[1][1] > 0:
        ratio = sorted_types[1][1] / top_score if top_score > 0 else 0
        if ratio > 0.6:
            secondary_type = sorted_types[1][0]
            top_type = "MULTI-HAZARD"

    if top_type == "MULTI-HAZARD":
        color = "#ef4444"
        icon = "M"
    else:
        sig = DISASTER_SIGNATURES.get(top_type, {})
        color = sig.get("color", "#64748b")
        icon = sig.get("icon", "?")

    if top_score > 2.0:
        confidence = "high"
        confidence_pct = 90
    elif top_score > 1.0:
        confidence = "moderate"
        confidence_pct = 65
    elif top_score > 0:
        confidence = "low"
        confidence_pct = 35
    else:
        confidence = "none"
        confidence_pct = 0

    return {
        "disaster_type": top_type,
        "confidence": confidence,
        "confidence_pct": confidence_pct,
        "color": color,
        "icon": icon,
        "scores": scores,
        "secondary_type": secondary_type,
    }


def classify_all_zones():
    """
    Classify all zones and return sorted results.

    Returns:
        list of dicts, each with zone_id, center_lat, center_lon, and classification fields.
    """
    from detection.models import Zone

    results = []
    for zone in Zone.objects.all():
        classification = classify_zone(zone)
        results.append({
            "zone_id": zone.zone_id,
            "center_lat": zone.center_lat,
            "center_lon": zone.center_lon,
            **classification,
        })

    results.sort(key=lambda x: x["confidence_pct"], reverse=True)
    return results
