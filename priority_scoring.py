"""
Priority Scoring Engine — Category 12 of the Disaster Management Framework.

Computes per-zone risk scores by combining hazard detections, damage severity,
route disruption, people/animal presence, infrastructure exposure, and data
confidence. Populates ZoneRiskSnapshot records for the report and map layers.
"""

import math
from django.utils import timezone
from detection.models import Detection, Zone, ZoneRiskSnapshot, FlightPass
from infrastructure_layer import fetch_critical_infrastructure
from freshness_tracking import compute_zone_freshness

WEIGHTS = {
    "people": 0.30,
    "structures": 0.25,
    "fire": 0.15,
    "flood": 0.10,
    "electrical": 0.10,
    "debris": 0.05,
    "infrastructure": 0.05,
}

RISK_THRESHOLDS = [
    (0.70, "CRITICAL"),
    (0.40, "HIGH"),
    (0.15, "MODERATE"),
    (0.00, "LOW"),
]

RISK_COLOR_MAP = {
    "CRITICAL": "red",
    "HIGH": "orange",
    "MODERATE": "yellow",
    "LOW": "green",
}


def _sigmoid_scale(count, midpoint=5.0, steepness=0.6):
    """Map a raw detection count to 0-1 using a sigmoid curve.
    midpoint=5 means 5 detections → 0.5 score; steepness controls ramp."""
    if count <= 0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-steepness * (count - midpoint)))


def _compute_zone_score(zone):
    """Compute raw risk score for a single zone from its detections."""
    dets = Detection.objects.filter(zone=zone)

    person_count = dets.filter(object_class="person").count()
    animal_count = dets.filter(object_class__in=["dog", "cat", "cow", "horse"]).count()
    people_score = _sigmoid_scale(person_count + animal_count, midpoint=3.0)

    collapsed = dets.filter(object_class="collapsed_building").count()
    damaged = dets.filter(object_class="damaged_building").count()
    structures_score = _sigmoid_scale(collapsed * 2 + damaged, midpoint=3.0)

    fire_count = dets.filter(object_class="fire").count()
    smoke_count = dets.filter(object_class="smoke").count()
    high_fire = dets.filter(object_class="fire", fire_intensity="high").count()
    fire_score = _sigmoid_scale(fire_count + smoke_count * 0.5 + high_fire * 2, midpoint=3.0)

    flood_count = dets.filter(object_class="flood_water").count()
    fast_water = dets.filter(object_class="flood_water", flow_speed="fast").count()
    flood_score = _sigmoid_scale(flood_count + fast_water * 2, midpoint=4.0)

    fallen_poles = dets.filter(object_class="fallen_power_pole").count()
    power_lines = dets.filter(object_class="power_line").count()
    near_water = 1 if (fallen_poles > 0 and flood_count > 0) else 0
    electrical_score = _sigmoid_scale(fallen_poles + power_lines + near_water * 3, midpoint=2.0)

    debris_count = dets.filter(object_class="debris").count()
    fallen_trees = dets.filter(object_class="fallen_tree").count()
    debris_score = _sigmoid_scale(debris_count + fallen_trees, midpoint=4.0)

    infra_items = fetch_critical_infrastructure(zone.center_lat, zone.center_lon, radius_m=500)
    infra_nearby = len([i for i in infra_items if i.get("type") in ("hospital", "shelter")])
    has_critical_infra = 1.0 if infra_nearby > 0 else 0.0
    hazard_present = 1.0 if (fire_count + flood_count + collapsed) > 0 else 0.0
    infra_score = has_critical_infra * hazard_present

    weighted_score = (
        WEIGHTS["people"] * people_score
        + WEIGHTS["structures"] * structures_score
        + WEIGHTS["fire"] * fire_score
        + WEIGHTS["flood"] * flood_score
        + WEIGHTS["electrical"] * electrical_score
        + WEIGHTS["debris"] * debris_score
        + WEIGHTS["infrastructure"] * infra_score
    )

    weighted_score = min(1.0, max(0.0, weighted_score))

    total_dets = dets.count()
    component_scores = {
        "people": round(people_score, 3),
        "structures": round(structures_score, 3),
        "fire": round(fire_score, 3),
        "flood": round(flood_score, 3),
        "electrical": round(electrical_score, 3),
        "debris": round(debris_score, 3),
        "infrastructure": round(infra_score, 3),
    }

    return weighted_score, total_dets, component_scores


def _score_to_label(score):
    """Map a 0-1 risk score to a risk label."""
    for threshold, label in RISK_THRESHOLDS:
        if score >= threshold:
            return label
    return "LOW"


def _compute_confidence(detection_count, freshness_info):
    """Compute confidence level from detection count and data freshness."""
    if detection_count == 0:
        return "none"

    age_minutes = freshness_info.get("age_minutes", 999)
    is_stale = freshness_info.get("stale_flag", True)

    if is_stale:
        return "low"
    if detection_count < 3:
        return "low"
    if detection_count < 10:
        if age_minutes < 15:
            return "high"
        return "moderate"
    if age_minutes < 10:
        return "high"
    return "moderate"


def compute_zone_priorities(flight_pass=None):
    """
    Compute risk scores for all zones and save as ZoneRiskSnapshot records.

    Parameters:
        flight_pass (FlightPass, optional): Associate snapshots with this flight pass.
            If None, uses the latest active or completed pass.

    Returns:
        list of dict: Scored zones sorted by risk (highest first).
    """
    if flight_pass is None:
        flight_pass = FlightPass.objects.order_by("-start_time").first()

    zones = Zone.objects.all()
    results = []

    for zone in zones:
        score, det_count, components = _compute_zone_score(zone)
        label = _score_to_label(score)

        latest_det = Detection.objects.filter(zone=zone).order_by("-timestamp").first()
        computed_at = latest_det.timestamp if latest_det else timezone.now()
        freshness = compute_zone_freshness(computed_at)
        confidence = _compute_confidence(det_count, freshness)

        if flight_pass:
            ZoneRiskSnapshot.objects.update_or_create(
                zone=zone,
                flight_pass=flight_pass,
                defaults={
                    "risk_score": round(score, 4),
                    "risk_label": label,
                    "confidence": confidence,
                    "detection_count": det_count,
                },
            )

        results.append({
            "zone_id": zone.zone_id,
            "center_lat": zone.center_lat,
            "center_lon": zone.center_lon,
            "risk_score": round(score, 4),
            "risk_label": label,
            "risk_color": RISK_COLOR_MAP.get(label, "green"),
            "confidence": confidence,
            "detection_count": det_count,
            "freshness": freshness,
            "components": components,
        })

    results.sort(key=lambda x: x["risk_score"], reverse=True)
    return results
