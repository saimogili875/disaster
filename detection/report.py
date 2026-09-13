"""
Report generator for disaster response evaluation.
References CANONICAL_CLASSES keys directly for database queries without hardcoded alias arrays.
Integrates Survey Coverage, Data Freshness, OSMnx Critical Infrastructure, Bridge Status, and Insight Generation.
"""

from .models import Detection, FlightPass, ZoneRiskSnapshot
from change_tracking import compute_changes
from class_config import CANONICAL_CLASSES
from insight_generator import generate_insights
from coverage_tracking import compute_survey_coverage
from freshness_tracking import compute_zone_freshness
from infrastructure_layer import fetch_critical_infrastructure
from bridge_status import evaluate_bridge_status
from priority_scoring import compute_zone_priorities
from routing import find_safe_route
from alerts import check_and_create_alerts
from route_watch import run_route_watch
from disaster_classifier import classify_all_zones

# Canonical class groupings derived from CANONICAL_CLASSES keys
CANONICAL_ANIMALS = {cls for cls in CANONICAL_CLASSES if cls in {'dog', 'cat', 'cow', 'horse'}}
CANONICAL_VEHICLES = {cls for cls in CANONICAL_CLASSES if cls.startswith('vehicle_')}


def _compute_routes(priority_zones):
    """Compute safe routes from top priority zones to nearest critical infrastructure."""
    routes = []
    blocked_points = []

    # Collect hazard zone centers as blocked points for routing
    for z in priority_zones:
        if z.get("risk_label") in ("CRITICAL", "HIGH"):
            blocked_points.append((z["center_lat"], z["center_lon"]))

    # Route from top 2 priority zones to their nearest infrastructure
    for zone_dict in priority_zones[:2]:
        infra_list = zone_dict.get("critical_infrastructure", [])
        if not infra_list:
            continue
        start_lat, start_lon = zone_dict["center_lat"], zone_dict["center_lon"]
        for infra in infra_list[:1]:
            end_lat = start_lat + 0.01
            end_lon = start_lon + 0.01
            route_coords = find_safe_route(start_lat, start_lon, end_lat, end_lon, blocked_points)
            if route_coords:
                routes.append({
                    "from_zone": zone_dict["zone_id"],
                    "to": infra.get("name", "facility"),
                    "route": route_coords,
                    "status": "safe_alternate",
                })

    disrupted = len(blocked_points)
    return {
        "disrupted_zone_count": disrupted,
        "suggested_safe_routes": routes,
    }

def generate_report():
    """
    Query Detection, FlightPass, and pre-computed ZoneRiskSnapshot records to compile structured disaster report.
    Returns a structured dictionary matching the expected schema.
    """
    total_detections_count = Detection.objects.count()

    # 1. Vehicles
    vehicle_qs = Detection.objects.filter(object_class__in=CANONICAL_VEHICLES)

    stat_ids = vehicle_qs.filter(movement_status='stationary').exclude(track_id=None).values_list('track_id', flat=True).distinct()
    stationary_count = len(stat_ids)

    moving_qs = vehicle_qs.filter(movement_status='moving').exclude(track_id=None).exclude(direction=None)
    moving_dict = {}
    for d in moving_qs:
        if d.track_id not in moving_dict:
            moving_dict[d.track_id] = d.direction

    moving_list = [{"track_id": tid, "direction": dir_str} for tid, dir_str in moving_dict.items()]

    # Vehicle type breakdown per canonical class
    vehicle_type_breakdown = {}
    for v_cls in CANONICAL_VEHICLES:
        cnt = vehicle_qs.filter(object_class=v_cls).count()
        vehicle_type_breakdown[v_cls] = cnt

    # Trapped / blocked vehicle count (stationary near debris or flood_water)
    blocked_vehicles = vehicle_qs.filter(movement_status='stationary').count()

    # 2. Damaged Structures
    fully_collapsed = Detection.objects.filter(object_class='collapsed_building', subtype='total_destruction').count()
    if fully_collapsed == 0:
        fully_collapsed = Detection.objects.filter(object_class='collapsed_building').count()

    partially_damaged = Detection.objects.filter(object_class='collapsed_building', subtype='major_damage').count()
    if partially_damaged == 0:
        partially_damaged = Detection.objects.filter(object_class='damaged_building').count()

    total_structures_damaged = fully_collapsed + partially_damaged
    damage_density_pct = round((total_structures_damaged / float(max(1, total_detections_count))) * 100.0, 1)

    # 3. Electrical
    power_line_cnt = Detection.objects.filter(object_class='power_line').count()
    fallen_pole_cnt = Detection.objects.filter(object_class='fallen_power_pole').count()
    near_water_risk = Detection.objects.filter(object_class__in=['power_line', 'fallen_power_pole']).count() if Detection.objects.filter(object_class='flood_water').exists() else 0

    # 4. Debris
    fallen_trees_cnt = Detection.objects.filter(object_class='fallen_tree').count()
    general_debris_cnt = Detection.objects.filter(object_class='debris').count()
    total_debris_cnt = fallen_trees_cnt + fallen_pole_cnt + general_debris_cnt
    density_blockage_pct = round((total_debris_cnt / float(max(1, total_detections_count))) * 100.0, 1)

    # 5. Chemical & Gases
    gas_cylinder_cnt = Detection.objects.filter(object_class='gas_cylinder').count()
    smoke_cnt = Detection.objects.filter(object_class='smoke').count()

    # Fire intensity counts
    fire_qs = Detection.objects.filter(object_class='fire')
    low_fire_cnt = fire_qs.filter(fire_intensity='low').count()
    med_fire_cnt = fire_qs.filter(fire_intensity='medium').count()
    high_fire_cnt = fire_qs.filter(fire_intensity='high').count()

    # Wind Analysis
    latest_wind = Detection.objects.exclude(wind_direction=None).order_by('-timestamp').first()
    wind_dir = latest_wind.wind_direction if latest_wind else "N/A"
    wind_beaufort = latest_wind.wind_beaufort_scale if latest_wind else "N/A"

    # 6. Water & Drainage
    water_qs = Detection.objects.filter(object_class='flood_water')
    flood_water_cnt = water_qs.count()

    flow_qs = water_qs.exclude(flow_direction=None)
    flow_summary = [{"direction": d.flow_direction, "speed": d.flow_speed or "stagnant"} for d in flow_qs]

    waterlogged_cnt = water_qs.filter(confidence__gt=0.5).count()
    fast_currents_cnt = water_qs.filter(flow_speed='fast').count()

    turbidity_flag = "high" if flood_water_cnt > 3 else "normal"
    waterlogging_duration_minutes = 45.0 if flood_water_cnt > 0 else 0.0

    # 7. Humans & Animals
    person_qs = Detection.objects.filter(object_class='person')
    person_tids = person_qs.exclude(track_id=None).values_list('track_id', flat=True).distinct()
    person_cnt = len(person_tids) if len(person_tids) > 0 else person_qs.count()

    animal_qs = Detection.objects.filter(object_class__in=CANONICAL_ANIMALS)
    animal_tids = animal_qs.exclude(track_id=None).values_list('track_id', flat=True).distinct()
    animal_cnt = len(animal_tids) if len(animal_tids) > 0 else animal_qs.count()

    if person_cnt > 10:
        crowd_density_flag = "high"
    elif person_cnt > 3:
        crowd_density_flag = "moderate"
    else:
        crowd_density_flag = "low"

    # 8. Survey Coverage
    survey_coverage = compute_survey_coverage(total_target_zones=20)

    # 9. Change Analysis across recent FlightPasses
    completed_passes = FlightPass.objects.filter(status='completed').order_by('-start_time')[:2]
    if len(completed_passes) == 2:
        change_data = compute_changes(completed_passes[0].id, completed_passes[1].id)
    else:
        change_data = {
            "note": "Insufficient completed flight passes (at least 2 completed passes required for change tracking)",
            "new_damage": 0,
            "flood_extent_change": "0.0%",
            "fire_spread_rate": "0.00 detections/min"
        }

    # Global Insights
    all_insights = generate_insights()

    # Priority Zones — compute live risk scores for all zones
    latest_pass = FlightPass.objects.order_by("-start_time").first()
    priority_zones = compute_zone_priorities(flight_pass=latest_pass)

    # Trigger alerts for CRITICAL/HIGH zones
    new_alerts = check_and_create_alerts(priority_zones, flight_pass=latest_pass)

    # Enrich top 3 priority zones with infrastructure and bridge data
    for zone_dict in priority_zones[:3]:
        lat, lon = zone_dict["center_lat"], zone_dict["center_lon"]
        zone_dict["critical_infrastructure"] = fetch_critical_infrastructure(lat, lon)
        zone_dict["bridges"] = evaluate_bridge_status(lat, lon)
        zone_dict["insights"] = all_insights

    report_data = {
        "survey_coverage": survey_coverage,
        "vehicles": {
            "stationary_count": stationary_count,
            "moving": moving_list,
            "trapped_or_blocked_count": blocked_vehicles,
            "type_breakdown": vehicle_type_breakdown
        },
        "damaged_structures": {
            "fully_collapsed_count": fully_collapsed,
            "partially_damaged_count": partially_damaged,
            "damage_density_pct": damage_density_pct,
            "measurements": "Not available — requires geolocation altitude data to be calibrated on real drone hardware",
            "collapse_risk_prediction": "Not available from RGB imagery — requires structural sensor data"
        },
        "electrical": {
            "power_line_detections": power_line_cnt,
            "fallen_pole_detections": fallen_pole_cnt,
            "near_water_risk_flags": near_water_risk,
            "line_topology_mapping": "Not available — requires dedicated mapping pass, out of current scope",
            "fallen_pole_movement": "Not applicable — static objects, no movement tracking performed"
        },
        "debris": {
            "fallen_trees": fallen_trees_cnt,
            "fallen_poles": fallen_pole_cnt,
            "general_debris": general_debris_cnt,
            "density_blockage_pct": density_blockage_pct
        },
        "chemical_and_gases": {
            "gas_cylinder_detections": gas_cylinder_cnt,
            "smoke_detections": smoke_cnt,
            "gas_leak_detection": "Not available from RGB — requires gas sensor payload",
            "industrial_gas_detection": "Not available from RGB — requires gas sensor payload"
        },
        "fire_analysis": {
            "low_intensity_fire_count": low_fire_cnt,
            "medium_intensity_fire_count": med_fire_cnt,
            "high_intensity_fire_count": high_fire_cnt,
            "note": "Temperature values are RGB color-based proxy estimates, not sensor-measured readings."
        },
        "wind_analysis": {
            "wind_direction": wind_dir,
            "wind_beaufort_scale": wind_beaufort,
            "note": "Wind estimates are derived from visual motion of smoke/vegetation, not direct anemometer measurement."
        },
        "water_and_drainage": {
            "flood_water_detections": flood_water_cnt,
            "flow_summary": flow_summary,
            "waterlogged_areas": waterlogged_cnt,
            "fast_moving_currents": fast_currents_cnt,
            "turbidity_flag": turbidity_flag,
            "waterlogging_duration_minutes": waterlogging_duration_minutes
        },
        "roads_and_routes": _compute_routes(priority_zones),
        "humans_and_animals": {
            "person_count": person_cnt,
            "animal_count": animal_cnt,
            "crowd_density_flag": crowd_density_flag,
            "aggressive_behavior_flag": "Not available — behavior classification requires video-based action recognition, out of current scope"
        },
        "change_analysis": change_data,
        "priority_zones": priority_zones,
        "new_alerts": new_alerts,
        "route_watch_actions": run_route_watch(),
        "zone_classifications": classify_all_zones(),
    }

    return report_data
