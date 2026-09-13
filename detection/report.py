from .models import Detection, FlightPass, ZoneRiskSnapshot
from change_tracking import compute_changes
from class_config import CANONICAL_CLASSES

# Canonical class groupings derived from CANONICAL_CLASSES keys
CANONICAL_ANIMALS = {cls for cls in CANONICAL_CLASSES if cls in {'dog', 'cat', 'cow', 'horse'}}
CANONICAL_VEHICLES = {cls for cls in CANONICAL_CLASSES if cls.startswith('vehicle_')}

def generate_report():
    """
    Query Detection, FlightPass, and pre-computed ZoneRiskSnapshot records to compile structured disaster report.
    Returns a structured dictionary matching the expected schema.
    """
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

    # 2. Damaged Structures
    fully_collapsed = Detection.objects.filter(object_class='collapsed_building', subtype='total_destruction').count()
    if fully_collapsed == 0:
        fully_collapsed = Detection.objects.filter(object_class='collapsed_building').count()

    partially_damaged = Detection.objects.filter(object_class='collapsed_building', subtype='major_damage').count()
    if partially_damaged == 0:
        partially_damaged = Detection.objects.filter(object_class='damaged_building').count()

    # 3. Electrical
    power_line_cnt = Detection.objects.filter(object_class='power_line').count()
    fallen_pole_cnt = Detection.objects.filter(object_class='fallen_power_pole').count()

    # 4. Debris
    fallen_trees_cnt = Detection.objects.filter(object_class='fallen_tree').count()
    general_debris_cnt = Detection.objects.filter(object_class='debris').count()

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

    # 6. Water & Drainage — Single reference to canonical 'flood_water'
    water_qs = Detection.objects.filter(object_class='flood_water')
    flood_water_cnt = water_qs.count()

    flow_qs = water_qs.exclude(flow_direction=None)
    flow_summary = [{"direction": d.flow_direction, "speed": d.flow_speed or "stagnant"} for d in flow_qs]

    waterlogged_cnt = water_qs.filter(confidence__gt=0.5).count()
    fast_currents_cnt = water_qs.filter(flow_speed='fast').count()

    # 7. Humans & Animals
    person_qs = Detection.objects.filter(object_class='person')
    person_tids = person_qs.exclude(track_id=None).values_list('track_id', flat=True).distinct()
    person_cnt = len(person_tids) if len(person_tids) > 0 else person_qs.count()

    animal_qs = Detection.objects.filter(object_class__in=CANONICAL_ANIMALS)
    animal_tids = animal_qs.exclude(track_id=None).values_list('track_id', flat=True).distinct()
    animal_cnt = len(animal_tids) if len(animal_tids) > 0 else animal_qs.count()

    # 8. Change Analysis across recent FlightPasses
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

    # Priority Zones (Query pre-computed ZoneRiskSnapshot per zone)
    snapshots = ZoneRiskSnapshot.objects.select_related('zone', 'flight_pass').order_by('-computed_at')
    zone_map = {}
    for s in snapshots:
        if s.zone_id not in zone_map:
            zone_map[s.zone_id] = {
                "zone_id": s.zone.zone_id,
                "center_lat": s.zone.center_lat,
                "center_lon": s.zone.center_lon,
                "risk_score": round(s.risk_score, 2),
                "risk_label": s.risk_label,
                "confidence": s.confidence,
                "detection_count": s.detection_count,
                "flight_pass_id": s.flight_pass_id
            }

    priority_zones = sorted(zone_map.values(), key=lambda x: x['risk_score'], reverse=True)

    report_data = {
        "vehicles": {
            "stationary_count": stationary_count,
            "moving": moving_list
        },
        "damaged_structures": {
            "fully_collapsed_count": fully_collapsed,
            "partially_damaged_count": partially_damaged,
            "measurements": "Not available — requires geolocation altitude data to be calibrated on real drone hardware",
            "collapse_risk_prediction": "Not available from RGB imagery — requires structural sensor data"
        },
        "electrical": {
            "power_line_detections": power_line_cnt,
            "fallen_pole_detections": fallen_pole_cnt,
            "line_topology_mapping": "Not available — requires dedicated mapping pass, out of current scope",
            "fallen_pole_movement": "Not applicable — static objects, no movement tracking performed"
        },
        "debris": {
            "fallen_trees": fallen_trees_cnt,
            "fallen_poles": fallen_pole_cnt,
            "general_debris": general_debris_cnt
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
            "fast_moving_currents": fast_currents_cnt
        },
        "roads_and_routes": {
            "disrupted_road_segments": "Populated by calling find_safe_route() from routing.py once start/end points are provided — leave as empty list [] if not called yet",
            "suggested_safe_routes": []
        },
        "humans_and_animals": {
            "person_count": person_cnt,
            "animal_count": animal_cnt,
            "aggressive_behavior_flag": "Not available — behavior classification requires video-based action recognition, out of current scope"
        },
        "change_analysis": change_data,
        "priority_zones": priority_zones
    }

    return report_data
