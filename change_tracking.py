from detection.models import FlightPass, Detection
from class_config import CANONICAL_CLASSES

def compute_changes(current_pass_id, previous_pass_id):
    """
    Compute comparative change analysis metrics between two FlightPass runs:
      - new_damage: count of damage detections in current pass in zones not present in previous pass.
      - flood_extent_change: percentage change in total flood water detections between passes.
      - fire_spread_rate: change in fire detections divided by elapsed minutes between pass start times.

    Parameters:
        current_pass_id (int): FlightPass ID of current/latest run.
        previous_pass_id (int): FlightPass ID of previous baseline run.

    Returns:
        dict: {"new_damage": int, "flood_extent_change": str, "fire_spread_rate": str} or note dict if insufficient passes.
    """
    try:
        curr_pass = FlightPass.objects.get(id=current_pass_id)
        prev_pass = FlightPass.objects.get(id=previous_pass_id)
    except FlightPass.DoesNotExist:
        return {
            "note": "Insufficient completed flight passes (at least 2 completed passes required for change tracking)",
            "new_damage": 0,
            "flood_extent_change": "0.0%",
            "fire_spread_rate": "0.00 detections/min"
        }

    # 1. New Damage Detections
    prev_zones = Detection.objects.filter(flight_pass=prev_pass).exclude(zone=None).values_list('zone_id', flat=True).distinct()
    curr_damage_dets = Detection.objects.filter(
        flight_pass=curr_pass,
        object_class__in=['collapsed_building', 'damaged_building']
    ).exclude(zone_id__in=prev_zones)
    new_damage_count = curr_damage_dets.count()

    # 2. Flood Extent Percentage Change (Uses canonical 'flood_water')
    prev_flood_cnt = Detection.objects.filter(flight_pass=prev_pass, object_class='flood_water').count()
    curr_flood_cnt = Detection.objects.filter(flight_pass=curr_pass, object_class='flood_water').count()

    if prev_flood_cnt > 0:
        pct_change = ((curr_flood_cnt - prev_flood_cnt) / float(prev_flood_cnt)) * 100.0
        flood_extent_str = f"{pct_change:+.1f}%"
    elif curr_flood_cnt > 0:
        flood_extent_str = "+100.0%"
    else:
        flood_extent_str = "0.0%"

    # 3. Fire Spread Rate (Detections per Minute)
    prev_fire_cnt = Detection.objects.filter(flight_pass=prev_pass, object_class='fire').count()
    curr_fire_cnt = Detection.objects.filter(flight_pass=curr_pass, object_class='fire').count()

    elapsed_seconds = (curr_pass.start_time - prev_pass.start_time).total_seconds()
    elapsed_minutes = elapsed_seconds / 60.0

    if elapsed_minutes > 0.01:
        rate = (curr_fire_cnt - prev_fire_cnt) / elapsed_minutes
        fire_rate_str = f"{rate:+.2f} detections/min"
    else:
        fire_rate_str = "0.00 detections/min"

    return {
        "new_damage": new_damage_count,
        "flood_extent_change": flood_extent_str,
        "fire_spread_rate": fire_rate_str
    }
