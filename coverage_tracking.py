"""
Survey Coverage Tracking Module.
Computes operational drone survey metrics including distinct zones scanned,
timestamp of the last flight pass, and percentage of overall grid area covered.
"""

from detection.models import FlightPass, Detection

def compute_survey_coverage(total_target_zones=20):
    """
    Calculate survey coverage across all flight passes.

    Parameters:
        total_target_zones (int): Expected total operational grid zones (default: 20).

    Returns:
        dict: {
            "zones_scanned": int,
            "last_pass_time": str or None,
            "pct_area_scanned": float
        }
    """
    scanned_zone_ids = Detection.objects.exclude(zone=None).values_list('zone_id', flat=True).distinct()
    zones_scanned = len(scanned_zone_ids)

    latest_pass = FlightPass.objects.order_by('-start_time').first()
    last_pass_time = None
    if latest_pass:
        t = latest_pass.end_time or latest_pass.start_time
        last_pass_time = t.isoformat() if t else None

    pct_area_scanned = round((zones_scanned / float(total_target_zones)) * 100.0, 1) if total_target_zones > 0 else 0.0
    if pct_area_scanned > 100.0:
        pct_area_scanned = 100.0

    return {
        "zones_scanned": zones_scanned,
        "last_pass_time": last_pass_time,
        "pct_area_scanned": pct_area_scanned
    }
