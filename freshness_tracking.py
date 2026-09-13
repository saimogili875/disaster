"""
Data Freshness Tracking Module.
Evaluates data staleness per zone by calculating time elapsed since last zone detection or snapshot.
"""

from django.utils import timezone

def compute_zone_freshness(computed_at_dt, stale_threshold_minutes=30.0):
    """
    Calculate age in minutes and staleness flag for a zone record or snapshot.

    Parameters:
        computed_at_dt (datetime): Datetime when the zone snapshot/detection was recorded.
        stale_threshold_minutes (float): Age in minutes after which data is considered stale.

    Returns:
        dict: {
            "age_minutes": float,
            "stale_flag": bool
        }
    """
    if not computed_at_dt:
        return {"age_minutes": 0.0, "stale_flag": True}

    now = timezone.now()
    delta_seconds = (now - computed_at_dt).total_seconds()
    age_minutes = round(max(0.0, delta_seconds / 60.0), 1)
    stale_flag = age_minutes > stale_threshold_minutes

    return {
        "age_minutes": age_minutes,
        "stale_flag": stale_flag
    }
