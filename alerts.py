"""
Alerts Module — threshold-based alert generation for disaster zones.

Called after zone priority scoring to create alerts for zones at CRITICAL or HIGH risk.
Avoids duplicate alerts: only creates a new alert if the zone has no active alert at
the same or higher severity.
"""

from detection.models import Alert, Zone, FlightPass


ALERT_THRESHOLDS = {"CRITICAL", "HIGH"}


def _build_alert_message(zone_id, risk_label, risk_score, components):
    """Build a human-readable alert message from zone scoring data."""
    parts = [f"Zone {zone_id} has reached {risk_label} risk (score: {risk_score:.2f})."]

    top_factors = sorted(
        [(k, v) for k, v in components.items() if v > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:3]

    if top_factors:
        factor_strs = [f"{k} ({v:.2f})" for k, v in top_factors]
        parts.append("Top risk factors: " + ", ".join(factor_strs) + ".")

    return " ".join(parts)


def check_and_create_alerts(priority_zones, flight_pass=None):
    """
    Check scored zones and create alerts for any at CRITICAL or HIGH risk.

    Parameters:
        priority_zones (list of dict): Output from compute_zone_priorities().
        flight_pass (FlightPass, optional): Associated flight pass.

    Returns:
        list of dict: Newly created alerts with zone_id, severity, message.
    """
    new_alerts = []

    for z in priority_zones:
        if z["risk_label"] not in ALERT_THRESHOLDS:
            continue

        zone_obj = Zone.objects.filter(zone_id=z["zone_id"]).first()
        if not zone_obj:
            continue

        existing_active = Alert.objects.filter(
            zone=zone_obj,
            status="active",
            severity=z["risk_label"],
        ).exists()

        if existing_active:
            continue

        message = _build_alert_message(
            z["zone_id"],
            z["risk_label"],
            z["risk_score"],
            z.get("components", {}),
        )

        alert = Alert.objects.create(
            zone=zone_obj,
            flight_pass=flight_pass,
            severity=z["risk_label"],
            risk_score=z["risk_score"],
            message=message,
            status="active",
        )

        new_alerts.append({
            "alert_id": alert.id,
            "zone_id": z["zone_id"],
            "severity": z["risk_label"],
            "risk_score": z["risk_score"],
            "message": message,
        })

    return new_alerts
