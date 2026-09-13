"""
Route Watch — continuous monitoring of active mission routes for new hazards.

Checks every active/rerouting mission:
1. Gets the team's planned route (list of lat/lon waypoints)
2. Finds hazard detections within a buffer zone of the route
3. If the route is compromised, flags the mission for rerouting
4. Computes a new route from the team's CURRENT position (not base)

Can be called periodically (management command, cron, or celery task)
or triggered after new detections land.
"""

import json
import math
from django.utils import timezone


ROUTE_BUFFER_DEG = 0.0008  # ~80m buffer around route waypoints
HAZARD_CLASSES = {"fire", "flood_water", "collapsed_building", "fallen_power_pole", "debris"}


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _point_near_route(lat, lon, route_coords, buffer_deg=ROUTE_BUFFER_DEG):
    for rlat, rlon in route_coords:
        if abs(lat - rlat) < buffer_deg and abs(lon - rlon) < buffer_deg:
            return True
    return False


def check_route_safety(mission):
    """
    Check if a mission's route has been compromised by new hazard detections.

    Returns dict:
        {
            "safe": bool,
            "threats": [{"class": str, "lat": float, "lon": float, "confidence": float}, ...],
            "threat_count": int,
        }
    """
    from detection.models import Detection

    route_coords = json.loads(mission.route_json) if mission.route_json else []
    if not route_coords:
        return {"safe": True, "threats": [], "threat_count": 0}

    recent_hazards = Detection.objects.filter(
        object_class__in=HAZARD_CLASSES,
        timestamp__gte=mission.started_at,
    ).exclude(latitude__isnull=True).values_list(
        "object_class", "latitude", "longitude", "confidence"
    )

    threats = []
    for obj_class, lat, lon, conf in recent_hazards:
        if _point_near_route(lat, lon, route_coords):
            threats.append({
                "class": obj_class,
                "lat": lat,
                "lon": lon,
                "confidence": conf,
            })

    return {
        "safe": len(threats) == 0,
        "threats": threats,
        "threat_count": len(threats),
    }


def check_escape_routes(team_lat, team_lon, zone_center_lat, zone_center_lon, blocked_points):
    """
    Bidirectional escape routing — check both forward (to zone) and backward (to base).

    Returns dict with forward_route, backward_route, and trapped flag.
    """
    from routing import find_safe_route

    base_lat = 17.385
    base_lon = 78.4867

    forward = find_safe_route(team_lat, team_lon, zone_center_lat, zone_center_lon, blocked_points)
    backward = find_safe_route(team_lat, team_lon, base_lat, base_lon, blocked_points)

    trapped = not forward and not backward

    return {
        "forward_route": forward,
        "backward_route": backward,
        "forward_viable": len(forward) > 0,
        "backward_viable": len(backward) > 0,
        "trapped": trapped,
    }


def run_route_watch():
    """
    Main route watch loop — check all active missions and handle rerouting.

    Returns list of actions taken.
    """
    from detection.models import Mission, Alert, Team

    active_missions = Mission.objects.filter(status__in=["active", "rerouting"])
    actions = []

    for mission in active_missions:
        safety = check_route_safety(mission)
        mission.last_route_check = timezone.now()

        if not safety["safe"]:
            mission.status = "rerouting"
            mission.reroute_count += 1
            mission.save()

            team = mission.team

            blocked = [(t["lat"], t["lon"]) for t in safety["threats"]]
            all_hazards = list(
                Detection.objects.filter(
                    object_class__in=HAZARD_CLASSES,
                ).exclude(latitude__isnull=True).values_list("latitude", "longitude")
            )
            blocked.extend(all_hazards)

            from routing import find_safe_route

            if team.latitude and team.longitude:
                new_route = find_safe_route(
                    team.latitude, team.longitude,
                    mission.zone.center_lat, mission.zone.center_lon,
                    blocked
                )
            else:
                new_route = []

            if new_route:
                mission.route_json = json.dumps(new_route)
                mission.status = "active"
                mission.save()
                actions.append({
                    "type": "rerouted",
                    "mission_id": mission.id,
                    "team": team.name,
                    "zone": mission.zone.zone_id,
                    "threats": safety["threat_count"],
                    "new_route_points": len(new_route),
                })
            else:
                escape = check_escape_routes(
                    team.latitude or mission.zone.center_lat,
                    team.longitude or mission.zone.center_lon,
                    mission.zone.center_lat,
                    mission.zone.center_lon,
                    blocked,
                )

                if escape["trapped"]:
                    team.status = "emergency"
                    team.save()
                    mission.status = "rerouting"
                    mission.save()

                    Alert.objects.create(
                        zone=mission.zone,
                        severity="CRITICAL",
                        risk_score=1.0,
                        message=f"TEAM TRAPPED: {team.name} cannot find forward or backward route from current position. Immediate aerial extraction needed.",
                        status="active",
                    )
                    actions.append({
                        "type": "trapped",
                        "mission_id": mission.id,
                        "team": team.name,
                        "zone": mission.zone.zone_id,
                    })
                elif escape["backward_viable"]:
                    mission.route_json = json.dumps(escape["backward_route"])
                    mission.save()
                    team.status = "returning"
                    team.save()
                    actions.append({
                        "type": "retreat",
                        "mission_id": mission.id,
                        "team": team.name,
                        "zone": mission.zone.zone_id,
                        "retreat_route_points": len(escape["backward_route"]),
                    })
                else:
                    mission.save()
                    actions.append({
                        "type": "no_route",
                        "mission_id": mission.id,
                        "team": team.name,
                        "zone": mission.zone.zone_id,
                    })
        else:
            mission.save()

    return actions
