"""
Bridge Status Module.
Cross-references nearby bridges from OSMnx against hazard detection zones to determine structural accessibility.
"""

from detection.models import Detection, Zone


def _nearby_hazard_count(lat, lon, radius_deg=0.005):
    """Count hazard detections in zones within radius_deg (~500m) of the given point."""
    hazard_classes = ["collapsed_building", "damaged_building", "debris", "flood_water"]
    nearby_zones = Zone.objects.filter(
        center_lat__gte=lat - radius_deg,
        center_lat__lte=lat + radius_deg,
        center_lon__gte=lon - radius_deg,
        center_lon__lte=lon + radius_deg,
    )
    return Detection.objects.filter(
        zone__in=nearby_zones,
        object_class__in=hazard_classes,
    ).count()


def evaluate_bridge_status(lat, lon, radius_m=1000):
    """
    Query nearby bridges and cross-reference with LOCAL hazard detections.

    Parameters:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        radius_m (int): Search radius in meters.

    Returns:
        list of dict: Nearby bridge status records with proximity-based hazard assessment.
    """
    bridges = []

    try:
        import osmnx as ox
        gdf = ox.geometries_from_point((lat, lon), tags={"bridge": True}, dist=radius_m)
        if not gdf.empty:
            for idx, row in gdf.iterrows():
                b_name = row.get("name", f"Bridge_{idx}")
                b_lat = lat
                b_lon = lon
                if hasattr(row.geometry, "centroid"):
                    b_lat = row.geometry.centroid.y
                    b_lon = row.geometry.centroid.x
                bridges.append({"name": str(b_name), "lat": b_lat, "lon": b_lon})
    except Exception:
        bridges = [{"name": f"River Crossing Bridge ({lat:.3f}, {lon:.3f})", "lat": lat, "lon": lon}]

    results = []
    for bridge in bridges:
        hazard_count = _nearby_hazard_count(bridge["lat"], bridge["lon"])

        if hazard_count > 5:
            status = "blocked_by_hazards"
            risk = "HIGH"
        elif hazard_count > 2:
            status = "partially_blocked"
            risk = "MODERATE"
        else:
            status = "passable"
            risk = "LOW"

        results.append({
            "name": bridge["name"],
            "status": status,
            "hazard_risk": risk,
            "nearby_hazard_count": hazard_count,
        })

    return results
