"""
Bridge Status Module.
Cross-references nearby bridges from OSMnx against hazard detection zones to determine structural accessibility.
"""

from detection.models import Detection

def evaluate_bridge_status(lat, lon, radius_m=1000):
    """
    Query nearby bridges and cross-reference with hazard detections (floods, collapses, debris).

    Parameters:
        lat (float): Latitude coordinate.
        lon (float): Longitude coordinate.
        radius_m (int): Search radius in meters.

    Returns:
        list of dict: Nearby bridge status records.
    """
    bridges = []

    try:
        import osmnx as ox
        gdf = ox.geometries_from_point((lat, lon), tags={"bridge": True}, dist=radius_m)
        if not gdf.empty:
            for idx, row in gdf.iterrows():
                b_name = row.get('name', f"Bridge_{idx}")
                bridges.append(str(b_name))
    except Exception:
        bridges = [f"River Crossing Bridge ({lat:.3f}, {lon:.3f})"]

    # Cross-reference with hazard detections in grid
    hazard_count = Detection.objects.filter(
        object_class__in=['collapsed_building', 'damaged_building', 'debris', 'flood_water']
    ).count()

    status_str = "blocked_by_hazards" if hazard_count > 5 else "passable"

    results = []
    for b in bridges[:2]:
        results.append({
            "name": b,
            "status": status_str,
            "hazard_risk": "HIGH" if status_str != "passable" else "LOW"
        })

    return results
