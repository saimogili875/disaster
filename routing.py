import networkx as nx

def find_safe_route(start_lat, start_lon, end_lat, end_lon, blocked_points=None):
    """
    Download road network between start and end coordinates using OSMnx, mark blocked nodes as impassable,
    and compute a safe shortest path around hazards.

    Parameters:
        start_lat (float): Latitude of start location.
        start_lon (float): Longitude of start location.
        end_lat (float): Latitude of destination location.
        end_lon (float): Longitude of destination location.
        blocked_points (list of tuples): [(lat, lon), ...] list of blocked hazard points.

    Returns:
        list of tuples: [(lat, lon), ...] route coordinates or empty list [] if no route is found or osmnx unavailable.
    """
    if blocked_points is None:
        blocked_points = []

    try:
        import osmnx as ox
    except ImportError:
        print("Warning: osmnx is not installed. Returning empty route.")
        return []

    lats = [start_lat, end_lat] + [p[0] for p in blocked_points]
    lons = [start_lon, end_lon] + [p[1] for p in blocked_points]

    margin = 0.02  # ~2km bounding box margin
    north, south = max(lats) + margin, min(lats) - margin
    east, west = max(lons) + margin, min(lons) - margin

    try:
        # Download driving road network
        try:
            G = ox.graph_from_bbox(bbox=(north, south, east, west), network_type='drive')
        except TypeError:
            G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
    except Exception as e:
        print(f"Error downloading road network: {e}")
        return []

    # Identify and remove nodes nearest to blocked hazard points
    for b_lat, b_lon in blocked_points:
        try:
            blocked_node = ox.distance.nearest_nodes(G, b_lon, b_lat)
            if blocked_node in G:
                G.remove_node(blocked_node)
        except Exception:
            continue

    # Find nearest valid nodes for origin and destination
    try:
        orig_node = ox.distance.nearest_nodes(G, start_lon, start_lat)
        dest_node = ox.distance.nearest_nodes(G, end_lon, end_lat)
    except Exception as e:
        print(f"Error locating start/end nodes on network: {e}")
        return []

    # Compute shortest path navigating around blocked nodes
    try:
        path_nodes = nx.shortest_path(G, orig_node, dest_node, weight='length')
        route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in path_nodes]
        return route_coords
    except nx.NetworkXNoPath:
        print("No viable route found avoiding blocked hazards.")
        return []
    except Exception as e:
        print(f"Error computing shortest path: {e}")
        return []
