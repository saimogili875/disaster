import math

def pixel_to_ground(
    bbox_px,
    camera_altitude_m=50.0,
    camera_fov_deg=80.0,
    frame_width_px=1920,
    frame_height_px=1080,
    drone_lat=None,
    drone_lon=None,
    drone_heading_deg=0.0
):
    """
    Estimate real-world latitude and longitude of a detected bounding box center using pinhole projection.

    Parameters:
        bbox_px (tuple/list): (x1, y1, x2, y2) bounding box in frame pixel coordinates.
        camera_altitude_m (float): Altitude of the drone camera above ground level in meters.
        camera_fov_deg (float): Horizontal Field of View (FOV) of the camera in degrees.
        frame_width_px (int): Width of the video frame in pixels.
        frame_height_px (int): Height of the video frame in pixels.
        drone_lat (float or None): Drone latitude GPS coordinate.
        drone_lon (float or None): Drone longitude GPS coordinate.
        drone_heading_deg (float): Drone compass heading in degrees (0 = North, 90 = East).

    Returns:
        tuple: (estimated_lat, estimated_lon) or (None, None) if drone_lat/drone_lon are not provided.
    """
    if drone_lat is None or drone_lon is None:
        return None, None

    x1, y1, x2, y2 = bbox_px
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    # Calculate pixel offset from frame center
    frame_cx = frame_width_px / 2.0
    frame_cy = frame_height_px / 2.0
    offset_x = cx - frame_cx
    offset_y = cy - frame_cy  # positive is downwards in image

    # Horizontal and vertical FOV calculation assuming square pixels
    h_fov_rad = math.radians(camera_fov_deg)
    v_fov_rad = 2.0 * math.atan((frame_height_px / frame_width_px) * math.tan(h_fov_rad / 2.0))

    # Angular displacement per pixel
    angle_x = (offset_x / frame_cx) * (h_fov_rad / 2.0)
    angle_y = (offset_y / frame_cy) * (v_fov_rad / 2.0)

    # Relative ground displacement in camera frame (X = East-West, Y = North-South)
    # altitude * tan(angle) gives distance on ground
    dist_x = camera_altitude_m * math.tan(angle_x)
    dist_y = -camera_altitude_m * math.tan(angle_y)  # invert image Y to point North (up)

    # Rotate ground displacements by drone heading (heading 0 = North, 90 = East)
    heading_rad = math.radians(drone_heading_deg)
    east_offset_m = dist_x * math.cos(heading_rad) + dist_y * math.sin(heading_rad)
    north_offset_m = -dist_x * math.sin(heading_rad) + dist_y * math.cos(heading_rad)

    # Convert meter offsets to latitude and longitude delta
    # 1 degree latitude ≈ 111,139 meters
    delta_lat = north_offset_m / 111139.0
    # 1 degree longitude ≈ 111,139 * cos(lat) meters
    delta_lon = east_offset_m / (111139.0 * math.cos(math.radians(drone_lat)))

    target_lat = drone_lat + delta_lat
    target_lon = drone_lon + delta_lon

    return target_lat, target_lon
