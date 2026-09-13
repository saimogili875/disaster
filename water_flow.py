import math
import cv2
import numpy as np

def vector_to_cardinal(dx, dy):
    """
    Convert 2D displacement vector (dx, dy) to cardinal/intercardinal direction string.
    Note: OpenCV image coordinates have Y axis pointing DOWN, so -dy is North.
    """
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return "N/A"

    # Angle in degrees from positive X axis (East), math.atan2(y, x)
    # Using -dy for North (up) and dx for East (right)
    angle_rad = math.atan2(-dy, dx)
    angle_deg = math.degrees(angle_rad) % 360.0

    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(round(angle_deg / 45.0)) % 8
    return directions[idx]

def analyze_water_flow(prev_frame, curr_frame, flood_mask):
    """
    Calculate optical flow within the flood_mask region using Farneback's method.

    Parameters:
        prev_frame (np.ndarray): Previous BGR video frame.
        curr_frame (np.ndarray): Current BGR video frame.
        flood_mask (np.ndarray): Binary or single-channel mask where flood water is present (> 0).

    Returns:
        tuple: (flow_direction, flow_speed) where:
            - flow_direction (str): Cardinal direction string ('N', 'S', 'E', 'W', 'NE', etc.)
            - flow_speed (str): Speed classification ('stagnant', 'slow', 'fast')
    """
    if prev_frame is None or curr_frame is None or flood_mask is None:
        return "N/A", "stagnant"

    # Convert frames to grayscale for Farneback optical flow calculation
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    # Compute dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Ensure mask dimensions match flow
    if flood_mask.shape[:2] != flow.shape[:2]:
        flood_mask = cv2.resize(flood_mask, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Filter flow vectors inside the flood mask region
    mask_indices = flood_mask > 0
    if not np.any(mask_indices):
        return "N/A", "stagnant"

    flow_x = flow[..., 0][mask_indices]
    flow_y = flow[..., 1][mask_indices]

    mean_dx = float(np.mean(flow_x))
    mean_dy = float(np.mean(flow_y))
    avg_speed = math.sqrt(mean_dx**2 + mean_dy**2)

    # Speed classification thresholds
    if avg_speed < 0.5:
        speed_class = "stagnant"
        direction_str = "N/A"
    elif avg_speed < 2.5:
        speed_class = "slow"
        direction_str = vector_to_cardinal(mean_dx, mean_dy)
    else:
        speed_class = "fast"
        direction_str = vector_to_cardinal(mean_dx, mean_dy)

    return direction_str, speed_class
