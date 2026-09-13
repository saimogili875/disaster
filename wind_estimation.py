import math
import cv2
import numpy as np

def vector_to_cardinal(dx, dy):
    """
    Convert 2D displacement vector (dx, dy) to cardinal/intercardinal compass direction string.
    Note: Image coordinates have Y pointing down, so -dy is North (up) and dx is East (right).
    """
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return "N/A"

    angle_rad = math.atan2(-dy, dx)
    angle_deg = math.degrees(angle_rad) % 360.0

    directions = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx = int(round(angle_deg / 45.0)) % 8
    return directions[idx]

def classify_beaufort_scale(magnitude, region_type="smoke"):
    """
    Map optical flow magnitude to approximate Beaufort scale category with visual estimate disclaimer.

    Parameters:
        magnitude (float): Mean optical flow magnitude.
        region_type (str): "smoke", "flame", or "vegetation".

    Returns:
        str: Beaufort scale description string including disclaimer.
    """
    disclaimer = " (visual estimate, not anemometer-measured)"

    # Adjusted threshold multipliers per region type
    if region_type == "smoke":
        # Smoke shifts easily with light airflow
        t = [0.3, 1.2, 2.5, 4.2, 6.5, 9.0, 12.0, 15.0]
    elif region_type == "flame":
        # Flame flickers rapidly
        t = [0.5, 1.8, 3.5, 5.5, 8.0, 11.0, 14.0, 17.0]
    elif region_type == "vegetation":
        # Canopy pixels require stronger wind to shift
        t = [0.2, 0.8, 1.8, 3.0, 4.5, 6.5, 9.0, 12.0]
    else:
        t = [0.3, 1.2, 2.5, 4.2, 6.5, 9.0, 12.0, 15.0]

    if magnitude < t[0]:
        scale_str = "Force 0 (Calm)"
    elif magnitude < t[1]:
        scale_str = "Force 1 (Light air)"
    elif magnitude < t[2]:
        scale_str = "Force 2 (Light breeze)"
    elif magnitude < t[3]:
        scale_str = "Force 3 (Gentle breeze)"
    elif magnitude < t[4]:
        scale_str = "Force 4 (Moderate breeze)"
    elif magnitude < t[5]:
        scale_str = "Force 5 (Fresh breeze)"
    elif magnitude < t[6]:
        scale_str = "Force 6 (Strong breeze)"
    elif magnitude < t[7]:
        scale_str = "Force 7 (Near gale)"
    else:
        scale_str = "Force 8 (Gale)"

    return f"{scale_str}{disclaimer}"

def estimate_wind_from_region(prev_frame, curr_frame, mask, region_type="smoke"):
    """
    Compute optical flow within mask region and return wind direction and magnitude.

    Parameters:
        prev_frame (np.ndarray): Previous BGR frame.
        curr_frame (np.ndarray): Current BGR frame.
        mask (np.ndarray): Binary mask specifying region of interest.
        region_type (str): "smoke", "vegetation", or "flame".

    Returns:
        dict: {"direction": str, "magnitude": float, "mean_dx": float, "mean_dy": float, "region_type": str}
    """
    default_res = {
        "direction": "N/A",
        "magnitude": 0.0,
        "mean_dx": 0.0,
        "mean_dy": 0.0,
        "region_type": region_type
    }

    if prev_frame is None or curr_frame is None or mask is None:
        return default_res

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )

    if mask.shape[:2] != flow.shape[:2]:
        mask = cv2.resize(mask, (flow.shape[1], flow.shape[0]), interpolation=cv2.INTER_NEAREST)

    indices = mask > 0
    if not np.any(indices):
        return default_res

    flow_x = flow[..., 0][indices]
    flow_y = flow[..., 1][indices]

    mean_dx = float(np.mean(flow_x))
    mean_dy = float(np.mean(flow_y))
    magnitude = math.sqrt(mean_dx**2 + mean_dy**2)
    direction = vector_to_cardinal(mean_dx, mean_dy)

    return {
        "direction": direction,
        "magnitude": magnitude,
        "mean_dx": mean_dx,
        "mean_dy": mean_dy,
        "region_type": region_type
    }

def combine_wind_estimates(estimates_list):
    """
    Average displacement vectors from multiple region motion sources and compute combined Beaufort scale.

    Parameters:
        estimates_list (list of dicts): Output dictionaries from estimate_wind_from_region.

    Returns:
        dict: {"direction": str, "beaufort_scale": str, "confidence_note": str}
    """
    valid_estimates = [e for e in estimates_list if e.get("magnitude", 0) > 0 and e.get("direction") != "N/A"]

    if not valid_estimates:
        return {
            "direction": "N/A",
            "beaufort_scale": classify_beaufort_scale(0.0, "smoke"),
            "confidence_note": "No motion sources detected in frame"
        }

    avg_dx = float(np.mean([e["mean_dx"] for e in valid_estimates]))
    avg_dy = float(np.mean([e["mean_dy"] for e in valid_estimates]))
    combined_mag = math.sqrt(avg_dx**2 + avg_dy**2)
    combined_dir = vector_to_cardinal(avg_dx, avg_dy)

    primary_region = valid_estimates[0]["region_type"]
    beaufort_str = classify_beaufort_scale(combined_mag, primary_region)

    src_count = len(valid_estimates)
    if src_count == 1:
        conf_note = "Low confidence (single visual motion indicator)"
    elif src_count == 2:
        conf_note = "Moderate confidence (2 visual motion sources agreed)"
    else:
        conf_note = f"High confidence ({src_count}+ visual motion sources agreed)"

    return {
        "direction": combined_dir,
        "beaufort_scale": beaufort_str,
        "confidence_note": conf_note
    }
