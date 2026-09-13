import cv2
import numpy as np

def analyze_fire_intensity(frame, bbox_px):
    """
    Estimate fire intensity and temperature proxy based on flame color-ratio analysis in HSV space.

    Parameters:
        frame (np.ndarray): Current BGR video frame image.
        bbox_px (tuple/list): Bounding box coordinates (x1, y1, x2, y2).

    Returns:
        dict: {
            "intensity_level": str ("low", "medium", or "high"),
            "temp_estimate": str ("~500-700°C (proxy estimate)", etc.),
            "color_ratios": {"red": float, "orange": float, "yellow": float, "white": float}
        }
    """
    default_result = {
        "intensity_level": "low",
        "temp_estimate": "~500-700°C (proxy estimate)",
        "color_ratios": {"red": 0.0, "orange": 0.0, "yellow": 0.0, "white": 0.0}
    }

    if frame is None or len(bbox_px) < 4:
        return default_result

    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox_px)

    # Crop bounding box region with boundary checks
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w_img, x2), min(h_img, y2)

    if x2 <= x1 or y2 <= y1:
        return default_result

    crop = frame[y1:y2, x1:x2]
    total_pixels = crop.shape[0] * crop.shape[1]
    if total_pixels == 0:
        return default_result

    # Convert BGR crop to HSV color space
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Define color masks in HSV
    # Red: split across 0-10 and 170-180 hue
    mask_red1 = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 100, 100]), np.array([180, 255, 255]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Orange: Hue 11 to 25
    mask_orange = cv2.inRange(hsv, np.array([11, 100, 140]), np.array([25, 255, 255]))

    # Yellow: Hue 26 to 35
    mask_yellow = cv2.inRange(hsv, np.array([26, 80, 150]), np.array([35, 255, 255]))

    # White / Bright-White: Low saturation (< 60), high value (> 200)
    mask_white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 60, 255]))

    # Calculate pixel counts and proportions
    count_red = np.count_nonzero(mask_red)
    count_orange = np.count_nonzero(mask_orange)
    count_yellow = np.count_nonzero(mask_yellow)
    count_white = np.count_nonzero(mask_white)

    ratio_red = float(count_red) / total_pixels
    ratio_orange = float(count_orange) / total_pixels
    ratio_yellow = float(count_yellow) / total_pixels
    ratio_white = float(count_white) / total_pixels

    # Flame color intensity shift: Red -> Orange -> Yellow -> White
    if ratio_white > 0.15:
        intensity_level = "high"
        temp_estimate = "~1000°C+ (proxy estimate)"
    elif (ratio_yellow + ratio_orange) > 0.25 or ratio_yellow > 0.10:
        intensity_level = "medium"
        temp_estimate = "~700-1000°C (proxy estimate)"
    else:
        intensity_level = "low"
        temp_estimate = "~500-700°C (proxy estimate)"

    return {
        "intensity_level": intensity_level,
        "temp_estimate": temp_estimate,
        "color_ratios": {
            "red": round(ratio_red, 4),
            "orange": round(ratio_orange, 4),
            "yellow": round(ratio_yellow, 4),
            "white": round(ratio_white, 4)
        }
    }
