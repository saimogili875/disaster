"""
Thermal Camera Proxy Module.

Generates pseudo-thermal analysis from RGB frames when no dedicated thermal
camera is available. Outputs are clearly labeled as proxy estimates.

Swappable: when real thermal hardware (FLIR, DJI Zenmuse H20T) is available,
replace generate_thermal_heatmap() with actual radiometric data parsing.
"""

import cv2
import numpy as np


def generate_thermal_heatmap(frame):
    """
    Generate a pseudo-thermal heatmap from an RGB frame by isolating warm-color
    channels (red, yellow, white) and mapping intensity to a colormap.

    Parameters:
        frame (np.ndarray): BGR video frame.

    Returns:
        dict: {
            "heatmap": np.ndarray (BGR colorized heatmap, same size as frame),
            "hot_pixel_ratio": float (fraction of frame classified as 'hot'),
            "max_region_temp_proxy": str (estimated temp range based on color),
            "source": str ("rgb_proxy")
        }
    """
    if frame is None:
        return {"heatmap": None, "hot_pixel_ratio": 0.0,
                "max_region_temp_proxy": "N/A", "source": "rgb_proxy"}

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Warm regions: red + orange + yellow + white-hot
    mask_red1 = cv2.inRange(hsv, np.array([0, 80, 100]), np.array([10, 255, 255]))
    mask_red2 = cv2.inRange(hsv, np.array([170, 80, 100]), np.array([180, 255, 255]))
    mask_orange = cv2.inRange(hsv, np.array([11, 80, 120]), np.array([25, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([26, 60, 150]), np.array([35, 255, 255]))
    mask_white = cv2.inRange(hsv, np.array([0, 0, 220]), np.array([180, 40, 255]))

    warm_mask = cv2.bitwise_or(mask_red1, mask_red2)
    warm_mask = cv2.bitwise_or(warm_mask, mask_orange)
    warm_mask = cv2.bitwise_or(warm_mask, mask_yellow)
    warm_mask = cv2.bitwise_or(warm_mask, mask_white)

    # Build intensity map: brighter in V channel = hotter
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    intensity = np.zeros_like(gray, dtype=np.float32)
    intensity[warm_mask > 0] = gray[warm_mask > 0].astype(np.float32)

    # Smooth for visual coherence
    intensity = cv2.GaussianBlur(intensity, (15, 15), 0)

    # Normalize to 0-255 and apply thermal colormap
    if intensity.max() > 0:
        intensity = (intensity / intensity.max() * 255).astype(np.uint8)
    else:
        intensity = intensity.astype(np.uint8)

    heatmap = cv2.applyColorMap(intensity, cv2.COLORMAP_JET)

    total_pixels = frame.shape[0] * frame.shape[1]
    hot_pixels = np.count_nonzero(warm_mask)
    hot_ratio = round(float(hot_pixels) / total_pixels, 4)

    white_count = np.count_nonzero(mask_white)
    yellow_count = np.count_nonzero(mask_yellow)
    if white_count > total_pixels * 0.02:
        temp_proxy = "~1000°C+ (RGB proxy)"
    elif yellow_count > total_pixels * 0.02:
        temp_proxy = "~700-1000°C (RGB proxy)"
    elif hot_pixels > total_pixels * 0.01:
        temp_proxy = "~400-700°C (RGB proxy)"
    else:
        temp_proxy = "Ambient (RGB proxy)"

    return {
        "heatmap": heatmap,
        "hot_pixel_ratio": hot_ratio,
        "max_region_temp_proxy": temp_proxy,
        "source": "rgb_proxy",
    }


def enhance_low_visibility(frame, threshold=60):
    """
    Enhance frame for person detection in low-visibility conditions (smoke, night,
    dust). Applies CLAHE contrast enhancement on the luminance channel.

    Parameters:
        frame (np.ndarray): BGR video frame.
        threshold (int): Mean brightness below which enhancement activates.

    Returns:
        tuple: (enhanced_frame, was_enhanced)
            - enhanced_frame: np.ndarray, contrast-enhanced if dark, else original
            - was_enhanced: bool
    """
    if frame is None:
        return frame, False

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    if mean_brightness >= threshold:
        return frame, False

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    enhanced_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    return enhanced_frame, True
