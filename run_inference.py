import argparse
import json
import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
from fire_intensity import analyze_fire_intensity
from water_flow import analyze_water_flow
from wind_estimation import estimate_wind_from_region, combine_wind_estimates
from class_config import normalize_class_name, get_class_color, check_and_warn_unfinetuned_model
from thermal_proxy import generate_thermal_heatmap, enhance_low_visibility

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Drone Video Inference Script for Disaster Response")
    parser.add_argument('--model', type=str, default="yolov8m.pt", help="Path to YOLOv8 weights")
    parser.add_argument('--input', type=str, required=True, help="Path to input video file")
    parser.add_argument('--output', type=str, default="output_annotated.mp4", help="Path to save annotated output video")
    parser.add_argument('--conf', type=float, default=0.35, help="Confidence threshold for detection")
    parser.add_argument('--enhance-dark', action='store_true', help="Apply CLAHE enhancement on dark/smoky frames before inference")
    parser.add_argument('--thermal', action='store_true', help="Generate pseudo-thermal heatmap overlay on fire frames")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading YOLOv8 model from '{args.model}'...")
    model = YOLO(args.model)
    is_coco_model = check_and_warn_unfinetuned_model(model)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open input video file '{args.input}'")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input Video: {args.input} ({width}x{height} @ {fps:.2f} FPS, {total_frames} total frames)")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not create output video writer for '{args.output}'")
        cap.release()
        sys.exit(1)

    frame_count = 0
    summary = {}
    fire_intensity_breakdown = {"low": 0, "medium": 0, "high": 0}
    water_flow_summary = []
    wind_summary = []
    thermal_summary = {"frames_with_hotspots": 0, "peak_temp_proxy": "Ambient (RGB proxy)"}
    enhanced_frame_count = 0
    prev_frame = None

    print(f"Processing frames with confidence threshold = {args.conf}...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Low-visibility enhancement: boost dark/smoky frames before inference
            inference_frame = frame
            if args.enhance_dark:
                inference_frame, was_enhanced = enhance_low_visibility(frame)
                if was_enhanced:
                    enhanced_frame_count += 1

            results = model(inference_frame, conf=args.conf, verbose=False)

            flood_masks = []
            smoke_masks = []
            fire_masks = []

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    r = box.xyxy[0].astype(int)
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    raw_label = model.names[cls_id]
                    label = normalize_class_name(raw_label)

                    if is_coco_model and label.startswith('vehicle_') and conf_val < 0.55:
                        continue

                    summary[label] = summary.get(label, 0) + 1

                    text = f"{label} {conf_val:.2f}"

                    # Fire intensity & temperature proxy estimation
                    if label == 'fire':
                        fire_res = analyze_fire_intensity(frame, r)
                        level = fire_res['intensity_level']
                        temp_proxy = fire_res['temp_estimate']
                        fire_intensity_breakdown[level] = fire_intensity_breakdown.get(level, 0) + 1
                        temp_short = temp_proxy.split()[0]
                        text += f" | {level.upper()} {temp_short} (proxy)"
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        mask[r[1]:r[3], r[0]:r[2]] = 255
                        fire_masks.append(mask)

                    # Collect flood region masks for water flow analysis
                    if label == 'flood_water':
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        mask[r[1]:r[3], r[0]:r[2]] = 255
                        flood_masks.append(mask)

                    # Collect smoke region masks for wind estimation
                    if label == 'smoke':
                        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                        mask[r[1]:r[3], r[0]:r[2]] = 255
                        smoke_masks.append(mask)

                    color = get_class_color(label)
                    cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), color, 2)
                    cv2.putText(
                        frame,
                        text,
                        (r[0], max(r[1] - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

            # Water flow analysis (needs previous frame for optical flow)
            if prev_frame is not None and flood_masks:
                combined_flood_mask = flood_masks[0]
                for m in flood_masks[1:]:
                    combined_flood_mask = cv2.bitwise_or(combined_flood_mask, m)
                flow_dir, flow_spd = analyze_water_flow(prev_frame, frame, combined_flood_mask)
                if flow_dir != "N/A":
                    water_flow_summary.append({"direction": flow_dir, "speed": flow_spd})
                    cv2.putText(frame, f"FLOOD: {flow_dir} {flow_spd}", (10, height - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # Wind estimation from smoke and fire regions (needs previous frame)
            if prev_frame is not None and (smoke_masks or fire_masks):
                wind_estimates = []
                for mask in smoke_masks:
                    est = estimate_wind_from_region(prev_frame, frame, mask, region_type="smoke")
                    wind_estimates.append(est)
                for mask in fire_masks:
                    est = estimate_wind_from_region(prev_frame, frame, mask, region_type="flame")
                    wind_estimates.append(est)
                if wind_estimates:
                    combined = combine_wind_estimates(wind_estimates)
                    wind_summary.append(combined)
                    cv2.putText(frame, f"WIND: {combined['direction']} {combined['beaufort_scale'][:20]}",
                                (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

            # Thermal heatmap overlay on frames with fire detections
            if args.thermal and fire_masks:
                thermal_result = generate_thermal_heatmap(frame)
                if thermal_result["hot_pixel_ratio"] > 0.005:
                    thermal_summary["frames_with_hotspots"] += 1
                    thermal_summary["peak_temp_proxy"] = thermal_result["max_region_temp_proxy"]
                    heatmap = thermal_result["heatmap"]
                    frame = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
                    cv2.putText(frame, f"THERMAL: {thermal_result['max_region_temp_proxy']}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            prev_frame = frame.copy()
            out.write(frame)

            if frame_count % 30 == 0 or frame_count == total_frames:
                if total_frames > 0:
                    print(f"Frame {frame_count}/{total_frames} processed")
                else:
                    print(f"Frame {frame_count} processed")

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\nProcessing complete! Annotated video saved to '{args.output}'.")

    # Add analysis breakdowns to output JSON summary
    if summary.get('fire', 0) > 0:
        summary['fire_intensity_breakdown'] = fire_intensity_breakdown
    if water_flow_summary:
        summary['water_flow_analysis'] = water_flow_summary[-1]
    if wind_summary:
        summary['wind_analysis'] = wind_summary[-1]
    if args.thermal and thermal_summary["frames_with_hotspots"] > 0:
        summary['thermal_analysis'] = thermal_summary
    if args.enhance_dark and enhanced_frame_count > 0:
        summary['low_visibility_enhancement'] = {
            "enhanced_frames": enhanced_frame_count,
            "total_frames": frame_count,
        }

    summary_filename = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary JSON saved to '{summary_filename}'.")

    print("\n" + "=" * 45)
    print(f"{'Class':<25} | {'Count':<10}")
    print("-" * 45)
    for k, v in summary.items():
        if k == 'fire_intensity_breakdown':
            continue
        print(f"{k:<25} | {v:<10}")
    if 'fire_intensity_breakdown' in summary:
        print("-" * 45)
        print("Fire Intensity Breakdown:")
        for lvl, cnt in fire_intensity_breakdown.items():
            print(f"  - {lvl.capitalize()} Intensity: {cnt}")
    if water_flow_summary:
        print("-" * 45)
        latest_flow = water_flow_summary[-1]
        print(f"Water Flow: {latest_flow['direction']} ({latest_flow['speed']})")
    if wind_summary:
        print("-" * 45)
        latest_wind = wind_summary[-1]
        print(f"Wind: {latest_wind['direction']} — {latest_wind['beaufort_scale']}")
    if args.thermal and thermal_summary["frames_with_hotspots"] > 0:
        print("-" * 45)
        print(f"Thermal: {thermal_summary['frames_with_hotspots']} hotspot frames")
        print(f"  Peak Temp: {thermal_summary['peak_temp_proxy']}")
    if args.enhance_dark and enhanced_frame_count > 0:
        print("-" * 45)
        print(f"Low-Vis Enhancement: {enhanced_frame_count}/{frame_count} frames enhanced")
    print("=" * 45)

if __name__ == '__main__':
    main()
