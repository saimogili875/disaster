import argparse
import json
import os
import sys
import cv2
from ultralytics import YOLO
from fire_intensity import analyze_fire_intensity
from class_config import normalize_class_name

def get_class_color(label):
    """Return a distinct BGR color tuple for a given class label."""
    known_colors = {
        'person': (0, 0, 255),               # Red
        'fire': (0, 0, 255),                 # Red
        'flood': (255, 0, 0),                # Blue
        'collapsed_building': (0, 165, 255), # Orange
        'debris': (0, 255, 255),             # Yellow
        'car': (0, 255, 0),                  # Green
        'bicycle': (255, 255, 0),            # Cyan
        'motorcycle': (255, 0, 255),        # Magenta
        'bus': (128, 0, 128),                # Purple
        'truck': (128, 128, 0),              # Olive
    }
    if label in known_colors:
        return known_colors[label]
    h = hash(label)
    return ((h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 Drone Video Inference Script for Disaster Response")
    parser.add_argument('--model', type=str, default="yolov8n.pt", help="Path to YOLOv8 weights")
    parser.add_argument('--input', type=str, required=True, help="Path to input video file")
    parser.add_argument('--output', type=str, default="output_annotated.mp4", help="Path to save annotated output video")
    parser.add_argument('--conf', type=float, default=0.35, help="Confidence threshold for detection")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"Loading YOLOv8 model from '{args.model}'...")
    model = YOLO(args.model)

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

    print(f"Processing frames with confidence threshold = {args.conf}...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            results = model(frame, conf=args.conf, verbose=False)

            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    r = box.xyxy[0].astype(int)
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    raw_label = model.names[cls_id]
                    label = normalize_class_name(raw_label)

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

    # Add fire intensity breakdown to output JSON summary
    if summary.get('fire', 0) > 0:
        summary['fire_intensity_breakdown'] = fire_intensity_breakdown

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
    print("=" * 45)

if __name__ == '__main__':
    main()
