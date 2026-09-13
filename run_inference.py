import argparse
import json
import os
import sys
import cv2
from ultralytics import YOLO

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
    # Compute deterministic color from hash for any other classes
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

    # Load the YOLOv8 model
    print(f"Loading YOLOv8 model from '{args.model}'...")
    model = YOLO(args.model)

    # Open input video
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open input video file '{args.input}'")
        sys.exit(1)

    # Get input video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input Video: {args.input} ({width}x{height} @ {fps:.2f} FPS, {total_frames} total frames)")

    # Create VideoWriter using mp4v codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"Error: Could not create output video writer for '{args.output}'")
        cap.release()
        sys.exit(1)

    frame_count = 0
    summary = {}

    print(f"Processing frames with confidence threshold = {args.conf}...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Perform inference on current frame
            results = model(frame, conf=args.conf, verbose=False)

            # Draw bounding boxes and accumulate class counts
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    r = box.xyxy[0].astype(int)
                    cls_id = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    label = model.names[cls_id]

                    # Accumulate summary detection count
                    summary[label] = summary.get(label, 0) + 1

                    # Draw bounding box and label text
                    color = get_class_color(label)
                    cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), color, 2)
                    text = f"{label} {conf_val:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (r[0], max(r[1] - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2
                    )

            # Write annotated frame to output video
            out.write(frame)

            # Print progress every 30 frames
            if frame_count % 30 == 0 or frame_count == total_frames:
                if total_frames > 0:
                    print(f"Frame {frame_count}/{total_frames} processed")
                else:
                    print(f"Frame {frame_count} processed")

    finally:
        # Release video capture and writer cleanly
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"\nProcessing complete! Annotated video saved to '{args.output}'.")

    # Save detection summary as JSON file with _summary.json suffix
    summary_filename = os.path.splitext(args.output)[0] + "_summary.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"Summary JSON saved to '{summary_filename}'.")

    # Print summary as a clean console table
    print("\n" + "=" * 35)
    print(f"{'Class':<20} | {'Count':<10}")
    print("-" * 35)
    if summary:
        for cls_name, count in sorted(summary.items(), key=lambda x: x[1], reverse=True):
            print(f"{cls_name:<20} | {count:<10}")
    else:
        print(f"{'No detections':<20} | {0:<10}")
    print("-" * 35)
    total_detections = sum(summary.values())
    print(f"{'Total Detections':<20} | {total_detections:<10}")
    print("=" * 35)

if __name__ == '__main__':
    main()
