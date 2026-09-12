import argparse
import cv2
import json
import time
from ultralytics import YOLO

# Define colors for each hazard class
COLORS = {
    'fire': (0, 0, 255),  # Red
    'flood': (0, 255, 0),  # Green
    'collapsed_building': (255, 0, 0),  # Blue
    'debris': (255, 255, 0)  # Yellow
}

def detect_hazards(model, source, output_file):
    # Open the video capture source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open source {source}")
        return

    # Open the output log file
    with open(output_file, 'a') as log_file:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get the current timestamp
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

            # Perform inference
            results = model(frame)

            # Draw bounding boxes and log detections
            for result in results:
                boxes = result.boxes.cpu().numpy()
                for box in boxes:
                    r = box.xyxy[0].astype(int)
                    cls = int(box.cls[0])
                    conf = box.conf[0]
                    label = model.names[cls]

                    # Check if the class is one of the target hazards
                    if label in COLORS:
                        color = COLORS[label]
                        cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), color, 2)
                        cv2.putText(frame, f'{label}: {conf:.2f}', (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        # Log the detection
                        detection = {
                            'timestamp': timestamp,
                            'class': label,
                            'confidence': conf,
                            'bbox_px': [r[0], r[1], r[2], r[3]]
                        }
                        log_file.write(json.dumps(detection) + '\n')

            # Calculate and display FPS
            fps = cap.get(cv2.CAP_PROP_FPS)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the frame
            cv2.imshow('Hazard Detection', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hazard Detection with YOLOv8')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to the YOLOv8 model')
    parser.add_argument('--source', type=str, default=0, help='Source of the video or image (default: webcam)')
    parser.add_argument('--output', type=str, default='hazard_detections.log', help='Output log file for detections')
    args = parser.parse_args()

    model = YOLO(args.model)
    detect_hazards(model, args.source, args.output)
