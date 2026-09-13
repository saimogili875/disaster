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

def detect_hazards(image, output_file):
    # Load the model
    model = YOLO('yolov8n.pt')

    # Open the output log file
    with open(output_file, 'a') as log_file:
        # Perform inference
        results = model(image)

        # Draw bounding boxes and log detections
        detections = []
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
                    cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), color, 2)
                    cv2.putText(image, f'{label}: {conf:.2f}', (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Log the detection
                    detection = {
                        'label': label,
                        'confidence': conf,
                        'bbox': [r[0], r[1], r[2], r[3]]
                    }
                    detections.append(detection)

        # Save the image with bounding boxes
        cv2.imwrite('detections.jpg', image)

        return detections
