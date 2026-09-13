import argparse
import cv2
import json
import time
from ultralytics import YOLO

# Define colors for each class
COLORS = {
    'person': (0, 0, 255),  # Red
    'bicycle': (0, 255, 0),  # Green
    'car': (255, 0, 0),  # Blue
    'motorcycle': (255, 255, 0),  # Yellow
    'airplane': (255, 0, 255),  # Magenta
    'bus': (0, 255, 255),  # Cyan
    'train': (128, 0, 128),  # Purple
    'truck': (128, 128, 0),  # Olive
    'boat': (128, 128, 128),  # Gray
    'traffic_light': (255, 165, 0),  # Orange
    'fire_hydrant': (0, 128, 128),  # Teal
    'stop_sign': (255, 255, 255),  # White
    'parking_meter': (165, 42, 42),  # Brown
    'bench': (192, 192, 192),  # Silver
    'bird': (255, 140, 0),  # Dark Orange
    'cat': (255, 105, 180),  # Hot Pink
    'dog': (255, 215, 0),  # Gold
    'horse': (135, 206, 235),  # Sky Blue
    'sheep': (255, 228, 196),  # Misty Rose
    'cow': (255, 228, 181),  # Papaya Whip
    'elephant': (255, 228, 181),  # Papaya Whip
    'bear': (255, 228, 181),  # Papaya Whip
    'zebra': (255, 228, 181),  # Papaya Whip
    'giraffe': (255, 228, 181),  # Papaya Whip
    'backpack': (255, 228, 181),  # Papaya Whip
    'umbrella': (255, 228, 181),  # Papaya Whip
    'handbag': (255, 228, 181),  # Papaya Whip
    'tie': (255, 228, 181),  # Papaya Whip
    'suitcase': (255, 228, 181),  # Papaya Whip
    'frisbee': (255, 228, 181),  # Papaya Whip
    'skis': (255, 228, 181),  # Papaya Whip
    'snowboard': (255, 228, 181),  # Papaya Whip
    'sports_ball': (255, 228, 181),  # Papaya Whip
    'kite': (255, 228, 181),  # Papaya Whip
    'baseball_bat': (255, 228, 181),  # Papaya Whip
    'baseball_glove': (255, 228, 181),  # Papaya Whip
    'skateboard': (255, 228, 181),  # Papaya Whip
    'surfboard': (255, 228, 181),  # Papaya Whip
    'tennis_racket': (255, 228, 181),  # Papaya Whip
    'bottle': (255, 228, 181),  # Papaya Whip
    'wine_glass': (255, 228, 181),  # Papaya Whip
    'cup': (255, 228, 181),  # Papaya Whip
    'fork': (255, 228, 181),  # Papaya Whip
    'knife': (255, 228, 181),  # Papaya Whip
    'spoon': (255, 228, 181),  # Papaya Whip
    'bowl': (255, 228, 181),  # Papaya Whip
    'banana': (255, 228, 181),  # Papaya Whip
    'apple': (255, 228, 181),  # Papaya Whip
    'sandwich': (255, 228, 181),  # Papaya Whip
    'orange': (255, 228, 181),  # Papaya Whip
    'broccoli': (255, 228, 181),  # Papaya Whip
    'carrot': (255, 228, 181),  # Papaya Whip
    'hot_dog': (255, 228, 181),  # Papaya Whip
    'pizza': (255, 228, 181),  # Papaya Whip
    'donut': (255, 228, 181),  # Papaya Whip
    'cake': (255, 228, 181),  # Papaya Whip
    'chair': (255, 228, 181),  # Papaya Whip
    'couch': (255, 228, 181),  # Papaya Whip
    'potted_plant': (255, 228, 181),  # Papaya Whip
    'bed': (255, 228, 181),  # Papaya Whip
    'dining_table': (255, 228, 181),  # Papaya Whip
    'toilet': (255, 228, 181),  # Papaya Whip
    'tv': (255, 228, 181),  # Papaya Whip
    'laptop': (255, 228, 181),  # Papaya Whip
    'mouse': (255, 228, 181),  # Papaya Whip
    'keyboard': (255, 228, 181),  # Papaya Whip
    'cell_phone': (255, 228, 181),  # Papaya Whip
    'microwave': (255, 228, 181),  # Papaya Whip
    'oven': (255, 228, 181),  # Papaya Whip
    'toaster': (255, 228, 181),  # Papaya Whip
    'sink': (255, 228, 181),  # Papaya Whip
    'refrigerator': (255, 228, 181),  # Papaya Whip
    'book': (255, 228, 181),  # Papaya Whip
    'clock': (255, 228, 181),  # Papaya Whip
    'vase': (255, 228, 181),  # Papaya Whip
    'scissors': (255, 228, 181),  # Papaya Whip
    'teddy_bear': (255, 228, 181),  # Papaya Whip
    'hair_dryer': (255, 228, 181),  # Papaya Whip
    'toothbrush': (255, 228, 181)  # Papaya Whip
}

def detect_objects(model, source, output_file):
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

                    # Check if the class is one of the target classes
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
            cv2.imshow('Object Detection', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object Detection with YOLOv8')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to the YOLOv8 model')
    parser.add_argument('--source', type=str, default=0, help='Source of the video or image (default: webcam)')
    parser.add_argument('--output', type=str, default='detections.log', help='Output log file for detections')
    args = parser.parse_args()

    model = YOLO(args.model)
    detect_objects(model, args.source, args.output)
