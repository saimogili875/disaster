import time
import cv2
from django.core.management.base import BaseCommand
from ultralytics import YOLO
from detection.models import Detection

# Predefined colors (BGR format for OpenCV) per object class
CLASS_COLORS = {
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

def get_class_color(label):
    """Return a specific color for known classes or compute a deterministic color."""
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    h = hash(label)
    return ((h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF))

class Command(BaseCommand):
    help = "Unified object and hazard detection loop saving records to the database."

    def add_arguments(self, parser):
        # Model path argument (defaults to yolov8n.pt)
        parser.add_argument(
            '--model',
            type=str,
            default='yolov8n.pt',
            help='Path to the YOLOv8 model weights file'
        )
        # Source argument (defaults to webcam index 0)
        parser.add_argument(
            '--source',
            type=str,
            default='0',
            help='Video source (webcam index or video file path)'
        )

    def handle(self, *args, **options):
        model_path = options['model']
        source_input = options['source']

        # If --source is a digit string, convert to int before cv2.VideoCapture
        if isinstance(source_input, str) and source_input.isdigit():
            source = int(source_input)
        elif isinstance(source_input, int):
            source = source_input
        else:
            source = source_input

        # Initialize the single YOLOv8 model
        self.stdout.write(f"Loading YOLO model from {model_path}...")
        model = YOLO(model_path)

        # Define hazard classes to monitor
        hazard_classes = {'fire', 'flood', 'collapsed_building', 'debris'}
        model_class_names = set(model.names.values())

        # Check if hazard classes exist in model.names.values() and warn if none exist
        found_hazards = hazard_classes.intersection(model_class_names)
        if not found_hazards:
            self.stdout.write(
                self.style.WARNING(
                    "Warning: No hazard classes (fire, flood, collapsed_building, debris) exist in the model's classes."
                )
            )

        # Open video capture device or file
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.stderr.write(self.style.ERROR(f"Error: Could not open video source {source}"))
            return

        self.stdout.write(self.style.SUCCESS(f"Starting detection loop on source '{source}'... Press 'q' to quit."))

        prev_time = time.time()
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.stdout.write("End of video stream or cannot read frame.")
                    break

                # Calculate real FPS using time.time() deltas between frames
                curr_time = time.time()
                delta = curr_time - prev_time
                prev_time = curr_time
                fps = (1.0 / delta) if delta > 0 else 0.0

                # Run inference on current frame
                results = model(frame)

                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        r = box.xyxy[0].astype(float)
                        cls_id = int(box.cls[0])
                        # Cast confidence to a plain Python float (float(box.conf[0]))
                        conf = float(box.conf[0])
                        label = model.names[cls_id]

                        # Categorize source: "hazard" for hazard classes, "rgb" for person/others
                        source_type = "hazard" if label in hazard_classes else "rgb"

                        # Save detection record into Django database ORM
                        Detection.objects.create(
                            object_class=label,
                            confidence=conf,
                            bbox_x1=float(r[0]),
                            bbox_y1=float(r[1]),
                            bbox_x2=float(r[2]),
                            bbox_y2=float(r[3]),
                            source=source_type
                        )

                        # Draw bounding box and label with distinct color per class
                        color = get_class_color(label)
                        p1 = (int(r[0]), int(r[1]))
                        p2 = (int(r[2]), int(r[3]))
                        cv2.rectangle(frame, p1, p2, color, 2)
                        cv2.putText(
                            frame,
                            f"{label}: {conf:.2f}",
                            (p1[0], max(p1[1] - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            color,
                            2
                        )

                # Draw real-time FPS counter
                cv2.putText(
                    frame,
                    f"FPS: {fps:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

                # Display the video frame
                cv2.imshow("Unified Object & Hazard Detection", frame)

                # Quit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            self.stdout.write(self.style.SUCCESS("\nDetection interrupted by user."))
        finally:
            # Cleanly release video capture and close windows on exit/Ctrl+C
            cap.release()
            cv2.destroyAllWindows()
            self.stdout.write(self.style.SUCCESS("Video capture released and windows destroyed."))
