import math
import time
import cv2
import numpy as np
from django.core.management.base import BaseCommand
from ultralytics import YOLO
from detection.models import Detection
from water_flow import analyze_water_flow, vector_to_cardinal
from geolocation import pixel_to_ground
from fire_intensity import analyze_fire_intensity
from wind_estimation import estimate_wind_from_region, combine_wind_estimates

from class_config import normalize_class_name, CANONICAL_CLASSES

# Recognized canonical hazard classes
HAZARD_CLASSES = {'fire', 'smoke', 'flood_water', 'collapsed_building', 'damaged_building', 'debris', 'fallen_tree', 'power_line', 'fallen_power_pole'}
VEHICLE_CLASSES = {'vehicle_car', 'vehicle_truck', 'vehicle_bus', 'vehicle_motorcycle', 'vehicle_bicycle'}
VEGETATION_CLASSES = {'fallen_tree'}

CLASS_COLORS = {
    'person': (0, 0, 255),               # Red
    'fire': (0, 0, 255),                 # Red
    'smoke': (128, 128, 128),            # Gray
    'flood': (255, 0, 0),                # Blue
    'flood_water': (255, 0, 0),          # Blue
    'collapsed_building': (0, 165, 255), # Orange
    'debris': (0, 255, 255),             # Yellow
    'fallen_tree': (34, 139, 34),        # Forest Green
    'power_line': (255, 191, 0),         # Amber
    'fallen_power_pole': (139, 69, 19),  # Brown
    'vehicle': (0, 255, 0),              # Green
    'car': (0, 255, 0),                  # Green
    'bicycle': (255, 255, 0),            # Cyan
    'motorcycle': (255, 0, 255),        # Magenta
    'bus': (128, 0, 128),                # Purple
    'truck': (128, 128, 0),              # Olive
}

def get_class_color(label):
    """Retrieve distinct color for known class or derive deterministic BGR color."""
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    h = hash(label)
    return ((h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF))

def check_and_warn_unfinetuned_model(model):
    """
    Check if the loaded YOLO model is an un-fine-tuned ground-level COCO model (80 default COCO classes).
    If so, print a warning to console/logs and return True indicating un-fine-tuned status.
    """
    names = getattr(model, 'names', {})
    is_coco = (len(names) == 80 and names.get(0) == 'person' and names.get(79) == 'toothbrush')
    if is_coco:
        print(
            "WARNING: Using ground-level COCO-trained model on aerial/drone imagery — vehicle and building "
            "classifications may be unreliable (e.g. rooftops misidentified as vehicles). Fine-tune on aerial "
            "datasets (RescueNet/AIDER/VisDrone) for accurate results."
        )
    return is_coco

class Command(BaseCommand):
    help = "Unified object tracking, optical flow, fire intensity, wind estimation, and geolocation command."

    def add_arguments(self, parser):
        parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to YOLO model weights')
        parser.add_argument('--source', type=str, default='0', help='Video source (webcam index or video file path)')
        parser.add_argument('--drone-alt', type=float, default=50.0, help='Drone altitude above ground in meters (default: 50)')
        parser.add_argument('--drone-heading', type=float, default=0.0, help='Drone compass heading in degrees (0 = North)')
        parser.add_argument('--drone-lat', type=float, default=None, help='Drone latitude coordinate')
        parser.add_argument('--drone-lon', type=float, default=None, help='Drone longitude coordinate')
        parser.add_argument('--camera-fov', type=float, default=80.0, help='Camera horizontal FOV in degrees (default: 80)')

    def handle(self, *args, **options):
        model_path = options['model']
        source_input = options['source']
        drone_alt = options['drone_alt']
        drone_heading = options['drone_heading']
        drone_lat = options['drone_lat']
        drone_lon = options['drone_lon']
        camera_fov = options['camera_fov']

        if isinstance(source_input, str) and source_input.isdigit():
            source = int(source_input)
        elif isinstance(source_input, int):
            source = source_input
        else:
            source = source_input

        self.stdout.write(f"Loading YOLO model from {model_path}...")
        model = YOLO(model_path)
        is_coco_model = check_and_warn_unfinetuned_model(model)

        model_class_names = set(model.names.values())
        if not HAZARD_CLASSES.intersection(model_class_names):
            self.stdout.write(self.style.WARNING("Warning: No hazard classes found in model class dictionary."))

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            self.stderr.write(self.style.ERROR(f"Could not open video source {source}"))
            return

        self.stdout.write(self.style.SUCCESS(f"Starting detection, tracking & wind analysis loop on '{source}'... Press 'q' to quit."))

        vehicle_history = {}
        prev_frame = None
        prev_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                curr_time = time.time()
                delta = curr_time - prev_time
                prev_time = curr_time
                fps = (1.0 / delta) if delta > 0 else 0.0

                frame_h, frame_w = frame.shape[:2]

                # Run ByteTrack tracking
                results = model.track(frame, persist=True, verbose=False)

                # Collect per-frame motion estimates for wind combination
                frame_wind_estimates = []
                frame_detections_data = []

                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        r = box.xyxy[0].astype(float)
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        raw_label = model.names[cls_id]
                        label = normalize_class_name(raw_label)

                        # Filter false positive vehicle detections on rooftops for un-fine-tuned COCO model
                        if is_coco_model and label in VEHICLE_CLASSES and conf < 0.55:
                            continue

                        track_id = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else None

                        source_type = "hazard" if label in HAZARD_CLASSES else "rgb"
                        movement_status = None
                        direction_str = None
                        flow_dir = None
                        flow_spd = None
                        fire_int_level = None
                        fire_temp_proxy = None

                        cx = (r[0] + r[2]) / 2.0
                        cy = (r[1] + r[3]) / 2.0

                        # 1. Vehicle Movement Tracking
                        if label in VEHICLE_CLASSES and track_id is not None:
                            if track_id not in vehicle_history:
                                vehicle_history[track_id] = []
                            vehicle_history[track_id].append((cx, cy))
                            if len(vehicle_history[track_id]) > 30:
                                vehicle_history[track_id].pop(0)

                            pts = vehicle_history[track_id]
                            if len(pts) >= 5:
                                dx = pts[-1][0] - pts[0][0]
                                dy = pts[-1][1] - pts[0][1]
                                dist = math.sqrt(dx**2 + dy**2)

                                if dist < 15.0:
                                    movement_status = "stationary"
                                    direction_str = None
                                else:
                                    movement_status = "moving"
                                    direction_str = vector_to_cardinal(dx, dy)

                        # 2. Optical Flow for Flood Water
                        if label == 'flood_water' and prev_frame is not None:
                            flood_mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
                            x1_i, y1_i = max(0, int(r[0])), max(0, int(r[1]))
                            x2_i, y2_i = min(frame_w, int(r[2])), min(frame_h, int(r[3]))
                            flood_mask[y1_i:y2_i, x1_i:x2_i] = 255

                            flow_dir, flow_spd = analyze_water_flow(prev_frame, frame, flood_mask)

                        # 3. Fire Intensity Analysis & Flame Wind Motion
                        if label == 'fire':
                            fire_res = analyze_fire_intensity(frame, r)
                            fire_int_level = fire_res['intensity_level']
                            fire_temp_proxy = fire_res['temp_estimate']

                            if prev_frame is not None:
                                mask_flame = np.zeros((frame_h, frame_w), dtype=np.uint8)
                                x1_i, y1_i = max(0, int(r[0])), max(0, int(r[1]))
                                x2_i, y2_i = min(frame_w, int(r[2])), min(frame_h, int(r[3]))
                                mask_flame[y1_i:y2_i, x1_i:x2_i] = 255

                                est_flame = estimate_wind_from_region(prev_frame, frame, mask_flame, region_type="flame")
                                frame_wind_estimates.append(est_flame)

                        # 4. Smoke Wind Motion
                        if label == 'smoke' and prev_frame is not None:
                            mask_smoke = np.zeros((frame_h, frame_w), dtype=np.uint8)
                            x1_i, y1_i = max(0, int(r[0])), max(0, int(r[1]))
                            x2_i, y2_i = min(frame_w, int(r[2])), min(frame_h, int(r[3]))
                            mask_smoke[y1_i:y2_i, x1_i:x2_i] = 255

                            est_smoke = estimate_wind_from_region(prev_frame, frame, mask_smoke, region_type="smoke")
                            frame_wind_estimates.append(est_smoke)

                        # 5. Vegetation Wind Motion
                        if label in VEGETATION_CLASSES and prev_frame is not None:
                            mask_veg = np.zeros((frame_h, frame_w), dtype=np.uint8)
                            x1_i, y1_i = max(0, int(r[0])), max(0, int(r[1]))
                            x2_i, y2_i = min(frame_w, int(r[2])), min(frame_h, int(r[3]))
                            mask_veg[y1_i:y2_i, x1_i:x2_i] = 255

                            est_veg = estimate_wind_from_region(prev_frame, frame, mask_veg, region_type="vegetation")
                            frame_wind_estimates.append(est_veg)

                        # 6. Geolocation Estimation
                        target_lat, target_lon = None, None
                        if drone_lat is not None and drone_lon is not None:
                            target_lat, target_lon = pixel_to_ground(
                                bbox_px=r,
                                camera_altitude_m=drone_alt,
                                camera_fov_deg=camera_fov,
                                frame_width_px=frame_w,
                                frame_height_px=frame_h,
                                drone_lat=drone_lat,
                                drone_lon=drone_lon,
                                drone_heading_deg=drone_heading
                            )

                        frame_detections_data.append({
                            'r': r,
                            'cls_id': cls_id,
                            'conf': conf,
                            'label': label,
                            'track_id': track_id,
                            'source_type': source_type,
                            'movement_status': movement_status,
                            'direction_str': direction_str,
                            'flow_dir': flow_dir,
                            'flow_spd': flow_spd,
                            'fire_int_level': fire_int_level,
                            'fire_temp_proxy': fire_temp_proxy,
                            'target_lat': target_lat,
                            'target_lon': target_lon
                        })

                # Combine per-frame wind estimates across smoke, flame, and vegetation sources
                combined_wind = combine_wind_estimates(frame_wind_estimates)
                win_dir = combined_wind['direction'] if combined_wind['direction'] != 'N/A' else None
                win_beaufort = combined_wind['beaufort_scale'] if combined_wind['direction'] != 'N/A' else None

                for item in frame_detections_data:
                    label = item['label']
                    r = item['r']
                    conf = item['conf']

                    # Save combined wind estimate on fire/smoke detections
                    save_wind_dir = win_dir if label in {'fire', 'smoke'} else None
                    save_wind_beaufort = win_beaufort if label in {'fire', 'smoke'} else None

                    Detection.objects.create(
                        object_class=label,
                        confidence=conf,
                        bbox_x1=float(r[0]),
                        bbox_y1=float(r[1]),
                        bbox_x2=float(r[2]),
                        bbox_y2=float(r[3]),
                        source=item['source_type'],
                        track_id=item['track_id'],
                        movement_status=item['movement_status'],
                        direction=item['direction_str'],
                        flow_direction=item['flow_dir'],
                        flow_speed=item['flow_spd'],
                        fire_intensity=item['fire_int_level'],
                        fire_temp_estimate=item['fire_temp_proxy'],
                        wind_direction=save_wind_dir,
                        wind_beaufort_scale=save_wind_beaufort,
                        latitude=item['target_lat'],
                        longitude=item['target_lon']
                    )

                    # Render bounding box and on-screen text
                    color = get_class_color(label)
                    p1 = (int(r[0]), int(r[1]))
                    p2 = (int(r[2]), int(r[3]))
                    cv2.rectangle(frame, p1, p2, color, 2)

                    disp_text = f"{label} {conf:.2f}"
                    if item['fire_int_level']:
                        temp_short = item['fire_temp_proxy'].split()[0] if item['fire_temp_proxy'] else ""
                        disp_text += f" | {item['fire_int_level'].upper()} {temp_short} (proxy)"
                    elif item['track_id'] is not None:
                        disp_text += f" ID:{item['track_id']}"

                    cv2.putText(frame, disp_text, (p1[0], max(p1[1] - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                prev_frame = frame.copy()

                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("Drone Disaster Tracking & Wind Analysis", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            self.stdout.write(self.style.SUCCESS("\nLoop stopped by user."))
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.stdout.write(self.style.SUCCESS("Resources released cleanly."))
