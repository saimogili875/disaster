import os
import cv2
from django.shortcuts import render
from django.conf import settings
from ultralytics import YOLO
from class_config import normalize_class_name
from .forms import ImageUploadForm
from .models import Detection

# Hazard classes for categorization
HAZARD_CLASSES = {'fire', 'flood', 'collapsed_building', 'debris'}

# Predefined colors for detection bounding boxes
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
    """Return specific BGR color for class or fallback to deterministic hash color."""
    if label in CLASS_COLORS:
        return CLASS_COLORS[label]
    h = hash(label)
    return ((h & 0xFF), ((h >> 8) & 0xFF), ((h >> 16) & 0xFF))

def get_yolo_model():
    """Load fine-tuned model if available at runs/detect/train/weights/best.pt; fallback to yolov8n.pt."""
    custom_model_path = os.path.join(settings.BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
    if os.path.exists(custom_model_path):
        return YOLO(custom_model_path)
    return YOLO("yolov8n.pt")

def upload_and_detect(request):
    """View handling GET (upload form) and POST (save file, run YOLO inference, render results)."""
    context = {
        'form': ImageUploadForm(),
        'annotated_url': None,
        'summary': None,
        'is_video': False,
    }

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['file']

            # Create media upload and result directories if missing
            upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
            results_dir = os.path.join(settings.MEDIA_ROOT, 'results')
            os.makedirs(upload_dir, exist_ok=True)
            os.makedirs(results_dir, exist_ok=True)

            # Save uploaded file to media/uploads/
            filename = uploaded_file.name
            upload_path = os.path.join(upload_dir, filename)
            with open(upload_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            # Check if file is video by extension
            ext = os.path.splitext(filename)[1].lower()
            video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
            is_video = ext in video_extensions

            model = get_yolo_model()
            summary = {}

            if not is_video:
                # Process Image File
                image = cv2.imread(upload_path)
                if image is not None:
                    results = model(image)
                    for result in results:
                        boxes = result.boxes.cpu().numpy()
                        for box in boxes:
                            r = box.xyxy[0].astype(float)
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            raw_label = model.names[cls_id]
                            label = normalize_class_name(raw_label)

                            source_type = "hazard" if label in HAZARD_CLASSES else "rgb"
                            summary[label] = summary.get(label, 0) + 1

                            # Save detection record to ORM database
                            Detection.objects.create(
                                object_class=label,
                                confidence=conf,
                                bbox_x1=float(r[0]),
                                bbox_y1=float(r[1]),
                                bbox_x2=float(r[2]),
                                bbox_y2=float(r[3]),
                                source=source_type
                            )

                            # Draw annotation box and label
                            color = get_class_color(label)
                            p1 = (int(r[0]), int(r[1]))
                            p2 = (int(r[2]), int(r[3]))
                            cv2.rectangle(image, p1, p2, color, 2)
                            cv2.putText(
                                image,
                                f"{label}: {conf:.2f}",
                                (p1[0], max(p1[1] - 10, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2
                            )

                    # Save annotated image to media/results/
                    annotated_filename = f"annotated_{filename}"
                    output_path = os.path.join(results_dir, annotated_filename)
                    cv2.imwrite(output_path, image)

                    context['annotated_url'] = f"{settings.MEDIA_URL}results/{annotated_filename}"
                    context['summary'] = summary
                    context['is_video'] = False
            else:
                # Process Video File frame-by-frame
                cap = cv2.VideoCapture(upload_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

                    annotated_filename = f"annotated_{os.path.splitext(filename)[0]}.mp4"
                    output_path = os.path.join(results_dir, annotated_filename)

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        results = model(frame, verbose=False)
                        for result in results:
                            boxes = result.boxes.cpu().numpy()
                            for box in boxes:
                                r = box.xyxy[0].astype(float)
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                raw_label = model.names[cls_id]
                                label = normalize_class_name(raw_label)

                                source_type = "hazard" if label in HAZARD_CLASSES else "rgb"
                                summary[label] = summary.get(label, 0) + 1

                                Detection.objects.create(
                                    object_class=label,
                                    confidence=conf,
                                    bbox_x1=float(r[0]),
                                    bbox_y1=float(r[1]),
                                    bbox_x2=float(r[2]),
                                    bbox_y2=float(r[3]),
                                    source=source_type
                                )

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

                        out.write(frame)

                    cap.release()
                    out.release()

                    context['annotated_url'] = f"{settings.MEDIA_URL}results/{annotated_filename}"
                    context['summary'] = summary
                    context['is_video'] = True

        context['form'] = form

    return render(request, 'detection/upload.html', context)

def report_view(request):
    """View to generate and render structured disaster response report (HTML or JSON API)."""
    import json
    from django.http import JsonResponse
    from .report import generate_report

    report_data = generate_report()

    if request.GET.get('format') == 'json' or request.headers.get('Accept') == 'application/json':
        return JsonResponse(report_data, json_dumps_params={'indent': 2})

    context = {
        'report': report_data,
        'report_json': json.dumps(report_data, indent=2),
    }
    return render(request, 'detection/report.html', context)

