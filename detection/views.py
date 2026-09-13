import os
import cv2
from django.shortcuts import render
from django.conf import settings
from ultralytics import YOLO
from class_config import normalize_class_name, get_class_color, check_and_warn_unfinetuned_model
from thermal_proxy import enhance_low_visibility
from .forms import ImageUploadForm
from .models import Detection, Alert

HAZARD_CLASSES = {'fire', 'flood_water', 'collapsed_building', 'debris'}

def get_yolo_model():
    """Load fine-tuned model if available at runs/detect/train/weights/best.pt; fallback to yolov8n.pt."""
    custom_model_path = os.path.join(settings.BASE_DIR, "runs", "detect", "train", "weights", "best.pt")
    if os.path.exists(custom_model_path):
        return YOLO(custom_model_path)
    return YOLO("yolov8m.pt")

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
            is_coco_model = check_and_warn_unfinetuned_model(model)
            summary = {}

            if not is_video:
                # Process Image File
                image = cv2.imread(upload_path)
                if image is not None:
                    inference_image, _ = enhance_low_visibility(image)
                    results = model(inference_image)
                    for result in results:
                        boxes = result.boxes.cpu().numpy()
                        for box in boxes:
                            r = box.xyxy[0].astype(float)
                            cls_id = int(box.cls[0])
                            conf = float(box.conf[0])
                            raw_label = model.names[cls_id]
                            label = normalize_class_name(raw_label)

                            if is_coco_model and label.startswith('vehicle_') and conf < 0.55:
                                continue

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
                # Process Video File with frame sampling + deduplication
                SAMPLE_EVERY_N = 5
                cap = cv2.VideoCapture(upload_path)
                if cap.isOpened():
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

                    annotated_filename = f"annotated_{os.path.splitext(filename)[0]}.mp4"
                    output_path = os.path.join(results_dir, annotated_filename)

                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                    frame_idx = 0
                    prev_class_counts = {}
                    last_annotations = []

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_idx += 1
                        is_sample_frame = (frame_idx % SAMPLE_EVERY_N == 0)

                        if is_sample_frame:
                            inference_frame, _ = enhance_low_visibility(frame)
                            results = model(inference_frame, verbose=False)
                            current_detections = []
                            current_class_counts = {}

                            for result in results:
                                boxes = result.boxes.cpu().numpy()
                                for box in boxes:
                                    r = box.xyxy[0].astype(float)
                                    cls_id = int(box.cls[0])
                                    conf = float(box.conf[0])
                                    raw_label = model.names[cls_id]
                                    label = normalize_class_name(raw_label)

                                    if is_coco_model and label.startswith('vehicle_') and conf < 0.55:
                                        continue

                                    source_type = "hazard" if label in HAZARD_CLASSES else "rgb"
                                    summary[label] = summary.get(label, 0) + 1
                                    current_class_counts[label] = current_class_counts.get(label, 0) + 1

                                    current_detections.append({
                                        'label': label, 'conf': conf,
                                        'r': r, 'source': source_type,
                                    })

                            # Dedup: only write to DB when class counts change
                            if current_class_counts != prev_class_counts:
                                for det in current_detections:
                                    Detection.objects.create(
                                        object_class=det['label'],
                                        confidence=det['conf'],
                                        bbox_x1=float(det['r'][0]),
                                        bbox_y1=float(det['r'][1]),
                                        bbox_x2=float(det['r'][2]),
                                        bbox_y2=float(det['r'][3]),
                                        source=det['source'],
                                    )
                                prev_class_counts = current_class_counts

                            # Cache annotations for non-sample frames
                            last_annotations = [
                                (det['label'], det['conf'], det['r'])
                                for det in current_detections
                            ]

                        # Draw annotations (use cached for non-sample frames)
                        for label, conf, r in last_annotations:
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


def map_view(request):
    """View rendering a Leaflet-based live disaster map with zone risk overlays and detection markers."""
    import json
    from .report import generate_report

    report_data = generate_report()

    detection_markers = []
    for det in Detection.objects.exclude(latitude=None).exclude(longitude=None).order_by('-timestamp')[:500]:
        detection_markers.append({
            'object_class': det.object_class,
            'confidence': det.confidence,
            'source': det.source,
            'lat': det.latitude,
            'lon': det.longitude,
        })
    report_data['_detection_markers'] = detection_markers

    context = {
        'report_json': json.dumps(report_data, default=str),
    }
    return render(request, 'detection/map.html', context)


def alerts_view(request):
    """View listing all triggered alerts, sorted by severity then recency."""
    alerts = Alert.objects.select_related('zone').all()
    context = {
        'alerts': alerts,
        'critical_count': alerts.filter(severity='CRITICAL', status='active').count(),
        'high_count': alerts.filter(severity='HIGH', status='active').count(),
        'active_count': alerts.filter(status='active').count(),
        'resolved_count': alerts.filter(status='resolved').count(),
    }
    return render(request, 'detection/alerts.html', context)


def alert_acknowledge(request, alert_id):
    """POST action to acknowledge an alert."""
    from django.shortcuts import redirect
    from django.utils import timezone as tz
    if request.method == 'POST':
        alert = Alert.objects.filter(id=alert_id).first()
        if alert and alert.status == 'active':
            alert.status = 'acknowledged'
            alert.acknowledged_at = tz.now()
            alert.save()
    return redirect('alerts_view')


def alert_resolve(request, alert_id):
    """POST action to resolve an alert."""
    from django.shortcuts import redirect
    if request.method == 'POST':
        alert = Alert.objects.filter(id=alert_id).first()
        if alert and alert.status in ('active', 'acknowledged'):
            alert.status = 'resolved'
            alert.save()
    return redirect('alerts_view')


def dashboard_view(request):
    """Command Center Dashboard — unified overview for disaster coordinators."""
    import json
    from django.db.models import Count
    from .report import generate_report

    report_data = generate_report()

    detection_markers = []
    for det in Detection.objects.exclude(latitude=None).exclude(longitude=None).order_by('-timestamp')[:500]:
        detection_markers.append({
            'object_class': det.object_class,
            'confidence': det.confidence,
            'source': det.source,
            'lat': det.latitude,
            'lon': det.longitude,
        })
    report_data['_detection_markers'] = detection_markers

    priority_zones = report_data.get('priority_zones', [])
    critical_zones = sum(1 for z in priority_zones if z.get('risk_label') == 'CRITICAL')
    high_zones = sum(1 for z in priority_zones if z.get('risk_label') == 'HIGH')

    recent_alerts = Alert.objects.select_related('zone').filter(
        status__in=['active', 'acknowledged']
    ).order_by('-created_at')[:8]

    class_counts = (
        Detection.objects.values('object_class')
        .annotate(count=Count('id'))
        .order_by('-count')[:10]
    )
    total_dets = Detection.objects.count()
    det_colors = {
        'person': '#ef4444', 'fire': '#f97316', 'smoke': '#94a3b8',
        'flood_water': '#3b82f6', 'collapsed_building': '#a855f7',
        'damaged_building': '#d946ef', 'debris': '#eab308',
        'fallen_tree': '#22c55e', 'vehicle_car': '#06b6d4',
    }
    class_breakdown = []
    for row in class_counts:
        pct = round((row['count'] / max(1, total_dets)) * 100, 1)
        color = det_colors.get(row['object_class'], '#64748b')
        class_breakdown.append((row['object_class'], row['count'], pct, color))

    context = {
        'report_json': json.dumps(report_data, default=str),
        'total_zones': len(priority_zones),
        'critical_zones': critical_zones,
        'high_zones': high_zones,
        'total_detections': total_dets,
        'active_alerts': Alert.objects.filter(status='active').count(),
        'person_count': report_data.get('humans_and_animals', {}).get('person_count', 0),
        'recent_alerts': recent_alerts,
        'class_breakdown': class_breakdown,
    }
    return render(request, 'detection/dashboard.html', context)

