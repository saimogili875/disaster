import os
import random
import cv2
from django.shortcuts import render
from django.conf import settings
from ultralytics import YOLO
from class_config import normalize_class_name, get_class_color, check_and_warn_unfinetuned_model
from thermal_proxy import enhance_low_visibility
from .forms import ImageUploadForm
from .models import Detection, Alert, Team, Mission

HAZARD_CLASSES = {'fire', 'flood_water', 'collapsed_building', 'debris'}

UPLOAD_ZONE_CENTER = (17.385, 78.4867)
UPLOAD_ZONE_SCATTER = 0.002

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

                            det_lat = UPLOAD_ZONE_CENTER[0] + random.uniform(-UPLOAD_ZONE_SCATTER, UPLOAD_ZONE_SCATTER)
                            det_lon = UPLOAD_ZONE_CENTER[1] + random.uniform(-UPLOAD_ZONE_SCATTER, UPLOAD_ZONE_SCATTER)

                            Detection.objects.create(
                                object_class=label,
                                confidence=conf,
                                bbox_x1=float(r[0]),
                                bbox_y1=float(r[1]),
                                bbox_x2=float(r[2]),
                                bbox_y2=float(r[3]),
                                source=source_type,
                                latitude=round(det_lat, 6),
                                longitude=round(det_lon, 6),
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
                                    det_lat = UPLOAD_ZONE_CENTER[0] + random.uniform(-UPLOAD_ZONE_SCATTER, UPLOAD_ZONE_SCATTER)
                                    det_lon = UPLOAD_ZONE_CENTER[1] + random.uniform(-UPLOAD_ZONE_SCATTER, UPLOAD_ZONE_SCATTER)
                                    Detection.objects.create(
                                        object_class=det['label'],
                                        confidence=det['conf'],
                                        bbox_x1=float(det['r'][0]),
                                        bbox_y1=float(det['r'][1]),
                                        bbox_x2=float(det['r'][2]),
                                        bbox_y2=float(det['r'][3]),
                                        source=det['source'],
                                        latitude=round(det_lat, 6),
                                        longitude=round(det_lon, 6),
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

    from disaster_classifier import classify_all_zones
    zone_classifications = classify_all_zones()

    disaster_type_counts = {}
    for zc in zone_classifications:
        dt = zc["disaster_type"]
        disaster_type_counts[dt] = disaster_type_counts.get(dt, 0) + 1

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
        'zone_classifications': zone_classifications,
        'disaster_type_counts': disaster_type_counts,
    }
    return render(request, 'detection/dashboard.html', context)


def mission_view(request, zone_id):
    """Mission Brief — single-page rescue action card for a specific zone."""
    from django.http import Http404
    from django.utils import timezone as tz
    from .models import Zone
    from infrastructure_layer import fetch_critical_infrastructure
    from bridge_status import evaluate_bridge_status
    from freshness_tracking import compute_zone_freshness
    from routing import find_safe_route

    zone = Zone.objects.filter(zone_id=zone_id).first()
    if not zone:
        raise Http404(f"Zone '{zone_id}' not found")

    dets = Detection.objects.filter(zone=zone)
    det_count = dets.count()

    people_count = dets.filter(object_class="person").count()
    animal_count = dets.filter(object_class__in=["dog", "cat", "cow", "horse"]).count()
    fire_count = dets.filter(object_class="fire").count()
    high_fire = dets.filter(object_class="fire", fire_intensity="high").count()
    smoke_count = dets.filter(object_class="smoke").count()
    flood_count = dets.filter(object_class="flood_water").count()
    collapsed_count = dets.filter(object_class="collapsed_building").count()
    damaged_count = dets.filter(object_class="damaged_building").count()
    fallen_poles = dets.filter(object_class="fallen_power_pole").count()
    power_lines = dets.filter(object_class="power_line").count()
    debris_count = dets.filter(object_class="debris").count()
    fallen_trees = dets.filter(object_class="fallen_tree").count()
    gas_count = dets.filter(object_class="gas_cylinder").count()

    def _level(count, thresholds=(1, 3)):
        if count == 0:
            return "NONE"
        if count <= thresholds[0]:
            return "LOW"
        if count <= thresholds[1]:
            return "MODERATE"
        return "HIGH"

    fire_level = "HIGH" if high_fire > 0 else _level(fire_count + smoke_count)
    flood_level = _level(flood_count)
    electrical_level = _level(fallen_poles + power_lines)
    structure_level = "SEVERE" if collapsed_count > 0 else ("PARTIAL" if damaged_count > 0 else "NONE")
    debris_level = _level(debris_count + fallen_trees)
    chemical_level = _level(gas_count + smoke_count)

    def _chip_status(level):
        return "active" if level not in ("NONE",) else "inactive"

    # Risk scoring
    from priority_scoring import _compute_zone_score, _score_to_label, RISK_COLOR_MAP
    score, _, components = _compute_zone_score(zone)
    risk_label = _score_to_label(score)
    risk_color = RISK_COLOR_MAP.get(risk_label, "#22c55e")

    # Freshness
    latest_det = dets.order_by("-timestamp").first()
    if latest_det:
        freshness = compute_zone_freshness(latest_det.timestamp)
        last_person = dets.filter(object_class="person").order_by("-timestamp").first()
        last_person_seen = f"{last_person.timestamp.strftime('%H:%M')} ({int((tz.now() - last_person.timestamp).total_seconds() / 60)} min ago)" if last_person else "Not detected"
    else:
        freshness = {"age_minutes": 999, "stale_flag": True}
        last_person_seen = "Not detected"

    confidence_map = {"high": 85, "moderate": 55, "low": 25, "none": 0}
    from priority_scoring import _compute_confidence
    confidence = _compute_confidence(det_count, freshness)
    confidence_pct = confidence_map.get(confidence, 30)

    # Approach direction — away from heaviest hazard
    hazard_lats = []
    hazard_lons = []
    for d in dets.filter(object_class__in=["fire", "flood_water", "collapsed_building", "fallen_power_pole"]):
        if d.latitude and d.longitude:
            hazard_lats.append(d.latitude)
            hazard_lons.append(d.longitude)

    if hazard_lats:
        avg_hazard_lat = sum(hazard_lats) / len(hazard_lats)
        avg_hazard_lon = sum(hazard_lons) / len(hazard_lons)
        dlat = zone.center_lat - avg_hazard_lat
        dlon = zone.center_lon - avg_hazard_lon
        if abs(dlat) > abs(dlon):
            approach_direction = "NORTH" if dlat < 0 else "SOUTH"
        else:
            approach_direction = "EAST" if dlon < 0 else "WEST"
        approach_reason = f"Hazards concentrated on the opposite side"
    else:
        approach_direction = "ANY DIRECTION"
        approach_reason = "No concentrated hazard cluster detected"

    # Warnings
    warnings = []
    if fire_level == "HIGH":
        warnings.append("Active high-intensity fire detected — do NOT approach without fire team")
    if electrical_level in ("MODERATE", "HIGH"):
        warnings.append("Fallen electrical infrastructure — electrocution risk")
    if structure_level == "SEVERE":
        warnings.append("Severely damaged structures — do NOT enter buildings without structural clearance")
    if flood_level == "HIGH":
        fast_water = dets.filter(object_class="flood_water", flow_speed="fast").count()
        if fast_water > 0:
            warnings.append("Fast-moving flood water detected — drowning risk")
        else:
            warnings.append("Significant flooding — watch for submerged hazards")
    if chemical_level in ("MODERATE", "HIGH"):
        warnings.append("Gas cylinders or heavy smoke — possible gas leak risk")
    if freshness.get("stale_flag"):
        warnings.append("Data is stale (>30 min) — situation may have changed, re-scan recommended")

    # Situation summary
    parts = [f"Zone {zone_id} is at {risk_label} risk (score {score:.2f})."]
    if people_count > 0:
        parts.append(f"{people_count} {'person' if people_count == 1 else 'people'} last detected.")
    active_hazards = []
    if fire_level != "NONE":
        active_hazards.append(f"fire ({fire_level.lower()})")
    if flood_level != "NONE":
        active_hazards.append(f"flooding ({flood_level.lower()})")
    if electrical_level != "NONE":
        active_hazards.append("electrical hazard")
    if structure_level != "NONE":
        active_hazards.append(f"structural damage ({structure_level.lower()})")
    if active_hazards:
        parts.append("Active hazards: " + ", ".join(active_hazards) + ".")
    parts.append(f"Approach from {approach_direction.lower()}.")
    situation_summary = " ".join(parts)

    # Routes
    routes = []
    try:
        blocked_points = []
        if hazard_lats:
            for hl, hlo in zip(hazard_lats, hazard_lons):
                blocked_points.append((hl, hlo))
        infra = fetch_critical_infrastructure(zone.center_lat, zone.center_lon)
        if infra:
            end_lat = infra[0].get("lat", zone.center_lat + 0.01)
            end_lon = infra[0].get("lon", zone.center_lon + 0.01)
        else:
            end_lat = zone.center_lat + 0.01
            end_lon = zone.center_lon + 0.01
        route_coords = find_safe_route(zone.center_lat, zone.center_lon, end_lat, end_lon, blocked_points)
        if route_coords:
            routes.append({"label": "Safe alternate route to nearest facility", "status": "safe_alternate"})
    except Exception:
        pass

    infrastructure = fetch_critical_infrastructure(zone.center_lat, zone.center_lon)
    bridges = evaluate_bridge_status(zone.center_lat, zone.center_lon)

    context = {
        "zone": zone,
        "risk_label": risk_label,
        "risk_score": f"{score:.2f}",
        "risk_color": risk_color,
        "situation_summary": situation_summary,
        "warnings": warnings,
        "people_count": people_count,
        "animal_count": animal_count,
        "last_person_seen": last_person_seen,
        "fire_level": fire_level,
        "fire_status": _chip_status(fire_level),
        "flood_level": flood_level,
        "flood_status": _chip_status(flood_level),
        "electrical_level": electrical_level,
        "electrical_status": _chip_status(electrical_level),
        "structure_level": structure_level,
        "structure_status": _chip_status(structure_level),
        "debris_level": debris_level,
        "debris_status": _chip_status(debris_level),
        "chemical_level": chemical_level,
        "chemical_status": _chip_status(chemical_level),
        "approach_direction": approach_direction,
        "approach_reason": approach_reason,
        "routes": routes,
        "infrastructure": infrastructure,
        "bridges": bridges,
        "confidence": confidence,
        "confidence_pct": confidence_pct,
        "data_age": freshness.get("age_minutes", "N/A"),
        "stale": freshness.get("stale_flag", False),
        "detection_count": det_count,
    }

    from disaster_classifier import classify_zone
    classification = classify_zone(zone)
    context["disaster_type"] = classification["disaster_type"]
    context["disaster_type_color"] = classification["color"]
    context["disaster_type_icon"] = classification["icon"]
    context["secondary_type"] = classification["secondary_type"]

    return render(request, "detection/mission.html", context)


def volunteer_view(request):
    """Volunteer view — simple green/yellow/red zone status for untrained volunteers."""
    from .models import Zone
    from priority_scoring import compute_zone_priorities, RISK_COLOR_MAP

    priority_zones = compute_zone_priorities()

    volunteer_zones = []
    for z in priority_zones:
        label = z["risk_label"]
        if label == "CRITICAL":
            status = "red"
            color = "#ef4444"
            instruction = "DO NOT APPROACH"
            detail = "This area has critical hazards. Stay away and wait for rescue teams."
        elif label == "HIGH":
            status = "red"
            color = "#f97316"
            instruction = "DO NOT APPROACH"
            detail = "High-risk area. Only trained rescue teams should enter."
        elif label == "MODERATE":
            status = "yellow"
            color = "#eab308"
            instruction = "USE CAUTION"
            detail = "Some hazards detected. Only approach if you have protective gear and a buddy."
        else:
            status = "green"
            color = "#22c55e"
            instruction = "SAFE TO APPROACH"
            detail = "Low risk detected. You can help here — stay alert for changing conditions."

        people = Detection.objects.filter(
            zone__zone_id=z["zone_id"], object_class="person"
        ).count()

        volunteer_zones.append({
            "zone_id": z["zone_id"],
            "status": status,
            "color": color,
            "instruction": instruction,
            "detail": detail,
            "risk_label": label,
            "people_count": people,
        })

    context = {
        "zones": volunteer_zones,
        "total_zones": len(volunteer_zones),
        "safe_count": sum(1 for z in volunteer_zones if z["status"] == "green"),
        "caution_count": sum(1 for z in volunteer_zones if z["status"] == "yellow"),
        "danger_count": sum(1 for z in volunteer_zones if z["status"] == "red"),
    }
    return render(request, "detection/volunteer.html", context)


def teams_view(request):
    """Teams dashboard — view all rescue teams, their positions, assigned missions."""
    import json
    from .models import Zone

    teams = Team.objects.all().order_by("status", "name")
    active_missions = Mission.objects.filter(status__in=["active", "rerouting", "planned"])

    team_data = []
    for t in teams:
        active_mission = active_missions.filter(team=t).first()
        team_data.append({
            "id": t.id,
            "name": t.name,
            "status": t.status,
            "members": t.members,
            "lat": t.latitude,
            "lon": t.longitude,
            "assigned_zone": t.assigned_zone.zone_id if t.assigned_zone else None,
            "mission_status": active_mission.status if active_mission else None,
            "mission_zone": active_mission.zone.zone_id if active_mission else None,
            "reroute_count": active_mission.reroute_count if active_mission else 0,
        })

    context = {
        "teams": teams,
        "team_json": json.dumps(team_data, default=str),
        "total_teams": teams.count(),
        "active_count": teams.filter(status__in=["en_route", "on_site"]).count(),
        "standby_count": teams.filter(status="standby").count(),
        "emergency_count": teams.filter(status="emergency").count(),
        "active_missions": active_missions.count(),
    }
    return render(request, "detection/teams.html", context)


def team_dispatch(request, team_id):
    """Dispatch a team to a zone — creates a Mission and computes initial route."""
    from django.http import JsonResponse
    from .models import Zone
    from routing import find_safe_route

    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    team = Team.objects.filter(id=team_id).first()
    if not team:
        return JsonResponse({"error": "Team not found"}, status=404)

    zone_id = request.POST.get("zone_id")
    zone = Zone.objects.filter(zone_id=zone_id).first()
    if not zone:
        return JsonResponse({"error": f"Zone '{zone_id}' not found"}, status=404)

    blocked = list(
        Detection.objects.filter(
            zone=zone,
            object_class__in=["fire", "flood_water", "collapsed_building", "fallen_power_pole"]
        ).exclude(latitude__isnull=True).values_list("latitude", "longitude")
    )

    route = []
    if team.latitude and team.longitude:
        route = find_safe_route(team.latitude, team.longitude, zone.center_lat, zone.center_lon, blocked)

    import json
    mission = Mission.objects.create(
        team=team,
        zone=zone,
        status="active",
        route_json=json.dumps(route),
    )
    team.status = "en_route"
    team.assigned_zone = zone
    team.save()

    return JsonResponse({
        "mission_id": mission.id,
        "team": team.name,
        "zone": zone_id,
        "route_points": len(route),
        "status": "active",
    })


def team_position_update(request, team_id):
    """Update a team's GPS position (called from team's mobile device)."""
    from django.http import JsonResponse

    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    team = Team.objects.filter(id=team_id).first()
    if not team:
        return JsonResponse({"error": "Team not found"}, status=404)

    lat = request.POST.get("lat")
    lon = request.POST.get("lon")
    if lat is None or lon is None:
        return JsonResponse({"error": "lat and lon required"}, status=400)

    team.latitude = float(lat)
    team.longitude = float(lon)
    team.save()

    return JsonResponse({
        "team": team.name,
        "lat": team.latitude,
        "lon": team.longitude,
        "status": team.status,
    })


def route_status_api(request):
    """API: current route safety status for all active missions."""
    from django.http import JsonResponse
    from route_watch import check_route_safety
    import json

    missions = Mission.objects.filter(status__in=["active", "rerouting"])
    results = []

    for m in missions:
        safety = check_route_safety(m)
        route_coords = json.loads(m.route_json) if m.route_json else []
        results.append({
            "mission_id": m.id,
            "team": m.team.name,
            "team_lat": m.team.latitude,
            "team_lon": m.team.longitude,
            "team_status": m.team.status,
            "zone": m.zone.zone_id,
            "mission_status": m.status,
            "route_safe": safety["safe"],
            "threat_count": safety["threat_count"],
            "threats": safety["threats"],
            "route_points": len(route_coords),
            "reroute_count": m.reroute_count,
            "last_check": m.last_route_check.isoformat() if m.last_route_check else None,
        })

    return JsonResponse({"missions": results, "count": len(results)})


def trigger_route_watch(request):
    """API: manually trigger a route watch cycle."""
    from django.http import JsonResponse
    from route_watch import run_route_watch

    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    actions = run_route_watch()
    return JsonResponse({"actions": actions, "count": len(actions)})
