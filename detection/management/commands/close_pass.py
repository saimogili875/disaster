from django.core.management.base import BaseCommand
from django.utils import timezone
from detection.models import FlightPass, Zone, ZoneRiskSnapshot, Detection

HAZARD_WEIGHTS = {
    'fire': 3.0,
    'smoke': 2.0,
    'collapsed_building': 3.0,
    'building-total-destruction': 3.0,
    'building-major-damage': 2.0,
    'flood_water': 2.5,
    'water': 2.5,
    'flood': 2.5,
    'debris': 1.5,
    'fallen_tree': 1.0,
    'power_line': 2.0,
    'fallen_power_pole': 2.5,
    'gas_cylinder': 2.5,
}

def compute_zone_risk_score(detections_qs):
    """
    Calculate aggregate risk score, 4-tier risk label (SAFE/CAUTION/HIGH-RISK/DANGER),
    and confidence level for a set of zone detections.

    Note: Any existing ZoneRiskSnapshot records created prior to this update retain their
    historical labels until re-evaluated in subsequent close_pass runs.
    """
    count = detections_qs.count()
    if count == 0:
        return 0.0, "SAFE", "low"

    total_score = 0.0
    for d in detections_qs:
        weight = HAZARD_WEIGHTS.get(d.object_class, 1.0)
        total_score += weight * d.confidence

    # 4-Tier Risk Classification
    if total_score < 3.0:
        risk_label = "SAFE"
    elif total_score < 8.0:
        risk_label = "CAUTION"
    elif total_score < 15.0:
        risk_label = "HIGH-RISK"
    else:
        risk_label = "DANGER"

    if count >= 5:
        confidence_label = "high"
    elif count >= 2:
        confidence_label = "moderate"
    else:
        confidence_label = "low"

    return total_score, risk_label, confidence_label

class Command(BaseCommand):
    help = "Close active FlightPass, aggregate batch statistics, and compute 4-tier ZoneRiskSnapshot records."

    def add_arguments(self, parser):
        parser.add_argument('--pass-id', type=int, default=None, help='Specific FlightPass ID to close')

    def handle(self, *args, **options):
        pass_id = options['pass_id']

        if pass_id:
            flight_pass = FlightPass.objects.filter(id=pass_id).first()
            if not flight_pass:
                self.stderr.write(self.style.ERROR(f"FlightPass with ID {pass_id} not found."))
                return
        else:
            flight_pass = FlightPass.objects.filter(status="active").order_by('-start_time').first()
            if not flight_pass:
                self.stdout.write("No active FlightPass found. Creating a new completed pass record.")
                flight_pass = FlightPass.objects.create(status="active", drone_id="drone_01")

        # Mark pass as completed with end_time
        flight_pass.status = "completed"
        flight_pass.end_time = timezone.now()
        flight_pass.save()

        self.stdout.write(self.style.SUCCESS(f"Closed FlightPass ID {flight_pass.id} at {flight_pass.end_time}."))

        # Find all zones touched by detections in this flight pass
        pass_detections = Detection.objects.filter(flight_pass=flight_pass)
        if not pass_detections.exists():
            pass_detections = Detection.objects.filter(flight_pass=None)

        touched_zone_ids = pass_detections.exclude(zone=None).values_list('zone_id', flat=True).distinct()
        touched_zones = Zone.objects.filter(id__in=touched_zone_ids)

        snapshot_count = 0
        for zone in touched_zones:
            zone_dets = pass_detections.filter(zone=zone)
            det_count = zone_dets.count()
            score, label, conf = compute_zone_risk_score(zone_dets)

            ZoneRiskSnapshot.objects.create(
                zone=zone,
                flight_pass=flight_pass,
                risk_score=score,
                risk_label=label,
                confidence=conf,
                detection_count=det_count
            )
            snapshot_count += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"Successfully computed batch risk aggregation for {snapshot_count} zones on FlightPass {flight_pass.id}."
            )
        )
