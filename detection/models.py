from django.db import models

class FlightPass(models.Model):
    """Represents a single drone flight pass mission."""
    STATUS_CHOICES = [
        ("active", "Active"),
        ("completed", "Completed"),
    ]

    start_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(null=True, blank=True)
    drone_id = models.CharField(max_length=50, default="drone_01")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="active")

    def __str__(self):
        return f"FlightPass {self.id} ({self.drone_id}) [{self.status}]"

class Zone(models.Model):
    """Geographic grid cell zone for spatial risk aggregation."""
    zone_id = models.CharField(max_length=50, unique=True)
    center_lat = models.FloatField()
    center_lon = models.FloatField()
    grid_size_deg = models.FloatField(default=0.001)

    def __str__(self):
        return f"Zone {self.zone_id}"

class ZoneRiskSnapshot(models.Model):
    """Pre-computed risk assessment snapshot for a Zone during a FlightPass."""
    zone = models.ForeignKey(Zone, on_delete=models.CASCADE)
    flight_pass = models.ForeignKey(FlightPass, on_delete=models.CASCADE)
    risk_score = models.FloatField()
    risk_label = models.CharField(max_length=20)
    confidence = models.CharField(max_length=20)
    detection_count = models.IntegerField()
    computed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ZoneRiskSnapshot {self.zone.zone_id} Pass {self.flight_pass.id} Risk: {self.risk_label} ({self.risk_score:.2f})"

class Detection(models.Model):
    # Foreign key references to FlightPass and Zone
    flight_pass = models.ForeignKey(FlightPass, on_delete=models.SET_NULL, null=True, blank=True)
    zone = models.ForeignKey(Zone, on_delete=models.SET_NULL, null=True, blank=True)

    # Timestamp when the object detection occurred (indexed)
    timestamp = models.DateTimeField(auto_now_add=True, db_index=True)

    # Class name of the detected object (indexed)
    object_class = models.CharField(max_length=50, db_index=True)

    # Optional subcategory/damage level
    subtype = models.CharField(max_length=50, null=True, blank=True)

    # Confidence score of the detection
    confidence = models.FloatField()

    # Bounding box coordinates (x1, y1, x2, y2)
    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()

    # Source type of detection (indexed)
    source = models.CharField(max_length=20, default="rgb", db_index=True)

    # Optional object tracking ID
    track_id = models.IntegerField(null=True, blank=True)

    # Vehicle movement tracking attributes
    movement_status = models.CharField(max_length=20, null=True, blank=True)
    direction = models.CharField(max_length=20, null=True, blank=True)

    # Optical flow attributes for water/flood detections
    flow_direction = models.CharField(max_length=20, null=True, blank=True)
    flow_speed = models.CharField(max_length=20, null=True, blank=True)

    # Fire intensity and temperature proxy estimate
    fire_intensity = models.CharField(max_length=20, null=True, blank=True)
    fire_temp_estimate = models.CharField(max_length=50, null=True, blank=True)

    # Wind estimation attributes
    wind_direction = models.CharField(max_length=20, null=True, blank=True)
    wind_beaufort_scale = models.CharField(max_length=100, null=True, blank=True)

    # Geographic location coordinates
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    def save(self, *args, **kwargs):
        # Automatically determine or create Zone for detection based on lat/lon
        if self.latitude is not None and self.longitude is not None and not self.zone:
            grid_size = 0.001
            grid_lat = round(round(self.latitude / grid_size) * grid_size, 3)
            grid_lon = round(round(self.longitude / grid_size) * grid_size, 3)
            zone_identifier = f"{grid_lat:.3f}_{grid_lon:.3f}"
            zone_obj, _ = Zone.objects.get_or_create(
                zone_id=zone_identifier,
                defaults={
                    'center_lat': grid_lat,
                    'center_lon': grid_lon,
                    'grid_size_deg': grid_size
                }
            )
            self.zone = zone_obj
        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.object_class} ({self.confidence:.2f}) [{self.source}] at {self.timestamp}"
