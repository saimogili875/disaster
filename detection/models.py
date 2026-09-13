from django.db import models

class Detection(models.Model):
    # Timestamp when the object detection occurred
    timestamp = models.DateTimeField(auto_now_add=True)

    # Class name of the detected object (e.g., 'person', 'fire', 'flood', etc.)
    object_class = models.CharField(max_length=50)

    # Confidence score of the detection (0.0 to 1.0)
    confidence = models.FloatField()

    # Bounding box coordinates (x1, y1, x2, y2)
    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()

    # Source type of detection: "rgb" for person/general, "hazard" for fire/flood/collapsed_building/debris
    source = models.CharField(max_length=20, default="rgb")

    # Optional geographic location coordinates
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    def __str__(self):
        return f"{self.object_class} ({self.confidence:.2f}) [{self.source}] at {self.timestamp}"
