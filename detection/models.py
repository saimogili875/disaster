from django.db import models

class Detection(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    label = models.CharField(max_length=100)
    confidence = models.FloatField()
    bbox = models.JSONField()
