from django.contrib import admin
from .models import Detection

@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    # Display fields in Django admin panel
    list_display = ('timestamp', 'object_class', 'confidence', 'source', 'latitude', 'longitude')
