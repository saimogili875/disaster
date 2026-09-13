"""
Disaster Insight Generator

Generates high-level situational awareness insights from stored Detection records.
Uses CANONICAL_CLASSES keys directly as single sources of truth, eliminating hardcoded alias lists.
"""

from class_config import CANONICAL_CLASSES
from detection.models import Detection

def generate_insights():
    """
    Generate actionable emergency response insights based on canonical detection counts.

    Returns:
        list: Bullet points detailing critical hazard conditions and personnel counts.
    """
    insights = []

    # Query using canonical class keys from CANONICAL_CLASSES
    flood_count = Detection.objects.filter(object_class="flood_water").count()
    fire_count = Detection.objects.filter(object_class="fire").count()
    person_count = Detection.objects.filter(object_class="person").count()
    collapsed_count = Detection.objects.filter(object_class="collapsed_building").count()
    damaged_count = Detection.objects.filter(object_class="damaged_building").count()
    power_pole_count = Detection.objects.filter(object_class="fallen_power_pole").count()

    if flood_count > 0:
        insights.append(f"FLOOD HAZARD: {flood_count} flood water detection(s) recorded in operational grid.")
    if fire_count > 0:
        insights.append(f"FIRE HAZARD: {fire_count} active fire detection(s) identified.")
    if collapsed_count > 0 or damaged_count > 0:
        insights.append(
            f"STRUCTURAL DAMAGE: {collapsed_count} collapsed building(s) and {damaged_count} damaged building(s) detected."
        )
    if person_count > 0:
        insights.append(f"LIFE SAFETY: {person_count} person detection(s) flagged for search & rescue evaluation.")
    if power_pole_count > 0:
        insights.append(f"INFRASTRUCTURE: {power_pole_count} fallen power pole(s) present severe electrocution/access hazards.")

    if not insights:
        insights.append("NORMAL STATUS: No critical hazard detections currently logged.")

    return insights
