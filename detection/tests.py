from django.test import TestCase
from . import detect_hazards

class DetectionTestCase(TestCase):
    def test_detection(self):
        image = open('path/to/test/image.jpg', 'rb')
        results = detect_hazards.detect_hazards(image, 'detections.log')
        self.assertTrue(len(results) > 0)
