import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from . import detect_hazards

@csrf_exempt
def detect_hazards(request):
    if request.method == 'POST':
        if 'file' not in request.FILES:
            return JsonResponse({'success': False, 'error': 'No image uploaded'}, status=400)

        image = request.FILES['file']
        results = detect_hazards.detect_hazards(image, 'detections.log')

        return JsonResponse({'success': True, 'detections': results})
    else:
        return JsonResponse({'success': False, 'error': 'Invalid request method'}, status=405)
