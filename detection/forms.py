from django import forms

class ImageUploadForm(forms.Form):
    # File upload field supporting images and videos
    file = forms.FileField(label="Select an image or video file")
