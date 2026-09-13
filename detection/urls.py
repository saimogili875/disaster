from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_and_detect, name='upload_and_detect'),
    path('upload/', views.upload_and_detect, name='upload_and_detect_alt'),
    path('report/', views.report_view, name='report_view'),
    path('map/', views.map_view, name='map_view'),
    path('alerts/', views.alerts_view, name='alerts_view'),
    path('alerts/<int:alert_id>/acknowledge/', views.alert_acknowledge, name='alert_acknowledge'),
    path('alerts/<int:alert_id>/resolve/', views.alert_resolve, name='alert_resolve'),
    path('dashboard/', views.dashboard_view, name='dashboard_view'),
]
