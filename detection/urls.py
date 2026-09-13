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
    path('mission/<str:zone_id>/', views.mission_view, name='mission_view'),
    path('volunteer/', views.volunteer_view, name='volunteer_view'),
    path('teams/', views.teams_view, name='teams_view'),
    path('teams/<int:team_id>/dispatch/', views.team_dispatch, name='team_dispatch'),
    path('teams/<int:team_id>/position/', views.team_position_update, name='team_position_update'),
    path('api/route-status/', views.route_status_api, name='route_status_api'),
    path('api/route-watch/', views.trigger_route_watch, name='trigger_route_watch'),
]
