"""
Drone Flight Simulator — generates realistic detection data along a flight path.

Usage:
    python manage.py simulate_drone
    python manage.py simulate_drone --lat 17.385 --lon 78.4867 --duration 90
    python manage.py simulate_drone --reset  (clears previous simulation data first)

Creates a FlightPass, flies a grid pattern over the target area, and generates
Detection records with real lat/lon. Zones auto-create via Detection.save().
The map, dashboard, alerts, and scoring all come alive from this data.
"""

import math
import random
import time
from django.core.management.base import BaseCommand
from django.utils import timezone
from detection.models import Detection, FlightPass, Zone, Team, Mission


DISASTER_SCENARIOS = [
    {
        "name": "Urban Fire + Structural Collapse",
        "classes": [
            ("fire", 0.85, 0.95, "high"),
            ("fire", 0.75, 0.90, "medium"),
            ("smoke", 0.80, 0.95, None),
            ("smoke", 0.70, 0.88, None),
            ("collapsed_building", 0.78, 0.92, None),
            ("damaged_building", 0.72, 0.88, None),
            ("debris", 0.80, 0.93, None),
            ("person", 0.82, 0.96, None),
            ("person", 0.65, 0.85, None),
            ("person", 0.70, 0.90, None),
            ("fallen_power_pole", 0.75, 0.90, None),
            ("gas_cylinder", 0.68, 0.82, None),
        ],
    },
    {
        "name": "Flood Zone",
        "classes": [
            ("flood_water", 0.85, 0.97, None),
            ("flood_water", 0.80, 0.95, None),
            ("flood_water", 0.75, 0.92, None),
            ("person", 0.60, 0.85, None),
            ("person", 0.55, 0.80, None),
            ("vehicle_car", 0.78, 0.93, None),
            ("vehicle_truck", 0.72, 0.88, None),
            ("debris", 0.70, 0.85, None),
            ("fallen_tree", 0.80, 0.92, None),
            ("fallen_tree", 0.75, 0.88, None),
            ("dog", 0.62, 0.78, None),
        ],
    },
    {
        "name": "Earthquake Damage",
        "classes": [
            ("collapsed_building", 0.85, 0.96, None),
            ("collapsed_building", 0.80, 0.93, None),
            ("damaged_building", 0.82, 0.94, None),
            ("damaged_building", 0.75, 0.90, None),
            ("debris", 0.88, 0.97, None),
            ("debris", 0.80, 0.92, None),
            ("person", 0.72, 0.90, None),
            ("person", 0.65, 0.85, None),
            ("person", 0.60, 0.80, None),
            ("person", 0.55, 0.78, None),
            ("fallen_power_pole", 0.78, 0.90, None),
            ("power_line", 0.70, 0.85, None),
            ("vehicle_car", 0.80, 0.92, None),
        ],
    },
    {
        "name": "Mixed Hazard — Moderate",
        "classes": [
            ("person", 0.70, 0.88, None),
            ("vehicle_car", 0.82, 0.95, None),
            ("vehicle_bus", 0.75, 0.90, None),
            ("debris", 0.65, 0.82, None),
            ("fallen_tree", 0.72, 0.88, None),
            ("smoke", 0.60, 0.78, None),
        ],
    },
    {
        "name": "Clear / Low Risk",
        "classes": [
            ("person", 0.80, 0.95, None),
            ("vehicle_car", 0.85, 0.97, None),
            ("vehicle_motorcycle", 0.78, 0.92, None),
        ],
    },
]

FLOW_DIRECTIONS = ["north", "south", "east", "west", "northeast", "northwest"]
FLOW_SPEEDS = ["slow", "moderate", "fast"]
WIND_DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


HYDERABAD_LANDMARKS = [
    # (lat, lon, name, scenario_index)
    # Scenario 0: Urban Fire, 1: Flood, 2: Earthquake, 3: Mixed, 4: Clear
    (17.3850, 78.4867, "Charminar Area", 0),
    (17.3950, 78.4740, "Nizam's Museum / Purani Haveli", 2),
    (17.4000, 78.4800, "Osmania General Hospital Road", 0),
    (17.3616, 78.4747, "Falaknuma Palace Area", 3),
    (17.4375, 78.4483, "Hussain Sagar Lake Edge", 1),
    (17.4400, 78.4980, "Secunderabad Railway Station", 2),
    (17.3984, 78.4861, "Mecca Masjid / High Court", 3),
    (17.4260, 78.4530, "Necklace Road / NTR Gardens", 1),
    (17.4156, 78.4347, "KBR National Park Perimeter", 4),
    (17.3780, 78.4920, "Salar Jung Museum Area", 0),
    (17.4480, 78.3910, "HITEC City / Cyber Towers", 4),
    (17.4440, 78.3490, "Gachibowli Stadium Area", 4),
    (17.3520, 78.5340, "LB Nagar / NH65 Junction", 3),
    (17.4240, 78.5500, "Uppal Flyover Area", 2),
    (17.3900, 78.5100, "Chaderghat Bridge", 1),
    (17.4100, 78.4700, "Abids / GPO Area", 0),
    (17.4310, 78.4200, "Banjara Hills Road No. 12", 3),
    (17.4560, 78.3620, "Kondapur / ORR Junction", 4),
    (17.3700, 78.4800, "Afzalgunj Bus Station", 2),
    (17.4050, 78.4500, "Lakdi-ka-pul Railway Bridge", 1),
]


def _generate_flight_path(center_lat, center_lon, grid_size=0.004, legs=4):
    """Generate a flight path through real Hyderabad landmarks with connecting waypoints."""
    points = []

    for i, (lat, lon, name, scenario_idx) in enumerate(HYDERABAD_LANDMARKS):
        points.append((round(lat, 6), round(lon, 6), scenario_idx, name))

        for _ in range(3):
            jitter_lat = lat + random.uniform(-0.001, 0.001)
            jitter_lon = lon + random.uniform(-0.001, 0.001)
            points.append((round(jitter_lat, 6), round(jitter_lon, 6), scenario_idx, name))

        if i < len(HYDERABAD_LANDMARKS) - 1:
            next_lat, next_lon = HYDERABAD_LANDMARKS[i + 1][0], HYDERABAD_LANDMARKS[i + 1][1]
            mid_lat = (lat + next_lat) / 2 + random.uniform(-0.001, 0.001)
            mid_lon = (lon + next_lon) / 2 + random.uniform(-0.001, 0.001)
            transit_scenario = random.choice([3, 4])
            points.append((round(mid_lat, 6), round(mid_lon, 6), transit_scenario, "Transit"))

    return points


def _pick_scenario_for_position(position_index, total_positions):
    """Fallback — used only if landmark data not available."""
    progress = position_index / max(1, total_positions - 1)

    if progress < 0.15:
        return random.choice([DISASTER_SCENARIOS[3], DISASTER_SCENARIOS[4]])
    elif progress < 0.35:
        return DISASTER_SCENARIOS[0]
    elif progress < 0.55:
        return DISASTER_SCENARIOS[1]
    elif progress < 0.75:
        return DISASTER_SCENARIOS[2]
    else:
        return random.choice([DISASTER_SCENARIOS[3], DISASTER_SCENARIOS[4]])


class Command(BaseCommand):
    help = "Simulate a drone flight mission with realistic disaster detections."

    def add_arguments(self, parser):
        parser.add_argument('--lat', type=float, default=17.385,
                            help='Center latitude (default: Hyderabad 17.385)')
        parser.add_argument('--lon', type=float, default=78.4867,
                            help='Center longitude (default: Hyderabad 78.4867)')
        parser.add_argument('--duration', type=int, default=60,
                            help='Simulated mission duration in seconds (default: 60)')
        parser.add_argument('--drone-id', type=str, default='drone_01',
                            help='Drone identifier')
        parser.add_argument('--reset', action='store_true',
                            help='Clear all existing detections and zones before simulating')
        parser.add_argument('--realtime', action='store_true',
                            help='Add real delays between positions (for live demo)')

    def handle(self, *args, **options):
        center_lat = options['lat']
        center_lon = options['lon']
        duration = options['duration']
        drone_id = options['drone_id']

        if options['reset']:
            det_count = Detection.objects.count()
            zone_count = Zone.objects.count()
            team_count = Team.objects.count()
            mission_count = Mission.objects.count()
            Detection.objects.all().delete()
            Zone.objects.all().delete()
            FlightPass.objects.all().delete()
            Mission.objects.all().delete()
            Team.objects.all().delete()
            from detection.models import Alert
            Alert.objects.all().delete()
            self.stdout.write(self.style.WARNING(
                f"Reset: deleted {det_count} detections, {zone_count} zones, "
                f"{team_count} teams, {mission_count} missions, all flight passes and alerts."
            ))

        flight_pass = FlightPass.objects.create(
            drone_id=drone_id,
            status="active",
        )
        self.stdout.write(self.style.SUCCESS(
            f"FlightPass #{flight_pass.id} started — drone '{drone_id}'"
        ))

        path = _generate_flight_path(center_lat, center_lon)
        total_points = len(path)
        delay = duration / max(1, total_points) if options['realtime'] else 0

        self.stdout.write(f"Flight path: {total_points} waypoints across {len(HYDERABAD_LANDMARKS)} Hyderabad landmarks")
        self.stdout.write(f"Simulating {'with real-time delays' if options['realtime'] else 'instant batch'}...")

        total_detections = 0
        zones_created = set()
        current_landmark = ""

        for idx, waypoint in enumerate(path):
            lat, lon, scenario_idx, landmark_name = waypoint
            scenario = DISASTER_SCENARIOS[scenario_idx]

            if landmark_name != current_landmark and landmark_name != "Transit":
                current_landmark = landmark_name
                self.stdout.write(f"  Flying over: {landmark_name} ({scenario['name']})")

            num_detections = random.randint(1, min(5, len(scenario["classes"])))
            selected = random.sample(scenario["classes"], num_detections)

            for obj_class, conf_min, conf_max, fire_intensity in selected:
                conf = round(random.uniform(conf_min, conf_max), 3)
                det_lat = lat + random.uniform(-0.0008, 0.0008)
                det_lon = lon + random.uniform(-0.0008, 0.0008)

                det_kwargs = {
                    "object_class": obj_class,
                    "confidence": conf,
                    "bbox_x1": random.uniform(50, 400),
                    "bbox_y1": random.uniform(50, 300),
                    "bbox_x2": random.uniform(450, 800),
                    "bbox_y2": random.uniform(350, 600),
                    "source": "hazard" if obj_class in (
                        "fire", "smoke", "flood_water", "collapsed_building", "debris"
                    ) else "rgb",
                    "latitude": round(det_lat, 6),
                    "longitude": round(det_lon, 6),
                    "flight_pass": flight_pass,
                }

                if fire_intensity and obj_class == "fire":
                    det_kwargs["fire_intensity"] = fire_intensity
                    if fire_intensity == "high":
                        det_kwargs["fire_temp_estimate"] = "~1000C+ (RGB proxy)"
                    elif fire_intensity == "medium":
                        det_kwargs["fire_temp_estimate"] = "~700-1000C (RGB proxy)"
                    else:
                        det_kwargs["fire_temp_estimate"] = "~400-700C (RGB proxy)"

                if obj_class == "flood_water":
                    det_kwargs["flow_direction"] = random.choice(FLOW_DIRECTIONS)
                    det_kwargs["flow_speed"] = random.choice(FLOW_SPEEDS)

                if obj_class == "smoke":
                    det_kwargs["wind_direction"] = random.choice(WIND_DIRECTIONS)
                    det_kwargs["wind_beaufort_scale"] = random.choice([
                        "Light breeze (2)", "Gentle breeze (3)",
                        "Moderate breeze (4)", "Fresh breeze (5)",
                    ])

                det = Detection(**det_kwargs)
                det.save()
                total_detections += 1

                if det.zone:
                    zones_created.add(det.zone.zone_id)

            if options['realtime']:
                time.sleep(delay)

            if (idx + 1) % 10 == 0 or idx == total_points - 1:
                self.stdout.write(
                    f"  Waypoint {idx + 1}/{total_points} — "
                    f"{total_detections} detections, {len(zones_created)} zones"
                )

        flight_pass.status = "completed"
        flight_pass.end_time = timezone.now()
        flight_pass.save()

        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=" * 50))
        self.stdout.write(self.style.SUCCESS(f"Mission complete!"))
        self.stdout.write(f"  FlightPass:  #{flight_pass.id}")
        self.stdout.write(f"  Detections:  {total_detections}")
        self.stdout.write(f"  Zones:       {len(zones_created)}")
        self.stdout.write(f"  Area:        {center_lat}, {center_lon}")
        self.stdout.write(self.style.SUCCESS("=" * 50))
        # Create demo rescue teams
        TEAM_DEFS = [
            {"name": "Alpha Team — Charminar", "members": 5, "status": "en_route",
             "base_lat": 17.3610, "base_lon": 78.4740},
            {"name": "Bravo Team — Secunderabad", "members": 4, "status": "on_site",
             "base_lat": 17.4400, "base_lon": 78.4980},
            {"name": "Charlie Team — HITEC City", "members": 6, "status": "standby",
             "base_lat": 17.4480, "base_lon": 78.3910},
            {"name": "Delta Team — Hussain Sagar", "members": 3, "status": "returning",
             "base_lat": 17.4260, "base_lon": 78.4530},
            {"name": "Echo Team — LB Nagar", "members": 4, "status": "standby",
             "base_lat": 17.3520, "base_lon": 78.5340},
        ]

        all_zones = list(Zone.objects.all())
        import json

        teams_created = 0
        missions_created = 0

        for i, tdef in enumerate(TEAM_DEFS):
            team_lat = tdef["base_lat"] + random.uniform(-0.002, 0.002)
            team_lon = tdef["base_lon"] + random.uniform(-0.002, 0.002)

            assigned_zone = None
            if tdef["status"] in ("en_route", "on_site") and all_zones:
                assigned_zone = random.choice(all_zones)
                if tdef["status"] == "on_site":
                    team_lat = assigned_zone.center_lat + random.uniform(-0.0005, 0.0005)
                    team_lon = assigned_zone.center_lon + random.uniform(-0.0005, 0.0005)

            team = Team.objects.create(
                name=tdef["name"],
                members=tdef["members"],
                status=tdef["status"],
                latitude=round(team_lat, 6),
                longitude=round(team_lon, 6),
                assigned_zone=assigned_zone,
            )
            teams_created += 1

            if assigned_zone and tdef["status"] in ("en_route", "on_site"):
                route = [
                    [team_lat, team_lon],
                    [assigned_zone.center_lat, assigned_zone.center_lon],
                ]
                Mission.objects.create(
                    team=team,
                    zone=assigned_zone,
                    status="active",
                    route_json=json.dumps(route),
                )
                missions_created += 1

        self.stdout.write(self.style.SUCCESS(f"  Teams:       {teams_created}"))
        self.stdout.write(f"  Missions:    {missions_created}")
        self.stdout.write(self.style.SUCCESS("=" * 50))
        self.stdout.write("")
        self.stdout.write("Next steps:")
        self.stdout.write("  python manage.py runserver")
        self.stdout.write("  Open /dashboard/ or /map/ to see the data")
        self.stdout.write("  Open /teams/ to see rescue team positions")
        self.stdout.write("  Open /volunteer/ for volunteer zone guide")
        self.stdout.write("  Open /report/ to see the full situation report")
        self.stdout.write("  Open /alerts/ to see triggered alerts")
