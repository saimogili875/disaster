"""
Route Watch Command — monitors active mission routes for new hazards.

Usage:
    python manage.py watch_routes              # Single check
    python manage.py watch_routes --loop       # Continuous monitoring (every 30s)
    python manage.py watch_routes --interval 10  # Custom interval in seconds
"""

import time
from django.core.management.base import BaseCommand
from route_watch import run_route_watch


class Command(BaseCommand):
    help = "Check active mission routes for new hazards and trigger rerouting if needed."

    def add_arguments(self, parser):
        parser.add_argument('--loop', action='store_true',
                            help='Run continuously in a loop')
        parser.add_argument('--interval', type=int, default=30,
                            help='Seconds between checks in loop mode (default: 30)')

    def handle(self, *args, **options):
        if options['loop']:
            self.stdout.write(self.style.SUCCESS(
                f"Route Watch running (checking every {options['interval']}s)..."
            ))
            while True:
                self._run_check()
                time.sleep(options['interval'])
        else:
            self._run_check()

    def _run_check(self):
        actions = run_route_watch()
        if actions:
            for action in actions:
                if action["type"] == "rerouted":
                    self.stdout.write(self.style.WARNING(
                        f"REROUTED: Team {action['team']} -> Zone {action['zone']} "
                        f"({action['threats']} threats, {action['new_route_points']} new waypoints)"
                    ))
                elif action["type"] == "trapped":
                    self.stdout.write(self.style.ERROR(
                        f"TRAPPED: Team {action['team']} in Zone {action['zone']} — "
                        f"NO VIABLE ROUTES. Emergency alert created."
                    ))
                elif action["type"] == "retreat":
                    self.stdout.write(self.style.WARNING(
                        f"RETREAT: Team {action['team']} returning from Zone {action['zone']} "
                        f"({action['retreat_route_points']} waypoints)"
                    ))
                elif action["type"] == "no_route":
                    self.stdout.write(self.style.ERROR(
                        f"NO ROUTE: Team {action['team']} -> Zone {action['zone']} — "
                        f"Forward blocked, checking backward..."
                    ))
        else:
            self.stdout.write("Route check: all routes clear.")
