"""CrowdGuard simulation engine.

Everything produced by this module is synthetic. It moves a set of virtual
people around a virtual venue, derives zone density from their positions and
invents flood telemetry from a rainfall model. No camera, sensor or radio
hardware is read and nothing is sent over a network.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque
from dataclasses import dataclass

VENUE_NAME = "Tech Fest Grounds"
VENUE_W_M = 20.0
VENUE_H_M = 11.0
VENUE_AREA_M2 = VENUE_W_M * VENUE_H_M

RISK_GREEN = "GREEN"
RISK_YELLOW = "YELLOW"
RISK_RED = "RED"

RED_LOAD = 0.85
YELLOW_LOAD = 0.60

FLOOD_WATCH_CM = 85.0
FLOOD_WARN_CM = 105.0
FLOOD_CRIT_CM = 125.0
FLOOD_MAX_CM = 170.0


def risk_for_load(load: float) -> str:
    """Map a zone occupancy ratio onto the three presentation risk levels."""
    if load >= RED_LOAD:
        return RISK_RED
    if load >= YELLOW_LOAD:
        return RISK_YELLOW
    return RISK_GREEN


@dataclass
class Zone:
    """One monitored area of the venue, stored in normalised venue coords."""

    zone_id: str
    name: str
    x: float
    y: float
    w: float
    h: float
    capacity: int
    attraction: float = 0.5
    count: int = 0

    @property
    def area_m2(self) -> float:
        return max(4.0, self.w * self.h * VENUE_AREA_M2)

    @property
    def density(self) -> float:
        return self.count / self.area_m2

    @property
    def load(self) -> float:
        return self.count / self.capacity

    @property
    def risk(self) -> str:
        return risk_for_load(self.load)

    @property
    def label(self) -> str:
        return f"{self.zone_id}  {self.name}"


def _default_zones() -> list[Zone]:
    layout = [
        ("Z1", "Main Gate", 0.02, 0.02, 0.20, 0.24),
        ("Z2", "Entry Corridor", 0.24, 0.02, 0.16, 0.24),
        ("Z3", "Food Court", 0.42, 0.02, 0.22, 0.24),
        ("Z4", "Main Stage", 0.66, 0.02, 0.32, 0.24),
        ("Z5", "Central Plaza", 0.02, 0.28, 0.44, 0.40),
        ("Z6", "VIP Lounge", 0.48, 0.28, 0.20, 0.24),
        ("Z7", "Media Bay", 0.70, 0.28, 0.28, 0.24),
        ("Z8", "East Exit", 0.48, 0.54, 0.20, 0.14),
        ("Z9", "Parking Walk", 0.02, 0.70, 0.96, 0.28),
    ]
    zones = []
    areas = [max(4.0, w * h * VENUE_AREA_M2) for _, _, _, _, w, h in layout]
    biggest = max(areas)
    for (zone_id, name, x, y, w, h), area in zip(layout, areas):
        jitter = random.Random(zone_id).uniform(0.6, 1.4)
        zones.append(
            Zone(
                zone_id=zone_id,
                name=name,
                x=x,
                y=y,
                w=w,
                h=h,
                capacity=max(12, int(area * 3.4)),
                attraction=round(min(2.0, max(0.15, (area / biggest) * jitter)), 3),
            )
        )
    return zones


@dataclass
class Person:
    """A single virtual attendee. Kept tiny on purpose, this is a demo."""

    x: float
    y: float
    tx: float
    ty: float
    speed: float
    phase: float
    conf: float
    zone_id: str = ""


class CrowdSimulation:
    """Holds all demo state and advances it one step at a time."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.started_at = time.time()
        self.tick = 0
        self.zones = _default_zones()
        self.zone_by_id = {z.zone_id: z for z in self.zones}
        self.people: list[Person] = []
        self.pop_target = 320
        self.rain_target = 4.0
        self.rain_mmh = 4.0
        self.river_cm = 48.0
        self.river_prev = 48.0
        self.pumps_on = False
        self.alerts: list[dict] = []
        self.logs: list[dict] = []
        self.history: deque[dict] = deque(maxlen=90)
        self.radio: list[dict] = []
        self._alert_seq = 0
        self._seed_population(320)
        self._recount()
        self.log("SYSTEM", "CrowdGuard ready - synthetic telemetry only", "info")

    def log(self, source: str, message: str, level: str = "info") -> None:
        self.logs.append(
            {
                "ts": time.time(),
                "source": source,
                "message": message,
                "level": level,
            }
        )
        if len(self.logs) > 400:
            del self.logs[: len(self.logs) - 400]

    def _zone_for(self, x: float, y: float) -> Zone:
        for z in self.zones:
            if z.x <= x <= z.x + z.w and z.y <= y <= z.y + z.h:
                return z
        best = min(
            self.zones,
            key=lambda z: (x - (z.x + z.w / 2)) ** 2 + (y - (z.y + z.h / 2)) ** 2,
        )
        return best

    def _pick_zone(self) -> Zone:
        total = sum(z.attraction for z in self.zones)
        roll = self.rng.random() * total
        for z in self.zones:
            roll -= z.attraction
            if roll <= 0:
                return z
        return self.zones[-1]

    def _point_in(self, zone: Zone) -> tuple[float, float]:
        return (
            min(0.99, max(0.01, zone.x + self.rng.random() * zone.w)),
            min(0.99, max(0.01, zone.y + self.rng.random() * zone.h)),
        )

    def _seed_population(self, n: int) -> None:
        for _ in range(n):
            zone = self._pick_zone()
            x, y = self._point_in(zone)
            tx, ty = self._point_in(zone)
            self.people.append(
                Person(
                    x=x,
                    y=y,
                    tx=tx,
                    ty=ty,
                    speed=self.rng.uniform(0.020, 0.055),
                    phase=self.rng.uniform(0, math.tau),
                    conf=round(self.rng.uniform(0.86, 0.99), 3),
                    zone_id=zone.zone_id,
                )
            )

    def _spawn(self, n: int, zone: Zone | None = None) -> None:
        for _ in range(n):
            target = zone or self._pick_zone()
            x, y = self._point_in(target)
            tx, ty = self._point_in(target)
            self.people.append(
                Person(
                    x=x,
                    y=y,
                    tx=tx,
                    ty=ty,
                    speed=self.rng.uniform(0.020, 0.055),
                    phase=self.rng.uniform(0, math.tau),
                    conf=round(self.rng.uniform(0.86, 0.99), 3),
                    zone_id=target.zone_id,
                )
            )

    def _remove(self, n: int) -> None:
        for _ in range(n):
            if not self.people:
                return
            self.people.pop(self.rng.randrange(len(self.people)))

    def _recount(self) -> None:
        for z in self.zones:
            z.count = 0
        for p in self.people:
            zone = self._zone_for(p.x, p.y)
            p.zone_id = zone.zone_id
            zone.count += 1

    def set_population(self, n: int) -> None:
        self.pop_target = max(0, min(900, int(n)))

    def set_rain(self, mmh: float) -> None:
        self.rain_target = max(0.0, min(45.0, float(mmh)))

    def surge(self, zone_id: str | None = None) -> None:
        zone = self.zone_by_id.get(zone_id or "") or max(
            self.zones, key=lambda z: z.load
        )
        zone.attraction = 2.0
        self._spawn(60, zone)
        self.log("CROWD", f"Simulated surge building at {zone.label}", "warn")
        self.announce(f"Attention. Crowd surge at {zone.name}. Officers move to zone {zone.zone_id}.", "alert")

    def all_clear(self) -> None:
        for a in self.alerts:
            if a["status"] in ("OPEN", "ACK"):
                a["status"] = "RESOLVED"
                a["resolved_at"] = time.time()
        self.log("SYSTEM", "All zones marked clear by the officer on duty", "info")
        self.announce("All clear. Resume normal patrol.", "clear")

    def inject_test_alert(self, zone_id: str | None = None) -> dict:
        zone = self.zone_by_id.get(zone_id or "") or max(
            self.zones, key=lambda z: z.load
        )
        alert = self._new_alert(
            kind="test",
            zone_id=zone.zone_id,
            severity=RISK_RED,
            message=f"TEST ALERT - simulated crowd overflow at {zone.name}",
        )
        self.log("ALERT", alert["message"], "crit")
        self.announce(
            f"Testing. Testing. Alert at {zone.name}, zone {zone.zone_id}. "
            "Security officer respond immediately.",
            "alert",
        )
        return alert

    def acknowledge(self, alert_id: str) -> None:
        for a in self.alerts:
            if a["id"] == alert_id and a["status"] == "OPEN":
                a["status"] = "ACK"
                a["acked_at"] = time.time()
                self.log("OFFICER", f"Acknowledged {alert_id}", "info")
                return

    def open_alerts(self) -> list[dict]:
        return [a for a in self.alerts if a["status"] in ("OPEN", "ACK")]

    def _new_alert(self, kind: str, zone_id: str, severity: str, message: str) -> dict:
        self._alert_seq += 1
        alert = {
            "id": f"A-{self._alert_seq:03d}",
            "ts": time.time(),
            "kind": kind,
            "zone_id": zone_id,
            "severity": severity,
            "message": message,
            "status": "OPEN",
        }
        self.alerts.append(alert)
        return alert

    def step(self, dt: float = 1.0) -> None:
        self.tick += 1
        self.river_prev = self.river_cm

        for z in self.zones:
            z.attraction = min(
                2.0, max(0.15, z.attraction * math.exp(self.rng.gauss(0, 0.022)))
            )

        delta = int(round(self.pop_target - len(self.people)))
        if delta > 0:
            self._spawn(min(delta, 12))
        elif delta < 0:
            self._remove(min(-delta, 10))

        self._move(dt)
        self._recount()
        self._update_flood()
        self._evaluate()

        self.history.append(
            {
                "tick": self.tick,
                "total": len(self.people),
                "load": max(z.load for z in self.zones),
                "river": self.river_cm,
            }
        )

    def _move(self, dt: float) -> None:
        for p in self.people:
            if self.rng.random() < 0.012:
                zone = self._pick_zone()
                p.zone_id = zone.zone_id
                p.tx, p.ty = self._point_in(zone)
            dx = p.tx - p.x
            dy = p.ty - p.y
            dist = math.hypot(dx, dy)
            if dist < 0.015:
                p.tx, p.ty = self._point_in(self.zone_by_id.get(p.zone_id, self.zones[0]))
                continue
            travel = min(p.speed * dt, dist)
            p.x += dx / dist * travel + self.rng.gauss(0, 0.0011)
            p.y += dy / dist * travel + self.rng.gauss(0, 0.0011)
            p.x = min(0.995, max(0.005, p.x))
            p.y = min(0.995, max(0.005, p.y))

    def _update_flood(self) -> None:
        self.rain_mmh = max(
            0.0, self.rain_mmh + (self.rain_target - self.rain_mmh) * 0.15 + self.rng.gauss(0, 0.25)
        )
        self.pumps_on = self.river_cm > FLOOD_WATCH_CM
        target = 40.0 + self.rain_mmh * 3.2 - (30.0 if self.pumps_on else 0.0)
        self.river_cm = min(
            FLOOD_MAX_CM,
            max(20.0, self.river_cm + (target - self.river_cm) * 0.08 + self.rng.gauss(0, 0.5)),
        )

    def flood_status(self) -> str:
        if self.river_cm >= FLOOD_CRIT_CM:
            return "CRITICAL"
        if self.river_cm >= FLOOD_WARN_CM:
            return "WARNING"
        if self.river_cm >= FLOOD_WATCH_CM:
            return "WATCH"
        return "NORMAL"

    def _evaluate(self) -> None:
        for z in self.zones:
            open_one = next(
                (
                    a
                    for a in self.alerts
                    if a["kind"] == "crowd"
                    and a["zone_id"] == z.zone_id
                    and a["status"] in ("OPEN", "ACK")
                ),
                None,
            )
            if z.risk == RISK_RED and open_one is None:
                alert = self._new_alert(
                    kind="crowd",
                    zone_id=z.zone_id,
                    severity=RISK_RED,
                    message=f"Crowd density {z.load * 100:.0f}% of safe capacity at {z.name}",
                )
                self.log("ALERT", alert["message"], "crit")
                self.announce(
                    f"Alert. High crowd density at {z.name}, zone {z.zone_id}. "
                    "Open the side gate and direct flow outwards.",
                    "alert",
                )
            elif z.risk != RISK_RED and open_one is not None:
                open_one["status"] = "RESOLVED"
                open_one["resolved_at"] = time.time()
                self.log("SYSTEM", f"{z.name} back below the red threshold", "info")

        status = self.flood_status()
        flood_open = next(
            (
                a
                for a in self.alerts
                if a["kind"] == "flood" and a["status"] in ("OPEN", "ACK")
            ),
            None,
        )
        if flood_open is not None and status == "CRITICAL" and flood_open["severity"] != RISK_RED:
            flood_open["severity"] = RISK_RED
            flood_open["message"] = (
                f"Flood water {self.river_cm:.0f} cm at the Parking Walk (CRITICAL)"
            )
            self.log("FLOOD", flood_open["message"], "crit")
        if status in ("WARNING", "CRITICAL") and flood_open is None:
            alert = self._new_alert(
                kind="flood",
                zone_id="Z9",
                severity=RISK_RED if status == "CRITICAL" else RISK_YELLOW,
                message=f"Flood water {self.river_cm:.0f} cm at the Parking Walk ({status})",
            )
            self.log("FLOOD", alert["message"], "crit" if status == "CRITICAL" else "warn")
            self.announce(
                f"Flood warning. Water level {self.river_cm:.0f} centimetres at the parking walk. "
                "Move vehicles to the upper lot.",
                "alert",
            )
        elif status in ("NORMAL", "WATCH") and flood_open is not None:
            flood_open["status"] = "RESOLVED"
            flood_open["resolved_at"] = time.time()
            self.log("FLOOD", "Flood level back inside the safe band", "info")

    def camera_view(self, zone_id: str = "Z1", limit: int = 30) -> dict:
        zone = self.zone_by_id.get(zone_id, self.zones[0])
        x0 = max(0.0, zone.x - zone.w * 0.45)
        x1 = min(1.0, zone.x + zone.w * 1.45)
        y0 = max(0.0, zone.y - zone.h * 0.65)
        y1 = min(1.0, zone.y + zone.h * 1.25)
        found = []
        for p in self.people:
            if not (x0 <= p.x <= x1 and y0 <= p.y <= y1):
                continue
            found.append(
                {
                    "u": (p.x - x0) / (x1 - x0),
                    "v": (p.y - y0) / (y1 - y0),
                    "conf": p.conf,
                    "phase": p.phase,
                    "risk": zone.risk,
                }
            )
        found.sort(key=lambda d: -d["v"])
        if len(found) > limit:
            stride = len(found) / limit
            found = [found[int(i * stride)] for i in range(limit)]
        return {
            "name": f"SIM CAM {zone.zone_id} - {zone.name.upper()}",
            "zone_id": zone.zone_id,
            "risk": zone.risk,
            "count": zone.count,
            "people": found,
        }

    def announce(self, text: str, kind: str = "alert") -> dict:
        self.radio.append({"ts": time.time(), "text": text, "kind": kind})
        if len(self.radio) > 40:
            del self.radio[: len(self.radio) - 40]
        return self.radio[-1]

    def totals(self) -> dict:
        total = len(self.people)
        return {
            "total": total,
            "density": total / VENUE_AREA_M2,
            "risk": max((z.risk for z in self.zones), key=lambda r: (RISK_GREEN, RISK_YELLOW, RISK_RED).index(r)),
            "alerts": len(self.open_alerts()),
        }
