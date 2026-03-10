"""
ROS2 publisher + RViz2 launcher — Office scenario.

Publishes:
  /office/environment   MarkerArray — floor tiles, walls, puddle
  /office/path          nav_msgs/Path
  /office/events        MarkerArray — mission site markers (puddle, carpet, charger)
  /office/robot         MarkerArray — wheeled robot model
  /office/effects       MarkerArray — dynamic: puddle ripples, charge sparks

Simulation logic:
  - puddle (floor_blue): puddle ripple effect always visible
  - charger (floor_yellow): charge sparks appear when robot arrives
  - carpet (floor_grey): carpet texture tile marker
"""

import json
import math
import os
import subprocess
import sys
import threading
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from builtin_interfaces.msg import Duration

# ---------------------------------------------------------------------------
# Palette — office colours
# ---------------------------------------------------------------------------
FLOOR_COLOR   = ColorRGBA(r=1.0, g=1.0, b=0.98, a=1.0)   # light beige floor
WALL_COLOR    = ColorRGBA(r=0.55, g=0.50, b=0.45, a=1.0)   # warm grey walls
PUDDLE_COLOR  = ColorRGBA(r=0.18, g=0.45, b=0.82, a=0.75)  # translucent blue
CARPET_COLOR  = ColorRGBA(r=0.45, g=0.38, b=0.62, a=1.0)   # muted purple-grey
CHARGER_COLOR = ColorRGBA(r=0.95, g=0.80, b=0.10, a=1.0)   # bright yellow

ROBOT_BODY    = ColorRGBA(r=0.20, g=0.20, b=0.22, a=1.0)   # dark grey chassis
ROBOT_ACCENT  = ColorRGBA(r=0.10, g=0.65, b=0.90, a=1.0)   # cyan accent
ROBOT_WHEEL   = ColorRGBA(r=0.15, g=0.15, b=0.15, a=1.0)   # black wheels
ROBOT_SENSOR  = ColorRGBA(r=0.90, g=0.20, b=0.10, a=1.0)   # red sensor

_EVENT_COLORS = {
    "initial_state0": ColorRGBA(r=0.75, g=0.75, b=0.75, a=1.0),
    "floor_blue":     PUDDLE_COLOR,
    "floor_grey":     CARPET_COLOR,
    "floor_yellow":   CHARGER_COLOR,
}

_EVENT_BADGE = {
    "initial_state0": "S",
    "floor_blue":     "puddle",
    "floor_grey":     "carpet",
    "floor_yellow":   "charger",
}

GRID_W, GRID_H = 12, 8   # office grid dimensions (matches OfficeEnv: 12×8 including walls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _m(ns, mid, mtype):
    mk = Marker()
    mk.header.frame_id = "map"
    mk.ns = ns
    mk.id = mid
    mk.type = mtype
    mk.action = Marker.ADD
    mk.pose.orientation.w = 1.0
    mk.lifetime = Duration(sec=0, nanosec=0)
    return mk

def _pos(mk, x, y, z=0.0):
    mk.pose.position.x = float(x)
    mk.pose.position.y = float(y)
    mk.pose.position.z = float(z)

def _sc(mk, x, y=None, z=None):
    mk.scale.x = float(x)
    mk.scale.y = float(y if y is not None else x)
    mk.scale.z = float(z if z is not None else x)

def _quat(mk, roll=0.0, pitch=0.0, yaw=0.0):
    cr, sr = math.cos(roll/2),  math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2),   math.sin(yaw/2)
    mk.pose.orientation.w = cr*cp*cy + sr*sp*sy
    mk.pose.orientation.x = sr*cp*cy - cr*sp*sy
    mk.pose.orientation.y = cr*sp*cy + sr*cp*sy
    mk.pose.orientation.z = cr*cp*sy - sr*sp*cy


# ---------------------------------------------------------------------------
# Environment: floor tiles + walls + puddle overlay
# ---------------------------------------------------------------------------
def build_environment(puddle_positions):
    ma = MarkerArray()
    mid = 0
    puddle_set = {(int(px), int(py)) for px, py in puddle_positions}

    # Floor tiles — one cube per cell
    for gx in range(GRID_W):
        for gy in range(GRID_H):
            col = ColorRGBA(
                r=FLOOR_COLOR.r + 0.03 * math.sin(gx * 1.7 + gy * 2.3),
                g=FLOOR_COLOR.g + 0.02 * math.cos(gx * 2.1 + gy * 1.5),
                b=FLOOR_COLOR.b,
                a=1.0,
            )
            tile = _m("floor", mid, Marker.CUBE); mid += 1
            _pos(tile, gx + 0.5, gy + 0.5, -0.02)
            _sc(tile, 1.0, 1.0, 0.04)
            tile.color = col
            ma.markers.append(tile)

    # Wall border — tall thin slabs around the perimeter
    WALL_H = 0.30
    WALL_T = 0.10
    for gx in range(GRID_W):
        for gy, sign in [(0, -1), (GRID_H - 1, +1)]:
            w = _m("walls", mid, Marker.CUBE); mid += 1
            _pos(w, gx + 0.5, gy + 0.5 + sign * 0.45, WALL_H / 2)
            _sc(w, 1.0, WALL_T, WALL_H)
            w.color = WALL_COLOR
            ma.markers.append(w)
    for gy in range(GRID_H):
        for gx, sign in [(0, -1), (GRID_W - 1, +1)]:
            w = _m("walls", mid, Marker.CUBE); mid += 1
            _pos(w, gx + 0.5 + sign * 0.45, gy + 0.5, WALL_H / 2)
            _sc(w, WALL_T, 1.0, WALL_H)
            w.color = WALL_COLOR
            ma.markers.append(w)

    # Puddle overlay tiles — slightly raised, translucent blue
    for px, py in puddle_positions:
        puddle = _m("puddle", mid, Marker.CYLINDER); mid += 1
        _pos(puddle, px + 0.5, py + 0.5, 0.01)
        _sc(puddle, 0.82, 0.82, 0.02)
        puddle.color = PUDDLE_COLOR
        ma.markers.append(puddle)

    return ma, mid


# ---------------------------------------------------------------------------
# Office decorations — desks, chairs, plant
# ---------------------------------------------------------------------------
DESK_COLOR  = ColorRGBA(r=0.62, g=0.45, b=0.28, a=1.0)   # warm wood
LEG_COLOR   = ColorRGBA(r=0.30, g=0.28, b=0.26, a=1.0)   # dark metal
CHAIR_COLOR = ColorRGBA(r=0.22, g=0.22, b=0.25, a=1.0)   # charcoal seat
PLANT_STEM  = ColorRGBA(r=0.35, g=0.22, b=0.12, a=1.0)   # brown pot
PLANT_GREEN = ColorRGBA(r=0.18, g=0.58, b=0.22, a=1.0)   # leaf green
PLANT_DARK  = ColorRGBA(r=0.10, g=0.40, b=0.14, a=1.0)   # darker leaf


def _desk(mid, cx, cy, yaw=0.0):
    """A desk: flat top surface + four legs."""
    markers = []
    top = _m("decor_desk", mid, Marker.CUBE); mid += 1
    _pos(top, cx, cy, 0.38); _sc(top, 0.80, 0.50, 0.05)
    _quat(top, yaw=yaw); top.color = DESK_COLOR
    markers.append(top)
    # Thin modesty panel on the back edge
    c, s = math.cos(yaw), math.sin(yaw)
    panel_x = cx - 0.38 * c
    panel_y = cy - 0.38 * s
    panel = _m("decor_desk", mid, Marker.CUBE); mid += 1
    _pos(panel, panel_x, panel_y, 0.20); _sc(panel, 0.06, 0.50, 0.36)
    _quat(panel, yaw=yaw); panel.color = DESK_COLOR
    markers.append(panel)
    # Four legs
    right = math.pi / 2
    for fw, sw in [(+0.36, +0.20), (+0.36, -0.20), (-0.36, +0.20), (-0.36, -0.20)]:
        lx = cx + fw * c - sw * s
        ly = cy + fw * s + sw * c
        leg = _m("decor_desk", mid, Marker.CYLINDER); mid += 1
        _pos(leg, lx, ly, 0.18); _sc(leg, 0.05, 0.05, 0.36)
        leg.color = LEG_COLOR
        markers.append(leg)
    return markers, mid


def _chair(mid, cx, cy, yaw=0.0):
    """A simple office chair: seat + back + four legs."""
    markers = []
    seat = _m("decor_chair", mid, Marker.CUBE); mid += 1
    _pos(seat, cx, cy, 0.24); _sc(seat, 0.36, 0.36, 0.05)
    _quat(seat, yaw=yaw); seat.color = CHAIR_COLOR
    markers.append(seat)
    c, s = math.cos(yaw), math.sin(yaw)
    back_x = cx - 0.16 * c
    back_y = cy - 0.16 * s
    back = _m("decor_chair", mid, Marker.CUBE); mid += 1
    _pos(back, back_x, back_y, 0.40); _sc(back, 0.06, 0.34, 0.30)
    _quat(back, yaw=yaw); back.color = CHAIR_COLOR
    markers.append(back)
    for fw, sw in [(+0.15, +0.15), (+0.15, -0.15), (-0.15, +0.15), (-0.15, -0.15)]:
        lx = cx + fw * c - sw * s
        ly = cy + fw * s + sw * c
        leg = _m("decor_chair", mid, Marker.CYLINDER); mid += 1
        _pos(leg, lx, ly, 0.11); _sc(leg, 0.04, 0.04, 0.22)
        leg.color = LEG_COLOR
        markers.append(leg)
    return markers, mid


def _plant(mid, cx, cy):
    """A potted plant: terracotta pot + layered leaf spheres."""
    markers = []
    pot = _m("decor_plant", mid, Marker.CYLINDER); mid += 1
    _pos(pot, cx, cy, 0.10); _sc(pot, 0.20, 0.20, 0.20)
    pot.color = ColorRGBA(r=0.72, g=0.38, b=0.22, a=1.0)
    markers.append(pot)
    rim = _m("decor_plant", mid, Marker.CYLINDER); mid += 1
    _pos(rim, cx, cy, 0.21); _sc(rim, 0.24, 0.24, 0.03)
    rim.color = ColorRGBA(r=0.60, g=0.30, b=0.18, a=1.0)
    markers.append(rim)
    soil = _m("decor_plant", mid, Marker.CYLINDER); mid += 1
    _pos(soil, cx, cy, 0.22); _sc(soil, 0.18, 0.18, 0.02)
    soil.color = ColorRGBA(r=0.28, g=0.20, b=0.12, a=1.0)
    markers.append(soil)
    # Main foliage — three overlapping spheres
    for dx, dy, dz, sc, col in [
        ( 0.00,  0.00, 0.48, 0.30, PLANT_GREEN),
        ( 0.10,  0.06, 0.40, 0.22, PLANT_DARK),
        (-0.08,  0.09, 0.44, 0.20, PLANT_GREEN),
        ( 0.04, -0.10, 0.42, 0.18, PLANT_DARK),
    ]:
        leaf = _m("decor_plant", mid, Marker.SPHERE); mid += 1
        _pos(leaf, cx + dx, cy + dy, dz); _sc(leaf, sc, sc, sc * 0.85)
        leaf.color = col
        markers.append(leaf)
    return markers, mid


def build_decorations(mid):
    """Place desks, chairs, and a plant in empty wall-adjacent cells."""
    ma = MarkerArray()

    # --- Desk + chair cluster top-left corner (clear of all event regions) ---
    # Desk at (2, 1) facing right (yaw=0), chair pulled in front
    new, mid = _desk(mid, 2.5, 1.5, yaw=0.0)
    ma.markers.extend(new)
    new, mid = _chair(mid, 2.5, 0.8, yaw=-30)   # chair faces the desk
    ma.markers.extend(new)

    # --- Desk + chair along the bottom wall (y=6 area, x=7-8) ---
    new, mid = _desk(mid, 7.5, 6.5, yaw=0.0)
    ma.markers.extend(new)
    new, mid = _chair(mid, 7.5, 7.6, yaw=30.0)       # chair tucked under
    ma.markers.extend(new)

    # --- Second desk top-right area (x=10-11, y=1) ---
    new, mid = _desk(mid, 10.5, 1.5, yaw=math.pi / 2)
    ma.markers.extend(new)
    new, mid = _chair(mid, 10.5, 2.4, yaw=-math.pi / 2)
    ma.markers.extend(new)

    # --- Plant in bottom-left corner (x=1, y=6) ---
    new, mid = _plant(mid, 1.5, 6.5)
    ma.markers.extend(new)

    # --- Plant in top-right corner (x=11, y=1) ---
    new, mid = _plant(mid, 11.5, 1.5)
    ma.markers.extend(new)

    return ma, mid


# ---------------------------------------------------------------------------
# Event markers — reuse mars rover helper shapes, office palette
# ---------------------------------------------------------------------------
def _charger_marker(mid, ex, ey, color):
    markers = []
    base = _m("site_pad", mid, Marker.CUBE); mid += 1
    _pos(base, ex, ey, 0.04); _sc(base, 0.72, 0.72, 0.07)
    base.color = ColorRGBA(r=color.r * 0.8, g=color.g * 0.8, b=color.b * 0.1, a=1.0)
    markers.append(base)
    pole = _m("site_pole", mid, Marker.CYLINDER); mid += 1
    _pos(pole, ex, ey, 0.32); _sc(pole, 0.06, 0.06, 0.52)
    pole.color = ColorRGBA(r=0.70, g=0.70, b=0.70, a=1.0)
    markers.append(pole)
    head = _m("site_light", mid, Marker.SPHERE); mid += 1
    _pos(head, ex, ey, 0.60); _sc(head, 0.16)
    head.color = color
    markers.append(head)
    for angle in [0, math.pi]:
        leg = _m("site_legs", mid, Marker.CUBE); mid += 1
        lx = ex + 0.20 * math.cos(angle)
        ly = ey + 0.20 * math.sin(angle)
        _pos(leg, lx, ly, 0.05); _sc(leg, 0.22, 0.06, 0.05)
        _quat(leg, yaw=angle)
        leg.color = ColorRGBA(r=0.65, g=0.65, b=0.65, a=1.0)
        markers.append(leg)
    return markers, mid


def _carpet_marker(mid, ex, ey, color):
    markers = []
    carpet = _m("site_disc", mid, Marker.CUBE); mid += 1
    _pos(carpet, ex, ey, 0.01); _sc(carpet, 0.88, 0.88, 0.03)
    carpet.color = color
    markers.append(carpet)
    # Grid lines on carpet (2×2 pattern, kept within single cell)
    LINE_COLOR = ColorRGBA(r=color.r * 0.75, g=color.g * 0.75, b=color.b * 0.75, a=1.0)
    for i in range(2):
        hline = _m("site_ring", mid, Marker.CUBE); mid += 1
        _pos(hline, ex, ey - 0.22 + i * 0.44, 0.03)
        _sc(hline, 0.88, 0.03, 0.01)
        hline.color = LINE_COLOR
        markers.append(hline)
        vline = _m("site_ring", mid, Marker.CUBE); mid += 1
        _pos(vline, ex - 0.22 + i * 0.44, ey, 0.03)
        _sc(vline, 0.03, 0.88, 0.01)
        vline.color = LINE_COLOR
        markers.append(vline)
    return markers, mid


def _puddle_site_marker(mid, ex, ey, color):
    markers = []
    puddle = _m("site_tile", mid, Marker.CYLINDER); mid += 1
    _pos(puddle, ex, ey, 0.02); _sc(puddle, 0.80, 0.80, 0.04)
    puddle.color = color
    markers.append(puddle)
    ring = _m("site_ring", mid, Marker.CYLINDER); mid += 1
    _pos(ring, ex, ey, 0.05); _sc(ring, 0.86, 0.86, 0.02)
    ring.color = ColorRGBA(r=color.r * 1.2, g=color.g * 1.2,
                           b=min(1.0, color.b * 1.3), a=0.5)
    markers.append(ring)
    return markers, mid


def build_event_markers(event_nodes, id_offset):
    ma = MarkerArray()
    mid = id_offset

    for ev in event_nodes:
        color = _EVENT_COLORS.get(ev["obs"], ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))
        ex, ey = ev["x"] + 0.5, ev["y"] + 0.5
        obs = ev["obs"]

        if obs == "floor_yellow":
            new_markers, mid = _charger_marker(mid, ex, ey, color)
        elif obs == "floor_grey":
            new_markers, mid = _carpet_marker(mid, ex, ey, color)
        elif obs == "floor_blue":
            new_markers, mid = _puddle_site_marker(mid, ex, ey, color)
        else:
            # depot / start — simple disc
            disc = _m("site_disc", mid, Marker.CYLINDER); mid += 1
            _pos(disc, ex, ey, 0.02); _sc(disc, 0.50, 0.50, 0.04)
            disc.color = color
            new_markers = [disc]

        for mk in new_markers:
            ma.markers.append(mk)

        badge = _m("site_label", mid, Marker.TEXT_VIEW_FACING); mid += 1
        _pos(badge, ex + 0.30, ey + 0.30, 0.55)
        badge.scale.z = 0.22
        badge.color = ColorRGBA(r=0.1, g=0.1, b=0.1, a=0.95)
        badge.text = _EVENT_BADGE.get(obs, obs)
        ma.markers.append(badge)

    return ma, mid


# ---------------------------------------------------------------------------
# Robot — simple wheeled differential-drive robot
# ---------------------------------------------------------------------------
def build_robot_markers(x, y, heading=0.0, id_offset=0):
    ma = MarkerArray()
    mid = id_offset

    # Main body — rounded box
    body = _m("robot_body", mid, Marker.CUBE); mid += 1
    _pos(body, x, y, 0.18); _sc(body, 0.44, 0.36, 0.22)
    _quat(body, yaw=heading); body.color = ROBOT_BODY
    ma.markers.append(body)

    # Cyan accent stripe on top
    stripe = _m("robot_body", mid, Marker.CUBE); mid += 1
    _pos(stripe, x, y, 0.30); _sc(stripe, 0.44, 0.36, 0.03)
    _quat(stripe, yaw=heading); stripe.color = ROBOT_ACCENT
    ma.markers.append(stripe)

    # Head / sensor dome
    dome = _m("robot_head", mid, Marker.SPHERE); mid += 1
    _pos(dome, x, y, 0.38); _sc(dome, 0.24, 0.24, 0.18)
    dome.color = ColorRGBA(r=0.28, g=0.28, b=0.30, a=1.0)
    ma.markers.append(dome)

    # Red sensor lens on front
    fwd_x = x + 0.15 * math.cos(heading)
    fwd_y = y + 0.15 * math.sin(heading)
    sensor = _m("robot_sensor", mid, Marker.CYLINDER); mid += 1
    _pos(sensor, fwd_x, fwd_y, 0.25); _sc(sensor, 0.08, 0.08, 0.06)
    _quat(sensor, pitch=math.pi / 2, yaw=heading); sensor.color = ROBOT_SENSOR
    ma.markers.append(sensor)

    # Left and right wheels
    right = math.pi / 2
    c, s = math.cos(heading), math.sin(heading)
    for side in [+1, -1]:
        wx = x + side * 0.20 * math.cos(heading + right)
        wy = y + side * 0.20 * math.sin(heading + right)
        wheel = _m("robot_wheel_l" if side > 0 else "robot_wheel_r",
                   mid, Marker.CYLINDER); mid += 1
        _pos(wheel, wx, wy, 0.07); _sc(wheel, 0.12, 0.12, 0.08)
        _quat(wheel, pitch=math.pi / 2, yaw=heading); wheel.color = ROBOT_WHEEL
        ma.markers.append(wheel)

    return ma, mid


# ---------------------------------------------------------------------------
# Effects: puddle ripple, charge sparks
# ---------------------------------------------------------------------------
def _puddle_ripple(mid, ex, ey, t):
    markers = []
    for i in range(3):
        phase = (t * 0.5 + i * 0.33) % 1.0
        r = 0.10 + phase * 0.50
        alpha = max(0.0, 0.55 * (1.0 - phase))
        ring = _m("puddle_ripple", mid, Marker.CYLINDER); mid += 1
        _pos(ring, ex, ey, 0.03)
        _sc(ring, r * 2, r * 2, 0.01)
        ring.color = ColorRGBA(r=0.18, g=0.55, b=0.92, a=alpha)
        markers.append(ring)
    return markers, mid


def _charge_sparks(mid, ex, ey, t, visible=False):
    markers = []
    sparks = [
        ( 0.00,  0.00, 0.65, 0.06),
        ( 0.08, -0.06, 0.75, 0.05),
        (-0.07,  0.08, 0.80, 0.04),
        ( 0.05,  0.10, 0.70, 0.05),
    ]
    for i, (dx, dy, dz, sc) in enumerate(sparks):
        phase = (t * 1.5 + i * 0.25) % 1.0
        alpha = max(0.0, 0.90 * (1.0 - phase)) if visible else 0.0
        spark = _m("charge_spark", mid, Marker.SPHERE); mid += 1
        _pos(spark, ex + dx + 0.04 * math.sin(t * 3 + i),
             ey + dy + 0.04 * math.cos(t * 2.5 + i),
             dz + phase * 0.10)
        _sc(spark, sc, sc, sc * 0.6)
        spark.color = ColorRGBA(r=1.0, g=0.90, b=0.20, a=alpha)
        markers.append(spark)
    return markers, mid


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------
class TourPublisher(Node):
    ROBOT_SPEED = 1.6  # cells per second

    @staticmethod
    def _cr(p0, p1, p2, p3, t):
        t2, t3 = t * t, t * t * t
        return (0.5 * (-t3 + 2 * t2 - t) * p0 +
                0.5 * (3 * t3 - 5 * t2 + 2) * p1 +
                0.5 * (-3 * t3 + 4 * t2 + t) * p2 +
                0.5 * (t3 - t2) * p3)

    @classmethod
    def _build_spline(cls, pts, sps=24):
        n = len(pts)
        if n < 2:
            return list(pts)
        out = []
        for i in range(n - 1):
            p0 = pts[max(i - 1, 0)]
            p1 = pts[i]
            p2 = pts[i + 1]
            p3 = pts[min(i + 2, n - 1)]
            for s in range(sps):
                t = s / sps
                out.append((cls._cr(p0[0], p1[0], p2[0], p3[0], t),
                             cls._cr(p0[1], p1[1], p2[1], p3[1], t)))
        out.append(pts[-1])
        return out

    def __init__(self, tour_data, step_delay):
        super().__init__("office_tour")

        self.pub_env     = self.create_publisher(MarkerArray, "/office/environment", 10)
        self.pub_path    = self.create_publisher(Path,        "/office/path",        10)
        self.pub_events  = self.create_publisher(MarkerArray, "/office/events",      10)
        self.pub_robot   = self.create_publisher(MarkerArray, "/office/robot",       10)
        self.pub_effects = self.create_publisher(MarkerArray, "/office/effects",     10)

        raw = tour_data["path"]
        raw_wps = [(float(x) + 0.5, float(y) + 0.5) for x, y in raw]
        deduped = [raw_wps[0]]
        for wp in raw_wps[1:]:
            if wp != deduped[-1]:
                deduped.append(wp)
        self._waypoints = deduped

        event_pos = {(ev["x"] + 0.5, ev["y"] + 0.5) for ev in tour_data["event_nodes"]}

        jittered = []
        for i, (wx, wy) in enumerate(self._waypoints):
            if (wx, wy) in event_pos or i == 0 or i == len(self._waypoints) - 1:
                jittered.append((wx, wy))
            else:
                ox = 0.08 * math.sin(wx * 2.3 + wy * 1.7 + 0.5)
                oy = 0.08 * math.cos(wx * 1.9 + wy * 2.1 + 1.2)
                jittered.append((wx + ox, wy + oy))

        self._curve = self._build_spline(jittered, sps=16)

        self._arc = [0.0]
        for i in range(1, len(self._curve)):
            ax, ay = self._curve[i - 1]
            bx, by = self._curve[i]
            self._arc.append(self._arc[-1] + math.hypot(bx - ax, by - ay))
        self._total_arc = self._arc[-1]

        self._wp_arc_s = []
        for wx, wy in self._waypoints:
            best_s, best_d = 0.0, float("inf")
            for i, (cx, cy) in enumerate(self._curve):
                d = math.hypot(cx - wx, cy - wy)
                if d < best_d:
                    best_d, best_s = d, self._arc[i]
            self._wp_arc_s.append(best_s)
        self._wp_trigger_order = sorted(range(len(self._waypoints)),
                                        key=lambda i: self._wp_arc_s[i])
        self._next_trigger_idx = 0

        self._event_nodes = tour_data["event_nodes"]
        self._puddle      = [tuple(p) for p in tour_data.get("puddle_positions", [])]
        self._t           = 0.0

        self._s    = 0.0
        self._cx   = self._curve[0][0]
        self._cy   = self._curve[0][1]
        self._heading = 0.0
        self._last_animate_time = None

        self._pos_to_obs = {
            (int(ev["x"]), int(ev["y"])): ev["obs"]
            for ev in self._event_nodes
        }
        self._obs_to_pos = {
            ev["obs"]: (ev["x"] + 0.5, ev["y"] + 0.5)
            for ev in self._event_nodes
        }

        self._visited = set()

        env_ma, env_id_end = build_environment(self._puddle)
        decor_ma, decor_id_end = build_decorations(env_id_end)
        env_ma.markers.extend(decor_ma.markers)
        event_ma, _ = build_event_markers(self._event_nodes, decor_id_end)
        self._env_msg    = env_ma
        self._events_msg = event_ma
        self._path_msg   = self._build_path_msg()

        self.create_timer(1.0,  self._publish_static)
        self.create_timer(0.05, self._animate)
        self.create_timer(0.08, self._publish_effects)

        self.get_logger().info(
            f"Office tour ready ({len(self._curve)}-pt spline). Close RViz2 to stop.")

    def _build_path_msg(self):
        msg = Path()
        msg.header.frame_id = "map"
        for cx, cy in self._curve:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = cx
            ps.pose.position.y = cy
            ps.pose.position.z = 0.10
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    def _publish_static(self):
        now = self.get_clock().now().to_msg()
        self._env_msg.markers[0].header.stamp    = now
        self._events_msg.markers[0].header.stamp = now
        self._path_msg.header.stamp              = now
        self.pub_env.publish(self._env_msg)
        self.pub_path.publish(self._path_msg)
        self.pub_events.publish(self._events_msg)

    def _fire_event(self, wp_idx):
        wx, wy = self._waypoints[wp_idx]
        obs = self._pos_to_obs.get((int(wx - 0.5), int(wy - 0.5)))
        if obs and obs not in self._visited:
            self._visited.add(obs)

    def _arc_to_curve_pos(self, s):
        lo, hi = 0, len(self._arc) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if self._arc[mid] <= s:
                lo = mid
            else:
                hi = mid
        seg_len = self._arc[hi] - self._arc[lo]
        f = (s - self._arc[lo]) / seg_len if seg_len > 1e-9 else 0.0
        ax, ay = self._curve[lo]
        bx, by = self._curve[hi]
        x = ax + f * (bx - ax)
        y = ay + f * (by - ay)
        hdg = math.atan2(by - ay, bx - ax) if seg_len > 1e-9 else self._heading
        return x, y, hdg

    def _animate(self):
        now_time = time.monotonic()
        if self._last_animate_time is None:
            self._last_animate_time = now_time
            if self._wp_trigger_order:
                first = self._wp_trigger_order[0]
                if self._wp_arc_s[first] <= 0.0:
                    self._fire_event(first)
                    self._next_trigger_idx = 1
            return
        dt = min(now_time - self._last_animate_time, 0.2)
        self._last_animate_time = now_time

        self._s += self.ROBOT_SPEED * dt

        if self._s >= self._total_arc:
            self._s = 0.0
            self._next_trigger_idx = 0
            self._visited.clear()

        order = self._wp_trigger_order
        while (self._next_trigger_idx < len(order) and
               self._wp_arc_s[order[self._next_trigger_idx]] <= self._s):
            self._fire_event(order[self._next_trigger_idx])
            self._next_trigger_idx += 1

        self._cx, self._cy, self._heading = self._arc_to_curve_pos(self._s)

        now = self.get_clock().now().to_msg()
        robot_ma, _ = build_robot_markers(self._cx, self._cy, self._heading, id_offset=8000)
        for mk in robot_ma.markers:
            mk.header.stamp = now
        self.pub_robot.publish(robot_ma)

    def _publish_effects(self):
        self._t += 0.08
        t = self._t
        ma = MarkerArray()
        mid = 0
        now = self.get_clock().now().to_msg()

        # Puddle ripple — always visible at puddle event position
        if "floor_blue" in self._obs_to_pos:
            ex, ey = self._obs_to_pos["floor_blue"]
            new_markers, mid = _puddle_ripple(mid, ex, ey, t)
            for mk in new_markers:
                mk.header.stamp = now
                ma.markers.append(mk)

        # Charge sparks — only visible after robot arrives at charger
        if "floor_yellow" in self._obs_to_pos:
            ex, ey = self._obs_to_pos["floor_yellow"]
            new_markers, mid = _charge_sparks(mid, ex, ey, t,
                                              visible="floor_yellow" in self._visited)
            for mk in new_markers:
                mk.header.stamp = now
                ma.markers.append(mk)

        if ma.markers:
            self.pub_effects.publish(ma)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    json_path   = sys.argv[1] if len(sys.argv) > 1 else "visualization/tour_data.json"
    rviz_config = sys.argv[2] if len(sys.argv) > 2 else "visualization/office.rviz"
    step_delay  = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

    with open(json_path) as f:
        tour_data = json.load(f)

    rclpy.init()
    node = TourPublisher(tour_data, step_delay)

    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    time.sleep(1.5)

    rviz2_bin = os.path.join(os.path.dirname(sys.executable), "rviz2")
    rviz_proc = subprocess.Popen([rviz2_bin, "-d", rviz_config])
    print(f"[RViz2] Launched (pid {rviz_proc.pid}). Close RViz2 window or Ctrl-C to stop.")

    try:
        rviz_proc.wait()
    except KeyboardInterrupt:
        rviz_proc.terminate()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
