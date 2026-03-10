"""
ROS2 publisher + RViz2 launcher — Mars Rover scenario.

Publishes:
  /mars_rover/terrain   MarkerArray — terrain, rocks, mountains, walls
  /mars_rover/path      nav_msgs/Path
  /mars_rover/events    MarkerArray — mission site markers
  /mars_rover/rover     MarkerArray — detailed rover model
  /mars_rover/effects   MarkerArray — dynamic: smoke, drill, gas cloud

Simulation logic:
  - e4 (atmospheric): persistent gas cloud always visible
  - e2 (soil heating): drilling animation when rover arrives,
    smoke above e2+e3 appears after rover leaves e2
  - e6 / e7: flat colored tiles (outcrop region markers)
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
# Palette
# ---------------------------------------------------------------------------
MARS_GROUND   = ColorRGBA(r=0.65, g=0.34, b=0.18, a=1.0)
MARS_DARK     = ColorRGBA(r=0.48, g=0.24, b=0.11, a=1.0)
ROCK_COLOR    = ColorRGBA(r=0.52, g=0.32, b=0.20, a=1.0)
MOUNTAIN_COL  = ColorRGBA(r=0.58, g=0.30, b=0.16, a=1.0)
LAVA_COLOR    = ColorRGBA(r=0.72, g=0.28, b=0.05, a=1.0)  # dark burnt-orange, opaque
WALL_COLOR    = ColorRGBA(r=0.32, g=0.19, b=0.10, a=1.0)

ROVER_CHASSIS = ColorRGBA(r=0.88, g=0.86, b=0.80, a=1.0)
ROVER_GOLD    = ColorRGBA(r=0.90, g=0.72, b=0.10, a=1.0)
ROVER_PANEL   = ColorRGBA(r=0.08, g=0.28, b=0.72, a=1.0)
ROVER_DARK    = ColorRGBA(r=0.18, g=0.18, b=0.18, a=1.0)
ROVER_RTG     = ColorRGBA(r=0.50, g=0.50, b=0.52, a=1.0)
ROVER_LENS    = ColorRGBA(r=0.10, g=0.60, b=0.90, a=1.0)

_EVENT_COLORS = {
    "initial_state0": ColorRGBA(r=0.75, g=0.75, b=0.75, a=1.0),
    "floor_green":    ColorRGBA(r=0.10, g=0.88, b=0.25, a=1.0),
    "floor_red":      ColorRGBA(r=0.92, g=0.18, b=0.10, a=1.0),
    "floor_purple":   ColorRGBA(r=0.60, g=0.05, b=0.88, a=1.0),
    "floor_blue":     ColorRGBA(r=0.05, g=0.38, b=0.92, a=1.0),
    "floor_yellow":   ColorRGBA(r=1.00, g=0.78, b=0.05, a=1.0),
    "floor_grey":     ColorRGBA(r=0.80, g=0.45, b=0.10, a=1.0),   # e6 — burnt orange tile
    "goal_green":     ColorRGBA(r=0.20, g=0.70, b=0.90, a=1.0),   # e7 — teal sample tile
}

_EVENT_BADGE = {
    "initial_state0": "S",
    "floor_green":    "e1",
    "floor_red":      "e2",
    "floor_purple":   "e3",
    "floor_blue":     "e4",
    "floor_yellow":   "e5",
    "floor_grey":     "e6",
    "goal_green":     "e7",
}

GRID_W, GRID_H = 16, 7

# ---------------------------------------------------------------------------
# Terrain heightmap — deterministic smooth noise via summed harmonics
# ---------------------------------------------------------------------------
def _height(x, y):
    """Pseudo-random smooth height in [0, 1] at world position (x, y)."""
    h  = 0.50 * math.sin(x * 0.55 + 0.3) * math.cos(y * 0.70 + 0.7)
    h += 0.25 * math.sin(x * 1.10 + 1.1) * math.cos(y * 1.40 + 0.2)
    h += 0.15 * math.sin(x * 2.30 + 2.3) * math.cos(y * 2.10 + 1.5)
    h += 0.10 * math.sin(x * 4.50 + 0.9) * math.cos(y * 3.80 + 0.4)
    return (h + 1.0) * 0.5   # remap to [0, 1]

def _terrain_color(h):
    """Mars surface colour: warm ochre in valleys, sandy orange on ridges."""
    # r = 0.62 + 0.24 * h
    # g = 0.32 + 0.18 * h
    # b = 0.12 + 0.08 * h
    r = 0.85 + 0.1 * h
    g = 0.5 + 0.1 * h
    b = 0.1 + 0.1 * h
    return ColorRGBA(r=min(r,1.0), g=min(g,1.0), b=min(b,1.0), a=1.0)

# Two mountains: (cx, cy, base_radius, peak_height)
MOUNTAINS = [
    (2.5, 1.5, 1.0, 1.8),   # left — taller, narrower
    # (6.0, 5.5, 1.2, 1.3),   # upper-centre — wider, shorter
    (7.1, 4.0, 1.0, 1.7),
    (14.0,6.0, 0.8, 1.3),
    (10.2,4.3, 0.8, 1.3),
    (11.0,2.2, 0.8, 1.3)
]

def _rocks():
    """Return list of (wx, wy, size_class) for scattered surface rocks.
    size_class: 0=pebble, 1=rock, 2=boulder"""
    out = []
    # Dense scatter — multiple candidates per cell at sub-cell offsets
    seeds = [
        (7, 13, 0.17, 0.29,  9, 0),   # pebble field
        (5, 11, 0.72, 0.18, 11, 0),
        (3, 17, 0.44, 0.61,  7, 0),
        (11, 7, 0.83, 0.47,  8, 0),
        (13, 5, 0.31, 0.78, 13, 0),
        (17, 3, 0.58, 0.12,  6, 0),
        (7, 19, 0.91, 0.55, 10, 0),
        (23, 7, 0.25, 0.88,  9, 0),
        (9, 23, 0.66, 0.33,  7, 0),
        # # medium rocks — sparser
        # (7, 13, 0.52, 0.74,  5, 1),
        (11,17, 0.38, 0.21,  4, 1),
        # (13, 7, 0.79, 0.63,  6, 1),
        # (17,11, 0.14, 0.47,  5, 1),
        (19, 5, 0.87, 0.32,  4, 1),
        # # large boulders — rare
        # (7, 13, 0.43, 0.81,  3, 2),
        # (11,17, 0.76, 0.24,  2, 2),
        # (19,23, 0.21, 0.67,  2, 2),
    ]
    for ax, ay, dx, dy, mod, sc in seeds:
        for x in range(1, 15):
            for y in range(1, 6):
                if (x * ax + y * ay) % mod == 0:
                    out.append((x + dx, y + dy, sc))
    return out


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
# Terrain
# ---------------------------------------------------------------------------
def build_terrain(lava_positions):
    ma = MarkerArray()
    mid = 0

    lava_set = {(int(lx), int(ly)) for lx, ly in lava_positions}
    TILE  = 0.4    # patch size inside the play area
    BRIM  = 4      # how many tiles beyond the grid to extend (valley walls)
    INNER_W = GRID_W
    INNER_H = GRID_H

    def _valley_edge_height(wx, wy):
        """Extra height added near / outside the boundary — creates valley walls."""
        # Distance from the nearest interior edge (negative = inside)
        dx = max(0.0 - wx, wx - INNER_W, 0.0)
        dy = max(0.0 - wy, wy - INNER_H, 0.0)
        dist = math.hypot(dx, dy)
        # Quadratic ramp: 0 at boundary, reaches ~3.0 at BRIM tiles out
        return (dist / BRIM) ** 1.6 * 2.0 if dist > 0 else 0.0

    # ---- Ground + valley walls -------------------------------------------
    # Extend BRIM tiles in every direction beyond the grid
    x_min = int(-BRIM / TILE)
    x_max = int((INNER_W + BRIM) / TILE)
    y_min = int(-BRIM / TILE)
    y_max = int((INNER_H + BRIM) / TILE)

    for ix in range(x_min, x_max):
        for iy in range(y_min, y_max):
            wx = (ix + 0.5) * TILE
            wy = (iy + 0.5) * TILE
            gx, gy = int(wx), int(wy)

            inside = (0 <= wx <= INNER_W) and (0 <= wy <= INNER_H)

            if inside and any(math.hypot(wx - mcx, wy - mcy) < mrad * 0.7
                              for mcx, mcy, mrad, _ in MOUNTAINS):
                continue

            h = _height(wx, wy)
            wall_h = _valley_edge_height(wx, wy)
            tile_z = h * 0.38 - 0.11 + wall_h

            if inside and (gx, gy) in lava_set:
                # Dark burnt-orange for outcrop/lava region — no separate tile
                col = ColorRGBA(r=0.5, g=0.5, b=0.5, a=1.0)
            else:
                wall_blend = min(wall_h / 2.5, 1.0)
                base_col = _terrain_color(h)
                col = ColorRGBA(
                    r=base_col.r * (1.0 - wall_blend * 0.18),
                    g=base_col.g * (1.0 - wall_blend * 0.25),
                    b=base_col.b * (1.0 - wall_blend * 0.12),
                    a=1.0,
                )

            mk = _m("ground", mid, Marker.CUBE); mid += 1
            _pos(mk, wx, wy, tile_z * 0.5)
            _sc(mk, TILE * 1.01, TILE * 1.01, max(tile_z, 0.1))
            mk.color = col
            ma.markers.append(mk)

    # Large outcrop rocks on lava/outcrop tiles — anchored to heightmap
    OUTCROP_ROCK_COLOR = ColorRGBA(r=0.38, g=0.22, b=0.12, a=1.0)
    for i, (lx, ly) in enumerate(lava_positions):
        offsets = [
            (0.20 + (i*0.13)%0.35, 0.18 + (i*0.17)%0.30, 0.24 + (i*0.07)%0.20),
            (-0.22 + (i*0.11)%0.20, -0.15 + (i*0.19)%0.25, 0.20 + (i*0.09)%0.16),
        ]
        if i % 3 == 0:
            offsets.append((0.05, -0.28, 0.17))
        for dx, dy, sz in offsets:
            rx, ry = lx+0.5+dx, ly+0.5+dy
            h = _height(rx, ry)
            ground_top = h * 0.38 - 0.14
            rock = _m("outcrop_rocks", mid, Marker.SPHERE); mid += 1
            _pos(rock, rx, ry, ground_top + sz*0.5)
            _sc(rock, sz, sz*0.85, sz*0.70)
            rock.color = OUTCROP_ROCK_COLOR
            ma.markers.append(rock)

    # ---- Mountains — 4 layers each, clearly visible peak ----------------
    MCOL_BASE = ColorRGBA(r=0.42, g=0.22, b=0.10, a=1.0)
    MCOL_MID  = ColorRGBA(r=0.55, g=0.30, b=0.14, a=1.0)
    MCOL_TOP  = ColorRGBA(r=0.70, g=0.42, b=0.20, a=1.0)
    MCOL_CAP  = ColorRGBA(r=0.82, g=0.58, b=0.34, a=1.0)

    for mcx, mcy, mrad, mpeak in MOUNTAINS:
        # 4 layers: base → mid → top → narrow bright cap
        layers = [
            (0.00,        mrad * 1.00, mpeak * 0.28, MCOL_BASE),
            (mpeak * 0.28, mrad * 0.65, mpeak * 0.28, MCOL_MID),
            (mpeak * 0.56, mrad * 0.35, mpeak * 0.26, MCOL_TOP),
            # (mpeak * 0.82, mrad * 0.14, mpeak * 0.24, MCOL_CAP),
        ]
        for z, rad, h, col in layers:
            mk = _m("mountains", mid, Marker.SPHERE); mid += 1
            _pos(mk, mcx, mcy, z)
            _sc(mk, rad * 2.0, rad * 2.0, h)
            mk.color = col
            ma.markers.append(mk)

        # Skirt boulders around the base
        for i in range(8):
            ang = i * (2 * math.pi / 8)
            rr = mrad * (0.75 + ((i * 7) % 3) * 0.06)
            rx = mcx + rr * math.cos(ang)
            ry = mcy + rr * math.sin(ang)
            sz = 0.14 + ((i * 11) % 5) * 0.04
            rock = _m("mountains", mid, Marker.SPHERE); mid += 1
            _pos(rock, rx, ry, sz * 0.4)
            _sc(rock, sz, sz * 0.9, sz * 0.6)
            rock.color = MCOL_BASE
            ma.markers.append(rock)

    # ---- Scattered surface rocks -----------------------------------------
    # Event cells to keep clear (1-cell radius)
    EVENT_CELLS = {
        (4,5),(5,3),(8,1),(9,1),(8,5),(12,4),(13,4),(12,3),
    }
    SIZE_PARAMS = [
        # (base_w, base_d, base_h, w_jitter, h_jitter)  — pebble, rock, boulder
        (0.08, 0.07, 0.05, 0.06, 0.04),
        (0.18, 0.14, 0.10, 0.10, 0.07),
        (0.38, 0.30, 0.22, 0.14, 0.10),
    ]
    ROCK_COLORS = [
        ColorRGBA(r=0.50, g=0.30, b=0.18, a=1.0),  # warm rust
        ColorRGBA(r=0.42, g=0.24, b=0.13, a=1.0),  # dark brown
        ColorRGBA(r=0.60, g=0.38, b=0.22, a=1.0),  # ochre
        ColorRGBA(r=0.35, g=0.20, b=0.12, a=1.0),  # very dark
    ]
    for i, (rx, ry, sc) in enumerate(_rocks()):
        gx, gy = int(rx), int(ry)
        # Skip event cells and their immediate neighbours
        if any(abs(gx-ex) <= 0 and abs(gy-ey) <= 0 for ex, ey in EVENT_CELLS):
            continue
        if any(math.hypot(rx - mcx, ry - mcy) < mrad * 0.80
               for mcx, mcy, mrad, _ in MOUNTAINS):
            continue
        if (gx, gy) in lava_set:
            continue
        bw, bd, bh, wj, hj = SIZE_PARAMS[sc]
        w  = bw + ((i * 37) % 7) * wj / 6
        d  = bd + ((i * 53) % 5) * wj / 5
        rh = bh + ((i * 71) % 9) * hj / 8
        h  = _height(rx, ry)
        ground_z = h * 0.18 - 0.04
        col = ROCK_COLORS[(i * 3 + sc) % len(ROCK_COLORS)]
        # Slightly tilt boulders for realism
        yaw = ((i * 29) % 12) * math.pi / 6 if sc == 2 else 0.0
        mk = _m("rocks", mid, Marker.SPHERE); mid += 1
        _pos(mk, rx, ry, ground_z + rh * 0.5)
        _sc(mk, w, d, rh)
        _quat(mk, yaw=yaw)
        mk.color = col
        ma.markers.append(mk)

    return ma, mid


# ---------------------------------------------------------------------------
# Event markers
# ---------------------------------------------------------------------------
def _lander_marker(mid, ex, ey, color):
    markers = []
    pad = _m("site_pad", mid, Marker.CYLINDER); mid += 1
    _pos(pad, ex, ey, 0.04); _sc(pad, 0.70, 0.70, 0.06)
    pad.color = ColorRGBA(r=color.r*0.9, g=color.g*0.9, b=color.b*0.9, a=1.0)
    markers.append(pad)
    for angle in [0, 2.094, 4.189]:
        leg = _m("site_legs", mid, Marker.CUBE); mid += 1
        lx = ex + 0.38*math.cos(angle)
        ly = ey + 0.38*math.sin(angle)
        _pos(leg, lx, ly, 0.06); _sc(leg, 0.38, 0.06, 0.06)
        _quat(leg, yaw=angle)
        leg.color = ColorRGBA(r=0.75, g=0.75, b=0.75, a=1.0)
        markers.append(leg)
    cone = _m("site_cone", mid, Marker.ARROW); mid += 1
    cone.points = [Point(x=ex, y=ey, z=0.10), Point(x=ex, y=ey, z=0.55)]
    cone.scale.x = 0.10; cone.scale.y = 0.18; cone.scale.z = 0.14
    cone.color = color
    markers.append(cone)
    light = _m("site_light", mid, Marker.SPHERE); mid += 1
    _pos(light, ex, ey, 0.62); _sc(light, 0.14)
    light.color = ColorRGBA(r=1.0, g=0.95, b=0.6, a=1.0)
    markers.append(light)
    return markers, mid


def _science_flag(mid, ex, ey, color):
    markers = []
    disc = _m("site_disc", mid, Marker.CYLINDER); mid += 1
    _pos(disc, ex, ey, 0.03); _sc(disc, 0.55, 0.55, 0.05)
    disc.color = ColorRGBA(r=color.r*0.6, g=color.g*0.6, b=color.b*0.6, a=0.85)
    markers.append(disc)
    pole = _m("site_pole", mid, Marker.CYLINDER); mid += 1
    _pos(pole, ex-0.10, ey, 0.35); _sc(pole, 0.05, 0.05, 0.62)
    pole.color = ColorRGBA(r=0.80, g=0.80, b=0.80, a=1.0)
    markers.append(pole)
    flag = _m("site_flag", mid, Marker.CUBE); mid += 1
    _pos(flag, ex+0.05, ey, 0.60); _sc(flag, 0.28, 0.04, 0.18)
    flag.color = color
    markers.append(flag)
    return markers, mid


def _colored_tile(mid, ex, ey, color, is_sample=False):
    """e6/e7: flat glowing floor tile.
    is_sample=True (e7) adds a large prominent rock on top."""
    markers = []
    tile = _m("site_tile", mid, Marker.CUBE); mid += 1
    _pos(tile, ex, ey, 0.03); _sc(tile, 0.90, 0.90, 0.07)
    tile.color = color
    markers.append(tile)
    # Border ring
    ring = _m("site_ring", mid, Marker.CYLINDER); mid += 1
    _pos(ring, ex, ey, 0.07); _sc(ring, 0.95, 0.95, 0.02)
    ring.color = ColorRGBA(r=min(1.0,color.r*1.3), g=min(1.0,color.g*1.3),
                           b=min(1.0,color.b*1.3), a=0.7)
    markers.append(ring)
    if is_sample:
        # Large prominent sample rock sitting on the tile
        rock = _m("sample_rock", mid, Marker.SPHERE); mid += 1
        _pos(rock, ex, ey, 0.28); _sc(rock, 0.42, 0.36, 0.32)
        rock.color = ColorRGBA(r=0.40, g=0.26, b=0.16, a=1.0)
        markers.append(rock)
        # Smaller accent rock beside it
        rock2 = _m("sample_rock", mid, Marker.SPHERE); mid += 1
        _pos(rock2, ex+0.18, ey-0.12, 0.16); _sc(rock2, 0.22, 0.18, 0.18)
        rock2.color = ColorRGBA(r=0.48, g=0.30, b=0.18, a=1.0)
        markers.append(rock2)
    return markers, mid


def build_event_markers(event_nodes, id_offset):
    ma = MarkerArray()
    mid = id_offset
    e7_rock_ids = []   # marker IDs for the sample rocks at e7

    for ev in event_nodes:
        color = _EVENT_COLORS.get(ev["obs"], ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0))
        ex, ey = ev["x"] + 0.5, ev["y"] + 0.5
        obs = ev["obs"]

        if obs == "floor_yellow":
            new_markers, mid = _lander_marker(mid, ex, ey, color)
        elif obs in ("floor_grey", "goal_green"):
            new_markers, mid = _colored_tile(mid, ex, ey, color, is_sample=(obs == "goal_green"))
            if obs == "goal_green":
                # Record the IDs of the sample_rock markers so we can delete them on collection
                for mk in new_markers:
                    if mk.ns == "sample_rock":
                        e7_rock_ids.append(mk.id)
        else:
            new_markers, mid = _science_flag(mid, ex, ey, color)

        for mk in new_markers:
            ma.markers.append(mk)

        badge = _m("site_label", mid, Marker.TEXT_VIEW_FACING); mid += 1
        _pos(badge, ex + 0.30, ey + 0.30, 0.55)
        badge.scale.z = 0.22
        badge.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.95)
        badge.text = _EVENT_BADGE.get(obs, obs)
        ma.markers.append(badge)

    return ma, mid, e7_rock_ids


# ---------------------------------------------------------------------------
# Rover
# ---------------------------------------------------------------------------
def build_rover_markers(x, y, heading=0.0, id_offset=0):
    ma = MarkerArray()
    mid = id_offset

    body = _m("rover_body", mid, Marker.CUBE); mid += 1
    _pos(body, x, y, 0.22); _sc(body, 0.52, 0.36, 0.20)
    _quat(body, yaw=heading); body.color = ROVER_CHASSIS
    ma.markers.append(body)

    blanket = _m("rover_blanket", mid, Marker.CUBE); mid += 1
    _pos(blanket, x, y, 0.33); _sc(blanket, 0.48, 0.32, 0.04)
    _quat(blanket, yaw=heading); blanket.color = ROVER_GOLD
    ma.markers.append(blanket)

    fwd_x = x + 0.08*math.cos(heading)
    fwd_y = y + 0.08*math.sin(heading)
    panel = _m("rover_panel", mid, Marker.CUBE); mid += 1
    _pos(panel, fwd_x, fwd_y, 0.40); _sc(panel, 0.60, 0.42, 0.04)
    _quat(panel, pitch=0.18, yaw=heading); panel.color = ROVER_PANEL
    ma.markers.append(panel)

    rtg_x = x - 0.22*math.cos(heading)
    rtg_y = y - 0.22*math.sin(heading)
    rtg = _m("rover_rtg", mid, Marker.CYLINDER); mid += 1
    _pos(rtg, rtg_x, rtg_y, 0.22); _sc(rtg, 0.13, 0.13, 0.22)
    rtg.color = ROVER_RTG
    ma.markers.append(rtg)

    mast_x = x + 0.15*math.cos(heading)
    mast_y = y + 0.15*math.sin(heading)
    mast = _m("rover_mast", mid, Marker.CYLINDER); mid += 1
    _pos(mast, mast_x, mast_y, 0.54); _sc(mast, 0.05, 0.05, 0.32)
    mast.color = ROVER_DARK
    ma.markers.append(mast)

    head = _m("rover_head", mid, Marker.CUBE); mid += 1
    _pos(head, mast_x, mast_y, 0.73); _sc(head, 0.14, 0.14, 0.10)
    head.color = ROVER_CHASSIS
    ma.markers.append(head)

    right = math.pi/2
    for side in [+1, -1]:
        lens_x = mast_x + side * 0.07 * math.cos(heading + right)
        lens_y = mast_y + side * 0.07 * math.sin(heading + right)
        lens = _m("rover_lens", mid, Marker.CYLINDER); mid += 1
        _pos(lens, lens_x, lens_y, 0.73); _sc(lens, 0.05, 0.05, 0.06)
        _quat(lens, pitch=math.pi/2, yaw=heading); lens.color = ROVER_LENS
        ma.markers.append(lens)

    c, s = math.cos(heading), math.sin(heading)
    for fw, sw in [(+0.20,+0.22),(+0.20,-0.22),(0.00,+0.24),(0.00,-0.24),(-0.20,+0.22),(-0.20,-0.22)]:
        wx = x + fw*c - sw*s
        wy = y + fw*s + sw*c
        wheel = _m("rover_wheels", mid, Marker.CYLINDER); mid += 1
        _pos(wheel, wx, wy, 0.09); _sc(wheel, 0.13, 0.13, 0.09)
        _quat(wheel, pitch=math.pi/2, yaw=heading); wheel.color = ROVER_DARK
        ma.markers.append(wheel)
        hub = _m("rover_hubs", mid, Marker.CYLINDER); mid += 1
        _pos(hub, wx, wy, 0.09); _sc(hub, 0.07, 0.07, 0.10)
        _quat(hub, pitch=math.pi/2, yaw=heading); hub.color = ROVER_CHASSIS
        ma.markers.append(hub)

    for side in [+1, -1]:
        arm_x = x + side * 0.23 * math.cos(heading + right) * 0.5
        arm_y = y + side * 0.23 * math.sin(heading + right) * 0.5
        arm = _m("rover_rocker", mid, Marker.CUBE); mid += 1
        _pos(arm, arm_x, arm_y, 0.16); _sc(arm, 0.38, 0.04, 0.05)
        _quat(arm, yaw=heading); arm.color = ROVER_DARK
        ma.markers.append(arm)

    return ma, mid


# ---------------------------------------------------------------------------
# Effects: smoke cloud, drill, gas cloud
# ---------------------------------------------------------------------------
def _smoke_puff(mid, ex, ey, t, ns="smoke", base_z=0.4, color=None, visible=True):
    """Animated smoke: several spheres at varying z/offset driven by time t.
    When visible=False all alphas are zero (markers still published to clear stale ones)."""
    markers = []
    if color is None:
        color = ColorRGBA(r=0.65, g=0.60, b=0.55, a=0.0)
    puffs = [
        (0.00,  0.00, 0.00, 0.38),
        (0.12, -0.08, 0.22, 0.30),
        (-0.10, 0.10, 0.40, 0.25),
        (0.05,  0.15, 0.58, 0.20),
        (-0.05, -0.12, 0.74, 0.16),
    ]
    for i, (dx, dy, dz, sc) in enumerate(puffs):
        phase = (t * 0.6 + i * 0.4) % 1.0
        alpha = max(0.0, 0.70 * (1.0 - phase)) if visible else 0.0
        rise  = dz + phase * 0.15
        wobble_x = dx + 0.04 * math.sin(t * 1.2 + i)
        wobble_y = dy + 0.04 * math.cos(t * 1.1 + i)
        mk = _m(ns, mid, Marker.SPHERE); mid += 1
        _pos(mk, ex + wobble_x, ey + wobble_y, base_z + rise)
        size = sc * (0.8 + 0.2 * phase)
        _sc(mk, size, size, size * 0.7)
        mk.color = ColorRGBA(r=color.r, g=color.g, b=color.b, a=alpha)
        markers.append(mk)
    return markers, mid


def _drill_marker(mid, ex, ey, t):
    """Spinning drill bit: a narrow cylinder that rotates."""
    markers = []
    angle = t * 8.0   # spin fast
    drill = _m("drill_bit", mid, Marker.CYLINDER); mid += 1
    _pos(drill, ex, ey, 0.10)
    _sc(drill, 0.06, 0.06, 0.22)
    _quat(drill, pitch=0.2, yaw=angle)
    drill.color = ColorRGBA(r=0.85, g=0.75, b=0.20, a=1.0)  # gold drill
    markers.append(drill)
    # Dust ring at base
    dust = _m("drill_dust", mid, Marker.CYLINDER); mid += 1
    _pos(dust, ex, ey, 0.04)
    _sc(dust, 0.25 + 0.05*math.sin(t*4), 0.25 + 0.05*math.sin(t*4), 0.02)
    dust.color = ColorRGBA(r=0.75, g=0.55, b=0.30, a=0.55)
    markers.append(dust)
    return markers, mid


def _gas_cloud(mid, ex, ey, t):
    """Persistent atmospheric gas cloud at e4 — slow drifting wisps."""
    markers = []
    wisps = [
        ( 0.00,  0.00, 0.50, 0.45),
        ( 0.18,  0.10, 0.72, 0.32),
        (-0.15,  0.08, 0.88, 0.28),
        ( 0.08, -0.18, 0.62, 0.35),
        (-0.08, -0.05, 1.00, 0.22),
    ]
    for i, (dx, dy, dz, sc) in enumerate(wisps):
        drift = 0.06 * math.sin(t * 0.4 + i * 1.3)
        mk = _m("gas_cloud", mid, Marker.SPHERE); mid += 1
        _pos(mk, ex + dx + drift, ey + dy + drift*0.5, dz)
        _sc(mk, sc, sc, sc*0.55)
        alpha = 0.30 + 0.12 * math.sin(t * 0.5 + i)
        mk.color = ColorRGBA(r=0.70, g=0.85, b=0.72, a=alpha)  # faint greenish gas
        markers.append(mk)
    return markers, mid


# ---------------------------------------------------------------------------
# ROS2 Node
# ---------------------------------------------------------------------------
class TourPublisher(Node):
    # World-space speed: cells per second
    ROVER_SPEED = 1.8

    # ---------------------------------------------------------------------------
    # Catmull-Rom spline helpers
    # ---------------------------------------------------------------------------
    @staticmethod
    def _cr(p0, p1, p2, p3, t):
        t2, t3 = t*t, t*t*t
        return (0.5*(        -t3 + 2*t2 -   t     ) * p0 +
                0.5*( 3*t3   - 5*t2          + 2  ) * p1 +
                0.5*(-3*t3   + 4*t2 +   t          ) * p2 +
                0.5*(   t3   -   t2                ) * p3)

    @classmethod
    def _build_spline(cls, pts, sps=24):
        """Sample Catmull-Rom spline; returns dense (x,y) list."""
        n = len(pts)
        if n < 2:
            return list(pts)
        out = []
        for i in range(n - 1):
            p0 = pts[max(i-1, 0)]
            p1 = pts[i]
            p2 = pts[i+1]
            p3 = pts[min(i+2, n-1)]
            for s in range(sps):
                t = s / sps
                out.append((cls._cr(p0[0],p1[0],p2[0],p3[0],t),
                             cls._cr(p0[1],p1[1],p2[1],p3[1],t)))
        out.append(pts[-1])
        return out

    # ---------------------------------------------------------------------------
    def __init__(self, tour_data, step_delay):
        super().__init__("mars_rover_tour")

        self.pub_terrain = self.create_publisher(MarkerArray, "/mars_rover/terrain", 10)
        self.pub_path    = self.create_publisher(Path,        "/mars_rover/path",    10)
        self.pub_events  = self.create_publisher(MarkerArray, "/mars_rover/events",  10)
        self.pub_rover   = self.create_publisher(MarkerArray, "/mars_rover/rover",   10)
        self.pub_effects = self.create_publisher(MarkerArray, "/mars_rover/effects", 10)

        # Convert raw grid coords to world-space cell centres, dedup consecutive
        raw = tour_data["path"]
        raw_wps = [(float(x)+0.5, float(y)+0.5) for x, y in raw]
        deduped = [raw_wps[0]]
        for wp in raw_wps[1:]:
            if wp != deduped[-1]:
                deduped.append(wp)
        self._waypoints = deduped  # original grid waypoints (for event detection)

        # Build set of event positions so we don't jitter them
        event_pos = {(ev["x"]+0.5, ev["y"]+0.5) for ev in tour_data["event_nodes"]}

        # Jitter non-event intermediate waypoints off-axis so the spline
        # doesn't run in straight grid-aligned segments
        jittered = []
        for i, (wx, wy) in enumerate(self._waypoints):
            if (wx, wy) in event_pos or i == 0 or i == len(self._waypoints)-1:
                jittered.append((wx, wy))
            else:
                # deterministic offset based on position, max ±0.35 cells
                ox = 0.35 * math.sin(wx * 2.3 + wy * 1.7 + 0.5)
                oy = 0.35 * math.cos(wx * 1.9 + wy * 2.1 + 1.2)
                jittered.append((wx + ox, wy + oy))

        # Smooth spline curve through jittered waypoints
        self._curve = self._build_spline(jittered, sps=30)

        # Cumulative arc-lengths along the spline curve
        self._arc = [0.0]
        for i in range(1, len(self._curve)):
            ax, ay = self._curve[i-1]
            bx, by = self._curve[i]
            self._arc.append(self._arc[-1] + math.hypot(bx-ax, by-ay))
        self._total_arc = self._arc[-1]

        # For each waypoint, find its closest arc-length on the spline
        self._wp_arc_s = []
        for wx, wy in self._waypoints:
            best_s, best_d = 0.0, float("inf")
            for i, (cx, cy) in enumerate(self._curve):
                d = math.hypot(cx-wx, cy-wy)
                if d < best_d:
                    best_d, best_s = d, self._arc[i]
            self._wp_arc_s.append(best_s)
        # Sort so we can scan forward efficiently
        self._wp_trigger_order = sorted(range(len(self._waypoints)),
                                        key=lambda i: self._wp_arc_s[i])
        self._next_trigger_idx = 0  # index into _wp_trigger_order

        self._event_nodes = tour_data["event_nodes"]
        self._lava        = [tuple(p) for p in tour_data.get("lava_positions", [])]
        self._t           = 0.0

        # Rover state along the spline arc
        self._s    = 0.0
        self._ci   = 0     # current index into self._curve
        self._cx   = self._curve[0][0]
        self._cy   = self._curve[0][1]
        self._heading = 0.0
        self._last_animate_time = None

        # Event lookups
        self._pos_to_obs = {
            (int(ev["x"]), int(ev["y"])): ev["obs"]
            for ev in self._event_nodes
        }
        self._obs_to_pos = {
            ev["obs"]: (ev["x"]+0.5, ev["y"]+0.5)
            for ev in self._event_nodes
        }

        # Dynamic effect state
        self._visited        = set()
        self._drilling       = False
        self._drill_start    = None
        self._e7_collected   = False
        self._e7_collect_t   = None
        self._e7_deleted_ids = []   # IDs of sample_rock markers that have been DELETEd

        # Pre-build static messages
        terrain_ma, terrain_id_end = build_terrain(self._lava)
        event_ma, _, self._e7_rock_ids = build_event_markers(self._event_nodes, terrain_id_end)
        self._terrain_msg = terrain_ma
        self._events_msg  = event_ma
        self._path_msg    = self._build_path_msg()

        self.create_timer(1.0,  self._publish_static)
        self.create_timer(0.05, self._animate)          # 20 Hz
        self.create_timer(0.08, self._publish_effects)  # ~12 fps effects

        self.get_logger().info(
            f"Mars Rover tour ready ({len(self._curve)}-pt spline). Close RViz2 to stop.")

    def _build_path_msg(self):
        """Publish the spline curve as the RViz path."""
        msg = Path()
        msg.header.frame_id = "map"
        for cx, cy in self._curve:
            ps = PoseStamped()
            ps.header.frame_id = "map"
            ps.pose.position.x = cx
            ps.pose.position.y = cy
            ps.pose.position.z = 0.2
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        return msg

    def _publish_static(self):
        now = self.get_clock().now().to_msg()
        self._terrain_msg.markers[0].header.stamp = now
        self._events_msg.markers[0].header.stamp  = now
        self._path_msg.header.stamp               = now
        self.pub_terrain.publish(self._terrain_msg)
        self.pub_path.publish(self._path_msg)
        # If e7 already collected, re-send DELETEs every static tick so the
        # 1 Hz republish never resurrects the rock after the one-shot DELETE
        if self._e7_collected and hasattr(self, '_e7_deleted_ids') and self._e7_deleted_ids:
            combined = MarkerArray()
            combined.markers = list(self._events_msg.markers)
            for rock_id in self._e7_deleted_ids:
                dk = _m("sample_rock", rock_id, Marker.SPHERE)
                dk.action = Marker.DELETE
                dk.header.stamp = now
                combined.markers.append(dk)
            combined.markers[0].header.stamp = now
            self.pub_events.publish(combined)
        else:
            self.pub_events.publish(self._events_msg)

    def _delete_e7_rocks(self):
        if not self._e7_rock_ids:
            return
        rock_id_set = set(self._e7_rock_ids)
        now = self.get_clock().now().to_msg()

        # Strip ADD markers from static message so future static publishes don't re-add them
        self._events_msg.markers = [
            mk for mk in self._events_msg.markers
            if not (mk.ns == "sample_rock" and mk.id in rock_id_set)
        ]

        # Remember which IDs were deleted so _publish_static keeps re-deleting them
        self._e7_deleted_ids = list(self._e7_rock_ids)
        self._e7_rock_ids = []

        # Send DELETE immediately
        combined = MarkerArray()
        combined.markers = list(self._events_msg.markers)
        for rock_id in self._e7_deleted_ids:
            dk = _m("sample_rock", rock_id, Marker.SPHERE)
            dk.action = Marker.DELETE
            dk.header.stamp = now
            combined.markers.append(dk)
        if combined.markers:
            combined.markers[0].header.stamp = now
        self.pub_events.publish(combined)

    def _fire_event(self, wp_idx):
        wx, wy = self._waypoints[wp_idx]
        obs = self._pos_to_obs.get((int(wx-0.5), int(wy-0.5)))
        if obs and obs not in self._visited:
            self._visited.add(obs)
            if obs == "floor_red":
                self._drilling    = True
                self._drill_start = self._t
            elif obs == "floor_purple":
                self._drilling = False
            elif obs == "goal_green":
                self._e7_collected = True
                self._e7_collect_t = self._t
                self._delete_e7_rocks()

    def _arc_to_curve_pos(self, s):
        """Binary-search the precomputed arc table; return (x, y, heading)."""
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
            # Fire event for starting waypoint
            if self._wp_trigger_order:
                first = self._wp_trigger_order[0]
                if self._wp_arc_s[first] <= 0.0:
                    self._fire_event(first)
                    self._next_trigger_idx = 1
            return
        dt = min(now_time - self._last_animate_time, 0.2)  # cap at 200 ms
        self._last_animate_time = now_time

        self._s += self.ROVER_SPEED * dt

        # Loop when reaching end of spline
        if self._s >= self._total_arc:
            self._s = 0.0
            self._next_trigger_idx = 0
            self._visited.clear()
            self._drilling     = False
            self._e7_collected = False
            self._e7_collect_t = None
            # Rebuild event markers so rocks reappear on loop restart
            terrain_id_end = len(self._terrain_msg.markers)
            event_ma, _, self._e7_rock_ids = build_event_markers(
                self._event_nodes, terrain_id_end)
            self._events_msg     = event_ma
            self._e7_deleted_ids = []
            # Publish immediately — don't wait for the 1 Hz static tick
            now = self.get_clock().now().to_msg()
            self._events_msg.markers[0].header.stamp = now
            self.pub_events.publish(self._events_msg)

        # Fire events for any waypoints we've passed
        order = self._wp_trigger_order
        while (self._next_trigger_idx < len(order) and
               self._wp_arc_s[order[self._next_trigger_idx]] <= self._s):
            self._fire_event(order[self._next_trigger_idx])
            self._next_trigger_idx += 1

        self._cx, self._cy, self._heading = self._arc_to_curve_pos(self._s)

        now = self.get_clock().now().to_msg()
        rover_ma, _ = build_rover_markers(self._cx, self._cy, self._heading, id_offset=8000)
        for mk in rover_ma.markers:
            mk.header.stamp = now
        self.pub_rover.publish(rover_ma)

    def _publish_effects(self):
        self._t += 0.08
        t = self._t
        ma = MarkerArray()
        mid = 0
        now = self.get_clock().now().to_msg()

        # e4 — persistent atmospheric gas cloud
        if "floor_blue" in self._obs_to_pos:
            ex, ey = self._obs_to_pos["floor_blue"]
            new_markers, mid = _gas_cloud(mid, ex, ey, t)
            for mk in new_markers:
                mk.header.stamp = now
                ma.markers.append(mk)

        # e2 — drill animation while rover is drilling
        if self._drilling and "floor_red" in self._obs_to_pos:
            ex, ey = self._obs_to_pos["floor_red"]
            drill_t = t - (self._drill_start or 0)
            new_markers, mid = _drill_marker(mid, ex, ey, drill_t)
            for mk in new_markers:
                mk.header.stamp = now
                ma.markers.append(mk)

        # e2 smoke — only visible after rover arrives at e2
        if "floor_red" in self._obs_to_pos:
            ex, ey = self._obs_to_pos["floor_red"]
            new_markers, mid = _smoke_puff(mid, ex, ey, t, ns="smoke_e2",
                                           base_z=0.3,
                                           color=ColorRGBA(r=0.70,g=0.60,b=0.45,a=0.0),
                                           visible="floor_red" in self._visited)
            for mk in new_markers:
                mk.header.stamp = now
                ma.markers.append(mk)

        # e3 smoke — only visible after rover arrives at e2
        if "floor_purple" in self._obs_to_pos:
            ex, ey = self._obs_to_pos["floor_purple"]
            new_markers, mid = _smoke_puff(mid, ex, ey, t, ns="smoke_e3",
                                           base_z=0.3,
                                           color=ColorRGBA(r=0.60,g=0.68,b=0.55,a=0.0),
                                           visible="floor_red" in self._visited)
            for mk in new_markers:
                mk.header.stamp = now
                ma.markers.append(mk)

        # e7 — sample collection: rock rises and fades
        if self._e7_collected and "goal_green" in self._obs_to_pos:
            ex, ey = self._obs_to_pos["goal_green"]
            elapsed  = t - (self._e7_collect_t or t)
            progress = min(1.0, elapsed * 0.5)
            if progress < 1.0:
                sz   = 0.22 * (1.0 - progress * 0.85)
                rise = progress * 0.55
                rock = _m("sample_collect", mid, Marker.SPHERE); mid += 1
                _pos(rock, ex, ey, 0.10 + rise)
                _sc(rock, sz, sz*0.85, sz*0.70)
                rock.color = ColorRGBA(r=0.42, g=0.28, b=0.18, a=1.0 - progress*0.7)
                rock.header.stamp = now
                ma.markers.append(rock)
                ring = _m("sample_ring", mid, Marker.CYLINDER); mid += 1
                _pos(ring, ex, ey, 0.06)
                ring_r = progress * 0.45
                _sc(ring, ring_r, ring_r, 0.02)
                ring.color = ColorRGBA(r=0.20, g=0.70, b=0.90, a=max(0.0, 0.7 - progress))
                ring.header.stamp = now
                ma.markers.append(ring)

        if ma.markers:
            self.pub_effects.publish(ma)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    json_path   = sys.argv[1] if len(sys.argv) > 1 else "visualization/tour_data.json"
    rviz_config = sys.argv[2] if len(sys.argv) > 2 else "visualization/mars_rover.rviz"
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
