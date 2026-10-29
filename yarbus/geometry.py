"""Sphere geometry for a rotation-only head.

Head frame (as in f3d-vision): +X right, +Y up, -Z forward. Gaze (yaw, pitch) in degrees,
yaw positive to the right, pitch positive up; (0, 0) looks along -Z, the panorama's centre.
Equirect pixel coordinates are continuous with pixel centres at +0.5:
    u_px = (yaw/360 + 0.5) * W,   v_px = (0.5 - pitch/180) * H
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


def direction(yaw_deg, pitch_deg) -> np.ndarray:
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return np.stack([np.sin(y) * np.cos(p), np.sin(p), -np.cos(y) * np.cos(p)], axis=-1)


def yaw_pitch(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    yaw = np.degrees(np.arctan2(d[..., 0], -d[..., 2]))
    pitch = np.degrees(np.arcsin(np.clip(d[..., 1], -1.0, 1.0)))
    return yaw, pitch


def wrap_yaw(yaw: float) -> float:
    return (yaw + 180.0) % 360.0 - 180.0


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0))))


def pano_px(yaw, pitch, w: int, h: int):
    return (np.asarray(yaw) / 360.0 + 0.5) * w, (0.5 - np.asarray(pitch) / 180.0) * h


def gaze_basis(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """forward, right, up of a gaze; pitch is clamped to +-89.9 so `right` stays defined."""
    pitch = float(np.clip(pitch, -89.9, 89.9))
    g = direction(yaw, pitch)
    r = np.array([-g[2], 0.0, g[0]])
    r /= np.linalg.norm(r)
    u = np.cross(r, g)
    return g, r, u


def fovea_rays(yaw: float, pitch: float, fov_deg: float, n: int) -> np.ndarray:
    """(n, n, 3) unit rays of an n x n pinhole image of field fov_deg centred on the gaze."""
    g, r, u = gaze_basis(yaw, pitch)
    t = math.tan(math.radians(fov_deg) / 2)
    s = (np.arange(n) + 0.5 - n / 2) / (n / 2) * t
    x, y = np.meshgrid(s, -s)                       # row 0 is the top
    d = g[None, None] + x[..., None] * r[None, None] + y[..., None] * u[None, None]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def fovea_pixel_to_dir(x: float, y: float, yaw: float, pitch: float, fov_deg: float, n: int) -> np.ndarray:
    g, r, u = gaze_basis(yaw, pitch)
    t = math.tan(math.radians(fov_deg) / 2)
    sx = (x + 0.5 - n / 2) / (n / 2) * t
    sy = -(y + 0.5 - n / 2) / (n / 2) * t
    d = g + sx * r + sy * u
    return d / np.linalg.norm(d)


def dir_to_fovea_pixel(d: np.ndarray, yaw: float, pitch: float, fov_deg: float, n: int):
    """Pixel (x, y) of direction d in the fovea image, or None if behind or outside it."""
    g, r, u = gaze_basis(yaw, pitch)
    z = float(np.dot(d, g))
    if z <= 1e-6:
        return None
    t = math.tan(math.radians(fov_deg) / 2)
    sx, sy = float(np.dot(d, r)) / z, float(np.dot(d, u)) / z
    x = sx / t * (n / 2) + n / 2 - 0.5
    y = -sy / t * (n / 2) + n / 2 - 0.5
    if -0.5 <= x <= n - 0.5 and -0.5 <= y <= n - 0.5:
        return x, y
    return None


def sample_pano(img: np.ndarray, rays: np.ndarray, order: int = 1) -> np.ndarray:
    """Sample an equirect image (H, W[, C]) along rays (..., 3)."""
    h, w = img.shape[:2]
    yaw, pitch = yaw_pitch(rays)
    u, v = pano_px(yaw, pitch, w, h)
    coords = [v - 0.5, u - 0.5]
    if img.ndim == 2:
        return ndimage.map_coordinates(img, coords, order=order, mode="grid-wrap")
    return np.stack([ndimage.map_coordinates(img[..., c], coords, order=order, mode="grid-wrap")
                     for c in range(img.shape[2])], axis=-1)


def footprint(yaw: float, pitch: float, fov_deg: float, k: int = 16) -> np.ndarray:
    """(4k, 3) rays along the border of the fovea square, corner to corner, clockwise from top-left."""
    n = 2 * k
    lo, hi = -0.5, n - 0.5
    corners = [(lo, lo), (hi, lo), (hi, hi), (lo, hi)]
    pts = []
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        for t in np.arange(k) / k:
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return np.array([fovea_pixel_to_dir(x, y, yaw, pitch, fov_deg, n) for x, y in pts])


def great_circle(a: np.ndarray, b: np.ndarray, step_deg: float = 1.0) -> np.ndarray:
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    ang = math.radians(angle_deg(a, b))
    if ang < 1e-6:
        return np.array([a])
    m = max(2, int(math.degrees(ang) / step_deg) + 1)
    ts = np.linspace(0, 1, m)
    s = math.sin(ang)
    return np.array([(math.sin((1 - t) * ang) * a + math.sin(t * ang) * b) / s for t in ts])


def equirect_polyline(dirs: np.ndarray, w: int, h: int) -> list[list[tuple[float, float]]]:
    """Split a polyline of directions into runs that do not cross the yaw = +-180 seam."""
    yaw, pitch = yaw_pitch(dirs)
    u, v = pano_px(yaw, pitch, w, h)
    runs, cur = [], []
    for i in range(len(u)):
        if abs(pitch[i]) > 89.5:                     # at a pole yaw is undefined: break the line there
            if cur:
                runs.append(cur)
            cur = []
            continue
        if cur and abs(u[i] - cur[-1][0]) > w / 4:   # the yaw seam
            runs.append(cur)
            cur = []
        cur.append((float(u[i]), float(v[i])))
    if cur:
        runs.append(cur)
    return runs


def depth_at(depth: np.ndarray, d: np.ndarray, win: int = 2) -> float:
    """Median ray distance in a (2win+1)^2 window around direction d (NaN-aware)."""
    h, w = depth.shape
    yaw, pitch = yaw_pitch(d)
    u, v = pano_px(yaw, pitch, w, h)
    ui, vi = int(math.floor(float(u))) % w, int(np.clip(math.floor(float(v)), 0, h - 1))
    cols = [(ui + k) % w for k in range(-win, win + 1)]
    rows = [int(np.clip(vi + k, 0, h - 1)) for k in range(-win, win + 1)]
    patch = depth[np.ix_(rows, cols)]
    return float(np.nanmedian(patch)) if np.isfinite(patch).any() else float("nan")
