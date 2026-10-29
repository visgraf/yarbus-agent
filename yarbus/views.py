"""What the agent sees: a blurred panorama (periphery / parafovea) and a sharp fovea, plus, in the
3D variant, the same views of depth and a top-down plan of the RGB-D panorama.

All images are drawn with Pillow. Overlays (grid, scanpath, numbered marks) carry the agent's
spatial memory; the agent gets no other picture of what it has already seen.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from . import geometry as G

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
_FONTS: dict = {}


def font(size: int, bold: bool = False):
    key = (size, bold)
    if key not in _FONTS:
        p = FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        try:
            _FONTS[key] = ImageFont.truetype(str(p), size)
        except OSError:
            _FONTS[key] = ImageFont.load_default(size)
    return _FONTS[key]


# colours (Okabe-Ito, as in f3d-vision's visual language)
INK, MUTED, GRID = (20, 20, 20), (120, 120, 120), (255, 255, 255)
BLUE, ORANGE, VERM, GREEN, SKY, YELLOW = (0, 114, 178), (230, 159, 0), (213, 94, 0), (0, 158, 115), (86, 180, 233), (240, 228, 66)
PATH = (230, 159, 0)
MARK = (0, 158, 115)
GAZE = (213, 94, 0)

PERI_W, PERI_H = 1500, 750
PERI_ML, PERI_MT, PERI_MR, PERI_MB = 46, 26, 10, 10
FOV_N = 512
FOV_M = 30
PLAN_N = 768
DEPTH_MIN, DEPTH_MAX = 0.5, 16.0


# ---------------------------------------------------------------- the panorama
class Pano:
    def __init__(self, folder: str | Path, blur_deg: float = 1.0):
        folder = Path(folder)
        self.folder = folder
        self.rgb = np.asarray(Image.open(folder / "rgb.png").convert("RGB"))
        self.depth = np.load(folder / "depth.npy").astype(np.float32)
        self.meta = json.loads((folder / "meta.json").read_text()) if (folder / "meta.json").exists() else {}
        self.eye_height = float(self.meta.get("eye_height_above_floor_m") or 1.6)
        self.H, self.W = self.depth.shape
        if self.rgb.shape[:2] != (self.H, self.W):
            raise ValueError(f"rgb {self.rgb.shape[:2]} and depth {self.depth.shape} differ")
        self.blur_deg = blur_deg
        self._blur_rgb = None
        self._blur_depth_rgb = None
        self._depth_lo = None

    # -- periphery
    @staticmethod
    def _sphere_blur(img: np.ndarray, sigma_deg: float) -> np.ndarray:
        """Gaussian of sigma_deg degrees on the sphere, approximately: vertical sigma fixed in
        pixels, horizontal sigma widened by 1/cos(latitude) row by row (wrapping in yaw)."""
        h, w = img.shape[:2]
        ppd = w / 360.0
        s = sigma_deg * ppd
        out = ndimage.gaussian_filter1d(img.astype(np.float32), s, axis=0, mode="nearest")
        lat = (0.5 - (np.arange(h) + 0.5) / h) * math.pi
        for i in range(h):
            sr = min(s / max(math.cos(lat[i]), 0.05), w / 4)
            out[i] = ndimage.gaussian_filter1d(out[i], sr, axis=0, mode="wrap")
        return out

    def blurred_rgb(self) -> Image.Image:
        if self._blur_rgb is None:
            small = np.asarray(Image.fromarray(self.rgb).resize((2048, 1024), Image.BOX), np.float32)
            b = self._sphere_blur(small, self.blur_deg)
            self._blur_rgb = Image.fromarray(np.clip(b, 0, 255).astype(np.uint8)).resize((PERI_W, PERI_H), Image.BICUBIC)
        return self._blur_rgb

    def depth_lo(self) -> np.ndarray:
        """Depth at periphery resolution (block median), for the plan and the depth view."""
        if self._depth_lo is None:
            f = self.W // PERI_W if self.W >= PERI_W else 1
            if f > 1:
                h, w = (self.H // f) * f, (self.W // f) * f
                blocks = self.depth[:h, :w].reshape(h // f, f, w // f, f).transpose(0, 2, 1, 3).reshape(h // f, w // f, -1)
                with np.errstate(all="ignore"):
                    lo = np.nanmedian(blocks, axis=2)
            else:
                lo = self.depth
            self._depth_lo = lo.astype(np.float32)
        return self._depth_lo

    def blurred_depth_rgb(self) -> Image.Image:
        if self._blur_depth_rgb is None:
            lo = self.depth_lo()
            col = depth_colors(lo)
            col = np.asarray(Image.fromarray(col).resize((2048, 1024), Image.BILINEAR), np.float32)
            b = self._sphere_blur(col, self.blur_deg)
            self._blur_depth_rgb = Image.fromarray(np.clip(b, 0, 255).astype(np.uint8)).resize((PERI_W, PERI_H), Image.BICUBIC)
        return self._blur_depth_rgb

    # -- fovea
    def fovea_rgb(self, yaw: float, pitch: float, fov: float, n: int = FOV_N) -> Image.Image:
        rays = G.fovea_rays(yaw, pitch, fov, n)
        a = G.sample_pano(self.rgb.astype(np.float32), rays, order=1)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    def fovea_depth(self, yaw: float, pitch: float, fov: float, n: int = FOV_N) -> np.ndarray:
        rays = G.fovea_rays(yaw, pitch, fov, n)
        return G.sample_pano(np.nan_to_num(self.depth, nan=1e3), rays, order=0)

    def sharp_small(self, w: int) -> Image.Image:
        return Image.fromarray(self.rgb).resize((w, w // 2), Image.LANCZOS)

    def points_lo(self) -> tuple[np.ndarray, np.ndarray]:
        """Head-frame points of the periphery-resolution depth, and their height above the floor."""
        lo = self.depth_lo()
        h, w = lo.shape
        u = (np.arange(w) + 0.5) / w
        v = (np.arange(h) + 0.5) / h
        yaw = (u - 0.5) * 360.0
        pitch = (0.5 - v) * 180.0
        Y, P = np.meshgrid(yaw, pitch)
        d = G.direction(Y, P)
        pts = d * lo[..., None]
        ok = np.isfinite(lo)
        pts = pts[ok]
        return pts, pts[:, 1] + self.eye_height


def depth_colors(depth: np.ndarray) -> np.ndarray:
    from matplotlib import colormaps
    t = (np.log(np.clip(np.nan_to_num(depth, nan=DEPTH_MAX), DEPTH_MIN, DEPTH_MAX)) - math.log(DEPTH_MIN)) / \
        (math.log(DEPTH_MAX) - math.log(DEPTH_MIN))
    rgba = colormaps["viridis"](t)
    col = (rgba[..., :3] * 255).astype(np.uint8)
    col[~np.isfinite(depth)] = 0
    return col


# ---------------------------------------------------------------- run state
@dataclass
class Mark:
    id: int
    fixation: int
    dir: np.ndarray
    point: np.ndarray | None          # head-frame metres (3D variant only)
    merged_into: int | None = None

    def yaw_pitch(self) -> tuple[float, float]:
        y, p = G.yaw_pitch(self.dir)
        return float(y), float(p)

    def to_json(self) -> dict:
        y, p = self.yaw_pitch()
        return {"id": self.id, "fixation": self.fixation, "yaw": round(y, 3), "pitch": round(p, 3),
                "point_head_m": None if self.point is None else [round(float(x), 4) for x in self.point],
                "merged_into": self.merged_into}


@dataclass
class State:
    fov: float
    mode: str                                   # "2d" or "3d"
    gazes: list[tuple[float, float]] = field(default_factory=list)
    marks: list[Mark] = field(default_factory=list)

    def live_marks(self) -> list[Mark]:
        return [m for m in self.marks if m.merged_into is None]


# ---------------------------------------------------------------- drawing helpers
def _peri_xy(yaw, pitch):
    u, v = G.pano_px(yaw, pitch, PERI_W, PERI_H)
    return PERI_ML + u, PERI_MT + v


def _framed(base: Image.Image) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    W = PERI_ML + PERI_W + PERI_MR
    H = PERI_MT + PERI_H + PERI_MB
    im = Image.new("RGB", (W, H), (245, 245, 242))
    im.paste(base, (PERI_ML, PERI_MT))
    return im, ImageDraw.Draw(im, "RGBA")


def _grid(d: ImageDraw.ImageDraw):
    f = font(13)
    for yaw in range(-180, 181, 30):
        x, _ = _peri_xy(yaw, 0)
        d.line([(x, PERI_MT), (x, PERI_MT + PERI_H)], fill=(255, 255, 255, 70), width=1)
        d.text((x, PERI_MT - 4), f"{yaw}", fill=INK, font=f, anchor="mb")
    for pitch in range(-90, 91, 30):
        _, y = _peri_xy(0, pitch)
        d.line([(PERI_ML, y), (PERI_ML + PERI_W, y)], fill=(255, 255, 255, 70), width=1)
        d.text((PERI_ML - 5, y), f"{pitch}", fill=INK, font=f, anchor="rm")


def _poly_runs(d, dirs, fill, width):
    for run in G.equirect_polyline(dirs, PERI_W, PERI_H):
        pts = [(PERI_ML + x, PERI_MT + y) for x, y in run]
        if len(pts) > 1:
            d.line(pts, fill=fill, width=width, joint="curve")


def draw_scanpath(d, gazes, upto: int | None = None, color=PATH, numbers=True, width=2, r=7, numfont=12):
    gz = gazes if upto is None else gazes[:upto]
    for i in range(1, len(gz)):
        arc = G.great_circle(G.direction(*gz[i - 1]), G.direction(*gz[i]))
        _poly_runs(d, arc, (*color, 220), width)
    f = font(numfont, bold=True)
    for i, (yw, pt) in enumerate(gz):
        x, y = _peri_xy(yw, pt)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(*color, 235), outline=(30, 30, 30, 255))
        if numbers:
            d.text((x, y), str(i + 1), fill=(0, 0, 0), font=f, anchor="mm")


def draw_marks_peri(d, marks: list[Mark], numfont=11):
    f = font(numfont, bold=True)
    for m in marks:
        y, p = m.yaw_pitch()
        x, yy = _peri_xy(y, p)
        s = 6
        d.polygon([(x, yy - s), (x + s, yy), (x, yy + s), (x - s, yy)], fill=(*MARK, 255), outline=(255, 255, 255, 255))
        d.text((x + s + 1, yy), str(m.id), fill=(255, 255, 255), font=f, anchor="lm", stroke_width=2, stroke_fill=(0, 0, 0))


def draw_footprint(d, gaze, fov, color=GAZE, width=3):
    fp = G.footprint(gaze[0], gaze[1], fov)
    fp = np.vstack([fp, fp[:1]])
    _poly_runs(d, fp, (*color, 255), width)


def periphery_image(base: Image.Image, st: State, gaze) -> Image.Image:
    im, d = _framed(base)
    _grid(d)
    draw_scanpath(d, st.gazes)
    draw_marks_peri(d, st.live_marks())
    draw_footprint(d, gaze, st.fov)
    return im


def depth_legend(im: Image.Image, x: int, y: int, w: int, h: int = 12):
    d = ImageDraw.Draw(im)
    t = np.linspace(0, 1, w)
    dep = np.exp(math.log(DEPTH_MIN) + t * (math.log(DEPTH_MAX) - math.log(DEPTH_MIN)))
    bar = np.repeat(depth_colors(dep)[None], h, axis=0)
    im.paste(Image.fromarray(bar), (x, y))
    f = font(12)
    for m in (0.5, 1, 2, 4, 8, 16):
        tx = x + (math.log(m) - math.log(DEPTH_MIN)) / (math.log(DEPTH_MAX) - math.log(DEPTH_MIN)) * (w - 1)
        d.line([(tx, y + h), (tx, y + h + 3)], fill=INK)
        d.text((tx, y + h + 4), f"{m:g} m", fill=INK, font=f, anchor="mt")


def depth_periphery_image(base: Image.Image, st: State, gaze) -> Image.Image:
    im = periphery_image(base, st, gaze)
    out = Image.new("RGB", (im.width, im.height + 34), (245, 245, 242))
    out.paste(im, (0, 0))
    depth_legend(out, PERI_ML, im.height + 2, 420)
    return out


def _fovea_frame(content: Image.Image, gaze, fov) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    n = content.width
    im = Image.new("RGB", (FOV_M + n + 8, FOV_M + n + 8), (245, 245, 242))
    im.paste(content, (FOV_M, FOV_M))
    d = ImageDraw.Draw(im, "RGBA")
    f = font(12)
    for t in (0, 128, 256, 384, 511):
        x = FOV_M + t * n / 512
        d.line([(x, FOV_M - 5), (x, FOV_M)], fill=INK)
        d.text((x, FOV_M - 6), str(t), fill=INK, font=f, anchor="mb")
        d.line([(FOV_M - 5, x), (FOV_M, x)], fill=INK)
        d.text((FOV_M - 6, x), str(t), fill=INK, font=f, anchor="rm")
    return im, d


def draw_marks_fovea(d, st: State, gaze, n=FOV_N, off=FOV_M, scale=1.0, numfont=14):
    f = font(numfont, bold=True)
    for m in st.live_marks():
        px = G.dir_to_fovea_pixel(m.dir, gaze[0], gaze[1], st.fov, n)
        if px is None:
            continue
        x, y = off + px[0] * scale, off + px[1] * scale
        r = 9 * scale
        d.ellipse([x - r, y - r, x + r, y + r], outline=(*MARK, 255), width=max(2, int(3 * scale)))
        d.text((x + r + 2, y), str(m.id), fill=(255, 255, 255), font=f, anchor="lm", stroke_width=2, stroke_fill=(0, 0, 0))


def fovea_image(pano: Pano, st: State, gaze) -> Image.Image:
    im, d = _fovea_frame(pano.fovea_rgb(gaze[0], gaze[1], st.fov), gaze, st.fov)
    draw_marks_fovea(d, st, gaze)
    return im


def fovea_depth_image(pano: Pano, st: State, gaze) -> Image.Image:
    dep = pano.fovea_depth(gaze[0], gaze[1], st.fov)
    dep[dep > 999] = np.nan
    im, d = _fovea_frame(Image.fromarray(depth_colors(dep)), gaze, st.fov)
    draw_marks_fovea(d, st, gaze)
    c = dep[FOV_N // 2 - 2:FOV_N // 2 + 3, FOV_N // 2 - 2:FOV_N // 2 + 3]
    txt = f"centre {np.nanmedian(c):.2f} m" if np.isfinite(c).any() else "centre: no surface"
    d.text((FOV_M + 6, FOV_M + FOV_N - 6), txt, fill=(255, 255, 255), font=font(15, True), anchor="lb",
           stroke_width=2, stroke_fill=(0, 0, 0))
    return im


# ---------------------------------------------------------------- the plan (3D variant)
class Plan:
    """Top-down view of the RGB-D panorama in the head frame: plan x = +X (right), plan up = -Z (forward)."""

    def __init__(self, pano: Pano, n: int = PLAN_N):
        self.n = n
        pts, hgt = pano.points_lo()
        keep = (hgt > 0.08) & (hgt < 2.4)
        x, z = pts[keep, 0], -pts[keep, 2]
        r = float(np.percentile(np.maximum(np.abs(x), np.abs(z)), 99.5)) * 1.05 + 0.3
        self.r = max(r, 2.0)
        self.margin = 34
        self.s = (n - 2 * self.margin) / (2 * self.r)
        img = np.full((n, n, 3), 250, np.float32)
        for lo_h, hi_h, shade in ((1.3, 2.4, 190.0), (0.08, 1.3, 110.0)):
            sel = (hgt[keep] >= lo_h) & (hgt[keep] < hi_h)
            px, py = self.to_px(x[sel], z[sel])
            ok = (px >= 0) & (px < n) & (py >= 0) & (py < n)
            cnt = np.zeros((n, n), np.float32)
            np.add.at(cnt, (py[ok].astype(int), px[ok].astype(int)), 1.0)
            a = np.clip(cnt / 3.0, 0, 1)[..., None]
            img = img * (1 - a) + shade * a
        self.base = Image.fromarray(img.astype(np.uint8))

    def to_px(self, x, z):
        return self.n / 2 + np.asarray(x) * self.s, self.n / 2 - np.asarray(z) * self.s

    def image(self, st: State, gaze, pano: Pano, numfont=13) -> Image.Image:
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        f = font(12)
        step = 1 if self.r < 6 else 2
        for k in range(-int(self.r), int(self.r) + 1):
            if k % step:
                continue
            a, _ = self.to_px(k, 0)
            _, b = self.to_px(0, k)
            d.line([(a, 0), (a, self.n)], fill=(0, 0, 0, 22))
            d.line([(0, b), (self.n, b)], fill=(0, 0, 0, 22))
        # 1 m scale bar
        x0, y0 = 12, self.n - 14
        d.line([(x0, y0), (x0 + self.s, y0)], fill=INK, width=3)
        d.text((x0 + self.s / 2, y0 - 4), "1 m", fill=INK, font=f, anchor="mb")
        # gaze wedge
        dist = G.depth_at(pano.depth, G.direction(*gaze))
        L = (dist if np.isfinite(dist) else self.r) * math.cos(math.radians(gaze[1]))
        hx, hy = self.to_px(0, 0)
        for s in (-1, 1):
            a = math.radians(gaze[0] + s * st.fov / 2)
            ex, ey = self.to_px(L * math.sin(a), L * math.cos(a))
            d.line([(hx, hy), (ex, ey)], fill=(*GAZE, 200), width=2)
        # marks
        fb = font(numfont, bold=True)
        for m in st.live_marks():
            if m.point is None:
                continue
            px, py = self.to_px(m.point[0], -m.point[2])
            d.ellipse([px - 6, py - 6, px + 6, py + 6], fill=(*MARK, 255), outline=(255, 255, 255, 255))
            d.text((px + 8, py), str(m.id), fill=INK, font=fb, anchor="lm", stroke_width=2, stroke_fill=(255, 255, 255))
        # head
        d.polygon([(hx, hy - 10), (hx + 7, hy + 7), (hx - 7, hy + 7)], fill=(*BLUE, 255))
        d.text((8, 8), "plan (top view): head at the triangle, forward is up", fill=INK, font=f)
        return im


def jpeg_bytes(im: Image.Image, q: int = 88) -> bytes:
    import io
    b = io.BytesIO()
    im.save(b, "JPEG", quality=q)
    return b.getvalue()


def png_bytes(im: Image.Image) -> bytes:
    import io
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return b.getvalue()
