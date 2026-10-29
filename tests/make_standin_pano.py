"""A stand-in panorama folder for testing without Blender: any equirect image plus the depth of
an axis-aligned box room around the head (so the 3D plumbing has metric numbers to work with).

    python tests/make_standin_pano.py --image some_equirect.jpg --out /tmp/pano_test --width 4096
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yarbus import geometry as G  # noqa: E402


def box_depth(w: int, h: int, half_x=3.5, front=4.0, back=5.0, floor=1.6, ceil=1.4) -> np.ndarray:
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    Y, P = np.meshgrid((u - 0.5) * 360, (0.5 - v) * 180)
    d = G.direction(Y, P)
    t = np.full((h, w), np.inf)
    for axis, lo, hi in ((0, -half_x, half_x), (1, -floor, ceil), (2, -front, back)):
        c = d[..., axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = np.where(c > 0, hi / c, np.where(c < 0, lo / c, np.inf))
        t = np.minimum(t, tt)
    return t.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=4096)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    w, h = a.width, a.width // 2
    Image.open(a.image).convert("RGB").resize((w, h), Image.BICUBIC).save(out / "rgb.png")
    np.save(out / "depth.npy", box_depth(w, h))
    (out / "meta.json").write_text(json.dumps({"name": "standin", "eye_height_above_floor_m": 1.6,
                                               "note": "stand-in: image from elsewhere, box-room depth"}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
