"""Checks that can fail: geometry round trips, the fovea sampler against a known-answer panorama,
the 3D lift against a box room, and the mark / duplicate / merge bookkeeping.

    python tests/selftest.py            # prints PASS / FAIL per check, exits 1 on any failure
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yarbus import geometry as G  # noqa: E402
from yarbus import views as V  # noqa: E402
from yarbus.harness import apply_action  # noqa: E402
from make_standin_pano import box_depth  # noqa: E402

RESULTS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def main() -> None:
    rng = np.random.default_rng(0)
    # 1. direction <-> (yaw, pitch)
    yaw, pitch = rng.uniform(-180, 180, 1000), rng.uniform(-89, 89, 1000)
    y2, p2 = G.yaw_pitch(G.direction(yaw, pitch))
    err = max(np.abs(G.wrap_yaw(0) + ((y2 - yaw + 180) % 360 - 180)).max(), np.abs(p2 - pitch).max())
    check("direction <-> yaw/pitch round trip", err < 1e-9, f"max err {err:.2e} deg")
    check("yaw 0 is forward (-Z), yaw 90 is right (+X), pitch 90 is up (+Y)",
          np.allclose(G.direction(0, 0), [0, 0, -1]) and np.allclose(G.direction(90, 0), [1, 0, 0])
          and np.allclose(G.direction(0, 90), [0, 1, 0], atol=1e-12))

    # 2. fovea pixel <-> direction
    worst = 0.0
    for _ in range(200):
        gy, gp = rng.uniform(-180, 180), rng.uniform(-80, 80)
        x, y = rng.uniform(0, 511, 2)
        d = G.fovea_pixel_to_dir(x, y, gy, gp, 12.0, 512)
        back = G.dir_to_fovea_pixel(d, gy, gp, 12.0, 512)
        worst = max(worst, abs(back[0] - x) + abs(back[1] - y))
    check("fovea pixel <-> direction round trip", worst < 1e-6, f"max err {worst:.2e} px")
    c = G.fovea_pixel_to_dir(255.5, 255.5, 37.0, -12.0, 12.0, 512)
    check("fovea centre is the gaze", G.angle_deg(c, G.direction(37.0, -12.0)) < 1e-6)
    corner = G.fovea_pixel_to_dir(-0.5, 255.5, 0.0, 0.0, 12.0, 512)
    check("fovea half-width is fov/2 (left edge at yaw -6)", abs(G.yaw_pitch(corner)[0] + 6.0) < 1e-6,
          f"{float(G.yaw_pitch(corner)[0]):.6f}")
    check("fovea x grows right, y grows down",
          G.yaw_pitch(G.fovea_pixel_to_dir(400, 256, 0, 0, 12, 512))[0] > 0 and
          G.yaw_pitch(G.fovea_pixel_to_dir(256, 400, 0, 0, 12, 512))[1] < 0)

    # 3. the fovea sampler against a panorama that encodes its own yaw / pitch
    w, h = 4096, 2048
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    YAW, PIT = np.meshgrid((u - 0.5) * 360, (0.5 - v) * 180)
    code = np.stack([np.cos(np.radians(YAW)), np.sin(np.radians(YAW)), PIT / 90.0], -1).astype(np.float32)
    worst = 0.0
    for gy, gp in ((0, 0), (90, 20), (-170, -30), (179, 5)):
        rays = G.fovea_rays(gy, gp, 12.0, 64)
        s = G.sample_pano(code, rays)
        sy = np.degrees(np.arctan2(s[..., 1], s[..., 0]))
        sp = s[..., 2] * 90.0
        ty, tp = G.yaw_pitch(rays)
        e = np.maximum(np.abs((sy - ty + 180) % 360 - 180), np.abs(sp - tp))
        worst = max(worst, float(e.max()))
    check("fovea sampler reads the right panorama pixels (incl. the seam)", worst < 0.1, f"max err {worst:.3f} deg")

    # 4. 3D lift against a box room
    depth = box_depth(w, h, half_x=3.5, front=4.0, back=5.0, floor=1.6, ceil=1.4)
    with tempfile.TemporaryDirectory() as td:
        Image.fromarray((np.clip(code * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)).save(Path(td) / "rgb.png")
        np.save(Path(td) / "depth.npy", depth)
        (Path(td) / "meta.json").write_text(json.dumps({"eye_height_above_floor_m": 1.6}))
        pano = V.Pano(td)
        st = V.State(fov=12.0, mode="3d")
        gaze = (0.0, -30.0)          # 1.6 / tan 30 = 2.77 m < 4 m: the ray meets the floor
        nm, fb, mv = apply_action(st, {"marks": [{"x": 256, "y": 256}], "next_gaze": {"yaw": 3, "pitch": -30}}, gaze, 1, pano, 0.35)
        p = nm[0].point
        on_floor = abs(p[1] + 1.6) < 0.02
        check("3D mark at pitch -30 lands on the floor, 2.77 m ahead", on_floor and abs(p[2] + 1.6 / math.tan(math.radians(30))) < 0.02, f"point {np.round(p, 3).tolist()}")
        check("3D feedback reports height ~0 m above the floor", "0.00 m above the floor" in fb[0] or "-0.00 m" in fb[0], fb[0])
        check("next gaze is applied", mv == (3.0, -30.0))
        # a second mark 3 cm away on the floor -> flagged; a far one -> not flagged
        nm2, fb2, _ = apply_action(st, {"marks": [{"x": 262, "y": 256}, {"x": 500, "y": 500}], "next_gaze": {"yaw": 0, "pitch": 0}},
                                   gaze, 2, pano, 0.35)
        check("3D near-duplicate is flagged", "POSSIBLE DUPLICATE" in fb2[0], fb2[0])
        check("3D distant mark is not flagged", "POSSIBLE DUPLICATE" not in fb2[1], fb2[1])
        _, fb3, mv3 = apply_action(st, {"merges": [{"keep": 1, "drop": 2}], "answer": "2"}, gaze, 3, pano, 0.35)
        check("merge reduces the live count", len(st.live_marks()) == 2 and st.marks[1].merged_into == 1, str(fb3))
        check("an answer stops the run", mv3 is None)
        _, fb4, _ = apply_action(st, {"merges": [{"keep": 1, "drop": 2}], "next_gaze": {"yaw": 0, "pitch": 0}}, gaze, 4, pano, 0.35)
        check("merging an already merged mark is refused", "ignored" in fb4[0], fb4[0])
        _, fb5, _ = apply_action(st, {"marks": [{"x": 700, "y": 10}], "next_gaze": {"yaw": 0, "pitch": 0}}, gaze, 5, pano, 0.35)
        check("a mark outside the fovea is refused", "ignored" in fb5[0] and len(st.marks) == 3, fb5[0])
        # 2D: angular flag
        st2 = V.State(fov=12.0, mode="2d")
        apply_action(st2, {"marks": [{"x": 256, "y": 256}], "next_gaze": {"yaw": 0, "pitch": 0}}, (0.0, 0.0), 1, pano, 4.0)
        _, f2, _ = apply_action(st2, {"marks": [{"x": 60, "y": 256}]}, (5.0, 0.0), 2, pano, 4.0)
        ang = float(f2[0].split("at ")[-1].split(" deg")[0])
        check("2D separation is the true angle (fovea at yaw 5, x=60 -> ~0.4 deg)", abs(ang - 0.42) < 0.1, f2[0])
        check("2D near-duplicate is flagged", "POSSIBLE DUPLICATE" in f2[0])
        # the agent can never see the board text in the periphery: blur really removes fine detail
        stripes = np.zeros((h, w, 3), np.uint8)
        stripes[:, ::8] = 255                      # 0.7-degree period, like chalk strokes
        b = np.asarray(V.Pano._sphere_blur(stripes[::2, ::2].astype(np.float32), 1.0))
        contrast = float(b[h // 4, :, 0].std())
        check("peripheral blur (1 deg) wipes out sub-degree detail", contrast < 2.0, f"residual std {contrast:.2f} of ~110")

    bad = RESULTS.count(False)
    print(f"\n{len(RESULTS) - bad}/{len(RESULTS)} checks pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
