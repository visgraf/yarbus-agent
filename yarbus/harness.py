"""Applying the agent's action to the run state: merges, then new marks, then the move.

Kept apart from the loop so tests/selftest.py can check it without a model.
"""
from __future__ import annotations

import math

import numpy as np

from . import geometry as G
from . import views as V


def apply_action(st: V.State, resp: dict, gaze, k: int, pano: V.Pano, thr: float, n: int = V.FOV_N):
    """Returns (new_marks, feedback lines, next gaze or None if the agent answered)."""
    fb: list[str] = []
    for mg in resp.get("merges") or []:
        live = {m.id: m for m in st.live_marks()}
        try:
            keep, drop = int(mg.get("keep", -1)), int(mg.get("drop", -1))
        except (TypeError, ValueError):
            keep, drop = -1, -1
        if keep in live and drop in live and keep != drop:
            live[drop].merged_into = keep
            fb.append(f"merged #{drop} into #{keep}")
        else:
            fb.append(f"merge #{drop} into #{keep} ignored (not two different live marks)")
    new_marks = []
    for mk in resp.get("marks") or []:
        try:
            x, y = float(mk.get("x", -1)), float(mk.get("y", -1))
        except (TypeError, ValueError):
            x, y = -1.0, -1.0
        if not (0 <= x < n and 0 <= y < n):
            fb.append(f"mark at ({x:.0f}, {y:.0f}) ignored: outside the fovea")
            continue
        d = G.fovea_pixel_to_dir(x, y, gaze[0], gaze[1], st.fov, n)
        pt = None
        if st.mode == "3d":
            r = G.depth_at(pano.depth, d)
            pt = d * r if math.isfinite(r) else None
        others = st.live_marks()
        m = V.Mark(id=len(st.marks) + 1, fixation=k, dir=d, point=pt)
        st.marks.append(m)
        new_marks.append(m)
        yw, pc = m.yaw_pitch()
        s = f"mark #{m.id} at yaw {yw:.1f}, pitch {pc:.1f}"
        if st.mode == "3d":
            s += " (no depth there)" if pt is None else \
                f", {np.linalg.norm(pt):.2f} m away, {pt[1] + pano.eye_height:.2f} m above the floor"
        if others:
            if st.mode == "3d" and pt is not None:
                cand = [(float(np.hypot(*(o.point[[0, 2]] - pt[[0, 2]]))), o) for o in others if o.point is not None]
                if cand:
                    dist, o = min(cand, key=lambda t: t[0])
                    s += f"; nearest is #{o.id} at {dist:.2f} m" + (" -> POSSIBLE DUPLICATE" if dist < thr else "")
            elif st.mode == "2d":
                ang, o = min(((G.angle_deg(o.dir, d), o) for o in others), key=lambda t: t[0])
                s += f"; nearest is #{o.id} at {ang:.1f} deg" + (" -> POSSIBLE DUPLICATE" if ang < thr else "")
        fb.append(s)

    nxt = resp.get("next_gaze")
    if resp.get("answer"):
        return new_marks, fb, None
    try:
        moved = (G.wrap_yaw(float(nxt["yaw"])), float(np.clip(float(nxt["pitch"]), -89.9, 89.9)))
    except (TypeError, KeyError, ValueError):
        fb.append("you gave neither a usable next gaze nor an answer; the gaze stays where it is")
        moved = gaze
    return new_marks, fb, moved
