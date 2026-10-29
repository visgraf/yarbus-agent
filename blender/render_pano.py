"""Render one RGB-D equirectangular panorama of the Classroom from a chosen head position.

    blender -b CLASSROOM/classroom_eye.blend --python-exit-code 1 \
        -P blender/render_pano.py -- --auto near_board --out panos/near_board --dry-run
    blender -b CLASSROOM/classroom_eye.blend --python-exit-code 1 \
        -P blender/render_pano.py -- --auto near_board --out panos/near_board

The head only rotates, so one panorama from a position holds everything the eyes can see from
there. The panorama is centred on the blackboard: yaw 0 / pitch 0 (the image centre) is the
horizontal direction from the head to the board's centre.

Head frame, as in f3d-vision: +X right, +Y up, -Z forward. Equirect pixel (u, v) in [0,1]^2 with
v = 0 the top row maps to lon = (u - 0.5) * 2pi, lat = (0.5 - v) * pi and
d = (sin lon cos lat, sin lat, -cos lon cos lat). Positive yaw is to the right.

Placement (--auto). Uses two objects by name: the board (--board, default 'blackBoard') and the
floor (--floor, default 'sol'). The board's thin horizontal axis is its normal, pointing to the
floor's centre (into the room).
  near_board    on the normal through the board's centre, --board-distance metres out
  back_of_room  on the same line, --back-margin metres in from the floor's far edge
Eye height is the floor top plus --height. --position X Y Z overrides placement (world metres).
A clearance test (26 rays) refuses positions within --min-clearance of any surface or inside a
mesh; --force overrides it.

Writes to --out (new or empty directory):
    rgb.png      the Combined pass through the scene's colour management, 8-bit sRGB
    depth.npy    float32 (H, W), ray distance from the eye centre in metres, NaN where nothing is hit
    pano.exr     the raw multilayer EXR (uncompressed; Combined + Depth), only with --keep-exr
    meta.json    head position, frame, placement inputs, render settings, timing

Failures exit nonzero (Blender alone would exit 0 after a failed script).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exr_lite import read_uncompressed_exr  # noqa: E402

PREFIX = "[render_pano]"


def args_after_dashes() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def parse(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="render_pano.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--auto", choices=["near_board", "back_of_room"])
    ap.add_argument("--position", type=float, nargs=3, metavar=("X", "Y", "Z"))
    ap.add_argument("--look-at", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    help="world point the panorama centre faces (default: the board centre)")
    ap.add_argument("--board", default="blackBoard")
    ap.add_argument("--floor", default="sol")
    ap.add_argument("--board-distance", type=float, default=2.5)
    ap.add_argument("--back-margin", type=float, default=1.0)
    ap.add_argument("--height", type=float, default=1.6, help="eye height above the floor top (m)")
    ap.add_argument("--min-clearance", type=float, default=0.25)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--width", type=int, default=8192)
    ap.add_argument("--spp", type=int, default=512)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--device", default="OPTIX", choices=["OPTIX", "CUDA", "CPU"])
    ap.add_argument("--dry-run", action="store_true", help="print placement and clearance, render nothing")
    ap.add_argument("--keep-exr", action="store_true", help="keep the raw EXR (about 0.7 GB at 8192 wide)")
    a = ap.parse_args(argv)
    if (a.auto is None) == (a.position is None):
        ap.error("give exactly one of --auto or --position")
    return a


# ---------------------------------------------------------------- scene geometry
def world_bbox(ob) -> tuple[Vector, Vector]:
    pts = [ob.matrix_world @ Vector(c) for c in ob.bound_box]
    lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return lo, hi


def find_object(name: str):
    ob = bpy.data.objects.get(name)
    if ob is None:
        cands = sorted(o.name for o in bpy.data.objects if o.name.lower().startswith(name.lower()))
        if not cands:
            raise SystemExit(f"{PREFIX} STOP no object named {name!r} (or starting with it)")
        ob = bpy.data.objects[cands[0]]
    return ob


def board_frame(board_name: str, floor_name: str) -> dict:
    blo, bhi = world_bbox(find_object(board_name))
    flo, fhi = world_bbox(find_object(floor_name))
    c_b = (blo + bhi) / 2
    c_f = (flo + fhi) / 2
    ext = bhi - blo
    axis = 0 if ext.x < ext.y else 1                      # the thin horizontal axis = the normal
    sign = 1.0 if (c_f[axis] - c_b[axis]) > 0 else -1.0  # into the room
    n = Vector((0.0, 0.0, 0.0))
    n[axis] = sign
    # floor's far edge along n, measured from the board's centre
    far = (fhi[axis] if sign > 0 else flo[axis])
    depth_room = abs(far - c_b[axis])
    return {"board_lo": list(blo), "board_hi": list(bhi), "board_centre": list(c_b),
            "floor_lo": list(flo), "floor_hi": list(fhi), "floor_top_z": fhi.z,
            "normal": list(n), "room_depth_from_board_m": depth_room}


def placement(a, fr: dict) -> Vector:
    if a.position is not None:
        return Vector(a.position)
    c_b, n = Vector(fr["board_centre"]), Vector(fr["normal"])
    t = a.board_distance if a.auto == "near_board" else fr["room_depth_from_board_m"] - a.back_margin
    p = c_b + n * t
    p.z = fr["floor_top_z"] + a.height
    return p


def clearance(scene, p: Vector) -> dict:
    dg = bpy.context.evaluated_depsgraph_get()
    dists, back = [], 0
    dirs = [Vector((i, j, k)).normalized() for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)
            if (i, j, k) != (0, 0, 0)]
    misses = 0
    for d in dirs:
        hit, loc, nrm, _, ob, _ = scene.ray_cast(dg, p, d)
        if not hit:
            misses += 1
            continue
        dists.append((loc - p).length)
        if nrm.dot(d) > 0:
            back += 1
    return {"rays": len(dirs), "misses": misses, "min_hit_m": min(dists) if dists else None,
            "backface_hits": back}


def head_matrix(p: Vector, look: Vector) -> Matrix:
    f = look - p
    f.z = 0.0
    if f.length < 1e-6:
        raise SystemExit(f"{PREFIX} STOP look-at point is straight above/below the head")
    f.normalize()
    up = Vector((0.0, 0.0, 1.0))
    right = f.cross(up).normalized()
    m = Matrix.Identity(4)
    for i in range(3):          # columns: camera +X = right, +Y = up, +Z = -forward
        m[i][0], m[i][1], m[i][2], m[i][3] = right[i], up[i], -f[i], p[i]
    return m


# ---------------------------------------------------------------- render
def setup_device(scene, prefer: str) -> str:
    cprefs = bpy.context.preferences.addons["cycles"].preferences
    if prefer != "CPU":
        for backend in [prefer] + (["CUDA"] if prefer != "CUDA" else []):
            try:
                cprefs.compute_device_type = backend
            except TypeError:
                continue
            cprefs.refresh_devices()
            gpus = [d for d in cprefs.devices if d.type == backend]
            if gpus:
                for d in cprefs.devices:
                    d.use = d.type == backend
                scene.cycles.device = "GPU"
                return backend
    scene.cycles.device = "CPU"
    return "CPU"


def configure(scene, m: Matrix, a) -> tuple[object, str]:
    if "cycles" not in bpy.context.preferences.addons:
        import addon_utils
        addon_utils.enable("cycles", default_set=True)
    scene.render.engine = "CYCLES"
    backend = setup_device(scene, a.device)
    if backend != a.device:
        print(f"{PREFIX} WARNING requested {a.device}, using {backend}", flush=True)
    r = scene.render
    r.resolution_x, r.resolution_y = a.width, a.width // 2
    r.resolution_percentage = 100
    r.pixel_aspect_x = r.pixel_aspect_y = 1.0
    r.film_transparent = False
    r.use_motion_blur = False
    r.use_compositing = False
    r.use_sequencer = False
    r.use_border = False
    c = scene.cycles
    c.samples = a.spp
    c.use_adaptive_sampling = True
    c.use_denoising = not a.no_denoise
    vl = bpy.context.view_layer
    vl.use_pass_combined = True
    vl.use_pass_z = True
    im = r.image_settings
    im.media_type = "MULTI_LAYER_IMAGE"
    im.file_format = "OPEN_EXR_MULTILAYER"
    im.color_depth = "32"
    im.exr_codec = "NONE"           # exr_lite reads uncompressed EXR only
    im.use_exr_interleave = True    # one part, all passes

    data = bpy.data.cameras.new("YARBUS_HEAD")
    data.type = "PANO"
    data.panorama_type = "EQUIRECTANGULAR"
    data.longitude_min, data.longitude_max = -math.pi, math.pi
    data.latitude_min, data.latitude_max = -math.pi / 2, math.pi / 2
    data.clip_start, data.clip_end = 0.01, 1000.0
    data.dof.use_dof = False
    cam = bpy.data.objects.new("YARBUS_HEAD", data)
    scene.collection.objects.link(cam)
    cam.matrix_world = m
    scene.camera = cam
    bpy.context.view_layer.update()
    return cam, backend


def depth_channel(chans: dict) -> np.ndarray:
    keys = [k for k in chans if k.endswith("Depth.Z")] or [k for k in chans if k.endswith(".Z") or k == "Z"]
    if not keys:
        raise RuntimeError(f"no depth channel in EXR; channels: {sorted(chans)}")
    return chans[keys[0]]


def main() -> None:
    a = parse(args_after_dashes())
    scene = bpy.context.scene
    fr = board_frame(a.board, a.floor) if (a.auto or a.look_at is None) else {}
    p = placement(a, fr)
    look = Vector(a.look_at) if a.look_at else Vector(fr["board_centre"])
    cl = clearance(scene, p)
    print(f"{PREFIX} board/floor frame: {json.dumps(fr)}", flush=True)
    print(f"{PREFIX} head position (world m): ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}); looks at "
          f"({look.x:.3f}, {look.y:.3f}, {look.z:.3f}), distance {(look - p).length:.2f} m", flush=True)
    print(f"{PREFIX} clearance: {json.dumps(cl)}", flush=True)
    bad = (cl["min_hit_m"] is not None and cl["min_hit_m"] < a.min_clearance) or cl["backface_hits"] > 6
    if bad and not a.force:
        raise SystemExit(f"{PREFIX} STOP position fails the clearance test (use --position or --force)")
    if a.dry_run:
        print(f"{PREFIX} DRY RUN, nothing rendered", flush=True)
        return

    out = Path(a.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{PREFIX} STOP output must be new or empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    m = head_matrix(p, look)
    cam, backend = configure(scene, m, a)
    exr = out / "pano.exr"
    scene.render.filepath = str(exr)
    t0 = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    secs = time.perf_counter() - t0
    if not exr.is_file():
        raise RuntimeError(f"render produced no EXR at {exr}")
    # colour-managed 8-bit PNG of the same render
    im = scene.render.image_settings
    im.media_type = "IMAGE"
    im.file_format = "PNG"
    im.color_mode = "RGB"
    im.color_depth = "8"
    bpy.data.images["Render Result"].save_render(str(out / "rgb.png"), scene=scene)
    if not (out / "rgb.png").is_file():
        raise RuntimeError("save_render produced no rgb.png")

    depth = depth_channel(read_uncompressed_exr(str(exr))).astype(np.float32)
    depth[~np.isfinite(depth) | (depth > 1e4)] = np.nan
    if depth.shape != (a.width // 2, a.width):
        raise RuntimeError(f"depth shape {depth.shape} != {(a.width // 2, a.width)}")
    np.save(out / "depth.npy", depth)
    h, w = depth.shape
    centre = float(np.nanmedian(depth[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3]))
    board_dist = (look - p).length
    # known answer: the image centre looks horizontally at the board, so its depth is the head's
    # distance to the board plane (within half the board's thickness)
    check = None
    if fr:
        n, c_b = Vector(fr["normal"]), Vector(fr["board_centre"])
        thick = abs((Vector(fr["board_hi"]) - Vector(fr["board_lo"])).dot(n))
        expected = abs((p - c_b).dot(n)) - thick / 2
        ok = math.isfinite(centre) and abs(centre - expected) <= max(0.03, thick)
        check = {"expected_m": expected, "measured_m": centre, "ok": bool(ok)}
        print(f"{PREFIX} centre-depth check {'PASS' if ok else 'WARN'}: measured {centre:.3f} m, "
              f"expected {expected:.3f} m (board plane); WARN means something stands in front of the board "
              f"or the panorama is not oriented as assumed", flush=True)
    meta = {
        "name": a.auto or "custom", "head_position_w_m": list(p), "look_at_w_m": list(look),
        "eye_height_above_floor_m": p.z - fr["floor_top_z"] if fr else None,
        "camera_matrix_world": [list(row) for row in cam.matrix_world],
        "frame": "head: +X right, +Y up, -Z forward; equirect lon=(u-0.5)*2pi, lat=(0.5-v)*pi",
        "placement": {"auto": a.auto, "position_arg": a.position, "board_distance": a.board_distance,
                      "back_margin": a.back_margin, "height": a.height, "frame": fr},
        "clearance": cl,
        "render": {"width": w, "height": h, "spp": a.spp, "adaptive": True, "denoise": not a.no_denoise,
                   "device": backend, "seconds": round(secs, 2), "blender": bpy.app.version_string,
                   "blend": bpy.data.filepath},
        "depth": {"units": "m, ray distance from the eye centre", "nan_fraction": float(np.isnan(depth).mean()),
                  "min": float(np.nanmin(depth)), "median": float(np.nanmedian(depth)),
                  "max": float(np.nanmax(depth)), "centre_pixel": centre,
                  "look_at_distance": board_dist, "centre_check": check},
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    if not a.keep_exr:
        exr.unlink()
    print(f"{PREFIX} wrote {out} in {secs:.1f} s; depth at the image centre {centre:.3f} m "
          f"(look-at point {board_dist:.3f} m)", flush=True)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(str(e), flush=True)
            sys.stdout.flush()
            os._exit(2)
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)
