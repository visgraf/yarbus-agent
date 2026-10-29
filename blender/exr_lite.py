"""Read an uncompressed, single-part, scanline OpenEXR file with numpy alone.

Why this exists: Blender's bundled Python has no OpenEXR module, and `bpy.data.images.load`
returns a multilayer EXR as type MULTILAYER with size (0, 0) and no pixels (measured
2026-09-12). In background mode the Render Result has no pixel access either. So a
Blender-side tool that needs its own Depth pass back has to save the multilayer file with
`exr_codec = "NONE"` and parse it. The format for that case is small: a header of
attributes, an offset table, then one chunk per scanline holding every channel's row.

Verified against the OpenEXR reader host-side by tools/check_sequence.py on every fixation
file it checks, so a wrong byte here fails a check rather than silently shifting a sample.

    channels = read_uncompressed_exr(path)   # {"ViewLayer.Depth.Z": (h, w) float32, ...}

Refuses anything else (compressed, tiled, deep, multipart) with a clear error.
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = 20000630
PIXEL_TYPES = {0: ("<u4", 4), 1: ("<f2", 2), 2: ("<f4", 4)}


def _cstr(buf: bytes, pos: int) -> tuple[str, int]:
    end = buf.index(b"\x00", pos)
    return buf[pos:end].decode("utf-8"), end + 1


def read_header(buf: bytes) -> tuple[dict, int]:
    magic, version = struct.unpack_from("<ii", buf, 0)
    if magic != MAGIC:
        raise ValueError("not an OpenEXR file")
    if version & 0x200:
        raise ValueError("tiled EXR is not supported")
    if version & 0x800:
        raise ValueError("deep EXR is not supported")
    if version & 0x1000:
        raise ValueError("multipart EXR is not supported (Blender: use_exr_interleave must be on)")
    pos, attrs = 8, {}
    while True:
        name, pos = _cstr(buf, pos)
        if name == "":
            break
        typ, pos = _cstr(buf, pos)
        size = struct.unpack_from("<i", buf, pos)[0]
        pos += 4
        raw = buf[pos:pos + size]
        pos += size
        if typ == "chlist":
            chans, p = [], 0
            while True:
                cname, p = _cstr(raw, p)
                if cname == "":
                    break
                ptype, plin, xs, ys = struct.unpack_from("<iB3xii", raw, p)
                p += 16
                chans.append((cname, ptype, xs, ys))
            attrs[name] = chans
        elif typ == "compression":
            attrs[name] = raw[0]
        elif typ == "box2i":
            attrs[name] = struct.unpack("<iiii", raw)
        elif typ == "lineOrder":
            attrs[name] = raw[0]
        else:
            attrs[name] = raw
    return attrs, pos


def read_uncompressed_exr(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as fh:
        buf = fh.read()
    attrs, pos = read_header(buf)
    if attrs.get("compression", 0) != 0:
        raise ValueError(f"{path}: compression {attrs['compression']} is not NONE; save with exr_codec='NONE'")
    if attrs.get("lineOrder", 0) != 0:
        raise ValueError(f"{path}: only INCREASING_Y line order is supported")
    x0, y0, x1, y1 = attrs["dataWindow"]
    w, h = x1 - x0 + 1, y1 - y0 + 1
    chans = attrs["channels"]
    for cname, ptype, xs, ys in chans:
        if (xs, ys) != (1, 1):
            raise ValueError(f"{path}: subsampled channel {cname}")
    offsets = np.frombuffer(buf, dtype="<u8", count=h, offset=pos)
    out = {cname: np.empty((h, w), dtype=PIXEL_TYPES[ptype][0]) for cname, ptype, _, _ in chans}
    line_bytes = w * sum(PIXEL_TYPES[ptype][1] for _, ptype, _, _ in chans)
    for off in offsets:
        off = int(off)
        y, size = struct.unpack_from("<ii", buf, off)
        if size != line_bytes:
            raise ValueError(f"{path}: chunk at y={y} has {size} bytes, expected {line_bytes}")
        p = off + 8
        for cname, ptype, _, _ in chans:            # channel-major within the scanline
            dt, nb = PIXEL_TYPES[ptype]
            out[cname][y - y0] = np.frombuffer(buf, dtype=dt, count=w, offset=p)
            p += w * nb
    return {k: v.astype(np.float32) if v.dtype != np.float32 else v for k, v in out.items()}
