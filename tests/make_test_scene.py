"""Build a small stand-in classroom (for testing the pipeline without the real scene).

    blender -b --factory-startup --python-exit-code 1 -P tests/make_test_scene.py -- --out /tmp/test_room.blend

A 7 m x 9 m x 3 m room: floor 'sol', a dark board 'blackBoard' on the +Y wall with chalk text,
and N_ROWS x N_COLS simple chairs (seat + back + legs) plus one teacher's chair. The chair count
is printed and written beside the .blend as <out>.truth.json.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy

ROOM_X, ROOM_Y, ROOM_Z = 7.0, 9.0, 3.0
N_ROWS, N_COLS = 3, 4
BOARD_TEXT = "FOVEA 12 DEG\nYARBUS 1967\nLOOK  >  SEE"


def args() -> list[str]:
    return sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def mat(name, rgb, emission=0.0, rough=0.6):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = rough
    if emission:
        bsdf.inputs["Emission Color"].default_value = (*rgb, 1.0)
        bsdf.inputs["Emission Strength"].default_value = emission
    return m


def box(name, centre, size, material):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=centre)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = size
    bpy.ops.object.transform_apply(scale=True)
    ob.data.materials.append(material)
    return ob


def chair(name, x, y, yaw_deg, m_wood, m_metal):
    parts = []
    parts.append(box(f"{name}_seat", (x, y, 0.45), (0.42, 0.42, 0.04), m_wood))
    parts.append(box(f"{name}_back", (x, y - 0.2, 0.75), (0.42, 0.04, 0.4), m_wood))
    for dx in (-0.18, 0.18):
        for dy in (-0.18, 0.18):
            parts.append(box(f"{name}_leg", (x + dx, y + dy, 0.22), (0.03, 0.03, 0.44), m_metal))
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    ob = bpy.context.active_object
    ob.name = name
    # rotate about the seat centre
    bpy.context.scene.cursor.location = (x, y, 0.0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    ob.rotation_euler[2] = math.radians(yaw_deg)
    bpy.ops.object.select_all(action="DESELECT")
    return ob


def main():
    out = Path(args()[args().index("--out") + 1])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    m_floor = mat("floor", (0.35, 0.25, 0.18))
    m_wall = mat("wall", (0.8, 0.78, 0.72))
    m_board = mat("board", (0.05, 0.07, 0.06), rough=0.9)
    m_chalk = mat("chalk", (0.9, 0.9, 0.88), emission=0.6)
    m_wood = mat("wood", (0.65, 0.45, 0.25))
    m_metal = mat("metal", (0.6, 0.6, 0.62), rough=0.3)
    t = 0.1
    box("sol", (0, 0, -t / 2), (ROOM_X, ROOM_Y, t), m_floor)
    box("ceiling", (0, 0, ROOM_Z + t / 2), (ROOM_X, ROOM_Y, t), m_wall)
    box("wall_front", (0, ROOM_Y / 2 + t / 2, ROOM_Z / 2), (ROOM_X, t, ROOM_Z), m_wall)
    box("wall_back", (0, -ROOM_Y / 2 - t / 2, ROOM_Z / 2), (ROOM_X, t, ROOM_Z), m_wall)
    box("wall_left", (-ROOM_X / 2 - t / 2, 0, ROOM_Z / 2), (t, ROOM_Y, ROOM_Z), m_wall)
    box("wall_right", (ROOM_X / 2 + t / 2, 0, ROOM_Z / 2), (t, ROOM_Y, ROOM_Z), m_wall)
    box("blackBoard", (0, ROOM_Y / 2 - 0.02, 1.55), (3.0, 0.04, 1.2), m_board)
    # a text object faces +Z and reads along +X; rotated 90 deg about X it faces -Y (into the room)
    # and still reads along +X, which is to the right for an observer looking towards +Y
    bpy.ops.object.text_add(location=(-1.3, ROOM_Y / 2 - 0.045, 1.95), rotation=(math.radians(90), 0, 0))
    txt = bpy.context.active_object
    txt.name = "chalkText"
    txt.data.body = BOARD_TEXT
    txt.data.size = 0.22
    txt.data.extrude = 0.002
    txt.data.materials.append(m_chalk)
    n = 0
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x = -2.25 + c * 1.5
            y = 2.0 - r * 1.6
            chair(f"chair_{n:02d}", x, y, 0.0, m_wood, m_metal)
            n += 1
    chair("chair_teacher", 2.4, 3.6, 200.0, m_wood, m_metal)
    n += 1
    bpy.ops.object.light_add(type="AREA", location=(0, 0, ROOM_Z - 0.05))
    L = bpy.context.active_object
    L.data.energy = 1500
    L.data.size = 4.0
    bpy.ops.object.light_add(type="POINT", location=(0, 3.0, 2.6))
    bpy.context.active_object.data.energy = 300
    sc.render.engine = "CYCLES"
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out))
    Path(str(out) + ".truth.json").write_text(json.dumps({"chairs": n, "board_text": BOARD_TEXT}, indent=1))
    print(f"[make_test_scene] wrote {out} with {n} chairs")


if __name__ == "__main__":
    main()
