"""Look at a panorama the way the agent would, without running the agent.

    python peek.py --pano panos/near_board --gaze 0 5 --out peek/            # one fovea + the periphery
    python peek.py --pano panos/near_board --sweep -40 40 10 --pitch 5 --out peek/   # a contact sheet of foveae

Use it to check that the board's text is legible in a 12-degree fovea and illegible in the
periphery, and that the chairs are visible from the back of the room.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from yarbus import views as V


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fov", type=float, default=12.0)
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--gaze", type=float, nargs=2, metavar=("YAW", "PITCH"))
    ap.add_argument("--sweep", type=float, nargs=3, metavar=("FROM", "TO", "STEP"), help="yaw sweep")
    ap.add_argument("--pitch", type=float, default=0.0, help="pitch of the sweep")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pano = V.Pano(a.pano, blur_deg=a.blur)
    st = V.State(fov=a.fov, mode="2d")
    tag = Path(a.pano).name
    if a.gaze:
        g = (a.gaze[0], a.gaze[1])
        st.gazes.append(g)
        V.periphery_image(pano.blurred_rgb(), st, g).save(out / f"{tag}_periphery.jpg", quality=90)
        V.fovea_image(pano, st, g).save(out / f"{tag}_fovea_{g[0]:+.0f}_{g[1]:+.0f}.png")
        print(f"wrote {out}/{tag}_periphery.jpg and the fovea at {g}")
    if a.sweep:
        yaws = np.arange(a.sweep[0], a.sweep[1] + 1e-9, a.sweep[2])
        tiles = [pano.fovea_rgb(float(y), a.pitch, a.fov, 384) for y in yaws]
        cols = min(len(tiles), 5)
        rows = (len(tiles) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * 394, rows * 414), (250, 250, 248))
        d = ImageDraw.Draw(sheet)
        for i, (y, t) in enumerate(zip(yaws, tiles)):
            x0, y0 = (i % cols) * 394 + 5, (i // cols) * 414 + 25
            sheet.paste(t, (x0, y0))
            d.text((x0, y0 - 20), f"yaw {y:+.0f}, pitch {a.pitch:+.0f}", fill=(20, 20, 20), font=V.font(15, True))
        p = out / f"{tag}_sweep_pitch{a.pitch:+.0f}.png"
        sheet.save(p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
