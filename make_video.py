"""Assemble the Yarbus-style video from finished runs.

    python make_video.py --runs runs/board_2d runs/board_3d runs/chairs_2d runs/chairs_3d --out yarbus.mp4

Sections: title; for each run, a replay (the scene, the growing scanpath, the fovea, what the
agent said at each fixation); the Yarbus plate (all scanpaths side by side); each run's
explanation in the agent's words; for counting runs, where the marks lie in 3D; an end card.
--sec-per-fix sets the replay pace; --still writes a few PNG frames instead of the video.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from yarbus import geometry as G
from yarbus import views as V
from yarbus.views import font

W, H, FPS = 1920, 1080, 30
BG = (250, 250, 248)
INK, INK2, MUTED = (20, 20, 20), (70, 70, 70), (130, 130, 130)
PATH, GAZE, MARK, BLUE = V.PATH, V.GAZE, V.MARK, V.BLUE
TITLE = "Yarbus (1967), revisited by an AI agent with a foveal eye"


# ---------------------------------------------------------------- helpers
def wrap(text: str, f, width: int) -> list[str]:
    lines = []
    for para in str(text).split("\n"):
        cur = ""
        for w in para.split():
            t = (cur + " " + w).strip()
            if f.getlength(t) <= width:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def text_block(d, xy, text, f, width, fill=INK, spacing=6, max_lines=None):
    x, y = xy
    lines = wrap(text, f, width)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip() + " …"
    for ln in lines:
        d.text((x, y), ln, fill=fill, font=f)
        y += f.size + spacing
    return y


def pxy(yaw, pitch, ox, oy, w, h):
    u, v = G.pano_px(yaw, pitch, w, h)
    return ox + float(u), oy + float(v)


def poly(d, dirs, ox, oy, w, h, fill, width):
    for run in G.equirect_polyline(dirs, w, h):
        pts = [(ox + x, oy + y) for x, y in run]
        if len(pts) > 1:
            d.line(pts, fill=fill, width=width, joint="curve")


def draw_path(d, gazes, ox, oy, w, h, color=PATH, width=3, r=8, numbers=True, nf=13, upto_partial=None):
    for i in range(1, len(gazes)):
        poly(d, G.great_circle(G.direction(*gazes[i - 1]), G.direction(*gazes[i])), ox, oy, w, h, (*color, 230), width)
    f = font(nf, True)
    for i, g in enumerate(gazes):
        x, y = pxy(*g, ox, oy, w, h)
        d.ellipse([x - r, y - r, x + r, y + r], fill=(*color, 240), outline=(30, 30, 30, 255))
        if numbers:
            d.text((x, y), str(i + 1), fill=(0, 0, 0), font=f, anchor="mm")


def draw_footprint(d, gaze, fov, ox, oy, w, h, width=3):
    fp = G.footprint(gaze[0], gaze[1], fov)
    poly(d, np.vstack([fp, fp[:1]]), ox, oy, w, h, (*GAZE, 255), width)


def draw_marks(d, marks, ox, oy, w, h, s=7, nf=13):
    f = font(nf, True)
    for m in marks:
        x, y = pxy(m["yaw"], m["pitch"], ox, oy, w, h)
        d.polygon([(x, y - s), (x + s, y), (x, y + s), (x - s, y)], fill=(*MARK, 255), outline=(255, 255, 255, 255))
        d.text((x + s + 2, y), str(m["id"]), fill=(255, 255, 255), font=f, anchor="lm", stroke_width=2, stroke_fill=(0, 0, 0))


def washed(im: Image.Image, keep: float = 0.55, gray: bool = False) -> Image.Image:
    if gray:
        im = ImageOps.grayscale(im).convert("RGB")
    return Image.blend(Image.new("RGB", im.size, (255, 255, 255)), im, keep)


# ---------------------------------------------------------------- a run, replayable
class Run:
    def __init__(self, folder: Path, pano_cache: dict):
        self.dir = folder
        self.cfg = json.loads((folder / "run.json").read_text())
        self.steps = [json.loads(l) for l in (folder / "steps.jsonl").read_text().splitlines() if l.strip()]
        self.final = json.loads((folder / "final.json").read_text())
        self.q = self.cfg["question"]
        self.mode = self.cfg["mode"]
        self.fov = self.cfg["fov_deg"]
        p = self.cfg["pano"]
        if p not in pano_cache:
            pano_cache[p] = V.Pano(p, blur_deg=self.cfg["blur_sigma_deg"])
        self.pano = pano_cache[p]
        self.gazes = [tuple(s["gaze"]) for s in self.steps]
        self.marks = {m["id"]: m for m in self.final["marks"]}
        self.merge_at = {}                       # mark id -> fixation at which it was merged away
        for s in self.steps:
            for fb in s["feedback"]:
                mm = re.match(r"merged #(\d+) into #(\d+)", fb)
                if mm:
                    self.merge_at[int(mm.group(1))] = s["fixation"]
        expl = (folder / "explanation.md").read_text() if (folder / "explanation.md").exists() else ""
        self.summary = self.final.get("summary") or ""
        self.explanation = expl
        name = self.cfg.get("pano_meta", {}).get("name") or Path(p).name
        self.place = {"near_board": "head near the board", "back_of_room": "head at the back of the room"}.get(name, name)
        self.label = f"{'(i)' if self.q['kind'] == 'read' else '(ii)'} {self.q['question']}"
        self.variant = "2D: RGB panorama" if self.mode == "2d" else "3D: RGB-D panorama"
        self.model = self.cfg.get("model") or self.cfg.get("agent")

    def marks_at(self, k: int) -> list[dict]:
        """Marks live after fixation k's action."""
        return [m for m in self.marks.values() if m["fixation"] <= k and not (m["id"] in self.merge_at and self.merge_at[m["id"]] <= k)]

    def live_final(self) -> list[dict]:
        return [m for m in self.marks.values() if m["merged_into"] is None]


# ---------------------------------------------------------------- frames
PX, PY, PW, PH = 24, 86, 1280, 640          # scene panorama
FX, FY, FN = 1328, 86, 568                  # fovea


class Replay:
    def __init__(self, run: Run):
        self.r = run
        base = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(base)
        d.text((PX, 22), run.label, fill=INK, font=font(34, True))
        d.text((W - 24, 30), f"{run.variant}  ·  {run.place}", fill=INK2, font=font(24), anchor="ra")
        base.paste(washed(run.pano.sharp_small(PW), 0.8), (PX, PY))
        d.text((PX, PY + PH + 4), "the scene (sharp, for the viewer) with the agent's scanpath", fill=MUTED, font=font(17))
        d.text((FX, PY + FN + 4), f"the agent's fovea ({run.fov:g}°)", fill=MUTED, font=font(17))
        self.base = base
        self._fovea = {}
        self._side = {}

    def fovea(self, k: int) -> Image.Image:
        if k not in self._fovea:
            g = self.r.gazes[k - 1]
            self._fovea[k] = self.r.pano.fovea_rgb(g[0], g[1], self.r.fov, FN)
        return self._fovea[k]

    def side(self, k: int) -> Image.Image | None:
        """What else the agent saw at fixation k: periphery (2D) or plan + foveal depth (3D)."""
        if k in self._side:
            return self._side[k]
        sd = self.r.dir / "steps"
        im = None
        if self.r.mode == "3d" and (sd / f"{k:03d}_plan.png").exists():
            im = Image.new("RGB", (FN, 360), BG)
            pl = Image.open(sd / f"{k:03d}_plan.png").resize((340, 340), Image.LANCZOS)
            dp = Image.open(sd / f"{k:03d}_depth_fovea.jpg")
            dp = dp.crop((V.FOV_M, V.FOV_M, V.FOV_M + V.FOV_N, V.FOV_M + V.FOV_N)).resize((210, 210), Image.LANCZOS)
            im.paste(pl, (0, 0))
            im.paste(dp, (FN - 210, 0))
            dd = ImageDraw.Draw(im)
            dd.text((FN - 210, 214), "foveal depth", fill=MUTED, font=font(15))
            dd.text((0, 342), "plan the agent sees (top view)", fill=MUTED, font=font(15))
        elif (sd / f"{k:03d}_periphery.jpg").exists():
            p = Image.open(sd / f"{k:03d}_periphery.jpg")
            p = p.crop((V.PERI_ML, V.PERI_MT, V.PERI_ML + V.PERI_W, V.PERI_MT + V.PERI_H)).resize((FN, FN // 2), Image.LANCZOS)
            im = Image.new("RGB", (FN, 360), BG)
            im.paste(p, (0, 0))
            ImageDraw.Draw(im).text((0, FN // 2 + 4), "what the agent sees around it (blurred)", fill=MUTED, font=font(15))
        self._side[k] = im
        return im

    def frame(self, k: int, t: float, show_answer: bool = False) -> Image.Image:
        """Fixation k (1-based); t in [0,1] is the saccade's progress toward gaze k."""
        r = self.r
        im = self.base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        gz = r.gazes[:k]
        if t < 1 and k > 1:
            arc = G.great_circle(G.direction(*r.gazes[k - 2]), G.direction(*r.gazes[k - 1]))
            cur_dir = arc[min(len(arc) - 1, int(t * (len(arc) - 1)))]
            cy, cp = G.yaw_pitch(cur_dir)
            cur = (float(cy), float(cp))
            draw_path(d, r.gazes[:k - 1], PX, PY, PW, PH)
            poly(d, arc[:max(2, int(t * len(arc)))], PX, PY, PW, PH, (*PATH, 230), 3)
            marks = r.marks_at(k - 1)
        else:
            cur = r.gazes[k - 1]
            draw_path(d, gz, PX, PY, PW, PH)
            marks = r.marks_at(k)
        if r.q["kind"] == "count":
            draw_marks(d, marks, PX, PY, PW, PH)
        draw_footprint(d, cur, r.fov, PX, PY, PW, PH)
        # fovea
        if t >= 1 or k == 1:
            fv = self.fovea(k).copy()
            fd = ImageDraw.Draw(fv, "RGBA")
            if r.q["kind"] == "count":
                ff = font(16, True)
                for m in marks:
                    px = G.dir_to_fovea_pixel(G.direction(m["yaw"], m["pitch"]), cur[0], cur[1], r.fov, FN)
                    if px is None:
                        continue
                    x, y = px
                    fd.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(*MARK, 255), width=3)
                    fd.text((x + 12, y), str(m["id"]), fill=(255, 255, 255), font=ff, anchor="lm", stroke_width=2, stroke_fill=(0, 0, 0))
            im.paste(fv, (FX, FY))
        else:
            d.rectangle([FX, FY, FX + FN, FY + FN], fill=(60, 60, 60))
            d.text((FX + FN / 2, FY + FN / 2), "saccade", fill=(200, 200, 200), font=font(26), anchor="mm")
        d.rectangle([FX, FY, FX + FN, FY + FN], outline=(*GAZE, 255), width=3)
        # text panel
        s = r.steps[k - 1]
        y0 = PY + PH + 34
        d.text((PX, y0), f"fixation {k} of {len(r.steps)}   ·   gaze yaw {cur[0]:.0f}°, pitch {cur[1]:.0f}°", fill=INK, font=font(24, True))
        if r.q["kind"] == "count":
            d.text((PX + PW, y0), f"marked: {len(marks)}", fill=MARK, font=font(26, True), anchor="ra")
        y = y0 + 40
        y = text_block(d, (PX, y), f"sees: {s.get('observation') or ''}", font(21), PW, INK2, max_lines=3)
        y = text_block(d, (PX, y + 6), f"why next: {s.get('reason') or ''}", font(21), PW, INK, max_lines=3)
        if show_answer and r.final.get("answer"):
            fa = font(22, True)
            n = min(4, len(wrap(f"answer: {r.final['answer']}", fa, PW - 12)))
            d.rectangle([PX - 6, y + 4, PX + PW, y + 16 + n * (fa.size + 6)], fill=(255, 244, 214))
            text_block(d, (PX, y + 10), f"answer: {r.final['answer']}", fa, PW - 12, INK, max_lines=4)
        sd = self.side(k)
        if sd is not None:
            im.paste(sd, (FX, FY + FN + 30))
        return im


def card(lines: list[tuple[str, int, bool, tuple]], sub: str | None = None) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    total = sum(len(wrap(t, font(sz, b), W - 360)) * (sz + 12) + 6 for t, sz, b, _ in lines)
    y = (H - total) / 2
    for text, sz, bold, col in lines:
        for ln in wrap(text, font(sz, bold), W - 360):
            d.text((W / 2, y), ln, fill=col, font=font(sz, bold), anchor="ma")
            y += sz + 12
        y += 6
    if sub:
        d.text((W / 2, H - 50), sub, fill=MUTED, font=font(18), anchor="ma")
    return im


def plate(runs: list[Run]) -> Image.Image:
    """Yarbus's figure, redrawn: rows are questions, columns are 2D and 3D."""
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((W / 2, 18), "Same agent, same room: the question shapes the scanpath", fill=INK, font=font(32, True), anchor="ma")
    rows = sorted({r.q["id"] for r in runs}, key=lambda q: [r.q["id"] for r in runs].index(q))
    cols = ["2d", "3d"]
    cw, ch = 860, 430
    for i, qid in enumerate(rows[:2]):
        for j, mode in enumerate(cols):
            r = next((x for x in runs if x.q["id"] == qid and x.mode == mode), None)
            ox, oy = (W - 2 * cw - 40) // 2 + j * (cw + 40), 70 + i * (ch + 80)
            if r is None:
                d.rectangle([ox, oy, ox + cw, oy + ch], outline=MUTED)
                continue
            im.paste(washed(r.pano.sharp_small(cw), 0.45, gray=True), (ox, oy))
            draw_path(d, r.gazes, ox, oy, cw, ch, color=(25, 25, 25), width=2, r=4, numbers=False)
            if r.q["kind"] == "count":
                draw_marks(d, r.live_final(), ox, oy, cw, ch, s=5, nf=11)
            hdr = f"{r.label}   [{mode.upper()}, {r.place}]   {len(r.gazes)} fixations"
            d.text((ox, oy + ch + 6), hdr, fill=INK, font=font(18, True))
            ans = str(r.final.get("answer") or "")
            text_block(d, (ox, oy + ch + 30), f"answer: {ans}", font(16), cw, INK2, spacing=3, max_lines=1)
    return im


def explanation_card(r: Run) -> Image.Image:
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((60, 30), f"{r.label}  ·  {r.variant}", fill=INK, font=font(30, True))
    d.text((60, 74), "how the agent explains its scanpath (its own words, written after the run)", fill=MUTED, font=font(20))
    pw, ph = 1000, 500
    im.paste(washed(r.pano.sharp_small(pw), 0.6), (60, 120))
    draw_path(d, r.gazes, 60, 120, pw, ph, width=2, r=6, nf=10)
    if r.q["kind"] == "count":
        draw_marks(d, r.live_final(), 60, 120, pw, ph, s=5, nf=11)
    text_block(d, (1100, 130), r.summary or "(no summary)", font(28), W - 1160, INK, spacing=10, max_lines=14)
    text_block(d, (60, 650), f"answer: {r.final.get('answer')}", font(24, True), W - 120, INK, max_lines=4)
    d.text((60, H - 46), f"full explanation: {r.dir.name}/explanation.md   ·   model: {r.model}", fill=MUTED, font=font(17))
    return im


def counting_card(runs: list[Run]) -> Image.Image | None:
    cr = [r for r in runs if r.q["kind"] == "count"]
    if not cr:
        return None
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im, "RGBA")
    d.text((W / 2, 24), "Counting needs a memory of where things are: the marks in 3D, seen from above",
           fill=INK, font=font(30, True), anchor="ma")
    n = len(cr)
    size = 760
    for i, r in enumerate(cr[:2]):
        ox = (W - n * size - (n - 1) * 80) // 2 + i * (size + 80)
        oy = 110
        plan = V.Plan(r.pano, n=size)
        st = V.State(fov=r.fov, mode="3d")
        for m in r.live_final():
            dvec = G.direction(m["yaw"], m["pitch"])
            pt = np.array(m["point_head_m"]) if m["point_head_m"] is not None else None
            if pt is None:
                rr = G.depth_at(r.pano.depth, dvec)
                pt = dvec * rr if math.isfinite(rr) else None
            st.marks.append(V.Mark(id=m["id"], fixation=m["fixation"], dir=dvec, point=pt))
        pim = plan.image(st, r.gazes[-1], r.pano, numfont=15)
        im.paste(pim, (ox, oy))
        cap = (f"{r.variant}: {len(r.live_final())} marks"
               + ("   (lifted to 3D after the run; the 2D agent never saw this view)" if r.mode == "2d" else "   (the agent saw this plan)"))
        text_block(d, (ox, oy + size + 10), cap, font(20, True), size, INK)
        text_block(d, (ox, oy + size + 64), f"answer: {r.final.get('answer')}", font(18), size, INK2, max_lines=2)
    return im


# ---------------------------------------------------------------- build
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default="yarbus.mp4")
    ap.add_argument("--sec-per-fix", type=float, default=0.9)
    ap.add_argument("--saccade", type=float, default=0.2, help="seconds of each saccade within --sec-per-fix")
    ap.add_argument("--still", help="write sample frames as PNGs into this folder instead of a video")
    a = ap.parse_args()

    cache: dict = {}
    runs = [Run(Path(p), cache) for p in a.runs]
    model = next((r.model for r in runs if r.model), "")

    def hold(img, sec):
        for _ in range(int(round(sec * FPS))):
            yield img

    def replay(r: Run):
        rp = Replay(r)
        nsac = int(round(a.saccade * FPS))
        ndw = max(1, int(round((a.sec_per_fix - a.saccade) * FPS)))
        for k in range(1, len(r.gazes) + 1):
            if k > 1:
                for i in range(nsac):
                    yield rp.frame(k, (i + 1) / (nsac + 1))
            f = rp.frame(k, 1.0)
            for _ in range(ndw):
                yield f
        yield from hold(rp.frame(len(r.gazes), 1.0, show_answer=True), 3.5)

    title = card([(TITLE, 46, True, INK),
                  ("A rotation-only head in a synthetic classroom. Around it, the agent sees only a blurred 360° panorama; "
                   "detail comes from a sharp 12° fovea it must point. Two questions, asked from two head positions, "
                   "each answered with a colour (2D) and a colour-plus-depth (3D) view.", 26, False, INK2),
                  ("(i) What is written on the blackboard?    (ii) How many chairs are there?", 28, True, BLUE)],
                 sub=f"Visgraf / IMPA  ·  scene: Blender 'Classroom' by Christophe Seux (CC0)  ·  agent: {model}")
    segs = [hold(title, 7)]
    for r in runs:
        segs.append(hold(card([(r.label, 40, True, INK), (f"{r.variant}  ·  {r.place}", 28, False, INK2)]), 2.5))
        segs.append(replay(r))
    segs.append(hold(plate(runs), 12))
    for r in runs:
        segs.append(hold(explanation_card(r), 9))
    cc = counting_card(runs)
    if cc is not None:
        segs.append(hold(cc, 9))
    segs.append(hold(card([("After Yarbus, A. L. (1967). Eye Movements and Vision. Plenum Press.", 26, False, INK2),
                           ("Active foveal playground: github.com/visgraf/f3d-vision", 24, False, MUTED)]), 4))

    if a.still:
        out = Path(a.still)
        out.mkdir(parents=True, exist_ok=True)
        title.save(out / "00_title.png")
        for i, r in enumerate(runs):
            rp = Replay(r)
            rp.frame(min(3, len(r.gazes)), 1.0).save(out / f"{i + 1:02d}_{r.dir.name}_fix3.png")
            if len(r.gazes) > 1:
                rp.frame(2, 0.5).save(out / f"{i + 1:02d}_{r.dir.name}_saccade.png")
            rp.frame(len(r.gazes), 1.0, show_answer=True).save(out / f"{i + 1:02d}_{r.dir.name}_final.png")
            explanation_card(r).save(out / f"{i + 1:02d}_{r.dir.name}_explanation.png")
        plate(runs).save(out / "90_plate.png")
        if cc is not None:
            cc.save(out / "91_counting.png")
        print(f"stills in {out}")
        return

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", a.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n = 0
    last = None
    for seg in segs:
        for fr in seg:
            if fr is not last:
                buf = np.asarray(fr.convert("RGB"), np.uint8).tobytes()
                last = fr
            p.stdin.write(buf)
            n += 1
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {a.out}: {n} frames, {n / FPS:.1f} s")


if __name__ == "__main__":
    main()
