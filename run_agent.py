"""One agent run: a question, a panorama, 2D or 3D, a scanpath, an answer and an explanation.

    python run_agent.py --pano panos/near_board  --question board  --mode 2d --out runs/board_2d
    python run_agent.py --pano panos/back_of_room --question chairs --mode 3d --out runs/chairs_3d

Needs ANTHROPIC_API_KEY in the environment (or --agent scripted, which tests the plumbing only).

Writes to --out (new or empty):
    run.json           configuration, model, panorama meta, prompts
    steps.jsonl        one line per fixation: gaze, the agent's observation / reason / notes,
                       marks and merges, harness feedback, token usage
    steps/NNN_*.jpg    the exact images the agent saw at each fixation
    final.json         answer, number of fixations, marks (for counting), token totals
    explanation.md     the agent's own account of its scanpath (written after the run)
    scanpath.png       the scanpath on the blurred panorama (also sent with the explanation request)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from yarbus import geometry as G
from yarbus import views as V
from yarbus.harness import apply_action
from yarbus.agent import ClaudeAgent, ScriptedAgent, EXPLAIN, act_tool, parse_explanation, system_prompt

HERE = Path(__file__).resolve().parent


def load_question(qid: str) -> dict:
    qs = json.loads((HERE / "questions.json").read_text())
    for q in qs:
        if q["id"] == qid:
            return q
    raise SystemExit(f"unknown question {qid!r}; known: {[q['id'] for q in qs]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pano", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--mode", choices=["2d", "3d"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent", choices=["claude", "scripted"], default="claude")
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--max-fix", type=int, default=50)
    ap.add_argument("--fov", type=float, default=12.0, help="foveal field in degrees")
    ap.add_argument("--blur", type=float, default=1.0, help="peripheral blur, Gaussian sigma in degrees")
    ap.add_argument("--start", type=float, nargs=2, default=(0.0, 0.0), metavar=("YAW", "PITCH"))
    ap.add_argument("--dup-deg", type=float, default=4.0, help="2D duplicate flag threshold (degrees)")
    ap.add_argument("--dup-m", type=float, default=0.35, help="3D duplicate flag threshold (metres, horizontal)")
    a = ap.parse_args()

    q = load_question(a.question)
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"output must be new or empty: {out}")
    (out / "steps").mkdir(parents=True, exist_ok=True)

    pano = V.Pano(a.pano, blur_deg=a.blur)
    st = V.State(fov=a.fov, mode=a.mode)
    thr = a.dup_m if a.mode == "3d" else a.dup_deg
    system = system_prompt(q, a.mode, a.fov, V.FOV_N, a.max_fix, thr)
    tool = act_tool(q["kind"])
    agent = ClaudeAgent(a.model) if a.agent == "claude" else ScriptedAgent(q["kind"], a.max_fix)
    if isinstance(agent, ScriptedAgent):
        agent.state = st
    plan = V.Plan(pano) if a.mode == "3d" else None
    peri_base = pano.blurred_rgb()
    depth_base = pano.blurred_depth_rgb() if a.mode == "3d" else None

    (out / "run.json").write_text(json.dumps({
        "question": q, "mode": a.mode, "agent": a.agent, "model": a.model if a.agent == "claude" else None,
        "max_fix": a.max_fix, "fov_deg": a.fov, "blur_sigma_deg": a.blur, "start": a.start,
        "dup_threshold": thr, "pano": str(Path(a.pano).resolve()), "pano_meta": pano.meta,
        "system_prompt": system, "tool": tool, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1) + "\n")

    gaze = (G.wrap_yaw(a.start[0]), float(np.clip(a.start[1], -89.9, 89.9)))
    notes, feedback, answer = "(empty)", "This is your first fixation.", None
    totals = {"input_tokens": 0, "output_tokens": 0}
    log_lines = []
    steps_f = (out / "steps.jsonl").open("w")
    for k in range(1, a.max_fix + 1):
        st.gazes.append(gaze)
        imgs = []
        peri = V.periphery_image(peri_base, st, gaze)
        fov_im = V.fovea_image(pano, st, gaze)
        peri.save(out / "steps" / f"{k:03d}_periphery.jpg", quality=88)
        fov_im.save(out / "steps" / f"{k:03d}_fovea.jpg", quality=92)
        imgs.append(("Image: PERIPHERAL view (blurred panorama).", "image/jpeg", V.jpeg_bytes(peri)))
        imgs.append((f"Image: FOVEAL view at yaw {gaze[0]:.1f}, pitch {gaze[1]:.1f}.", "image/jpeg", V.jpeg_bytes(fov_im, 92)))
        if a.mode == "3d":
            dper = V.depth_periphery_image(depth_base, st, gaze)
            dfov = V.fovea_depth_image(pano, st, gaze)
            pim = plan.image(st, gaze, pano)
            dper.save(out / "steps" / f"{k:03d}_depth_periphery.jpg", quality=88)
            dfov.save(out / "steps" / f"{k:03d}_depth_fovea.jpg", quality=92)
            pim.save(out / "steps" / f"{k:03d}_plan.png")
            imgs.append(("Image: PERIPHERAL DEPTH (blurred depth panorama).", "image/jpeg", V.jpeg_bytes(dper)))
            imgs.append(("Image: FOVEAL DEPTH.", "image/jpeg", V.jpeg_bytes(dfov, 92)))
            imgs.append(("Image: PLAN (top view).", "image/png", V.png_bytes(pim)))

        last = k == a.max_fix
        hist = "; ".join(f"#{i + 1} ({y:.0f}, {p:.0f})" for i, (y, p) in enumerate(st.gazes))
        text = (f"QUESTION: {q['question']}\n\n"
                f"Fixation {k} of at most {a.max_fix}. Current gaze: yaw {gaze[0]:.1f}, pitch {gaze[1]:.1f}.\n"
                f"Fixations so far (yaw, pitch): {hist}\n\n"
                f"Feedback on your last action: {feedback}\n\n"
                f"Your working notes:\n{notes}\n")
        if q["kind"] == "count":
            text += f"\nMarks so far: {len(st.live_marks())} ({', '.join('#' + str(m.id) for m in st.live_marks()) or 'none'}).\n"
        if last:
            text += "\nThis is your LAST fixation: you must give your answer now.\n"

        resp, usage = agent.act(system, tool, text, imgs)
        for key in totals:
            totals[key] += usage.get(key, 0) or 0

        # apply merges, then marks, then the move
        new_marks, fb, moved = apply_action(st, resp, gaze, k, pano, thr)
        answer = resp.get("answer")
        if isinstance(agent, ScriptedAgent):
            agent.flags = [(int(f.split("nearest is #")[1].split()[0]), m.id)
                           for m, f in zip(new_marks, [x for x in fb if x.startswith("mark #")]) if "DUPLICATE" in f]
        rec = {"fixation": k, "gaze": [round(gaze[0], 3), round(gaze[1], 3)], "observation": resp.get("observation"),
               "reason": resp.get("reason"), "notes": resp.get("notes"), "marks_in": resp.get("marks"),
               "merges_in": resp.get("merges"), "new_marks": [m.to_json() for m in new_marks],
               "feedback": fb, "next_gaze": None if moved is None else [round(moved[0], 3), round(moved[1], 3)],
               "answer": answer, "live_marks": len(st.live_marks()), "usage": usage}
        steps_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        steps_f.flush()
        log_lines.append(f"#{k} gaze ({gaze[0]:.1f}, {gaze[1]:.1f}): saw: {resp.get('observation')} | reason: {resp.get('reason')}"
                         + (f" | marks: {[m.id for m in new_marks]}" if new_marks else ""))
        print(f"[{out.name}] #{k:02d} ({gaze[0]:6.1f},{gaze[1]:6.1f}) {str(resp.get('reason'))[:90]}", flush=True)
        notes = resp.get("notes") or notes
        feedback = "; ".join(fb) if fb else "none"
        if answer:
            break
        gaze = moved
    steps_f.close()

    # the explanation
    sp = V.periphery_image(peri_base, st, st.gazes[-1])
    sp.save(out / "scanpath.png")
    etext = EXPLAIN.format(question=q["question"], answer=answer or "(no answer)", log="\n".join(log_lines))
    raw, eu = agent.explain(system, etext, [("Image: your scanpath on the blurred panorama.", "image/png", V.png_bytes(sp))])
    summary, expl = parse_explanation(raw)
    (out / "explanation.md").write_text(f"# {q['question']} ({a.mode.upper()})\n\n**Answer.** {answer}\n\n"
                                        f"**Summary.** {summary}\n\n## Explanation\n\n{expl}\n")
    final = {"answer": answer, "fixations": len(st.gazes), "gazes": st.gazes,
             "marks": [m.to_json() for m in st.marks], "count_live_marks": len(st.live_marks()) if q["kind"] == "count" else None,
             "summary": summary, "tokens": totals, "explain_tokens": eu,
             "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "final.json").write_text(json.dumps(final, indent=1, ensure_ascii=False) + "\n")
    print(f"[{out.name}] done: {len(st.gazes)} fixations; answer: {answer}", flush=True)


if __name__ == "__main__":
    main()
