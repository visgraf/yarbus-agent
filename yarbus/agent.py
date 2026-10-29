"""The agent: prompts, the one tool it acts through, the Claude client and a scripted stand-in.

Each step is a single, stateless request: the system prompt, the question, the agent's own
working notes (which it rewrites every step), the list of fixations so far, the harness's
feedback on its last action, and the current images. The agent answers by calling `act`.
"""
from __future__ import annotations

import base64
import json
import time

SYSTEM = """You are the visual system of an observer standing at a fixed spot in a room. Your head can turn in any direction but cannot move from this spot. You have two kinds of vision:

- PERIPHERAL vision: a blurred 360-degree panorama of everything around you. It is too blurred to read writing or see fine detail, but shows the layout, large shapes and colours. Grid lines are every 30 degrees; yaw runs from -180 to 180 left to right (0 = straight ahead, positive = to your right, +-180 = behind you); pitch runs from 90 (up) to -90 (down).
- FOVEAL vision: a sharp {fov:g} x {fov:g} degree image at your current gaze. It is the only way to see detail. The fovea image is {n} x {n} pixels; pixel (0,0) is its top-left corner, x grows to the right and y downwards (the tick labels on its border give pixel coordinates).

On the panorama, your fixations so far are drawn as numbered orange dots joined by your scanpath, and the current fovea is the red outline.
{mode_text}
You have a budget of at most {max_fix} fixations, but the aim is to answer the question well; stop as soon as you are confident. At each fixation you call the tool `act` once: say what you see in the fovea, update your working notes (they are your only memory of the details you have seen, so keep everything you need), and either choose the next gaze or give your answer. Gaze directions are absolute (yaw, pitch) in degrees. Choose each next fixation deliberately and say why in `reason`."""

MODE_2D = """
You see colour only (no depth)."""

MODE_3D = """
You also see DEPTH: a second panorama and a second fovea image show distance from your eyes, colour-coded (dark purple = near, yellow = far; legend in metres), and the fovea's depth image prints the distance at its centre. A PLAN image shows the room from above, built from the depth panorama: you are the blue triangle at the centre, forward is up, grey is everything between 0.1 m and 1.3 m above the floor (furniture), light grey is 1.3 m to 2.4 m (walls), the grid is in metres, and the red wedge is your current gaze."""

COUNT_TEXT = """
This question asks you to count {target}s. Whenever the fovea shows a {target} that is not yet marked, add a mark for it at a distinctive point of that {target} (for example the middle of its seat), in fovea pixel coordinates. Each mark gets a number and stays drawn on every view (panorama{plan_word} and fovea) as a green marker, so you can see what you have already counted. Never mark the same {target} twice. A {target} that is only partly visible still counts once if you can identify it. After each mark the harness reports {sep_text}. If two marks belong to the same {target}, merge them. Your answer must state the number."""

SEP_2D = "its direction and the angle in degrees to the nearest existing mark (marks closer than {thr:g} degrees are flagged as possible duplicates)"
SEP_3D = ("its position in metres and the horizontal distance to the nearest existing mark (a {target} is about "
          "0.45 m wide; marks closer than {thr:g} m are flagged as possible duplicates)")

READ_TEXT = """
This question asks you to read text. Transcribe what you read into your notes as you go, keeping the layout (lines, columns). Your answer is the transcription."""


def system_prompt(q: dict, mode: str, fov: float, n: int, max_fix: int, thr: float) -> str:
    mode_text = MODE_3D if mode == "3d" else MODE_2D
    if q["kind"] == "count":
        sep = (SEP_3D if mode == "3d" else SEP_2D).format(thr=thr, target=q["target"])
        mode_text += COUNT_TEXT.format(target=q["target"], sep_text=sep,
                                       plan_word=", plan" if mode == "3d" else "")
    else:
        mode_text += READ_TEXT
    return SYSTEM.format(fov=fov, n=n, max_fix=max_fix, mode_text=mode_text)


def act_tool(kind: str) -> dict:
    props = {
        "observation": {"type": "string", "description": "What the fovea shows now (brief)."},
        "notes": {"type": "string", "description": "Your updated working notes. They replace the previous notes. Keep under 1500 characters."},
        "reason": {"type": "string", "description": "Why you choose the next gaze (or why you answer now). One or two sentences."},
        "next_gaze": {"type": "object", "description": "The next fixation, absolute degrees. Omit when answering.",
                      "properties": {"yaw": {"type": "number"}, "pitch": {"type": "number"}},
                      "required": ["yaw", "pitch"]},
        "answer": {"type": "string", "description": "Your final answer to the question. Give it only when you stop."},
    }
    if kind == "count":
        props["marks"] = {"type": "array", "description": "New, not-yet-marked targets visible in the current fovea, in fovea pixel coordinates.",
                          "items": {"type": "object", "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                                    "required": ["x", "y"]}}
        props["merges"] = {"type": "array", "description": "Pairs of existing marks that are the same target: mark `drop` is merged into mark `keep`.",
                           "items": {"type": "object", "properties": {"keep": {"type": "integer"}, "drop": {"type": "integer"}},
                                     "required": ["keep", "drop"]}}
    return {"name": "act", "description": "Report what you see, update your notes, and choose the next gaze or answer.",
            "input_schema": {"type": "object", "properties": props, "required": ["observation", "notes", "reason"]}}


EXPLAIN = """The run is over. Above is your whole scanpath on the blurred panorama, and below is the log of every fixation with the reason you gave at the time.

Question: {question}
Your answer: {answer}

Log:
{log}

Explain how you chose this scanpath: the overall strategy, how it followed from the question, and the role of particular fixations (cite their numbers). Be faithful to the reasons in the log; where a fixation turned out unnecessary or a mistake, say so. Reply in exactly two sections:
SUMMARY: at most 60 words, for a caption in a video.
EXPLANATION: at most 400 words."""


# ---------------------------------------------------------------- Claude
class ClaudeAgent:
    # Claude Opus 5.5 always thinks (adaptive) and the thinking counts against max_tokens, so the
    # ceiling must leave room for it; it also rejects a forced tool_choice, so `act` is requested
    # with "auto" (the system prompt names the tool) and a reply without it is re-sent unchanged.
    NO_TOOL_RETRIES = 2

    def __init__(self, model: str, max_tokens: int = 16000):
        import anthropic
        self.client = anthropic.Anthropic(max_retries=6)
        self.model = model
        self.max_tokens = max_tokens

    @staticmethod
    def _content(text_before: str, images: list[tuple[str, str, bytes]]) -> list[dict]:
        content = [{"type": "text", "text": text_before}]
        for label, media, data in images:
            content.append({"type": "text", "text": label})
            content.append({"type": "image", "source": {"type": "base64", "media_type": media,
                                                        "data": base64.b64encode(data).decode()}})
        return content

    def act(self, system: str, tool: dict, text: str, images) -> tuple[dict, dict]:
        t0 = time.time()
        content = self._content(text, images)
        tin = tout = 0
        misses = []
        for _ in range(1 + self.NO_TOOL_RETRIES):
            r = self.client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system, tools=[tool],
                tool_choice={"type": "auto", "disable_parallel_tool_use": True},
                messages=[{"role": "user", "content": content}])
            tin += r.usage.input_tokens
            tout += r.usage.output_tokens
            out = next((b.input for b in r.content if b.type == "tool_use" and b.name == "act"), None)
            if out is not None:
                return dict(out), {"input_tokens": tin, "output_tokens": tout,
                                   "seconds": round(time.time() - t0, 2), "stop_reason": r.stop_reason,
                                   "no_tool_retries": len(misses), "no_tool_replies": misses}
            misses.append({"stop_reason": r.stop_reason,
                           "text": "".join(b.text for b in r.content if b.type == "text")[:500]})
        raise RuntimeError(f"no act call in {len(misses)} replies: {misses}")

    def explain(self, system: str, text: str, images) -> tuple[str, dict]:
        r = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                        messages=[{"role": "user", "content": self._content(text, images)}])
        txt = "".join(b.text for b in r.content if b.type == "text")
        return txt, {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens}


# ---------------------------------------------------------------- scripted stand-in (tests only)
class ScriptedAgent:
    """Exercises the plumbing without a model: a fixed sweep, a mark at the fovea centre where
    the harness says something is near, merges of flagged duplicates. Not a vision policy."""

    def __init__(self, kind: str, max_fix: int):
        self.kind = kind
        if kind == "read":
            self.path = [(y, p) for p in (10, 0) for y in (-20, -8, 4, 16)]
        else:
            self.path = [(y, -25.0) for y in range(-150, 181, 30)] + [(0, -40), (8, -40)]
        self.path = self.path[:max_fix]
        self.k = 0
        self.state = None        # set by the harness

    def act(self, system, tool, text, images):
        self.k += 1
        out = {"observation": f"scripted fixation {self.k}", "notes": f"scripted notes after {self.k} fixations",
               "reason": "scripted sweep (test plumbing only)"}
        if self.kind == "count":
            out["marks"] = [{"x": 256, "y": 300}] if (self.k % 2 == 0 or self.k > len(self.path) - 2) else []
            dup = [m for m in getattr(self, "flags", [])]
            out["merges"] = [{"keep": a, "drop": b} for a, b in dup]
        if self.k > len(self.path):
            out["answer"] = "scripted answer" if self.kind == "read" else f"{len(self.state.live_marks())} (scripted)"
        else:
            y, p = self.path[self.k - 1]
            out["next_gaze"] = {"yaw": y, "pitch": p}
        return out, {"input_tokens": 0, "output_tokens": 0, "seconds": 0.0, "stop_reason": "scripted"}

    def explain(self, system, text, images):
        return ("SUMMARY: A scripted sweep used to test the pipeline; no model chose these fixations.\n"
                "EXPLANATION: The scripted agent follows a fixed list of gazes and marks the fovea centre "
                "on every second fixation. It exists only to exercise rendering, logging and the video."), {}


def parse_explanation(txt: str) -> tuple[str, str]:
    s, e = txt.find("SUMMARY:"), txt.find("EXPLANATION:")
    if s < 0 or e < 0:
        return txt.strip()[:400], txt.strip()
    return txt[s + 8:e].strip(), txt[e + 12:].strip()


def dumps(o) -> str:
    return json.dumps(o, ensure_ascii=False)
