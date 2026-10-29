# yarbus-agent: first full run (2026-10-29)

Result: `yarbus.mp4` was built from four runs of claude-opus-5-5, one run per cell. Each number below
says how it was obtained. MEASURED means taken from a command or file, named next to it.
ESTIMATED means a judgement or a calculation from a price list.

## Setup

- `tests/selftest.py`: 19/19, before and after every code change (MEASURED, its output).
- Blender: `/usr/local/bin/blender` → `/opt/blender-5.2/blender`, 5.2.1 LTS, Cycles OPTIX on an RTX 4090
  (MEASURED, `blender --version` and `render.device` in meta.json).
- f3d-vision: at the expected path, `/home/lvelho/rd/f3d-vision`. Only `scenes/classroom/classroom_eye.blend`
  was read. Nothing was written there.

## Panoramas

**Synthetic room, Blender binary with OptiX.** Exit 0. rgb.png, depth.npy and meta.json were written.
Centre-depth check PASS: 2.480 m measured, 2.480 m expected (MEASURED, render_pano output). The board is
at the image centre and the chalk text reads left to right. The asymmetric teacher's chair is at
x ≈ 671 px of 1024, which is yaw +55.9°; the geometry gives +56° (atan2(2.4, 1.62)). So +yaw is to the
right. **render_pano.py needed no changes** for Blender 5.2.1.

**Classroom placement.** `--auto` defaults were kept: `--board-distance 2.5`, `--back-margin 1.0`, `--height 1.6`.
**No placement changes.** All values are MEASURED, from the render_pano output and meta.json.

| | head (world m) | to board centre | min clearance | centre-depth check |
|---|---|---|---|---|
| near_board | (0.897, 0.759, 1.600) | 2.51 m | 1.30 m | PASS, 2.500 = 2.500 m |
| back_of_room | (0.897, −3.793, 1.600) | 7.06 m | 0.80 m | PASS, 7.052 = 7.052 m |

- Frame:
  - The board 'blackBoard' has its centre at (0.897, 3.259, 1.334). It is a zero-thickness plane, 2.20 × 1.14 m.
  - The normal points to −Y, into the room.
  - The top of the floor 'sol' is at z ≈ 0. From the board, the room is 8.05 m deep.
  - The board is off-centre in the room: the floor spans x −5.67 to 3.76.
- Previews (2048 wide, 32 spp), checked by eye:
  - near_board: the board is centred, about ±24° wide, and nothing stands in front of it.
  - back_of_room: the rows of desk-chairs fill the view ahead.
- Full renders (8192 × 4096, 512 spp, denoised):

  | | render | wall, incl. scene load and PNG/depth export |
  |---|---|---|
  | near_board | 239.3 s | 282.0 s |
  | back_of_room | 235.3 s | 281.0 s |

  Render times are MEASURED from meta.json `render.seconds`; wall times from `/usr/bin/time`. Each preview rendered in 4.6 s.

**Board legibility check (peek.py).** The sweeps covered yaw −40…40 at pitch +2, −2, −6 and −10.
Every chalk row is legible in the 12° foveae: "m Mika / un chat / papa / maman",
"une amie / samedi / il y a / la classe", "Jeudi 13 septembre", and the circled syllables. The far-left
column of single letters is faint chalk in the scene itself. In the periphery image (`--gaze 0 -2`), the
board is a uniform dark rectangle with no strokes visible. The check passes, so it was not a stop.

**Reference count (MEASURED, a read-only Blender listing of objects in classroom_eye.blend).** There are 22
chairs: 20 instances of the `schoolDesk` collection, which are combined desk-chairs in 5 rows × 4, and 2
`leatherChair` instances. One leather chair is by the coat rack at the front left; the other is behind the
teacher's desk. The back_of_room head is level with the back row. The nearest chair is about 0.6 m away,
so four chairs lie beside the head at steep downward angles.

## The four runs

Model: claude-opus-5-5. At most 50 fixations, start gaze (0, 0). The prompts, tool schema and thresholds
were not changed. Each cell was run once. Token counts are MEASURED, from final.json `tokens` and
`explain_tokens`. Wall time is MEASURED, from run.json `started` to final.json `finished`, and includes the
explanation request.

| run | fixations | answer | live chair marks | tokens in / out (fixations) | tokens in / out (explanation) | wall |
|---|---|---|---|---|---|---|
| board_2d | 11 | a transcription in 3 columns (below) | – | 39,502 / 6,242 | 3,852 / 789 | 84 s |
| board_3d | 8 | a transcription in 2 sections (below) | – | 52,954 / 3,696 | 3,505 / 838 | 76 s |
| chairs_2d | 28 | **22** | 22 (22 placed, 0 merged) | 114,695 / 16,950 | 5,261 / 920 | 219 s |
| chairs_3d | 4 | **22** | **3** (3 placed, 0 merged) | 27,893 / 2,111 | 3,137 / 719 | 40 s |

- Totals: 250,799 input and 32,265 output tokens (MEASURED, sum of the above). That is about $1.65 at
  $4 / $20 per million tokens (ESTIMATED, list price).
- `./run_all.sh` took 476.7 s in all, including the stills and the video (MEASURED, `/usr/bin/time`).
- None of the 51 fixation requests needed a re-send (MEASURED, `no_tool_retries` in steps.jsonl).
- Thinking: Opus 5.5 always thinks, and its thinking tokens are included in the output counts. Effort was
  not set, so the model default applied. The API documentation gives `medium` as that default; this is
  not measured.

**Board transcriptions.** These are judged against my own reading of the peek sheets (ESTIMATED).
- 2D: all 12 words and the date, in the right three-column layout, plus the circled syllables. The faint
  left column was read as "m n l s(?)", where I read "m r l …".
- 3D: "la classe" is missing. The middle column is grouped under the date, and each left-column letter is
  paired with a word ("m – Mika", "r – un chat", …). It had fewer fixations and read less accurately.
- Both scanpaths stayed on the board and read it left to right, row by row. 2D and 3D behaved alike,
  as the README expected for reading. The 2D agent spent its last fixation checking the empty lower left.

## What was surprising

1. **chairs_3d counted from the plan, not with the fovea.** In the plan, which is a top view built from
   depth, it saw a 4 × 5 grid of desk units. Its fixations went only to the two chairs outside the grid
   and to one student seat, to check that a seat is a separate chair. The answer is "20 + 2 = 22", with
   only 3 live marks. Its explanation says this openly: "Only one of the 20 student chairs was actually
   inspected … the mark count does not match the stated answer." With depth, the plan resolves every
   chair at 7 m, so the 12° fovea is not needed for counting. Whether that is acceptable is a design
   question for Luiz.
2. **chairs_2d gave the right number for the wrong reason.** I lifted its 22 marks onto the depth
   panorama after the run:
   - All 22 lie within 0.8 m of a student desk-chair (MEASURED: depth.npy, plus the object positions).
     None is near either leather chair; each leather chair is at least 2.3 m from every desk-chair.
   - So at least 2 marks are double counts (22 marks on 20 chairs), and the 2 leather chairs were never
     marked. The answer 22 matches the truth only because the two errors cancel.
   - Matching each mark to the nearest chair origin gives 16 distinct chairs: 6 marks are second marks on
     an already marked chair, and 6 chairs have no mark (ESTIMATED, depends on where each chair's origin
     sits).
   - The double counts are on the near chairs, beside and below the head. Each near chair spans tens of
     degrees, so marks on its seat, back and desk top are 10–30° apart. That is far outside the 4°
     duplicate flag, which never fired: the closest pairs were 4.6° and 4.7°.
   - The agent merged nothing. Its explanation says it "compared positions with earlier marks to avoid
     counting any chair twice". That is a reconstruction after the fact, and the log does not support it.
3. **Effort, 2D against 3D.** 3D was cheaper in both questions: 8 against 11 fixations for the board, and
   4 against 28 for the chairs.

## Fixes made

- **`yarbus/agent.py` (harness crash, delegated).** The first launch of `./run_all.sh` failed on its first
  request with HTTP 400: `tool_choice: type "tool" and "any" are not supported for this model`. The model
  id itself was accepted. Opus 5.5 does not allow forced tool use. Three changes:
  - `tool_choice` is now `{"type": "auto", "disable_parallel_tool_use": true}`. The system prompt already
    tells the agent to call `act` once per fixation, so no prompt text changed.
  - `max_tokens` went from 2000 to 16000. Opus 5.5 always thinks, and the thinking counts against
    `max_tokens`, so 2000 risked truncated replies with no `act` call.
  - A reply without an `act` call is re-sent unchanged, at most twice, and logged (`no_tool_retries`).
    This never happened.

  The tool schema, prompts and thresholds are unchanged. I did not add `strict: true`, an explicit effort,
  or a fallback to another model. The crashed folder `runs/board_2d` had 0 fixations; I deleted it and
  re-ran `./run_all.sh`. No cell had completed before the fix, so each cell has exactly one run.
- **`make_video.py` (layout, step 5).** A stress test with long text on copies of the scripted runs
  showed two problems:
  - On the final replay frame, the 4-line answer box ran past the bottom of the 1080-px frame. On that
    frame, "sees" and "why next" now get 1 line each, and the answer gets what fits, up to 4 lines.
  - The " …" added to a truncated line could run past the text column. Words are now trimmed so that
    it fits.

## Left as they are (for Luiz)

- **The plan the agent sees is centred on the head.** The prompt says "you are the blue triangle at the
  centre". From the back of the room, half of the plan is empty.
- **The board answer box in the replay shows only the agent's opening lines**, about "three columns". The
  transcription itself is in the explanation card and in final.json.
- **Two mark labels collide on the counting card:** #19 and #22, next to the head, read as "229".
- **README "Status" is out of date.** It still says the panorama script has not been run with the Blender
  binary, OptiX or the Classroom. I did not edit the README.

## Video

`yarbus.mp4`: 137.1 s, 6,840,221 bytes (6.5 MiB), 1920 × 1080, H.264, 30 fps, 4113 frames (MEASURED,
`ffprobe` and the make_video output). Each run's `final.json` and `explanation.md` is copied in
`report/<run>/`.
