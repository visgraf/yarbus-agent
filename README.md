# yarbus-agent

A proof-of-concept video: Alfred Yarbus's 1967 experiment, where the question asked changes the
scanpath, repeated with an AI agent that has a foveal eye. The agent stands in the f3d-vision
Classroom and answers two questions:

- (i) *What is written on the blackboard?*, with the head close to the board;
- (ii) *How many chairs are there?*, with the head at the back of the room.

The head stays at its position and only rotates. Around it, the agent sees a **blurred 360°
panorama** (the parafovea and periphery). Detail comes only from a **sharp 12° fovea** that it
points by choosing each next gaze. Each question is answered twice: once with colour only
(**2D**, an RGB panorama) and once with colour plus depth (**3D**, an RGB-D panorama). That gives
four runs. After each run the agent explains how it chose its scanpath.

This folder lives outside the f3d-vision repository. It only reads the scene
(`scenes/classroom/classroom_eye.blend`) and changes nothing in that repository.

## Files

| file | what |
|---|---|
| `blender/render_pano.py` | Blender 5.2: places the head and renders one RGB-D equirect panorama (`rgb.png`, `depth.npy`, `meta.json`) |
| `blender/exr_lite.py` | the uncompressed-EXR reader, copied unchanged from f3d-vision `tools/exr_lite.py` |
| `yarbus/geometry.py` | head-frame sphere geometry (the f3d-vision conventions), the fovea's pinhole projection |
| `yarbus/views.py` | what the agent sees: blurred periphery, fovea, depth views, the top-down plan |
| `yarbus/agent.py` | prompts, the `act` tool, the Claude client, a scripted stand-in for tests |
| `yarbus/harness.py` | applies an action: merges, new marks (lifted to 3D in the 3D variant), duplicate flags, the move |
| `run_agent.py` | one run: the loop, the logs, the explanation |
| `make_video.py` | the video (or `--still` sample frames) |
| `peek.py` | look at a panorama as the agent would (a fovea, a sweep of foveae, the periphery) |
| `questions.json` | the two questions |
| `run_all.sh` | the four runs, then stills and the video |
| `tests/selftest.py` | 19 checks that can fail (geometry, sampler, 3D lift, bookkeeping, blur) |
| `tests/make_test_scene.py`, `tests/make_standin_pano.py` | a synthetic room and a stand-in panorama for testing without the Classroom |

## How a run works

Every fixation is **one stateless request** to the model. It contains the system prompt, the
question, the agent's own working notes (it rewrites them at every step, and they are its only
memory of detail), the list of fixations so far, the harness's feedback on the last action, and
the current images:

- 2D: the periphery (the blurred panorama, 1500 × 750, with a 30° grid, the scanpath so far, the
  marks and the current fovea outline) and the fovea (512 × 512, 12°, sampled from the sharp
  panorama);
- 3D: the same two images, plus the blurred depth panorama, the fovea's depth, and a plan (a top
  view built from the depth panorama, with the marks at their metric positions).

The model answers by calling one tool, `act`. The call carries an observation, the new notes, a
reason, and either the next gaze (absolute yaw and pitch) or the answer. For counting it can also
carry new marks (fovea pixel coordinates) and merges of marks it judges to be the same chair. The
harness reports each new mark's nearest neighbour: as an angle in 2D, or as a horizontal distance
in metres in 3D. Near neighbours are flagged as possible duplicates (4° in 2D, 0.35 m in 3D). The
agent itself decides whether to merge.

When the agent answers, it gets one more request: its scanpath and the log of its reasons, with a
request to explain the strategy. That gives a 60-word summary for the video and a 400-word
explanation (`explanation.md`). The per-step `reason` fields were written at the time; the
explanation is written after the fact and may rationalize. Both are kept.

The agent is not told about Yarbus.

## Design choices and their limits

- **One panorama per head position.** The head only rotates, so one panorama holds everything the
  eyes can see from there. Foveae are resampled from it (8192 px wide, 22.8 px/deg; f3d-vision's
  full core is 21.3 px/deg), not rendered per fixation, and the loop has no stereo. That is the
  simplest route. Live foveated rendering through the playground's render server is the natural
  next step.
- **Periphery = Gaussian blur, σ = 1°, uniform over the sphere.** It is not an eccentricity-graded
  retina. `--blur` changes it. The self-test checks that 1° removes sub-degree detail such as
  chalk strokes.
- **What 3D adds.** With a fixed head, a direction alone already identifies a point, so 2D marks
  are not ambiguous by themselves. What depth adds is metric scale: a chair is about 0.45 m wide
  whether it is near or far, while its angular size changes about five-fold across the room. It
  also adds a plan view of the room's layout (rows, aisles) to sweep by. The 2D and 3D runs may
  turn out alike, and that would be a finding too.
- **Head placement** (`--auto`): `near_board` is 2.5 m out from the board's centre along its
  normal; `back_of_room` is 1.0 m in from the floor's far edge on the same line. Both are at 1.6 m
  eye height (standing), facing the board. A clearance test refuses positions inside or too close
  to geometry. `--position X Y Z` overrides placement.
- **One run per cell.** The model is stochastic, so a rerun gives a different scanpath.

## Status

Tested in Chat, without the Classroom and without any model calls:

- `tests/selftest.py` gives 19/19.
- `blender/render_pano.py` ran under the bpy 5.2.1 module (CPU) on `tests/make_test_scene.py`'s room.
  At both positions the centre-depth check gave measured = expected (2.480 m and 7.960 m). The chalk
  text reads left to right, so the panorama is not mirrored. The asymmetric teacher's chair appears at
  yaw +56° at 3.05 m (3.12 m expected at the seat centre), confirming that +yaw is right. Not yet run
  with the Blender binary, OptiX or the Classroom.
- The board text in the test room is legible in 12° foveae and illegible in the σ = 1° periphery.
- `run_agent.py --agent scripted` and `make_video.py` (stills and mp4) ran on those panoramas.

## Running it

1. Set up (once):

        python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
        .venv/bin/python tests/selftest.py                      # expect 19/19
        export ANTHROPIC_API_KEY=...

2. Render the panoramas. `CLASSROOM` is the f3d-vision `scenes/classroom` folder:

        blender -b $CLASSROOM/classroom_eye.blend --python-exit-code 1 -P blender/render_pano.py -- \
            --auto near_board --out panos/near_board --dry-run          # placement + clearance only
        blender -b $CLASSROOM/classroom_eye.blend --python-exit-code 1 -P blender/render_pano.py -- \
            --auto near_board --out panos/preview_near_board --width 2048 --spp 32
        blender -b $CLASSROOM/classroom_eye.blend --python-exit-code 1 -P blender/render_pano.py -- \
            --auto near_board --out panos/near_board                   # 8192 x 4096, 512 spp, denoised
        # the same three for back_of_room

3. Check the views:

        .venv/bin/python peek.py --pano panos/near_board --sweep -40 40 10 --pitch 5 --out peek

4. Run and assemble: `./run_all.sh` (or `./run_all.sh --agent scripted` to test the plumbing at no cost).

The outputs are `runs/<question>_<mode>/` (with `steps.jsonl`, the images the agent saw,
`final.json` and `explanation.md`), `stills/` and `yarbus.mp4`.
