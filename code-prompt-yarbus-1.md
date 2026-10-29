You are working in /home/lvelho/rd/yarbus-agent, a new standalone project (its own git repo, not f3d-vision). Read README.md and CLAUDE.md first. Goal: produce yarbus.mp4, a proof-of-concept video of an AI agent repeating Yarbus's experiment in the f3d-vision Classroom, and a short REPORT.md. Never write into the f3d-vision checkout; only read its scene. The f3d-vision checkout is assumed at /home/lvelho/rd/f3d-vision; if it is elsewhere, find it (it has scenes/classroom/classroom_eye.blend) and say where.

Steps, in order. Each step's stop conditions apply. If a step fails outside the delegated fixes, stop and report.

1. Setup. `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`, then `.venv/bin/python tests/selftest.py`. It must print 19/19. Find Blender 5.2.1 (the one f3d-vision uses) and record its path.

2. Blender plumbing on the synthetic room (seconds on the GPU). The panorama script was tested in Chat with the bpy 5.2.1 Python module on CPU, not with the Blender binary or OptiX, so repeat that test here first:
   - `blender -b --factory-startup --python-exit-code 1 -P tests/make_test_scene.py -- --out /tmp/yarbus_test/room.blend`
   - `blender -b /tmp/yarbus_test/room.blend --python-exit-code 1 -P blender/render_pano.py -- --auto near_board --out /tmp/yarbus_test/near --width 1024 --spp 16`
   - Expect: exit 0; rgb.png, depth.npy and meta.json written; "centre-depth check PASS"; and, opening rgb.png, the board at the image centre. If the board appears at the image edge, or the image is mirrored (the chalk text reads backwards), the yaw convention is wrong. In that case diagnose and fix render_pano.py's camera matrix only, so that the centre faces the board and +yaw is to the right. Re-run until it passes.
   - Delegated: fixes to render_pano.py for Blender 5.2 API differences (property names, enum values, pass names). Describe each fix. Nothing else.

3. Classroom placement (dry runs, seconds). Run render_pano.py with `--dry-run` for `--auto near_board` and `--auto back_of_room` on CLASSROOM/classroom_eye.blend. Record positions, the board/floor frame and clearance. Then render a preview of each (`--width 2048 --spp 32`, into panos/preview_*) and look at both:
   - near_board: the board is centred and large, no object stands between the head and the board, and the head is not inside furniture;
   - back_of_room: most of the room's chairs are visible ahead.
   - Delegated: if a check fails (clearance refused, view blocked, the floor object 'sol' or board 'blackBoard' named differently), adjust only --board-distance (2.0–3.5 m), --back-margin (0.5–1.5 m), the object names, or give --position near the same spot. Record what you changed and why. Keep --height 1.6.

4. Full panoramas (minutes each; report the measured time): `--width 8192 --spp 512` for both, into panos/near_board and panos/back_of_room. Then the board legibility check: `.venv/bin/python peek.py --pano panos/near_board --sweep -40 40 10 --pitch P --out peek`, with P at the board's text rows (find them from the preview; try a few). Look at the sheets.
   - STOP and report if the chalk text is not legible in the 12° foveae. Changing the fovea or the head distance is Luiz's decision.
   - Also run `peek.py --gaze 0 P` and confirm the text is illegible in the periphery image.

5. Plumbing on the real panoramas, at no cost: `./run_all.sh --agent scripted --max-fix 10`, then look at stills/ (the replay frames, the plate, the counting card). Fix only layout bugs in make_video.py (text overflow, overlaps). Then delete runs/, stills/ and yarbus.mp4.

6. The four agent runs. Requires ANTHROPIC_API_KEY. If it is not set, STOP and ask Luiz. Run `./run_all.sh` (default model claude-opus-5-5, at most 50 fixations). If the model id is refused, STOP and report the error; don't choose another model. Don't edit prompts, tool schema or thresholds between runs. One run per cell; no reruns to get a nicer result. If a run crashes, fix the crash (harness bugs are delegated, prompt changes are not), delete that run's folder and rerun only that cell. Say so in the report.

7. Report. Write REPORT.md with:
   - the Blender path, the head positions (world m) and any placement changes, and the render times;
   - per run: fixations, the answer, the live mark count for chairs, input and output tokens (from final.json; these are MEASURED), wall time;
   - the video's duration and file size;
   - anything surprising, for example an agent that never left the board, merges it made or refused, or 2D and 3D behaving alike;
   - every fix you made and why.
   Then `git add -A && git commit` in yarbus-agent (panos/, runs/, stills/ and the mp4 are gitignored, so copy each run's final.json and explanation.md into report/<run>/ before committing). Do not push. Send back REPORT.md's text.
