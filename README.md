# Snowboard Approach Analyzer

Offline analysis of snowboard jump approaches, from drop-in to the takeoff lip. It shows
per-frame biomechanics on a frame-accurate timeline next to the footage. The design is in
[docs/architecture.md](docs/architecture.md) and the requirements are in
[docs/requirement.md](docs/requirement.md).

## Quick start

```sh
uv pip install --python .venv/bin/python -e ".[dev]"
.venv/bin/sbanalyze process videos/*.MOV      # ~1 min per clip on Apple silicon
.venv/bin/sbanalyze serve                     # http://127.0.0.1:8000
```

On the first run, rtmlib downloads the YOLOX-m and RTMPose-m (HALPE-26) ONNX weights into
`~/.cache/rtmlib`. After that, everything runs offline. Results are written to `data/`, which is
gitignored:

```
data/catalog.sqlite            sessions, runs, clips, run windows, phase spans, QA
data/metrics/{clip}.parquet    long format: one row per metric per frame, with err_est + validity
data/frames/{clip}/NNNNN.jpg   run-window frame cache, 1080 px long edge
data/overlay/{clip}.json       per-frame skeleton / board / CoM geometry, drawn on canvas
data/logs/pipeline.jsonl       per-stage timings
```

## Pipeline

`ingest → trim → detect → calib → metrics → phase → fuse → render/store`

| Stage | What it does |
| --- | --- |
| ingest | Decodes with PyAV, applies the `rotation` side data explicitly, and keys frames on PTS. The clips are variable-rate. |
| trim | Builds a median background, then measures motion energy restricted to the mat mask. The run window is the longest sustained stretch of motion. |
| detect | Runs YOLOX person detection on a crop around the tracked rider, then RTMPose HALPE-26 on the full-resolution frame. It does not use background subtraction, because the floodlit shadow is rider-sized. |
| calib | Session-scoped. The mat's left edge comes from the background, and scale comes from the rider (see below). |
| metrics | Gap-fills, then applies an 8 Hz zero-phase Butterworth filter, then derives values. Speed uses a Savitzky–Golay derivative. Every value carries `err_est` and a validity flag. |
| phase | Phases are terrain. A frame's phase is where the rider's feet are relative to the session's terrain rows. A board-yaw state machine and a speed-profile split are fallbacks. The source of the split is recorded. |
| fuse | Selects the better source per metric when a second view exists. With one view it returns its input unchanged, and a test checks this. |

Use `sbanalyze process --reuse-poses` to recompute metrics and phases from cached poses in
about 2 s instead of about 30 s.

### Calibration: deviations from the architecture

The architecture assumes a ground homography fitted to the tile grid. On this venue the mat is
**not planar**: it has a steep in-run, a curve, then the kicker face up to the lip. A single
homography would be wrong over most of the run. Calibration uses these instead:

- **Mat edge.** The mat is segmented in the median background, and its left boundary gives a row→x
  map. `line_offset` is measured from it.
- **Terrain rows.** A straight 3D edge projects to a straight image line, so the bends in the mat
  edge are bends in the terrain. The edge's image slope is constant down the straight in-run,
  relaxes through the curve, and is vertical on the kicker face. Those two changes mark
  drop-in → transition → takeoff run. Because the rows are fixed per session, the boundaries sit
  in the same place for every run, which makes runs comparable.
- **Range and scale.** Each frame's body size gives `range = focal_px × stature / rider_px_h`.
  `rider_px_h` is the larger of the full-chain and torso-chain estimates, because foreshortening
  only ever shortens them. A robust smoothing spline maps the feet's image row to range. Range,
  and hence speed, is read off the feet, so it is immune to crouching and to the pop at the lip.

The architecture's yaw-driven phases assume a rider who starts across the fall line and turns.
The sample riders point down the fall line from the start: board yaw stays under 25°. So yaw
is the fallback, and `com_foreaft` is flagged invalid whenever board yaw is under 20°,
because it can't be observed from this camera then.

**Systematic uncertainty.** `err_est` covers keypoint noise only. Every value in metres,
including speed, scales with `rider.stature_m` and `camera.focal_px` in `config.toml`, and with
posture foreshortening. Assume ±10 % on those. Angles are unaffected. Speed on the kicker face is
flagged *degraded*: the path rises toward the camera's line of sight there, and the pop confounds
body scale.

## Importing videos

Click **＋ Import videos** in the header.

- **Choose a folder:** use the folder picker or file picker, or drag and drop a folder. The
  videos are uploaded into `data/videos/`.
- **Browse this computer:** navigate to a folder and process its videos in place, without
  copying.

Videos that are already imported are flagged. Processing runs in a background queue with live
progress, and new runs appear in the run list when they finish. The API behind it is
`PUT /uploads/{name}`, `GET /browse`, `POST /jobs` and `GET /jobs`.

`camera.focal_px` is for a 4K frame and is scaled to each clip's resolution. A clip shot with
the 0.5× or 2× lens needs its own focal length. The `g` chip (gravity refitted from the flight)
flags when the scale looks off.

## Flight, sudden movements, playback

- **Flight.** Tracking continues past the lip into an **Air** phase. The architecture scoped
  flight out; this adds it.
  - *Takeoff* is the first sustained frame with the feet above the lip row.
  - *Touchdown* is when the fall is arrested: the sink rate collapses after its peak.
  - *Air time* runs from takeoff to touchdown.
  - *Jump height* is the board's apex above the lip. Horizontal speed is carried off the lip
    unchanged, and camera pitch separates height from moving away. Pitch comes from the
    vertical vanishing point of background structures. The error bar includes ±2° of pitch.
  - As a check, `g` is refitted from the observed descent. Near 9.8 means the geometry is
    consistent. A warning chip appears outside 7.5–12.5.
- **Sudden movements** (`events.py`). CoM up or down, CoM sideways, the board sliding sideways,
  board bumps, knee snaps, trunk jerks, board twists, and sudden speed changes.
  - A frame is flagged when a signal's rate exceeds both an absolute threshold and five robust
    deviations above that run's typical rate.
  - Changes smaller than the error band or larger than physically possible are dropped (for
    example, a pose-model left/right swap mid-rotation).
  - Events are shown as red bands on the charts, ticks on the scrubber, a badge on the video,
    and a clickable list. Each entry has a ▶ 0.25× slow-motion replay.
- **Follow cam.** The main view is a crop cut from the full-resolution 4K frame, with the
  rider's box kept centred. The display cache would be mush at the lip: the rider is about
  57 px tall there. A small inset shows the whole frame. Press `f` or click the inset to swap.
  - When a sudden movement is on screen, the rider's box turns red (strong) or amber
    (moderate) and is labelled.
  - The body part involved is highlighted: the leg for a knee snap, the torso for a trunk jerk,
    the board for slides, bumps and twists. For CoM events, the CoM marker gets an arrow
    showing the direction.
  - *Slow-mo on events* (on by default) drops playback to 0.25× through each event.
- **Edges.** *CoM toe (+) / heel (−)* is the CoM's offset across the board, signed by
  stance. When it crosses zero, that's an **edge change**, which is detected and reported with
  how far the CoM moved, how long it took, and how close to takeoff it happened.
  *Board edge angle* (+ toe / − heel) is the board's roll, read from each foot's heel→toe line
  (seen from behind, it runs across the board), corrected for camera depression.
  - Near the camera it's good to about ±2°. On the takeoff run each foot is only ~25 px
    across in 4K, so the error grows to about ±6° and the value is marked degraded there.
  - It's shown as a chart and as a heel/toe tilt gauge on the video.
- **Playback speed.** 0.2×, 0.25×, 0.5×, 0.75× or 1×, with `[` / `]` to step between them.

## Not yet implemented

- Second view (milestone M8). `fuse()` and the lip-side validity table exist, but
  `POST /runs/{id}/views` and `PATCH /runs/{id}/sync` return 501.
- `board_pitch` is not computed. From the fall-line camera, the foot keypoints cannot separate
  pitch from foreshortening.
