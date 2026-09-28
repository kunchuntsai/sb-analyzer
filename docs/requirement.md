# Snowboard Approach Analyzer — Requirements

**Source:** extracted from [architecture.md](architecture.md) (Draft 1 · 2026-09-08)
**Target platform:** macOS on Apple silicon, single user, local-only

---

## 1. Purpose and scope

Import short video clips of a snowboard jump approach and produce per-frame biomechanical
metrics (joint angles, centre of mass, board kinematics). Show them on a scrubbable timeline
that stays in sync with the footage.

| In scope | Out of scope |
| --- | --- |
| Drop-in through takeoff: analysis starts when the rider drops in and ends when the board leaves the lip | Airborne phase, rotation, landing |
| Offline analysis of recorded clips | Live/streaming capture |
| Single user on a local machine | Auth, multi-tenancy, horizontal scaling |

### 1.1 View model

- A **run** is one descent.
- Each run has exactly one **primary view**: the fall-line camera at the top of the ramp.
- A run may also have one **secondary view**, filmed from beside the lip at the same time with matching timestamps.
- Every feature must work from the primary view alone. The secondary view is only a quality
  upgrade. Nothing may degrade or break when it is absent.
- The primary view defines the run timebase and the frame cache. The secondary view contributes values but never frames.

---

## 2. Functional requirements

| ID | Requirement |
| --- | --- |
| F-1 | Import a `.mov`/`.mp4` clip from local disk and register it against a session |
| F-2 | Automatically locate the run window and discard non-run frames |
| F-3 | Estimate 2D pose per frame with per-keypoint confidence |
| F-4 | Segment the run into drop-in, transition, and takeoff-run phases |
| F-5 | Map image coordinates to mat-plane metric coordinates |
| F-6 | Compute joint angles, centre of mass, and board kinematics with uncertainty |
| F-7 | Mark each metric valid or invalid according to phase geometry |
| F-8 | Present a frame-accurate timeline with charts synchronised to the footage |
| F-9 | Overlay skeleton, board axis and CoM on the footage, with each layer toggleable on its own |
| F-10 | Compare two runs on a phase-aligned common timeline |
| F-11 | Export metrics as CSV/Parquet for external analysis |
| F-12 | Accept an optional second simultaneous view and attach it to the same run |
| F-13 | Where a second view exists, select the better source per metric and record which was used |

### 2.1 Supporting functional details

| Ref | Detail | Relates to |
| --- | --- | --- |
| F-2.1 | When run-window detection confidence is low, fall back to manual in/out points | F-2 |
| F-2.2 | A clip that fails trimming is stored with status `needs_manual_window`, not discarded | F-2 |
| F-4.1 | Phase boundaries are persisted and can be corrected by hand and reused | F-4 |
| F-5.1 | Calibration is scoped to the session, not the clip | F-5 |
| F-5.2 | Calibration can be done manually (4 points on the tile grid) or automatically (Hough lines) | F-5 |
| F-6.1 | Every metric value carries `err_est` derived from rider pixel height (`3 × 1750 / rider_px_h` mm) | F-6 |
| F-7.1 | Validity follows a versioned phase validity matrix keyed on `(metric, phase, view)` | F-7 |
| F-12.1 | Two views are synchronised by container timestamp, then automatic CoM-height cross-correlation, then a manual nudge as fallback | F-12 |
| F-12.2 | The sync offset is persisted and can be overridden manually | F-12 |

---

## 3. Non-functional requirements

| ID | Requirement | Target |
| --- | --- | --- |
| N-1 | End-to-end processing of one clip | < 60 s on M-series |
| N-2 | Timeline scrub latency | < 16 ms per frame step |
| N-3 | Offline operation | No network dependency after model download |
| N-4 | Reproducibility | Same input + same config ⇒ byte-identical metrics |
| N-5 | Storage per clip | < 100 MB including frame cache |
| N-6 | Model swappability | Pose backend replaceable without touching metrics code |

---

## 4. UI / presentation requirements

| Ref | Requirement |
| --- | --- |
| U-1 | Phase bands render behind every chart and are always visible |
| U-2 | `err_est` renders as a band around each series and is always visible |
| U-3 | Where `valid` is false, the series is greyed out rather than hidden |
| U-4 | Keyboard: `←`/`→` step one frame, `,`/`.` step ten, `space` plays at real time |
| U-5 | When a second view exists, the frame viewer can toggle between views. Each series shows which camera produced it |
| U-6 | Calibration residual (`residual_px`) and mean keypoint confidence per phase are shown in the UI header |
| U-7 | Manual in/out points are offered for clips that need a manual run window |

---

## 5. Quality and operational requirements

| Ref | Requirement |
| --- | --- |
| Q-1 | Each processing run records the resolved config hash and `pipeline_version` (supports N-4) |
| Q-2 | Model weights are pinned by checksum. The ONNX Runtime thread count is fixed. No wall-clock values or random seeds are used in the metric path |
| Q-3 | Each stage returns either a result or a typed failure |
| Q-4 | Structured JSON-lines logging per stage, with frame counts and durations |
| Q-5 | Warn when calibration residual exceeds a threshold, since the tripod may have drifted |
| Q-6 | Rider stature and the segment-table choice can be configured (for youth riders) |
| Q-7 | `fuse(secondary=None)` must return its input unchanged. This is tested on every build |

---

## 6. Open questions

1. **Secondary view calibration.** Does the lip-side camera need its own ground homography?
2. **Board length.** Is it known and constant, so it can be used as a scale check?
3. **Session portability.** Should calibration be keyed to a named camera position that the user picks at import?
4. **Retention.** Should original clips be kept indefinitely, or should only the frame cache be kept after processing?
