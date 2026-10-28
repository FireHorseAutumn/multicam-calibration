# multicam-calibration — decisions & novel-feature notes

## Current status (Oct 27, 2026)

The Sep 29 "never compiled" statement below is historical. C1 compiled, the
noise-free numeric e2e passed, and the detector is now exercised
(`tests/test_detector.cpp` + `tests/board_renderer.hpp`, hand-written board
renderer, Eigen only per decision 1): full recall on frontal, tilted, and
close/steep synthetic views (27/27 markers, 40/40 corners, max corner error
~0.9 px vs ground truth), clean rejection on blank/noise/occluded/
garbled/frame-clipped inputs with zero false positives. Standard contiguous
boards are supported (no printed seams needed). COLMAP export is
covered by `tests/test_colmap_export.cpp` (parse-back round-trip, failure
paths, k3-drop note). Next: input adapters -> JSON schema -> novel features.

Two detector findings from first exercise, and the fixes:
- The 8-connected contour tracer merged diagonally-touching black squares:
  a contiguous board rendered undetectable (0–1 of 27 markers). Junction
  anti-aliasing pixels classify dark when the local mean is high, and the
  tracer is 8-connected, so king-move-adjacent squares become one blob.
  FIXED: 1-px 8-neighborhood foreground erosion after the adaptive
  threshold (distinct 4-connected components are then never 8-adjacent;
  the eroded corner pixel is recovered by the grayscale sub-pixel
  refinement). 27/27 on frontal and tilted contiguous boards. Renderer
  supersampling went 2 -> 4: hard 1-px AA steps classify inconsistently at
  junctions; real optics blur similarly.
- FIXED: close/steep recall. Was: `polygonal_epsilon` (0.04 * perimeter)
  gives Douglas-Peucker quad corners ~12 px of slack at close range,
  breaking the exact 6/8 marker-inside-square cell alignment, and
  threshold notches at junctions add spurious DP vertices (20/27 markers
  at 0.45 m, 25 deg yaw pre-erosion; 19/27 with erosion alone). Fix, all
  inside the quad stage: convex hull of the contour first (junction
  notches are concave and drop out), DP anchored on the hull's diameter
  pair (the old anchor kept the arbitrary trace-start point), quad
  vertices snapped to the intersection of total-least-squares side line
  fits, and a cornerSubPix pass on the grayscale image before the warp
  (the binarized blob edge sits +-1-2 px off the true edge; the decode
  grid needs the true outline). After: 27/27 at 0.45 m / 25 deg yaw;
  max charuco corner error 0.9 px. Decision 8 intact: every rejection
  path still explicit, zero false positives on noise.

Same day, later: `btl/cholesky.cuh` vendored from batched-tiny-linalg
(commit 3bb4194, provenance in VENDORING.md) behind the
`include/mcc/batched_solve.hpp` adapter — first real dependency-graph edge.
Wired into BA as `BoardSolveBackend::btl_gpu` (opt-in). The A/B
(`benchmarks/bench_solve_ab.cpp`, table in README) is a LOSS at every tested
scale (4.85x slower at the e2e rig's 14 systems, still 2.03x at 1024);
Eigen LDLT remains the default. Correctness through the adapter is exact
(ctest `ba_gpu_solve`). Revisit if frame counts grow by ~100x or the adapter
moves the whole LM step to the GPU.

## Sep 29 decisions and historical state

Written 2026-09-29. Captures the design decisions and the novel-feature space.
At that time the calibration core was written but never compiled or tested;
all modules were desk-checked only.

Files: headers in `include/mcc/`, impl in `src/`, target `mcc_core`.

## Design decisions (with reasons)

1. **Lean C++20 + Eigen only.** No OpenCV, no Ceres, no vendor SDK. OpenCV
   and Ceres were not installed and are now permanently excluded — own
   detector, own LM.
2. **Own ChArUco detector** (`src/detector.cpp`), own deterministic 6x6
   dictionary generator (`src/board.cpp`, max-Hamming seeded search). NOT
   OpenCV bit-compatible — boards must be printed from our generator.
3. **Pose = camera-from-world** (`pc = q*pw + t`), updates via
   right-composition `poseBoxplus` (q' = q*dq, t' = t + q*dt). ALL LM
   Jacobians are the boxplus-exact tangent `[-R[pw]_x | R]`. A left/right
   perturbation conflict surfaced in integration and was unified on this.
4. **Planar-exact pose init** (homography decomposition, not iterative
   PnP) — kills the archived stage-2 mirrored-minimum failure mode at init.
5. **Extrinsic init graph:** pairwise edges from shared frames, Markley
   quaternion averaging, Kruskal-best (lowest-RMS) spanning tree — not
   discovery-order BFS (the archived pipeline's ~30% chained-pair error
   accumulation).
6. **BA = custom LM**, analytic Jacobians, Huber, Schur-eliminate 6x6 board
   blocks, dense reduced camera system (rigs <= 32 cams → dense LDLT, no
   sparse lib). The archived pipeline's own-LM was 2035x slower than Ceres
   because it solved the full dense system; ours eliminates first.
7. **Corner cache invariant:** corners sorted ascending by id, enforced at
   insert. Archived Python/C++ ordering scar.
8. **Every detector rejection path explicit** — no silent garbage corners
   (archived TensorRT detector failure: 576 garbage corners, 100% detection
   rate, rejected).
9. **Intrinsics:** 5-param Brown-Conrady (OpenCV order), Zhang closed form
   + full LM. Hard floor on frames/coverage — the archived 14-param
   overfit scar.
10. **JSON schema: TBD.** Cache is in-memory only; no serialization written.
11. **COLMAP export:** OPENCV model, k3 dropped with non-fatal warning.

## Archived-pipeline negative results to respect (do not re-litigate)

- **17mm alignment is a physics floor**, not a bug: sigma_z = z^2/(f*b) *
  sigma_d. ICP already captures all rigid-body error; the rest is stereo
  noise. 9/9 improvement experiments failed.
- **Classical MVS as depth replacement: dead end** at 70–85 deg inter-camera
  angles (267mm vs 17.3mm). Plane-sweep only viable at moderate angles / as
  learned-cost substrate.
- **Rectified stereo collapses at wide angles** (valid ROI -> 0x0 at 70.8
  deg); plane-sweep got 25.8% valid px there, per-pixel-min fusion lifted
  4-cam 5.5% -> 28.8%.

## Novel-feature space (seed list)

Starting points, none committed:

- **Online drift correction done right**: the archived stage-3 drift work
  had the EKF + sliding-window pose graph but a stubbed neural matcher and
  innovation-gate bias slip. Ours could do it with the own-detector + BA
  core.
- **Target-free / self-calibration** from scene structure (the ICP
  pose-graph logic from refine_icp.py, ported to C++/Eigen + CUDA).
- **GPU BA** via the Tier 1 kernels (batched-tiny-linalg sibling repo,
  vendored header).
- **Wide-baseline multi-view depth**: plane-sweep stack (archived stage-4
  code was validated in isolation, never integrated) + per-pixel-min
  fusion as the classical baseline, positioned as substrate for a learned
  cost.
- **Multi-camera rig as a single virtual sensor**: synced capture,
  joint calibration, fused depth — the toolkit framing.
- Hand-eye AX=XB (small, robotics audience).

## Queue on resume

1. First compile pass + synthetic e2e test (recover known extrinsics).
2. Input adapters (image dir / ROS bag / TUM / synthetic).
3. JSON schema -> corner cache serialization.
4. Then: novel features.
