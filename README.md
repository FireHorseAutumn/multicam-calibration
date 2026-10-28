# multicam-calibration (C1–C4 — Stage B, the pipeline as library)

Vendor-independent multi-camera calibration core, built around the Tier 1
kernels (batched-tiny-linalg, gpu-bundle-adjustment). Synthetic/BAL/TUM-driven.
No vendor SDK anywhere.

## Components
- **C1. Calibration core** — own ChArUco detect (no OpenCV; CPU/Eigen,
  CUDA seams marked) -> corner cache -> intrinsics (Zhang + LM) ->
  extrinsics via custom LM BA (Schur, dense reduced camera system).
  COLMAP exporter unlocks nerfstudio/3DGS downstream.
- **C2. Point-cloud toolkit** — CUDA transform + voxel grid + voxel-covariance
  accumulation -> per-voxel normals -> GPU point-to-plane ICP.
- **C3. Plane-sweep MVS stack** — self-contained, host-callable; shipped WITH
  its negative result (learned stereo beat classical SAD MVS 15x at wide
  baselines; physics floor sigma_z = z^2/(f*b)*sigma_d).
- **C4. Hand-eye calibration AX=XB in Ceres** — small add, robotics audience.

## Data-model scars to not repeat
- Corner cache sorted-by-ID as a format invariant (Python/C++ ordering bug).
- Static-extrinsics data model, versioned JSON (stage2<->stage3 mismatch bug).
- Metric-scale provenance explicit in output (normalized-translation trap).

## Layout
- `include/mcc/` — corner cache, extrinsics types, input-adapter interface
- `src/`         — adapters, calibration driver, ICP, MVS (.cu)
- `tests/`       — synthetic-data end-to-end calibration check
- `data/`        — TUM RGB-D fetch, synthetic scene generator

## Standalone
This repo builds and runs with NO sibling repos and no archive code.
- Vendored header: `include/mcc/vendor/btl/cholesky.cuh` (from the
  batched-tiny-linalg sibling) — byte-identical copy, provenance in
  VENDORING.md. C1 code includes the adapter `include/mcc/batched_solve.hpp`,
  never the vendored header directly.

## Vendored btl A/B (2026-10-27, Ryzen 9 8940HX + RTX 5060 Laptop, machine idle)

Schur board-block solve (6x6 SPD, 25 rhs per system), median wall time,
`benchmarks/bench_solve_ab`:

| systems | Eigen LDLT | btl GPU  | slowdown |
|--------:|-----------:|---------:|---------:|
|      14 |    11.1 us |   54.0 us |    4.85x |
|     128 |   103.3 us |  335.5 us |    3.25x |
|    1024 |   848.5 us | 1722.8 us |    2.03x |

BA end-to-end (perturbed init, median):

| rig               | Eigen LDLT | btl GPU  |
|-------------------|-----------:|---------:|
| 2 cams x 14 frames |    1.40 ms |  1.93 ms |
| 4 cams x 120 frames|   20.17 ms | 23.69 ms |

Outcome: **Eigen stays the default** (`BoardSolveBackend::eigen`); btl is
opt-in via `BAConfig::board_solve`. The GPU path loses at every tested
scale — transfer + host-side packing overhead dominates the N=6 kernels.
The ratio trends toward 1 at 1024 systems, so any crossover sits beyond
realistic frame counts for this pipeline. Correctness is equivalent: same
iteration counts, final rms agrees to < 2e-15 px, adapter matches the Eigen
reference to 1.1e-16 (tests/test_ba_gpu_solve.cpp).
- No ZED SDK, no SVO formats, no vendor serials. The input-adapter interface
  (image dir / ROS bag / TUM / synthetic) replaces the whole vendor seam.

## Status / TODO (in order)
- [x] `include/mcc/types.hpp` — Pose (camera-from-world), Intrinsics
      (OpenCV-order 5-param Brown-Conrady), SE(3) boxplus
      (right-composition; all LM Jacobians match it)
- [x] `include/mcc/corner_cache.hpp` + `src/corner_cache.cpp` — sorted-by-ID
      invariant, in-memory only; serialization deferred (schema TBD)
- [x] `src/board.cpp` — own deterministic 6x6 dictionary generator +
      ChArUco board model (no OpenCV bits)
- [x] `src/detector.cpp` — own ChArUco detector (adaptive threshold,
      junction-separating erosion, Suzuki-Abe quads, convex-hull +
      diameter-anchored Douglas-Peucker, line-fit corner snapping, DLT
      warp, Otsu decode, sub-pixel refine, corner interpolation).
      CPU/Eigen; CUDA seams marked. Exercised via `tests/test_detector.cpp`
      + `tests/board_renderer.hpp`: full recall (27/27 markers, 40/40
      corners) on frontal, tilted, and close/steep synthetic views of
      standard contiguous boards, max corner error ~0.9 px vs ground
      truth, clean rejection on blank/noise/occluded/garbled/
      frame-clipped inputs. Fix history in NOTES.md.
- [x] `src/intrinsics.cpp` — Zhang closed form + full LM refinement,
      analytic Jacobians
- [x] `src/pose_init.cpp` — planar-DLT solveBoardPose, pairwise edges,
      Markley quaternion averaging, Kruskal-best spanning tree
- [x] `src/bundle_adjustment.cpp` — custom LM, Huber, Schur over board
      blocks, dense reduced camera system (<= 32 cams), no Ceres
- [x] `src/colmap_export.cpp` — COLMAP text export (OPENCV model)
- [x] `include/mcc/batched_solve.hpp` + `src/batched_solve.cu` — batched 6x6
      SPD solve adapter over the vendored btl kernels; wired into BA as
      `BoardSolveBackend::btl_gpu` (opt-in; Eigen LDLT stays default after
      the A/B above)
- [ ] `include/mcc/input_adapter.hpp` — one interface: image dir / ROS bag / TUM / synthetic
- [ ] `src/icp.cu` — voxel-covariance + point-to-plane
- [ ] `src/plane_sweep.cu` — MVS stack + negative-result numbers
- [ ] `src/hand_eye.cpp` — AX=XB in Ceres (or own LM)
- [x] `tests/test_synthetic_e2e.cpp` — recovers known synthetic extrinsics
      to ~1e-14 m / intrinsics to ~1e-11 px from analytically projected
      corners (zero noise). Scope: numeric pipeline only (detector coverage
      is in the two tests below)
- [x] `tests/board_renderer.hpp` + `tests/test_detector.cpp` — hand-written
      synthetic board-image renderer (inverse ray-cast, zero-distortion
      pinhole, supersampled; no OpenCV) driving the detector: positive views
      vs ground truth, rejection paths (blank/noise/occluded/garbled/
      frame-clipped/invalid args), stride handling
- [x] `tests/test_colmap_export.cpp` — COLMAP export smoke test: render ->
      detect -> cache -> export -> parse-back round-trip, failure paths,
      k3-drop note
- [x] `tests/test_ba_gpu_solve.cpp` — btl adapter unit checks + BA through
      both solve backends (same recovery gates, rms agreement)
- [x] `benchmarks/bench_solve_ab.cpp` — solve-path A/B (numbers in README)
- [x] FIRST COMPILE done (2026-10-10, g++-16 + nvcc 13.2 host g++-15):
      green, zero warnings; ctest synthetic_e2e PASS
