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
- Optional vendored header: `btl/cholesky.cuh` now exists in the
  batched-tiny-linalg sibling, but is not wired here. If this dependency is
  taken, copy it into `include/mcc/vendor/btl/`; header-only means one file
  copy, no linking.
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
      Suzuki-Abe quads, DLT warp, Otsu decode, sub-pixel refine, corner
      interpolation). CPU/Eigen; CUDA seams marked. Compiled; not yet
      exercised (needs a board-image renderer; see test note below)
- [x] `src/intrinsics.cpp` — Zhang closed form + full LM refinement,
      analytic Jacobians
- [x] `src/pose_init.cpp` — planar-DLT solveBoardPose, pairwise edges,
      Markley quaternion averaging, Kruskal-best spanning tree
- [x] `src/bundle_adjustment.cpp` — custom LM, Huber, Schur over board
      blocks, dense reduced camera system (<= 32 cams), no Ceres
- [x] `src/colmap_export.cpp` — COLMAP text export (OPENCV model)
- [ ] `include/mcc/input_adapter.hpp` — one interface: image dir / ROS bag / TUM / synthetic
- [ ] `src/icp.cu` — voxel-covariance + point-to-plane
- [ ] `src/plane_sweep.cu` — MVS stack + negative-result numbers
- [ ] `src/hand_eye.cpp` — AX=XB in Ceres (or own LM)
- [x] `tests/test_synthetic_e2e.cpp` — recovers known synthetic extrinsics
      to ~1e-14 m / intrinsics to ~1e-11 px from analytically projected
      corners (zero noise). Scope: numeric pipeline only — the detector is
      bypassed (no image renderer yet)
- [x] FIRST COMPILE done (2026-10-10, g++-16 + nvcc 13.2 host g++-15):
      green, zero warnings; ctest synthetic_e2e PASS
