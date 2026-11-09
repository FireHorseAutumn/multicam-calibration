# multicam-calibration (C1 to C4, Stage B: the pipeline as library)

Vendor-independent multi-camera calibration core, built around the Tier 1
kernels (batched-tiny-linalg, gpu-bundle-adjustment). Synthetic/BAL/TUM-driven.
No vendor SDK anywhere: no ZED SDK, no SVO formats, no vendor serials. The
input-adapter interface (image dir / ROS bag / TUM / synthetic) replaces the
whole vendor seam.

## Status

- C1 calibration core complete end to end: own ChArUco detector, Zhang + LM
  intrinsics, custom LM bundle adjustment (Schur, dense reduced camera
  system), COLMAP export. ctest synthetic_e2e green with zero warnings.
- Recorded on a Ryzen 9 8940HX + RTX 5060 Laptop, machine idle; g++-16 +
  nvcc 13.2 (host g++-15).

## Headline: the vendored btl A/B is a published loss

Schur board-block solve (6x6 SPD, 25 rhs per system), median wall time,
Eigen LDLT vs the vendored btl GPU kernels:

| systems | Eigen LDLT | btl GPU  | slowdown |
|--------:|-----------:|---------:|---------:|
|      14 |    11.1 us |   54.0 us |    4.85x |
|     128 |   103.3 us |  335.5 us |    3.25x |
|    1024 |   848.5 us | 1722.8 us |    2.03x |

Outcome: **Eigen stays the default**; btl is opt-in via
`BAConfig::board_solve`. The GPU path loses at every tested scale: transfer
+ host-side packing overhead dominates the N=6 kernels.

## Key numbers

- BA end-to-end (perturbed init, median): 2 cams x 14 frames, 1.40 ms Eigen
  vs 1.93 ms btl; 4 cams x 120 frames, 20.17 ms vs 23.69 ms.
- Correctness is equivalent across backends: same iteration counts, final
  rms agrees to < 2e-15 px, adapter matches the Eigen reference to 1.1e-16.
- Own ChArUco detector: full recall (27/27 markers, 40/40 corners) on
  frontal, tilted, and close/steep synthetic views, max corner error ~0.9
  px vs ground truth, clean rejection on blank/noise/occluded/garbled/
  frame-clipped inputs.
- Synthetic end-to-end: recovers known extrinsics to ~1e-14 m and
  intrinsics to ~1e-11 px from analytically projected corners (zero noise).

## Honest limits

- The GPU solve path is a loss at every tested scale; the ratio trends
  toward 1 at 1024 systems, so any crossover sits beyond realistic frame
  counts for this pipeline. Published, not buried.
- The synthetic e2e scope is the numeric pipeline only; detector coverage
  lives in the renderer-driven detector tests.
- C2 (ICP), C3 (plane-sweep MVS, which ships with its own negative result:
  learned stereo beat classical SAD MVS 15x at wide baselines), and C4
  (hand-eye AX=XB) are not yet implemented.

## Build and test

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Standalone: builds and runs with no sibling repos and no archive code. The
one vendored header (`include/mcc/vendor/btl/cholesky.cuh`) is a
byte-identical copy from batched-tiny-linalg, provenance in VENDORING.md;
C1 code includes the adapter `include/mcc/batched_solve.hpp`, never the
vendored header directly.

## Repository layout

- `include/mcc/`: corner cache, extrinsics types, input-adapter interface
- `src/`: adapters, calibration driver, ICP, MVS (.cu)
- `tests/`: synthetic-data end-to-end calibration check
- `data/`: TUM RGB-D fetch, synthetic scene generator

## Deeper reading

- `docs/RUNGS.md`: the full record: all four components, the data-model
  scars, both A/B tables verbatim, and the complete per-file status list.
- `NOTES.md`: detector fix history.
