#pragma once

// Batched tiny SPD solve adapter. C1 code includes THIS header, never the
// vendored btl header (see VENDORING.md) — the backend can be swapped or
// dropped without touching callers.
//
// Current backend: vendored btl/cholesky.cuh warp-per-matrix CUDA kernel
// (N = 6 fixed: the Schur board-pose blocks of bundle_adjustment.cpp).
// Layout matches the btl contract: full 6x6 row-major per system,
// batch-index-major.

namespace mcc {

// True if a CUDA device is available for the btl backend.
bool batchedSpdSolve6Available();

// Solves A_b x_bj = rhs_bj for n independent 6x6 SPD systems on the GPU.
//   A:     n * 36 doubles, row-major 6x6 per system
//   n_rhs: n_rhs[b] right-hand sides for system b (>= 1)
//   rhs:   concatenated 6-vectors, sum(n_rhs) * 6 doubles
//   x:     same layout as rhs (output)
// Returns false if any system is not numerically SPD (per-system info from
// the btl kernel); x is then undefined. No CUDA device -> returns false.
bool batchedSpdSolve6(const double* A, int n, const int* n_rhs, const double* rhs,
                      double* x);

}  // namespace mcc
