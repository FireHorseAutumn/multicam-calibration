// btl-backed implementation of the batched 6x6 SPD solve adapter.
// This is the only C1 translation unit that includes the vendored header
// (see VENDORING.md).
//
// The btl kernels take one rhs per batch element, so systems with multiple
// right-hand sides are expanded to one (system, rhs) pair per batch element
// and factored redundantly per rhs — at N = 6 the redundant factorization
// is ~100 FLOPs, far cheaper than a second launch.

#include "mcc/batched_solve.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "mcc/vendor/btl/cholesky.cuh"

namespace mcc {

namespace {

constexpr int N = 6;
constexpr int TILE = 8;  // btl tile width for N = 6 (next pow2)

struct DeviceBuffers {
  double* a = nullptr;     // expanded systems, cap * 36 doubles
  double* l = nullptr;     // factor scratch, cap * 36
  double* r = nullptr;     // rhs, cap * 6
  double* x = nullptr;     // solution, cap * 6
  int* info = nullptr;     // cap
  size_t cap = 0;

  bool ensure(size_t n) {
    if (n <= cap) return true;
    cudaFree(a);
    cudaFree(l);
    cudaFree(r);
    cudaFree(x);
    cudaFree(info);
    a = l = r = x = nullptr;
    info = nullptr;
    cap = 0;
    if (cudaMalloc(&a, n * 36 * sizeof(double)) != cudaSuccess) return false;
    if (cudaMalloc(&l, n * 36 * sizeof(double)) != cudaSuccess) return false;
    if (cudaMalloc(&r, n * 6 * sizeof(double)) != cudaSuccess) return false;
    if (cudaMalloc(&x, n * 6 * sizeof(double)) != cudaSuccess) return false;
    if (cudaMalloc(&info, n * sizeof(int)) != cudaSuccess) return false;
    cap = n;
    return true;
  }
};

// Process-lifetime device scratch; never freed (avoids cudaMalloc/free per
// LM iteration).
DeviceBuffers g_buf;

}  // namespace

bool batchedSpdSolve6Available() {
  int count = 0;
  return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool batchedSpdSolve6(const double* A, int n, const int* n_rhs, const double* rhs,
                      double* x) {
  if (!A || !n_rhs || !rhs || !x || n <= 0) return false;
  static const bool available = batchedSpdSolve6Available();
  if (!available) return false;

  int total = 0;
  for (int b = 0; b < n; ++b) {
    if (n_rhs[b] < 1) return false;
    total += n_rhs[b];
  }

  std::vector<double> a_exp(static_cast<size_t>(total) * 36);
  size_t off = 0;
  for (int b = 0; b < n; ++b)
    for (int j = 0; j < n_rhs[b]; ++j, ++off)
      std::copy_n(A + static_cast<size_t>(b) * 36, 36, a_exp.begin() + off * 36);

  if (!g_buf.ensure(static_cast<size_t>(total))) return false;
  const size_t a_bytes = static_cast<size_t>(total) * 36 * sizeof(double);
  const size_t r_bytes = static_cast<size_t>(total) * 6 * sizeof(double);
  if (cudaMemcpy(g_buf.a, a_exp.data(), a_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    return false;
  if (cudaMemcpy(g_buf.r, rhs, r_bytes, cudaMemcpyHostToDevice) != cudaSuccess)
    return false;

  const int threads = total * TILE;
  constexpr int block = 128;
  btl::factor_solve_warp_per_matrix<double, N><<<(threads + block - 1) / block, block>>>(
      g_buf.a, g_buf.r, g_buf.l, g_buf.x, total, g_buf.info);
  if (cudaGetLastError() != cudaSuccess) return false;
  if (cudaMemcpy(x, g_buf.x, r_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  std::vector<int> info(static_cast<size_t>(total));
  if (cudaMemcpy(info.data(), g_buf.info, total * sizeof(int), cudaMemcpyDeviceToHost) !=
      cudaSuccess)
    return false;
  for (int v : info)
    if (v != 0) return false;  // some system not numerically SPD
  return true;
}

}  // namespace mcc
