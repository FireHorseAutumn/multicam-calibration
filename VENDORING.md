# Vendored headers

## include/mcc/vendor/btl/cholesky.cuh

- Source: sibling repo `batched-tiny-linalg`,
  `include/btl/cholesky.cuh` at commit **3bb4194** (2026-10-22,
  "Batched tiny Cholesky factor+solve (thread- and warp-per-matrix)").
- Byte-identical copy; sha256
  `f155c58c082f24abc3de6dbb5ffbc65ab869a2e27c393f8e6e7ffdc392d19d53`.
- Vendored 2026-10-27. Header-only, no linking.
- C1 code includes the adapter `include/mcc/batched_solve.hpp`, never this
  header directly — the backend can be swapped or dropped without touching
  callers.
- Refresh procedure: copy the file over, update the commit/hash above,
  rerun the full ctest suite.
