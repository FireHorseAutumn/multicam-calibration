# Vendored headers

## include/mcc/vendor/btl/cholesky.cuh

- Source: sibling repo `batched-tiny-linalg`,
  `include/btl/cholesky.cuh` at commit **0f5672f** (2026-10-28, "Max-pass
  fixes: ..." — comment-only; the corrected failure-semantics paragraph is
  the content that matters).
- Byte-identical copy; sha256
  `c94afde6ef471fa0f18ce8fd675ea60c5e594919b9712705d4f34f36b5158ed3`.
- Vendored 2026-10-27, refreshed 2026-10-28. Header-only, no linking.
- C1 code includes the adapter `include/mcc/batched_solve.hpp`, never this
  header directly — the backend can be swapped or dropped without touching
  callers.
- Refresh procedure: copy the file over, update the commit/hash above,
  rerun the full ctest suite.
