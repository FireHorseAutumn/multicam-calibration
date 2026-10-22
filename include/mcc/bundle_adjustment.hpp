#pragma once

// Global bundle adjustment over the whole rig: camera poses, per-frame
// board poses, and (optionally) intrinsics — custom Levenberg-Marquardt,
// analytic Jacobians, Huber robust kernel, no Ceres.
//
// Structure exploits the multicam setting: the Schur complement eliminates
// board poses (independent 6x6 blocks per frame), leaving a dense reduced
// system over camera poses (and intrinsics blocks when enabled) of size
// 6*(ncams-1) [+ 9 per refined camera]. For any realistic rig (<= 32
// cameras) that system is small enough for a plain dense LDLT — no sparse
// library needed.
//
// Gauge: camera 0's pose is fixed (its parameters are not in the system).

#include <vector>

#include "mcc/corner_cache.hpp"
#include "mcc/board.hpp"
#include "mcc/types.hpp"

namespace mcc {

struct BAConfig {
  int max_iters = 50;
  double huber_delta = 1.0;      // pixels
  double init_lambda = 1e-3;     // Marquardt damping
  double min_step_quality = 1e-3;// gain-ratio floor for accepting a step
  bool refine_intrinsics = false;
  bool verbose = false;
};

struct BAResult {
  std::vector<Pose> camera_poses;       // world == camera 0 frame
  std::vector<Pose> board_poses;        // camera-0-from-board per frame
                                        // (rig-synchronized frame list)
  std::vector<Intrinsics> intrinsics; // refined, or copy of the input
  double initial_rms_px = 0.0;
  double final_rms_px = 0.0;
  int iterations = 0;
  bool converged = false;
};

class BundleAdjuster {
 public:
  explicit BundleAdjuster(BAConfig cfg = {}) : cfg_(cfg) {}

  // Frames are aligned across cameras by frame_index: the board pose of
  // frame f is shared by every camera whose cache contains frame f.
  // camera_poses_init must come from initializeCameraPoses (or a prior BA).
  BAResult run(const std::vector<CameraCache>& caches,
               const CharucoBoard& board,
               const std::vector<Intrinsics>& intrinsics_init,
               const std::vector<Pose>& camera_poses_init) const;

 private:
  BAConfig cfg_;
};

}  // namespace mcc
