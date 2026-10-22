#pragma once

// Per-camera intrinsic calibration, Zhang's method, pure Eigen.
//
//   1. Per-frame homography board->image via normalized DLT (frames with
//      too few corners or degenerate geometry are rejected).
//   2. Closed-form intrinsics from >= 3 homographies.
//   3. Per-frame board pose (camera-from-board) recovered from the
//      homography given the intrinsics.
//   4. Full LM refinement over {intrinsics, distortion, all board poses}
//      with analytic Jacobians — this is the step the archived pipeline's
//      `cv::calibrateCamera` performed; ours is transparent.
//
// Known archived-pipeline scar encoded as a guard: the 5-parameter model
// still overfits
// with very few views, so frames_used < min_frames is a hard error and a
// low parallax / low corner-coverage condition is reported, not silently
// accepted.

#include <vector>

#include "mcc/corner_cache.hpp"
#include "mcc/board.hpp"
#include "mcc/types.hpp"

namespace mcc {

struct IntrinsicsConfig {
  int min_frames = 8;          // hard minimum; below this we refuse
  int min_corners_per_frame = 6;
  int max_lm_iters = 50;
  bool fix_aspect_ratio = false;  // constrain fx == fy during refinement
  double min_coverage = 0.25;     // fraction of image area the corner
                                  // bounding boxes must cover, cumulative
};

struct IntrinsicsResult {
  Intrinsics intrinsics;
  std::vector<Pose> board_poses;    // camera-from-board, one per used frame
  std::vector<uint64_t> frame_indices;  // which cache frames were used
  double rms_px = 0.0;
  bool ok = false;
  std::string failure_reason;       // populated when !ok
};

IntrinsicsResult calibrateIntrinsics(const CameraCache& cache,
                                     const CharucoBoard& board,
                                     const IntrinsicsConfig& cfg = {});

// Convenience: calibrate every camera in a rig.
std::vector<IntrinsicsResult> calibrateAllIntrinsics(
    const std::vector<CameraCache>& caches, const CharucoBoard& board,
    const IntrinsicsConfig& cfg = {});

}  // namespace mcc
