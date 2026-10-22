#pragma once

// Extrinsic initialization: from corner caches to a connected camera pose
// graph with per-frame board poses — the deterministic seed for bundle
// adjustment. This is the Eigen port of the archived stage-2 init path,
// with its
// documented failure modes handled explicitly:
//   - solvePnP local minima: our per-frame solver is DLT + LM with a
//     planarity-exact DLT (homography decomposition), so the board being
//     planar cannot push the init into a mirrored minimum.
//   - BFS chain error accumulation: pairwise edges carry their RMS, and
//     the spanning tree is built by max-edge-weight (lowest-RMS-first
//     Kruskal), not breadth-first order of discovery.

#include <cstddef>
#include <vector>

#include "mcc/corner_cache.hpp"
#include "mcc/board.hpp"
#include "mcc/types.hpp"

namespace mcc {

// Per-frame board pose from 2D-3D corner correspondences.
// Planar DLT (homography) init + LM refinement with analytic Jacobians.
// Returns false when fewer than 4 correspondences or degenerate geometry.
bool solveBoardPose(const FrameDetections& frame, const CharucoBoard& board,
                    const Intrinsics& K, Pose& camera_from_board,
                    double* rms_px = nullptr);

struct PairwiseEdge {
  size_t cam_a = 0, cam_b = 0;   // cam_a < cam_b
  Pose a_from_b;                 // pose of camera b's frame in camera a's frame
  double rms_px = 0.0;           // mean of per-frame solve RMS over common frames
  int common_frames = 0;
};

struct PoseGraphInit {
  std::vector<Pose> camera_poses;   // world frame == camera 0 frame
  std::vector<PairwiseEdge> edges;  // all pairwise edges with >= min_common
  bool connected = false;
};

struct PoseInitConfig {
  int min_common_frames = 3;        // the archived pipeline used the same floor
  int min_common_corners = 8;       // per frame, for the pair to count
};

// Build pairwise relative poses from frames where both cameras saw the
// board, average them (quaternion mean + translation mean), then assemble
// a maximum-quality spanning tree rooted at camera 0.
PoseGraphInit initializeCameraPoses(const std::vector<CameraCache>& caches,
                                    const CharucoBoard& board,
                                    const std::vector<Intrinsics>& intrinsics,
                                    const PoseInitConfig& cfg = {});

}  // namespace mcc
