// Synthetic end-to-end calibration check (README TODO).
//
// Scope: this drives the numeric pipeline — corner cache -> Zhang + LM
// intrinsics -> pose graph init -> Schur LM bundle adjustment — with
// analytically projected corner observations (zero noise, zero distortion).
// The detector stage (src/detector.cpp) is NOT exercised here: that needs a
// board-image renderer, which is intentionally not fabricated in this pass.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "mcc/board.hpp"
#include "mcc/bundle_adjustment.hpp"
#include "mcc/corner_cache.hpp"
#include "mcc/intrinsics.hpp"
#include "mcc/pose_init.hpp"
#include "mcc/types.hpp"

using namespace mcc;

static int failures = 0;

static void check(bool cond, const std::string& name) {
  std::printf("  [%s] %s\n", cond ? "PASS" : "FAIL", name.c_str());
  if (!cond) ++failures;
}

static Pose makePose(double yaw, double pitch, double roll,
                     const Eigen::Vector3d& t) {
  Eigen::Quaterniond q =
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ());
  return {q.normalized(), t};
}

int main() {
  // --- Board: 9x6 squares, 40 ChArUco corners, 0.36 x 0.20 m ---
  CharucoBoard board;
  board.squares_x = 9;
  board.squares_y = 6;
  board.square_length = 0.04;
  board.marker_length = 0.03;
  board.dict = Dictionary::generate(board.numMarkers(), 6, /*seed=*/42);

  // --- Rig: two cameras, world == camera 0 frame ---
  Intrinsics K0{800.0, 800.0, 640.0, 360.0, {0, 0, 0, 0, 0}};
  Intrinsics K1{810.0, 805.0, 638.0, 362.0, {0, 0, 0, 0, 0}};
  const std::vector<Intrinsics> truth_K = {K0, K1};
  const Pose cam1_from_cam0 =
      makePose(0.15, 0.05, 0.0, {0.22, 0.03, 0.05});
  const std::vector<Pose> truth_cam = {Pose{}, cam1_from_cam0};
  constexpr int W = 1280, H = 720;

  // --- Cache invariant smoke check (sorted insert + duplicate reject) ---
  {
    FrameDetections f;
    check(insertCorner(f, {5, {0, 0}}) && insertCorner(f, {2, {1, 1}}) &&
              insertCorner(f, {9, {2, 2}}),
          "insertCorner accepts new ids");
    check(f.corners.size() == 3 && f.corners[0].id == 2 &&
              f.corners[1].id == 5 && f.corners[2].id == 9,
          "corners sorted ascending by id after unsorted inserts");
    check(!insertCorner(f, {5, {9, 9}}), "duplicate id rejected");
  }

  // --- Board poses (camera0-from-board), spread over the image ---
  std::vector<Pose> board_poses;
  for (int i = 0; i < 14; ++i) {
    const double yaw = -0.35 + 0.05 * (i % 5);
    const double pitch = -0.20 + 0.08 * ((i / 5) % 3);
    const double roll = -0.10 + 0.05 * (i % 3);
    const double tx = -0.25 + 0.07 * (i % 5);
    const double ty = -0.10 + 0.10 * ((i / 5) % 3);
    const double tz = 0.95 + 0.10 * (i % 4);
    board_poses.push_back(makePose(yaw, pitch, roll, {tx, ty, tz}));
  }

  // --- Project corners into per-camera caches (noise-free) ---
  std::vector<CameraCache> caches(2);
  for (int c = 0; c < 2; ++c) {
    caches[c].camera_name = "cam" + std::to_string(c);
    caches[c].image_width = W;
    caches[c].image_height = H;
  }
  for (size_t f = 0; f < board_poses.size(); ++f) {
    for (int c = 0; c < 2; ++c) {
      FrameDetections det;
      det.frame_index = f;
      det.timestamp = std::numeric_limits<double>::quiet_NaN();
      for (uint32_t id = 0; id < board.numCharucoCorners(); ++id) {
        Eigen::Vector3d pw;
        board.cornerPoint(id, pw);
        const Eigen::Vector3d pc =
            truth_cam[c].apply(board_poses[f].apply(pw));
        if (pc.z() < 0.3) continue;
        const Eigen::Vector2d uv = truth_K[c].project(pc);
        if (uv.x() < 2 || uv.x() > W - 3 || uv.y() < 2 || uv.y() > H - 3)
          continue;
        insertCorner(det, {id, uv});
      }
      caches[c].frames.push_back(std::move(det));
    }
  }
  for (int c = 0; c < 2; ++c) {
    int min_corners = 1000;
    for (const auto& f : caches[c].frames)
      min_corners = std::min(min_corners, (int)f.corners.size());
    std::printf("cam%d: %zu frames, min corners/frame = %d\n", c,
                caches[c].frames.size(), min_corners);
    check(min_corners >= 20, "synthetic frames have healthy corner counts");
  }

  // --- Stage 1: intrinsics per camera ---
  std::vector<Intrinsics> est_K(2);
  for (int c = 0; c < 2; ++c) {
    const IntrinsicsResult r = calibrateIntrinsics(caches[c], board);
    if (!r.ok) {
      std::printf("  [FAIL] cam%d intrinsics: %s\n", c,
                  r.failure_reason.c_str());
      ++failures;
      continue;
    }
    est_K[c] = r.intrinsics;
    const double dfx = std::abs(r.intrinsics.fx - truth_K[c].fx);
    const double dfy = std::abs(r.intrinsics.fy - truth_K[c].fy);
    const double dcx = std::abs(r.intrinsics.cx - truth_K[c].cx);
    const double dcy = std::abs(r.intrinsics.cy - truth_K[c].cy);
    std::printf("cam%d intrinsics: rms=%.6f px, |dfx|=%.4g |dfy|=%.4g "
                "|dcx|=%.4g |dcy|=%.4g\n",
                c, r.rms_px, dfx, dfy, dcx, dcy);
    check(r.rms_px < 1e-3, "cam" + std::to_string(c) + " rms ~ 0");
    check(dfx / truth_K[c].fx < 1e-3 && dfy / truth_K[c].fy < 1e-3,
          "cam" + std::to_string(c) + " focal recovered");
    check(dcx < 0.5 && dcy < 0.5,
          "cam" + std::to_string(c) + " principal point recovered");
  }

  // --- Stage 2: pose graph init ---
  const PoseGraphInit init = initializeCameraPoses(caches, board, est_K);
  std::printf("pose init: %zu edges, connected=%d\n", init.edges.size(),
              (int)init.connected);
  check(init.connected, "pose graph connected");

  // --- Stage 3: bundle adjustment ---
  const BAResult ba =
      BundleAdjuster().run(caches, board, est_K, init.camera_poses);
  std::printf("BA: iters=%d converged=%d rms %.6f -> %.6f px\n",
              ba.iterations, (int)ba.converged, ba.initial_rms_px,
              ba.final_rms_px);
  check(ba.converged, "BA converged");
  check(ba.final_rms_px < 1e-3, "BA final rms ~ 0");

  // --- Recovered extrinsics vs ground truth ---
  const Pose& pe = ba.camera_poses[1];
  const Pose& pt = truth_cam[1];
  const double rot_err_deg =
      quatToAngleAxis(pe.q.normalized() * pt.q.normalized().conjugate())
          .norm() *
      180.0 / M_PI;
  const double t_err = (pe.t - pt.t).norm();
  std::printf("cam1 extrinsics: rot err = %.3e deg, trans err = %.3e m\n",
              rot_err_deg, t_err);
  check(rot_err_deg < 0.1, "cam1 rotation recovered");
  check(t_err < 1e-3, "cam1 translation recovered");

  std::printf(failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECKS FAILED\n",
              failures);
  return failures == 0 ? 0 : 1;
}
