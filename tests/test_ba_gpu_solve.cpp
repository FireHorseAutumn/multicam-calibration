// btl-backed board-block solve: adapter unit checks + whole-application BA
// correctness through the adapter (BAConfig::board_solve = btl_gpu).
//
// Rig mirrors test_synthetic_e2e (2 cameras, noise-free analytic corners)
// but starts camera 1 from a perturbed pose so LM does real work. Both
// backends must converge to the same recovery gates; the eigen backend is
// the reference.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "mcc/batched_solve.hpp"
#include "mcc/board.hpp"
#include "mcc/bundle_adjustment.hpp"
#include "mcc/corner_cache.hpp"
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

static void adapter_unit_checks() {
  // Three SPD systems A = M M^T + I with 1, 3, 2 right-hand sides.
  std::vector<double> a(3 * 36);
  std::vector<int> n_rhs = {1, 3, 2};
  std::vector<double> rhs, x_ref;
  uint64_t s = 7;  // splitmix64
  auto rnd = [&] {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<double>((z ^ (z >> 31)) % 1000000ULL) / 1000000.0 - 0.5;
  };
  for (int b = 0; b < 3; ++b) {
    Eigen::Matrix<double, 6, 6> M;
    for (int i = 0; i < 36; ++i) M(i % 6, i / 6) = rnd();
    const Eigen::Matrix<double, 6, 6> A = M * M.transpose() + Eigen::Matrix<double, 6, 6>::Identity();
    const Eigen::Matrix<double, 6, 6> rm = A.transpose();  // to row-major
    std::copy_n(rm.data(), 36, a.begin() + b * 36);
    for (int j = 0; j < n_rhs[b]; ++j) {
      Eigen::Matrix<double, 6, 1> r;
      for (int k = 0; k < 6; ++k) r(k) = rnd();
      for (int k = 0; k < 6; ++k) rhs.push_back(r(k));
      const Eigen::Matrix<double, 6, 1> xr = A.ldlt().solve(r);
      for (int k = 0; k < 6; ++k) x_ref.push_back(xr(k));
    }
  }
  std::vector<double> x(rhs.size());
  check(batchedSpdSolve6(a.data(), 3, n_rhs.data(), rhs.data(), x.data()),
        "adapter: batched solve returns success");
  double max_err = 0.0;
  for (size_t i = 0; i < x.size(); ++i) max_err = std::max(max_err, std::abs(x[i] - x_ref[i]));
  std::printf("adapter: max |x - Eigen LDLT| = %.3e\n", max_err);
  check(max_err < 1e-9, "adapter: solutions match Eigen reference");

  // Non-SPD system (zero pivot) must report failure, not produce garbage.
  std::vector<double> bad(36, 0.0);
  const int one = 1;
  std::vector<double> r6(6, 1.0), x6(6, 0.0);
  check(!batchedSpdSolve6(bad.data(), 1, &one, r6.data(), x6.data()),
        "adapter: non-SPD system reports failure");

  check(!batchedSpdSolve6(nullptr, 1, &one, r6.data(), x6.data()) &&
            !batchedSpdSolve6(bad.data(), 0, &one, r6.data(), x6.data()),
        "adapter: invalid arguments rejected");
}

int main() {
  if (!batchedSpdSolve6Available()) {
    std::printf("no CUDA device; btl path untested (adapter must decline)\n");
    const int one = 1;
    std::vector<double> a(36, 0.0), r(6, 1.0), x(6, 0.0);
    check(!batchedSpdSolve6(a.data(), 1, &one, r.data(), x.data()),
          "adapter: declines cleanly without a CUDA device");
    std::printf(failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECKS FAILED\n",
                failures);
    return failures == 0 ? 0 : 1;
  }

  adapter_unit_checks();

  // --- Rig: same board/cameras/frames as test_synthetic_e2e ---
  CharucoBoard board;
  board.squares_x = 9;
  board.squares_y = 6;
  board.square_length = 0.04;
  board.marker_length = 0.03;
  board.dict = Dictionary::generate(board.numMarkers(), 6, /*seed=*/42);
  const std::vector<Intrinsics> truth_K = {
      Intrinsics{800.0, 800.0, 640.0, 360.0, {0, 0, 0, 0, 0}},
      Intrinsics{810.0, 805.0, 638.0, 362.0, {0, 0, 0, 0, 0}}};
  const Pose cam1_truth = makePose(0.15, 0.05, 0.0, {0.22, 0.03, 0.05});
  constexpr int W = 1280, H = 720;

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

  const std::vector<Pose> truth_cam = {Pose{}, cam1_truth};
  std::vector<CameraCache> caches(2);
  for (int c = 0; c < 2; ++c) {
    caches[c].camera_name = "cam" + std::to_string(c);
    caches[c].image_width = W;
    caches[c].image_height = H;
  }
  for (size_t f = 0; f < board_poses.size(); ++f)
    for (int c = 0; c < 2; ++c) {
      FrameDetections det;
      det.frame_index = f;
      det.timestamp = std::numeric_limits<double>::quiet_NaN();
      for (uint32_t id = 0; id < board.numCharucoCorners(); ++id) {
        Eigen::Vector3d pw;
        board.cornerPoint(id, pw);
        const Eigen::Vector3d pc = truth_cam[c].apply(board_poses[f].apply(pw));
        if (pc.z() < 0.3) continue;
        const Eigen::Vector2d uv = truth_K[c].project(pc);
        if (uv.x() < 2 || uv.x() > W - 3 || uv.y() < 2 || uv.y() > H - 3) continue;
        insertCorner(det, {id, uv});
      }
      caches[c].frames.push_back(std::move(det));
    }

  // Perturbed init for camera 1 so LM iterates for real.
  Eigen::Matrix<double, 6, 1> delta;
  delta << 0.01, -0.015, 0.008, 0.005, -0.004, 0.003;
  const std::vector<Pose> init_cam = {Pose{}, poseBoxplus(cam1_truth, delta)};

  BAResult res[2];
  const BoardSolveBackend backends[2] = {BoardSolveBackend::eigen,
                                         BoardSolveBackend::btl_gpu};
  const char* names[2] = {"eigen", "btl_gpu"};
  for (int k = 0; k < 2; ++k) {
    BAConfig cfg;
    cfg.board_solve = backends[k];
    res[k] = BundleAdjuster(cfg).run(caches, board, truth_K, init_cam);
    const double rot_err_deg =
        quatToAngleAxis(res[k].camera_poses[1].q.normalized() *
                        cam1_truth.q.normalized().conjugate())
            .norm() *
        180.0 / M_PI;
    const double t_err = (res[k].camera_poses[1].t - cam1_truth.t).norm();
    std::printf("%s: iters=%d converged=%d rms %.6g -> %.6g px, cam1 rot err %.3e deg, "
                "trans err %.3e m\n",
                names[k], res[k].iterations, static_cast<int>(res[k].converged),
                res[k].initial_rms_px, res[k].final_rms_px, rot_err_deg, t_err);
    check(res[k].converged, std::string(names[k]) + ": BA converged");
    check(res[k].final_rms_px < 1e-6, std::string(names[k]) + ": final rms ~ 0");
    check(rot_err_deg < 0.01, std::string(names[k]) + ": cam1 rotation recovered");
    check(t_err < 1e-4, std::string(names[k]) + ": cam1 translation recovered");
  }
  check(std::abs(res[0].final_rms_px - res[1].final_rms_px) < 1e-12,
        "backends agree on final rms");

  std::printf(failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECKS FAILED\n",
              failures);
  return failures == 0 ? 0 : 1;
}
