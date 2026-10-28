// Timing A/B: Eigen LDLT board-block solve (existing path) vs vendored-btl
// GPU batched Cholesky (adapter path), at the solve-micro level and at the
// whole-application (BundleAdjuster::run) level. Prints a table; the numbers
// are recorded in README.md (bench outputs are gitignored by convention).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include <Eigen/Dense>

#include "mcc/batched_solve.hpp"
#include "mcc/board.hpp"
#include "mcc/bundle_adjustment.hpp"
#include "mcc/corner_cache.hpp"
#include "mcc/types.hpp"

using namespace mcc;

static double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

static Pose makePose(double yaw, double pitch, double roll,
                     const Eigen::Vector3d& t) {
  Eigen::Quaterniond q =
      Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitY()) *
      Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitX()) *
      Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitZ());
  return {q.normalized(), t};
}

// --- Solve micro A/B -------------------------------------------------------

static void micro_ab() {
  std::printf("== batched 6x6 SPD solve (25 rhs per system) ==\n");
  uint64_t s = 99;  // splitmix64
  auto rnd = [&] {
    uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return static_cast<double>((z ^ (z >> 31)) % 1000000ULL) / 1000000.0 - 0.25;
  };
  constexpr int NR = 25;
  for (const int n : {14, 128, 1024}) {
    std::vector<double> a(n * 36), rhs(n * NR * 6), x(n * NR * 6);
    std::vector<int> n_rhs(n, NR);
    std::vector<Eigen::Matrix<double, 6, 6>> am(n);
    std::vector<Eigen::Matrix<double, 6, Eigen::Dynamic>> ar(n);
    for (int b = 0; b < n; ++b) {
      Eigen::Matrix<double, 6, 6> M;
      for (int i = 0; i < 36; ++i) M(i % 6, i / 6) = rnd();
      am[b] = M * M.transpose() + 6.0 * Eigen::Matrix<double, 6, 6>::Identity();
      const Eigen::Matrix<double, 6, 6> rm = am[b].transpose();
      std::copy_n(rm.data(), 36, a.begin() + b * 36);
      ar[b].resize(6, NR);
      for (int j = 0; j < NR; ++j)
        for (int k = 0; k < 6; ++k) ar[b](k, j) = rnd();
      std::copy_n(ar[b].data(), NR * 6, rhs.begin() + b * NR * 6);
    }
    const int reps = n <= 14 ? 200 : n <= 128 ? 100 : 30;
    // eigen side
    std::vector<double> te;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      for (int b = 0; b < n; ++b) {
        const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(am[b]);
        volatile double sink = ldlt.solve(ar[b]).sum();
        (void)sink;
      }
      te.push_back(
          std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
              .count());
    }
    // btl side (first call warms up CUDA; excluded)
    std::vector<double> tb;
    for (int r = 0; r < reps + 1; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      if (!batchedSpdSolve6(a.data(), n, n_rhs.data(), rhs.data(), x.data())) {
        std::printf("btl solve failed at n=%d\n", n);
        return;
      }
      if (r > 0)
        tb.push_back(
            std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0)
                .count());
    }
    const double me = median(te), mb = median(tb);
    std::printf("n=%5d: eigen %9.1f us   btl_gpu %9.1f us   ratio %6.2fx\n", n, me, mb,
                mb / me);
  }
}

// --- Whole-application A/B -------------------------------------------------

struct Rig {
  CharucoBoard board;
  std::vector<Intrinsics> intr;
  std::vector<Pose> truth_cam, init_cam;
  std::vector<CameraCache> caches;
};

static Rig makeRig(int ncams, int nframes) {
  Rig rig;
  rig.board.squares_x = 9;
  rig.board.squares_y = 6;
  rig.board.square_length = 0.04;
  rig.board.marker_length = 0.03;
  rig.board.dict = Dictionary::generate(rig.board.numMarkers(), 6, /*seed=*/42);
  constexpr int W = 1280, H = 720;
  for (int c = 0; c < ncams; ++c) {
    rig.intr.push_back(
        Intrinsics{800.0 + 5.0 * c, 800.0 + 3.0 * c, 640.0 - 2.0 * c, 360.0 + c,
                   {0, 0, 0, 0, 0}});
    rig.truth_cam.push_back(c == 0
                                ? Pose{}
                                : makePose(0.15 * c, 0.05, -0.02 * c,
                                           {0.20 * c, 0.03 * c, 0.05}));
    Eigen::Matrix<double, 6, 1> d;
    d << 0.01 * c, -0.015, 0.008, 0.005 * c, -0.004, 0.003;
    rig.init_cam.push_back(c == 0 ? Pose{} : poseBoxplus(rig.truth_cam[c], d));
    CameraCache cc;
    cc.camera_name = "cam" + std::to_string(c);
    cc.image_width = W;
    cc.image_height = H;
    rig.caches.push_back(std::move(cc));
  }
  for (int f = 0; f < nframes; ++f) {
    const Pose board_pose =
        makePose(-0.35 + 0.02 * (f % 7), -0.20 + 0.03 * ((f / 7) % 5),
                 -0.10 + 0.02 * (f % 3),
                 {-0.25 + 0.03 * (f % 5), -0.10 + 0.04 * ((f / 5) % 4),
                  0.95 + 0.05 * (f % 4)});
    for (int c = 0; c < ncams; ++c) {
      FrameDetections det;
      det.frame_index = f;
      det.timestamp = std::numeric_limits<double>::quiet_NaN();
      for (uint32_t id = 0; id < rig.board.numCharucoCorners(); ++id) {
        Eigen::Vector3d pw;
        rig.board.cornerPoint(id, pw);
        const Eigen::Vector3d pc = rig.truth_cam[c].apply(board_pose.apply(pw));
        if (pc.z() < 0.3) continue;
        const Eigen::Vector2d uv = rig.intr[c].project(pc);
        if (uv.x() < 2 || uv.x() > W - 3 || uv.y() < 2 || uv.y() > H - 3) continue;
        insertCorner(det, {id, uv});
      }
      rig.caches[c].frames.push_back(std::move(det));
    }
  }
  return rig;
}

static void app_ab(int ncams, int nframes, int reps) {
  const Rig rig = makeRig(ncams, nframes);
  std::printf("== BA end-to-end: %d cams x %d frames (init perturbed) ==\n", ncams,
              nframes);
  const BoardSolveBackend backends[2] = {BoardSolveBackend::eigen,
                                         BoardSolveBackend::btl_gpu};
  const char* names[2] = {"eigen", "btl_gpu"};
  for (int k = 0; k < 2; ++k) {
    BAConfig cfg;
    cfg.board_solve = backends[k];
    std::vector<double> times;
    BAResult res;
    for (int r = 0; r < reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      res = BundleAdjuster(cfg).run(rig.caches, rig.board, rig.intr, rig.init_cam);
      times.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
              .count());
    }
    std::printf("%-8s: median %8.3f ms over %d runs (iters=%d, final rms %.6g px)\n",
                names[k], median(times), reps, res.iterations, res.final_rms_px);
  }
}

int main() {
  std::printf("btl backend available: %s\n",
              batchedSpdSolve6Available() ? "yes" : "no");
  micro_ab();
  app_ab(2, 14, 9);    // the synthetic e2e rig size
  app_ab(4, 120, 5);   // a larger rig
  return 0;
}
