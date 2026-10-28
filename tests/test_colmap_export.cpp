// COLMAP export smoke test: renders a synthetic 2-camera rig, runs the
// detector to fill corner caches, exports, and parses cameras.txt /
// images.txt / points3D.txt back to verify they load and round-trip.
// Also covers the explicit failure paths (empty input, size mismatch) and
// the non-fatal k3-drop note (OPENCV model has no k3).

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "mcc/colmap_export.hpp"
#include "mcc/detector.hpp"

#include "board_renderer.hpp"

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

static std::vector<std::string> dataLines(const std::filesystem::path& p) {
  std::ifstream f(p);
  std::vector<std::string> lines;
  for (std::string line; std::getline(f, line);)
    if (line.empty() || line[0] != '#') lines.push_back(line);
  return lines;
}

int main() {
  CharucoBoard board;
  board.squares_x = 9;
  board.squares_y = 6;
  board.square_length = 0.04;
  board.marker_length = 0.03;
  board.dict = Dictionary::generate(board.numMarkers(), 6, /*seed=*/42);

  constexpr int W = 1280, H = 720;
  mcctest::RenderOptions opt;
  opt.seam_m = 0.0008;

  // Rig: two cameras, poses are camera-from-board (world == board frame).
  const std::vector<Intrinsics> intrinsics = {
      Intrinsics{800.0, 800.0, 640.0, 360.0, {0, 0, 0, 0, 0}},
      Intrinsics{810.0, 805.0, 638.0, 362.0, {0, 0, 0, 0, 0}}};
  const std::vector<Pose> cam_poses = {
      makePose(0.0, 0.0, 0.0, {-0.18, -0.10, 0.60}),
      makePose(0.30, 0.15, 0.05, {-0.15, -0.08, 0.62})};
  const Pose board_moves[] = {Pose{}, makePose(0.05, -0.05, 0.0, {0.02, 0.01, -0.05})};

  // Fill caches from detector output on rendered images.
  CharucoDetector det(board);
  std::vector<CameraCache> caches(2);
  for (int c = 0; c < 2; ++c) {
    caches[c].camera_name = "cam" + std::to_string(c);
    caches[c].image_width = W;
    caches[c].image_height = H;
    for (size_t f = 0; f < 2; ++f) {
      const Pose cam_from_board = cam_poses[c] * board_moves[f];
      const auto img = mcctest::renderBoardImage(board, intrinsics[c], cam_from_board, W, H, opt);
      const DetectionResult res = det.detect(img.data(), W, H, W);
      FrameDetections det_frame;
      det_frame.frame_index = f;
      det_frame.timestamp = std::numeric_limits<double>::quiet_NaN();
      bool all_inserted = true;
      for (const auto& corner : res.charuco_corners)
        if (!insertCorner(det_frame, corner)) all_inserted = false;
      check(all_inserted && det_frame.corners.size() == res.charuco_corners.size(),
            "cam" + std::to_string(c) + " frame " + std::to_string(f) +
                ": detector output respects the cache invariant");
      caches[c].frames.push_back(std::move(det_frame));
    }
  }

  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "mcc_colmap_smoke";
  std::filesystem::remove_all(dir);

  // --- Failure paths ---
  {
    std::string err;
    check(!exportColmap(dir.string(), {}, intrinsics, cam_poses, &err) && !err.empty(),
          "empty cache list rejected");
    check(!exportColmap(dir.string(), caches, {intrinsics[0]}, cam_poses, &err) &&
              !err.empty(),
          "intrinsics/caches size mismatch rejected");
    check(!exportColmap(dir.string(), caches, intrinsics, {cam_poses[0]}, &err) &&
              !err.empty(),
          "camera_poses/caches size mismatch rejected");
  }

  // --- k3 drop: success with a non-fatal note ---
  {
    auto k3_intrinsics = intrinsics;
    k3_intrinsics[0].dist[4] = 0.01;
    std::string err;
    check(exportColmap(dir.string(), caches, k3_intrinsics, cam_poses, &err),
          "k3 export succeeds");
    check(err.find("k3") != std::string::npos, "k3 drop reported in notes");
    std::filesystem::remove_all(dir);
  }

  // --- Successful export ---
  {
    std::string err;
    check(exportColmap(dir.string(), caches, intrinsics, cam_poses, &err),
          "export succeeds");
    check(err.empty(), "no notes for zero-distortion rig");

    const auto cams = dataLines(dir / "cameras.txt");
    check(cams.size() == 2, "cameras.txt has one line per camera");
    for (size_t i = 0; i < cams.size() && i < 2; ++i) {
      std::istringstream ls(cams[i]);
      int id, w, h;
      std::string model;
      double fx, fy, cx, cy, k1, k2, p1, p2;
      ls >> id >> model >> w >> h >> fx >> fy >> cx >> cy >> k1 >> k2 >> p1 >> p2;
      const Intrinsics& K = intrinsics[i];
      check(ls && id == static_cast<int>(i) + 1 && model == "OPENCV" && w == W && h == H,
            "cameras.txt line " + std::to_string(i + 1) + " parses");
      check(fx == K.fx && fy == K.fy && cx == K.cx && cy == K.cy && k1 == 0.0 && k2 == 0.0 &&
                p1 == 0.0 && p2 == 0.0,
            "cameras.txt line " + std::to_string(i + 1) + " params round-trip");
    }

    const auto imgs = dataLines(dir / "images.txt");
    check(imgs.size() == 8, "images.txt has two lines per image, 4 images");
    size_t expected_id = 1;
    bool poses_ok = true, names_ok = true, empty_points_ok = true;
    for (size_t i = 0; i + 1 < imgs.size(); i += 2) {
      std::istringstream ls(imgs[i]);
      size_t id, cam_id;
      double qw, qx, qy, qz, tx, ty, tz;
      std::string name;
      ls >> id >> qw >> qx >> qy >> qz >> tx >> ty >> tz >> cam_id >> name;
      if (id != expected_id || cam_id < 1 || cam_id > 2) poses_ok = false;
      const Pose& p = cam_poses[cam_id - 1];
      const Eigen::Quaterniond q = p.q.normalized();
      const double qnorm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
      if (std::abs(qnorm - 1.0) > 1e-12) poses_ok = false;
      // exporter writes q.normalized() directly; sign is not flipped
      if (std::abs(qw - q.w()) > 1e-15 || std::abs(qx - q.x()) > 1e-15 ||
          std::abs(qy - q.y()) > 1e-15 || std::abs(qz - q.z()) > 1e-15 ||
          std::abs(tx - p.t.x()) > 1e-15 || std::abs(ty - p.t.y()) > 1e-15 ||
          std::abs(tz - p.t.z()) > 1e-15)
        poses_ok = false;
      const size_t frame = (id - 1) % 2;
      if (name != "cam" + std::to_string(cam_id - 1) + "/" + std::to_string(frame) + ".png")
        names_ok = false;
      if (!imgs[i + 1].empty()) empty_points_ok = false;
      ++expected_id;
    }
    check(poses_ok, "images.txt poses round-trip (unit quaternions, camera-from-world)");
    check(names_ok, "images.txt names follow <camera>/<frame>.png");
    check(empty_points_ok, "images.txt POINTS2D lines empty (triangulation is C2/C3)");

    const auto pts = dataLines(dir / "points3D.txt");
    std::ifstream pts_raw(dir / "points3D.txt");
    check(pts_raw.good() && pts.empty(), "points3D.txt exists, header only");
  }

  std::filesystem::remove_all(dir);

  std::printf(failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECKS FAILED\n",
              failures);
  return failures == 0 ? 0 : 1;
}
