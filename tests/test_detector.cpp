// Detector stage check: renders synthetic ChArUco board images
// (tests/board_renderer.hpp, zero distortion) and runs src/detector.cpp on
// them.
//
// Primary case is the standard contiguous board (seam_m = 0); a seamed
// board is kept as a regression case. Positive views assert exact
// marker/corner counts and sub-pixel accuracy against analytically
// projected ground truth. Rejection paths (blank, noise, occluded, garbled,
// frame-clipped, invalid args) must return clean partial or empty results —
// never garbage corners (NOTES.md decision 8).

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

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

// All reported corners must sit within `tol` px of ground truth, and all
// marker ids must be real and unique. Returns max charuco corner error.
static double checkClean(const DetectionResult& res, const CharucoBoard& board,
                         const Intrinsics& K, const Pose& cam_from_board,
                         double seam_m, double tol, const std::string& tag) {
  std::vector<char> seen(board.numMarkers(), 0);
  bool ids_ok = true;
  for (const auto& m : res.markers) {
    if (m.id >= static_cast<uint32_t>(board.numMarkers()) || seen[m.id]) ids_ok = false;
    if (m.id < static_cast<uint32_t>(board.numMarkers())) seen[m.id] = 1;
  }
  check(ids_ok, tag + ": marker ids valid and unique");
  double max_cerr = 0.0;
  for (const auto& c : res.charuco_corners)
    max_cerr = std::max(
        max_cerr, (c.uv - mcctest::projectCorner(board, K, cam_from_board, c.id)).norm());
  check(max_cerr < tol, tag + ": charuco corners within tolerance");
  double max_merr = 0.0;
  for (const auto& m : res.markers) {
    const auto sq = mcctest::projectSquareCorners(board, K, cam_from_board, m.id,
                                                  0.5 * seam_m);
    for (int i = 0; i < 4; ++i)
      max_merr = std::max(max_merr, (m.corners[i] - sq[i]).norm());
  }
  check(max_merr < 2.0 * tol, tag + ": marker corners near square corners");
  return max_cerr;
}

int main() {
  CharucoBoard board;
  board.squares_x = 9;
  board.squares_y = 6;
  board.square_length = 0.04;
  board.marker_length = 0.03;
  board.dict = Dictionary::generate(board.numMarkers(), 6, /*seed=*/42);

  Intrinsics K{800.0, 800.0, 640.0, 360.0, {0, 0, 0, 0, 0}};
  constexpr int W = 1280, H = 720;
  CharucoDetector det(board);  // default config

  const Pose frontal = makePose(0.0, 0.0, 0.0, {-0.18, -0.10, 0.60});
  const Pose tilted = makePose(0.30, 0.15, 0.05, {-0.15, -0.08, 0.62});
  const Pose steep = makePose(-0.25, -0.10, 0.0, {-0.20, -0.12, 0.45});

  // --- Positive views, contiguous board (standard print) ---
  for (const auto& [name, pose] : {std::pair{"frontal", frontal},
                                   std::pair{"tilted", tilted},
                                   std::pair{"steep", steep}}) {
    const auto img = mcctest::renderBoardImage(board, K, pose, W, H);
    const DetectionResult res = det.detect(img.data(), W, H, W);
    std::printf("%s (contiguous): markers=%zu charuco=%zu\n", name,
                res.markers.size(), res.charuco_corners.size());
    check(static_cast<int>(res.markers.size()) == board.numMarkers(),
          std::string(name) + ": all markers detected on contiguous board");
    check(static_cast<int>(res.charuco_corners.size()) == board.numCharucoCorners(),
          std::string(name) + ": all charuco corners interpolated");
    bool sorted = true;
    for (size_t i = 1; i < res.charuco_corners.size(); ++i)
      if (res.charuco_corners[i].id <= res.charuco_corners[i - 1].id) sorted = false;
    check(sorted, std::string(name) + ": charuco corners sorted ascending by id");
    const double max_cerr = checkClean(res, board, K, pose, 0.0, 1.5, name);
    std::printf("%s: max charuco corner error %.3f px\n", name, max_cerr);
  }

  // --- Regression: seamed board (white hairlines between squares) ---
  {
    mcctest::RenderOptions opt;
    opt.seam_m = 0.0008;  // ~1 px at 0.6 m with f=800
    const auto img = mcctest::renderBoardImage(board, K, frontal, W, H, opt);
    const DetectionResult res = det.detect(img.data(), W, H, W);
    std::printf("frontal (seamed): markers=%zu charuco=%zu\n", res.markers.size(),
                res.charuco_corners.size());
    check(static_cast<int>(res.markers.size()) == board.numMarkers(),
          "seamed regression: all markers detected");
    checkClean(res, board, K, frontal, opt.seam_m, 1.5, "seamed");
  }

  // --- Invalid inputs: empty result, no crash ---
  {
    const auto img = mcctest::renderBoardImage(board, K, frontal, W, H);
    check(det.detect(nullptr, W, H, W).markers.empty(), "null image rejected");
    check(det.detect(img.data(), 0, H, W).markers.empty(), "zero width rejected");
    check(det.detect(img.data(), W, 0, W).markers.empty(), "zero height rejected");
    check(det.detect(img.data(), W, H, W - 1).markers.empty(),
          "stride < width rejected");
  }

  // --- Blank and noise images: no false positives ---
  {
    std::vector<uint8_t> white(W * H, 240), black(W * H, 15);
    check(det.detect(white.data(), W, H, W).markers.empty(), "blank white image: no markers");
    check(det.detect(black.data(), W, H, W).markers.empty(), "blank black image: no markers");
    std::vector<uint8_t> noise(W * H);
    uint64_t s = 12345;  // splitmix64
    for (auto& b : noise) {
      uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
      z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
      z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
      b = static_cast<uint8_t>((z ^ (z >> 31)) >> 32);
    }
    const DetectionResult res = det.detect(noise.data(), W, H, W);
    check(res.markers.empty() && res.charuco_corners.empty(),
          "uniform noise: no false-positive markers");
  }

  // --- Non-tight stride is honored ---
  {
    const auto img = mcctest::renderBoardImage(board, K, frontal, W, H);
    const int stride = W + 16;
    std::vector<uint8_t> padded(static_cast<size_t>(stride) * H, 0);
    for (int y = 0; y < H; ++y)
      std::copy_n(img.begin() + static_cast<size_t>(y) * W, W,
                  padded.begin() + static_cast<size_t>(y) * stride);
    const DetectionResult res = det.detect(padded.data(), W, H, stride);
    check(res.markers.size() == static_cast<size_t>(board.numMarkers()),
          "padded stride: all markers detected");
  }

  // --- Occlusion: covered markers vanish, the rest stay clean ---
  {
    auto img = mcctest::renderBoardImage(board, K, frontal, W, H);
    const int ox0 = 650, oy0 = 300, ox1 = 900, oy1 = 460;
    for (int y = oy0; y < oy1; ++y)
      for (int x = ox0; x < ox1; ++x) img[y * W + x] = 110;
    const DetectionResult res = det.detect(img.data(), W, H, W);
    std::printf("occluded: markers=%zu charuco=%zu\n", res.markers.size(),
                res.charuco_corners.size());
    check(res.markers.size() >= 10 &&
              res.markers.size() < static_cast<size_t>(board.numMarkers()),
          "occluded: partial detection");
    int inside = 0;
    for (const auto& m : res.markers) {
      const Eigen::Vector2d c = (m.corners[0] + m.corners[2]) / 2.0;
      if (c.x() > ox0 && c.x() < ox1 && c.y() > oy0 && c.y() < oy1) ++inside;
    }
    check(inside == 0, "occluded: no detections inside the occluder");
    checkClean(res, board, K, frontal, 0.0, 1.5, "occluded");
  }

  // --- Garbled marker: that id rejected, neighbors unaffected ---
  {
    auto img = mcctest::renderBoardImage(board, K, frontal, W, H);
    const int victim = 5;
    int p = 2 * victim, sx = p % board.squares_x, sy = p / board.squares_x;
    if ((sx + sy) & 1) {
      ++p;
      sx = p % board.squares_x;
      sy = p / board.squares_x;
    }
    const double sl = board.square_length, m = 0.5 * (sl - board.marker_length);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        // frontal pose is identity rotation; board z = 0 maps to camera z = 0.6
        const double bx = (x + 0.5 - K.cx) / K.fx * 0.6 - frontal.t.x();
        const double by = (y + 0.5 - K.cy) / K.fy * 0.6 - frontal.t.y();
        const double lx = bx - sx * sl - m, ly = by - sy * sl - m;
        if (lx > 0 && ly > 0 && lx < board.marker_length && ly < board.marker_length)
          img[y * W + x] = 240;  // erase the bit pattern
      }
    const DetectionResult res = det.detect(img.data(), W, H, W);
    bool victim_seen = false;
    for (const auto& mk : res.markers)
      if (mk.id == static_cast<uint32_t>(victim)) victim_seen = true;
    std::printf("garbled: markers=%zu victim_seen=%d\n", res.markers.size(),
                static_cast<int>(victim_seen));
    check(!victim_seen, "garbled: corrupted marker rejected, not mis-decoded");
    check(res.markers.size() >= static_cast<size_t>(board.numMarkers()) - 3,
          "garbled: other markers survive");
    checkClean(res, board, K, frontal, 0.0, 1.5, "garbled");
  }

  // --- Frame-clipped board: edge quads rejected, no garbage ---
  {
    const Pose clipped = makePose(0.0, 0.0, 0.0, {0.10, -0.10, 0.55});
    const auto img = mcctest::renderBoardImage(board, K, clipped, W, H);
    const DetectionResult res = det.detect(img.data(), W, H, W);
    std::printf("clipped: markers=%zu charuco=%zu\n", res.markers.size(),
                res.charuco_corners.size());
    check(res.markers.size() > 5 &&
              res.markers.size() < static_cast<size_t>(board.numMarkers()),
          "clipped: partial detection");
    bool near_edge = false;
    for (const auto& c : res.charuco_corners)
      if (c.uv.x() < 1 || c.uv.y() < 1 || c.uv.x() > W - 2 || c.uv.y() > H - 2)
        near_edge = true;
    check(!near_edge, "clipped: no corners glued to the frame edge");
    checkClean(res, board, K, clipped, 0.0, 1.5, "clipped");
  }

  std::printf(failures == 0 ? "\nALL CHECKS PASSED\n" : "\n%d CHECKS FAILED\n",
              failures);
  return failures == 0 ? 0 : 1;
}
