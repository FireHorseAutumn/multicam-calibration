#pragma once

// Synthetic ChArUco board renderer for tests: inverse ray-cast through a
// zero-distortion pinhole onto the board plane z=0, supersampled (default
// 4x4 sub-pixel samples — 1-2 px hard anti-aliasing steps classify
// inconsistently under the detector's adaptive threshold at square
// junctions; real optics blur similarly). No OpenCV (NOTES.md decision 1).
//
// Board drawing convention (must match what src/detector.cpp expects):
// even-parity squares are black and carry markers (board.hpp), odd squares
// white. The drawn marker is the bare 6x6 bit field filling marker_length —
// marker_length / square_length = 6/8 exactly, so the black square margin
// serves as the 1-cell border of the detector's (marker_size+2)^2 canonical
// grid. `seam_m` optionally draws a white hairline on square boundaries;
// contiguous boards (seam_m = 0) are the primary case, seamed boards a
// regression case.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Geometry>

#include "mcc/board.hpp"
#include "mcc/types.hpp"

namespace mcctest {

struct RenderOptions {
  double seam_m = 0.0;   // white hairline width on square boundaries, board meters
  uint8_t black = 15;
  uint8_t white = 240;
  int supersample = 4;   // sub-pixel samples per axis
};

// Board-plane intensity at (x, y) in meters, 1.0 = white. Outside the board
// is white page.
inline double boardIntensity(const mcc::CharucoBoard& board, double x, double y,
                             const RenderOptions& opt) {
  const double sl = board.square_length;
  if (x < 0.0 || y < 0.0 || x >= board.squares_x * sl || y >= board.squares_y * sl)
    return 1.0;
  const int sx = std::min(static_cast<int>(x / sl), board.squares_x - 1);
  const int sy = std::min(static_cast<int>(y / sl), board.squares_y - 1);
  const double lx = x - sx * sl, ly = y - sy * sl;
  const double h = 0.5 * opt.seam_m;
  if (h > 0.0 && (lx < h || ly < h || lx > sl - h || ly > sl - h)) return 1.0;
  if ((sx + sy) & 1) return 1.0;  // white square
  const double m = 0.5 * (sl - board.marker_length);
  const double mx = lx - m, my = ly - m;
  if (mx < 0.0 || my < 0.0 || mx >= board.marker_length || my >= board.marker_length)
    return 0.0;  // black square margin = marker border
  const int ms = board.dict.marker_size;
  const int c = std::min(static_cast<int>(mx / board.marker_length * ms), ms - 1);
  const int r = std::min(static_cast<int>(my / board.marker_length * ms), ms - 1);
  return board.dict.bit(board.markerAt(sx, sy), r, c) ? 1.0 : 0.0;
}

// Renders the board seen by a zero-distortion pinhole camera.
// cam_from_board: pc = q * pb + t (Pose convention, camera-from-board).
inline std::vector<uint8_t> renderBoardImage(const mcc::CharucoBoard& board,
                                             const mcc::Intrinsics& K,
                                             const mcc::Pose& cam_from_board, int width,
                                             int height, const RenderOptions& opt = {}) {
  std::vector<uint8_t> img(static_cast<size_t>(width) * height, opt.white);
  const Eigen::Quaterniond qi = cam_from_board.q.normalized().conjugate();
  const Eigen::Vector3d origin = qi * (-cam_from_board.t);  // camera center, board frame
  const int ss = std::max(1, opt.supersample);
  const int range = opt.white - opt.black;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      double acc = 0.0;
      for (int j = 0; j < ss; ++j) {
        for (int i = 0; i < ss; ++i) {
          const double u = x + (i + 0.5) / ss, v = y + (j + 0.5) / ss;
          const Eigen::Vector3d d_cam((u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0);
          const Eigen::Vector3d d = qi * d_cam;
          const double lambda =
              std::abs(d.z()) > 1e-12 ? -origin.z() / d.z() : -1.0;
          if (lambda <= 0.0) {
            acc += 1.0;  // ray misses the board plane in front of the camera
            continue;
          }
          const Eigen::Vector3d p = origin + lambda * d;
          acc += boardIntensity(board, p.x(), p.y(), opt);
        }
      }
      img[static_cast<size_t>(y) * width + x] =
          static_cast<uint8_t>(opt.black + range * acc / (ss * ss));
    }
  }
  return img;
}

// Ground-truth image position of a ChArUco corner id under the same camera.
inline Eigen::Vector2d projectCorner(const mcc::CharucoBoard& board, const mcc::Intrinsics& K,
                                     const mcc::Pose& cam_from_board, uint32_t id) {
  Eigen::Vector3d pw;
  board.cornerPoint(id, pw);
  return K.projectUndistorted(cam_from_board.apply(pw));
}

// Ground-truth image positions of a marker's 4 detected-quad corners
// (TL,TR,BR,BL). The detected quad is the black region of the square, which a
// seam of `inset_m` half-width pulls in from the square outline.
inline std::array<Eigen::Vector2d, 4> projectSquareCorners(const mcc::CharucoBoard& board,
                                                           const mcc::Intrinsics& K,
                                                           const mcc::Pose& cam_from_board,
                                                           int marker_index,
                                                           double inset_m = 0.0) {
  int p = 2 * marker_index;
  int sx = p % board.squares_x, sy = p / board.squares_x;
  if ((sx + sy) & 1) {
    ++p;
    sx = p % board.squares_x;
    sy = p / board.squares_x;
  }
  const double sl = board.square_length;
  const std::array<Eigen::Vector3d, 4> pts = {
      Eigen::Vector3d(sx * sl + inset_m, sy * sl + inset_m, 0.0),
      Eigen::Vector3d((sx + 1) * sl - inset_m, sy * sl + inset_m, 0.0),
      Eigen::Vector3d((sx + 1) * sl - inset_m, (sy + 1) * sl - inset_m, 0.0),
      Eigen::Vector3d(sx * sl + inset_m, (sy + 1) * sl - inset_m, 0.0)};
  std::array<Eigen::Vector2d, 4> out;
  for (int i = 0; i < 4; ++i) out[i] = K.projectUndistorted(cam_from_board.apply(pts[i]));
  return out;
}

}  // namespace mcctest
