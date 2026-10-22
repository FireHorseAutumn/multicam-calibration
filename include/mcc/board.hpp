#pragma once

// ChArUco board model and ArUco-style marker dictionary.
//
// The dictionary is OUR OWN, generated deterministically (seeded search
// maximizing pairwise Hamming distance) — not OpenCV's predefined bits.
// Marker size is fixed at 6x6 data cells with a 1-cell black border.
//
// Board geometry: squares_x * squares_y squares, chessboard alternation,
// markers on the black squares. ChArUco corner id = y * (squares_x - 1) + x
// for inner corner grid positions x in [0, squares_x-2], y in [0, squares_y-2].
// Corner id -> 3D board point: ((x+1) * square_length, (y+1) * square_length, 0)
// with the board lying in the z=0 plane, origin at the outer corner.

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace mcc {

struct Dictionary {
  int marker_size = 6;              // data cells per side (fixed 6 in this pass)
  std::vector<uint64_t> words;      // one word per marker, bit (r*size+c) = cell

  int size() const { return static_cast<int>(words.size()); }
  bool bit(int marker, int r, int c) const {
    return (words[marker] >> (r * marker_size + c)) & 1ULL;
  }

  // Generate a dictionary of n markers maximizing the minimum pairwise
  // Hamming distance, deterministic given `seed`. Also enforces a minimum
  // self-distance under the 4 rotations of each marker.
  static Dictionary generate(int n, int marker_size, uint64_t seed);

  // Hamming distance between marker words; rotations of `b` considered by
  // the caller via rotatedWord().
  static int hamming(uint64_t a, uint64_t b, int bits);
  static uint64_t rotatedWord(uint64_t w, int marker_size, int quarter_turns);
};

struct CharucoBoard {
  int squares_x = 0, squares_y = 0;   // squares per side
  double square_length = 0.0;         // meters
  double marker_length = 0.0;         // meters, < square_length
  Dictionary dict;

  int numMarkers() const { return squares_x * squares_y / 2; }
  int numCharucoCorners() const { return (squares_x - 1) * (squares_y - 1); }

  // 3D board-frame position of a ChArUco corner id. Returns false if id
  // is out of range.
  bool cornerPoint(uint32_t id, Eigen::Vector3d& out) const {
    const int nx = squares_x - 1;
    const int i = static_cast<int>(id);
    if (i < 0 || i >= numCharucoCorners()) return false;
    const int x = i % nx, y = i / nx;
    out = Eigen::Vector3d((x + 1) * square_length, (y + 1) * square_length, 0.0);
    return true;
  }

  // Marker index on square (sx, sy) if that square carries a marker,
  // else -1. Markers sit on squares where (sx + sy) is even.
  int markerAt(int sx, int sy) const {
    if (sx < 0 || sy < 0 || sx >= squares_x || sy >= squares_y) return -1;
    if ((sx + sy) % 2 != 0) return -1;
    return (sy * squares_x + sx) / 2;
  }

  // The 4 board-frame corners of a marker (counter-clockwise, starting at
  // the marker's top-left in its canonical orientation), used for decoding.
  void markerCorners(int marker_index, std::array<Eigen::Vector2d, 4>& out) const;
};

}  // namespace mcc
