#include "mcc/board.hpp"

#include <algorithm>
#include <bit>
#include <utility>

namespace mcc {

namespace {

struct Splitmix64 {
  uint64_t state;
  uint64_t next() {
    uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
};

uint64_t bitMask(int bits) {
  return (bits >= 64) ? ~0ULL : ((1ULL << bits) - 1ULL);
}

}  // namespace

int Dictionary::hamming(uint64_t a, uint64_t b, int bits) {
  return std::popcount((a ^ b) & bitMask(bits));
}

uint64_t Dictionary::rotatedWord(uint64_t w, int marker_size, int quarter_turns) {
  const int n = marker_size;
  int k = ((quarter_turns % 4) + 4) % 4;
  uint64_t cur = w;
  while (k-- > 0) {
    uint64_t rot = 0;
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c)
        if ((cur >> (r * n + c)) & 1ULL) rot |= 1ULL << (c * n + (n - 1 - r));
    cur = rot;
  }
  return cur;
}

Dictionary Dictionary::generate(int n, int marker_size, uint64_t seed) {
  Dictionary dict;
  dict.marker_size = marker_size;
  if (n <= 0 || marker_size <= 0) return dict;

  const int bits = marker_size * marker_size;
  const int self_floor = bits / 4 + 1;  // 10 for 6x6
  const uint64_t mask = bitMask(bits);
  Splitmix64 rng{seed};

  auto self_distance_ok = [&](uint64_t w) {
    uint64_t rw = w;
    for (int k = 1; k < 4; ++k) {
      rw = rotatedWord(rw, marker_size, 1);
      if (hamming(w, rw, bits) < self_floor) return false;
    }
    return true;
  };
  auto pair_distance = [&](uint64_t a, uint64_t b) {
    int d = bits;
    uint64_t rb = b;
    for (int k = 0; k < 4; ++k) {
      d = std::min(d, hamming(a, rb, bits));
      rb = rotatedWord(rb, marker_size, 1);
    }
    return d;
  };

  constexpr int kPoolSize = 4096;
  std::vector<uint64_t> chosen;
  chosen.reserve(n);
  while (static_cast<int>(chosen.size()) < n) {
    std::vector<uint64_t> pool;
    pool.reserve(kPoolSize);
    for (int i = 0; i < kPoolSize; ++i) {
      const uint64_t cand = rng.next() & mask;
      if (self_distance_ok(cand)) pool.push_back(cand);
    }
    if (pool.empty()) continue;

    int best = -1, best_score = 0;
    for (int i = 0; i < static_cast<int>(pool.size()); ++i) {
      int score = bits;
      for (uint64_t w : chosen) score = std::min(score, pair_distance(pool[i], w));
      if (score > best_score) {
        best_score = score;
        best = i;
      }
    }
    if (best < 0) continue;  // every candidate duplicates an existing word
    chosen.push_back(pool[best]);
  }
  dict.words = std::move(chosen);
  return dict;
}

void CharucoBoard::markerCorners(int marker_index, std::array<Eigen::Vector2d, 4>& out) const {
  out.fill(Eigen::Vector2d::Zero());
  if (marker_index < 0 || squares_x <= 0 || squares_y <= 0) return;

  int p = 2 * marker_index;
  int sx = p % squares_x, sy = p / squares_x;
  if ((sx + sy) & 1) {
    ++p;
    sx = p % squares_x;
    sy = p / squares_x;
  }
  if (sy >= squares_y) return;

  const double margin = 0.5 * (square_length - marker_length);
  const double ox = sx * square_length + margin;
  const double oy = sy * square_length + margin;
  const double l = marker_length;
  out[0] = {ox, oy};
  out[1] = {ox + l, oy};
  out[2] = {ox + l, oy + l};
  out[3] = {ox, oy + l};
}

}  // namespace mcc
