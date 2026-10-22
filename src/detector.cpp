#include "mcc/detector.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace mcc {

namespace {

constexpr double kBorderMargin = 1.0;  // px; quads touching the frame edge are rejected
constexpr int kPixelsPerCell = 8;      // canonical patch resolution per marker cell

struct Quad {
  std::array<Eigen::Vector2d, 4> corners;  // TL, TR, BR, BL in image coords
  Eigen::Vector2d center;
};

std::vector<uint8_t> adaptiveThreshold(const uint8_t* gray, int w, int h, int stride,
                                       int window, double offset) {
  std::vector<uint64_t> ii((w + 1) * (h + 1), 0);
  for (int y = 0; y < h; ++y) {
    uint64_t rowsum = 0;
    for (int x = 0; x < w; ++x) {
      rowsum += gray[y * stride + x];
      ii[(y + 1) * (w + 1) + (x + 1)] = ii[y * (w + 1) + (x + 1)] + rowsum;
    }
  }
  const int half = window / 2;
  std::vector<uint8_t> bin(w * h, 0);
  // CUDA seam: per-pixel windowed reduction over the integral image.
  for (int y = 0; y < h; ++y) {
    const int y0 = std::max(0, y - half), y1 = std::min(h - 1, y + half);
    for (int x = 0; x < w; ++x) {
      const int x0 = std::max(0, x - half), x1 = std::min(w - 1, x + half);
      const uint64_t s = ii[(y1 + 1) * (w + 1) + (x1 + 1)] - ii[y0 * (w + 1) + (x1 + 1)] -
                         ii[(y1 + 1) * (w + 1) + x0] + ii[y0 * (w + 1) + x0];
      const double mean = static_cast<double>(s) / ((y1 - y0 + 1) * (x1 - x0 + 1));
      bin[y * w + x] = (gray[y * stride + x] < mean - offset) ? 1 : 0;
    }
  }
  return bin;
}

// Clockwise 8-neighborhood offsets in image coords (y down), starting at NW.
constexpr int kNb[8][2] = {{-1, -1}, {0, -1}, {1, -1}, {1, 0},
                           {1, 1},   {0, 1},  {-1, 1}, {-1, 0}};

int neighborIndex(int dx, int dy) {
  for (int i = 0; i < 8; ++i)
    if (kNb[i][0] == dx && kNb[i][1] == dy) return i;
  return -1;
}

// Suzuki-Abe border following, outer borders only (RETR_EXTERNAL equivalent).
// f: 0 = background, 1 = unvisited foreground, +/-NBD = visited border.
std::vector<std::vector<Eigen::Vector2d>> outerContours(const std::vector<uint8_t>& bin,
                                                        int w, int h) {
  std::vector<int32_t> f(w * h);
  for (int i = 0; i < w * h; ++i) f[i] = bin[i] ? 1 : 0;
  std::vector<std::vector<Eigen::Vector2d>> contours;
  const int64_t max_steps = 4LL * w * h + 16;
  auto at = [&](int x, int y) -> int32_t {
    return (x < 0 || y < 0 || x >= w || y >= h) ? 0 : f[y * w + x];
  };
  int32_t nbd = 1;
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      if (f[y * w + x] != 1 || at(x - 1, y) != 0) continue;
      ++nbd;
      std::vector<Eigen::Vector2d> cont;

      int i1x = -1, i1y = -1;
      const int base0 = neighborIndex(-1, 0);
      for (int k = 1; k <= 8; ++k) {
        const int idx = (base0 + k) & 7;
        const int nx = x + kNb[idx][0], ny = y + kNb[idx][1];
        if (at(nx, ny) != 0) {
          i1x = nx;
          i1y = ny;
          break;
        }
      }
      if (i1x < 0) {
        f[y * w + x] = -nbd;
        cont.emplace_back(x, y);
        contours.push_back(std::move(cont));
        continue;
      }

      int px = i1x, py = i1y;  // (i2, j2)
      int cx = x, cy = y;      // (i3, j3)
      bool complete = false;
      for (int64_t step = 0; step < max_steps; ++step) {
        const int base = neighborIndex(px - cx, py - cy);
        int i4x = -1, i4y = -1;
        bool right_examined_zero = false;
        for (int k = 1; k <= 8; ++k) {
          const int idx = (((base - k) % 8) + 8) & 7;
          const int nx = cx + kNb[idx][0], ny = cy + kNb[idx][1];
          const int32_t v = at(nx, ny);
          if (nx == cx + 1 && ny == cy && v == 0) right_examined_zero = true;
          if (v != 0) {
            i4x = nx;
            i4y = ny;
            break;
          }
        }
        int32_t& fc = f[cy * w + cx];
        if (right_examined_zero)
          fc = -nbd;
        else if (fc == 1)
          fc = nbd;
        cont.emplace_back(cx, cy);
        if (i4x == x && i4y == y && cx == i1x && cy == i1y) {
          complete = true;
          break;
        }
        px = cx;
        py = cy;
        cx = i4x;
        cy = i4y;
      }
      if (complete) contours.push_back(std::move(cont));
    }
  }
  return contours;
}

double pointLineDistance(const Eigen::Vector2d& p, const Eigen::Vector2d& a,
                         const Eigen::Vector2d& b) {
  const Eigen::Vector2d ab = b - a;
  const double n = ab.norm();
  if (n < 1e-12) return (p - a).norm();
  return std::abs(ab.x() * (a.y() - p.y()) - ab.y() * (a.x() - p.x())) / n;
}

void dpRecurse(const std::vector<Eigen::Vector2d>& pts, int a, int b, double eps,
               std::vector<uint8_t>& keep) {
  double dmax = eps;
  int imax = -1;
  for (int i = a + 1; i < b; ++i) {
    const double d = pointLineDistance(pts[i], pts[a], pts[b]);
    if (d > dmax) {
      dmax = d;
      imax = i;
    }
  }
  if (imax < 0) return;
  keep[imax] = 1;
  dpRecurse(pts, a, imax, eps, keep);
  dpRecurse(pts, imax, b, eps, keep);
}

std::vector<Eigen::Vector2d> approxPolygon(const std::vector<Eigen::Vector2d>& cont,
                                           double eps) {
  const int n = static_cast<int>(cont.size());
  if (n < 3) return cont;
  std::vector<Eigen::Vector2d> ext = cont;
  ext.push_back(cont[0]);
  int far = 0;
  double dmax = -1.0;
  for (int i = 1; i < n; ++i) {
    const double d = (cont[i] - cont[0]).squaredNorm();
    if (d > dmax) {
      dmax = d;
      far = i;
    }
  }
  std::vector<uint8_t> keep(n + 1, 0);
  keep[0] = keep[far] = keep[n] = 1;
  dpRecurse(ext, 0, far, eps, keep);
  dpRecurse(ext, far, n, eps, keep);
  std::vector<Eigen::Vector2d> out;
  for (int i = 0; i < n; ++i)
    if (keep[i]) out.push_back(ext[i]);
  return out;
}

double polygonPerimeter(const std::vector<Eigen::Vector2d>& p) {
  double per = 0.0;
  for (size_t i = 0; i < p.size(); ++i) per += (p[(i + 1) % p.size()] - p[i]).norm();
  return per;
}

bool isConvexQuad(const std::vector<Eigen::Vector2d>& p) {
  int sign = 0;
  for (int i = 0; i < 4; ++i) {
    const Eigen::Vector2d e1 = p[(i + 1) & 3] - p[i];
    const Eigen::Vector2d e2 = p[(i + 2) & 3] - p[(i + 1) & 3];
    const double cr = e1.x() * e2.y() - e1.y() * e2.x();
    const int s = (cr > 1e-9) - (cr < -1e-9);
    if (s == 0) return false;
    if (sign == 0)
      sign = s;
    else if (s != sign)
      return false;
  }
  return true;
}

// Angle-sort around the centroid (ascending atan2 in y-down image coords walks
// TL, TR, BR, BL), then rotate so the corner with minimal x + y (top-left) leads.
std::array<Eigen::Vector2d, 4> orderCorners(const std::vector<Eigen::Vector2d>& p) {
  Eigen::Vector2d c = Eigen::Vector2d::Zero();
  for (const auto& v : p) c += v;
  c /= 4.0;
  std::array<int, 4> idx = {0, 1, 2, 3};
  std::sort(idx.begin(), idx.end(), [&](int a, int b) {
    return std::atan2(p[a].y() - c.y(), p[a].x() - c.x()) <
           std::atan2(p[b].y() - c.y(), p[b].x() - c.x());
  });
  int lead = 0;
  for (int i = 1; i < 4; ++i)
    if (p[idx[i]].x() + p[idx[i]].y() < p[idx[lead]].x() + p[idx[lead]].y()) lead = i;
  std::array<Eigen::Vector2d, 4> out;
  for (int i = 0; i < 4; ++i) out[i] = p[idx[(lead + i) & 3]];
  return out;
}

Eigen::Matrix3d homographyDlt(const std::array<Eigen::Vector2d, 4>& src,
                              const std::array<Eigen::Vector2d, 4>& dst) {
  Eigen::Matrix<double, 8, 9> A;
  for (int i = 0; i < 4; ++i) {
    const double x = src[i].x(), y = src[i].y();
    const double u = dst[i].x(), v = dst[i].y();
    A.row(2 * i) << -x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u;
    A.row(2 * i + 1) << 0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v;
  }
  const Eigen::JacobiSVD<Eigen::Matrix<double, 8, 9>> svd(A, Eigen::ComputeFullV);
  const Eigen::Matrix<double, 9, 1> hh = svd.matrixV().col(8);
  Eigen::Matrix3d H;
  H << hh(0), hh(1), hh(2), hh(3), hh(4), hh(5), hh(6), hh(7), hh(8);
  return H;
}

// Warp the (cells x cells) canonical grid out of the image with bilinear
// sampling. Returns false if any sample falls outside the frame.
bool samplePatch(const uint8_t* gray, int w, int h, int stride, const Eigen::Matrix3d& H,
                 int cells, std::vector<double>& patch) {
  const int ps = cells * kPixelsPerCell;
  patch.assign(ps * ps, 0.0);
  // CUDA seam: per-pixel projective warp + bilinear gather.
  for (int v = 0; v < ps; ++v) {
    for (int u = 0; u < ps; ++u) {
      const double s = (u + 0.5) / kPixelsPerCell;
      const double t = (v + 0.5) / kPixelsPerCell;
      const double Z = H(2, 0) * s + H(2, 1) * t + H(2, 2);
      if (std::abs(Z) < 1e-12) return false;
      const double X = (H(0, 0) * s + H(0, 1) * t + H(0, 2)) / Z;
      const double Y = (H(1, 0) * s + H(1, 1) * t + H(1, 2)) / Z;
      if (X < 0.0 || Y < 0.0 || X > w - 1 || Y > h - 1) return false;
      const int x0 = static_cast<int>(X), y0 = static_cast<int>(Y);
      const int x1 = std::min(x0 + 1, w - 1), y1 = std::min(y0 + 1, h - 1);
      const double fx = X - x0, fy = Y - y0;
      const double g00 = gray[y0 * stride + x0], g01 = gray[y0 * stride + x1];
      const double g10 = gray[y1 * stride + x0], g11 = gray[y1 * stride + x1];
      patch[v * ps + u] =
          (1.0 - fy) * ((1.0 - fx) * g00 + fx * g01) + fy * ((1.0 - fx) * g10 + fx * g11);
    }
  }
  return true;
}

double otsuThreshold(const std::vector<double>& vals) {
  int hist[256] = {};
  for (double v : vals) ++hist[std::clamp(static_cast<int>(std::lround(v)), 0, 255)];
  const double total = static_cast<double>(vals.size());
  double sum = 0.0;
  for (int i = 0; i < 256; ++i) sum += i * hist[i];
  double sum_b = 0.0, w_b = 0.0, best = -1.0;
  int thresh = 0;
  for (int i = 0; i < 256; ++i) {
    w_b += hist[i];
    if (w_b == 0.0) continue;
    const double w_f = total - w_b;
    if (w_f == 0.0) break;
    sum_b += i * hist[i];
    const double diff = sum_b / w_b - (sum - sum_b) / w_f;
    const double between = w_b * w_f * diff * diff;
    if (between > best) {
      best = between;
      thresh = i;
    }
  }
  return static_cast<double>(thresh);
}

// cornerSubPix fixed-point iteration: p minimizes sum_i (g_i . (p - q_i))^2
// over the window, with central-difference gradients on the grayscale image.
Eigen::Vector2d refineCornerSubpix(const uint8_t* gray, int w, int h, int stride,
                                   Eigen::Vector2d p, int win, int iters) {
  for (int it = 0; it < iters; ++it) {
    const int cx = static_cast<int>(std::lround(p.x()));
    const int cy = static_cast<int>(std::lround(p.y()));
    if (cx - win - 1 < 0 || cy - win - 1 < 0 || cx + win + 1 >= w || cy + win + 1 >= h)
      break;
    double a00 = 0.0, a01 = 0.0, a11 = 0.0, b0 = 0.0, b1 = 0.0;
    // CUDA seam: windowed gradient accumulation (atomics or block reduce).
    for (int y = cy - win; y <= cy + win; ++y) {
      for (int x = cx - win; x <= cx + win; ++x) {
        const double gx = 0.5 * (gray[y * stride + x + 1] - gray[y * stride + x - 1]);
        const double gy = 0.5 * (gray[(y + 1) * stride + x] - gray[(y - 1) * stride + x]);
        const double gq = gx * x + gy * y;
        a00 += gx * gx;
        a01 += gx * gy;
        a11 += gy * gy;
        b0 += gx * gq;
        b1 += gy * gq;
      }
    }
    const double det = a00 * a11 - a01 * a01;
    if (std::abs(det) < 1e-9) break;
    const Eigen::Vector2d np((a11 * b0 - a01 * b1) / det, (a00 * b1 - a01 * b0) / det);
    if (!np.allFinite()) break;
    const double delta = (np - p).norm();
    p = np;
    if (delta < 1e-3) break;
  }
  return p;
}

}  // namespace

CharucoDetector::CharucoDetector(const CharucoBoard& board, DetectorConfig cfg)
    : board_(board), cfg_(cfg) {}

DetectionResult CharucoDetector::detect(const uint8_t* gray, int width, int height,
                                        int stride) const {
  DetectionResult result;
  if (!gray || width <= 0 || height <= 0 || stride < width) return result;
  const int ms = board_.dict.marker_size;
  const int dict_size = board_.dict.size();
  if (ms <= 0 || dict_size == 0) return result;
  const int cells = ms + 2;
  const int bits = ms * ms;

  int window = cfg_.adaptive_window | 1;
  if (window < 3) window = 3;
  const std::vector<uint8_t> bin =
      adaptiveThreshold(gray, width, height, stride, window, cfg_.adaptive_offset);
  const auto contours = outerContours(bin, width, height);

  std::vector<Quad> quads;
  for (const auto& cont : contours) {
    if (cont.size() < 4) continue;
    const double per = polygonPerimeter(cont);
    if (per < cfg_.min_marker_perimeter || per > cfg_.max_marker_perimeter) continue;
    const auto approx = approxPolygon(cont, cfg_.polygonal_epsilon * per);
    if (approx.size() != 4 || !isConvexQuad(approx)) continue;
    const double quad_per = polygonPerimeter(approx);
    if (quad_per < cfg_.min_marker_perimeter || quad_per > cfg_.max_marker_perimeter)
      continue;
    bool at_edge = false;
    for (const auto& v : approx)
      if (v.x() < kBorderMargin || v.y() < kBorderMargin ||
          v.x() > width - 1 - kBorderMargin || v.y() > height - 1 - kBorderMargin)
        at_edge = true;
    if (at_edge) continue;
    Quad q;
    q.corners = orderCorners(approx);
    q.center = (q.corners[0] + q.corners[1] + q.corners[2] + q.corners[3]) / 4.0;
    bool duplicate = false;
    for (const auto& kept : quads)
      if ((kept.center - q.center).norm() < cfg_.min_corner_separation) duplicate = true;
    if (duplicate) continue;
    quads.push_back(q);
  }

  const std::array<Eigen::Vector2d, 4> canon = {
      Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(cells, 0.0),
      Eigen::Vector2d(cells, cells), Eigen::Vector2d(0.0, cells)};

  std::vector<char> id_used(dict_size, 0);
  for (const auto& q : quads) {
    const Eigen::Matrix3d H = homographyDlt(canon, q.corners);
    std::vector<double> patch;
    if (!samplePatch(gray, width, height, stride, H, cells, patch)) continue;
    const double thr = otsuThreshold(patch);
    const int ps = cells * kPixelsPerCell;
    const int inner = kPixelsPerCell / 4;
    auto cell_mean = [&](int r, int c) {
      double s = 0.0;
      const int u0 = c * kPixelsPerCell + inner, u1 = (c + 1) * kPixelsPerCell - inner;
      const int v0 = r * kPixelsPerCell + inner, v1 = (r + 1) * kPixelsPerCell - inner;
      for (int v = v0; v < v1; ++v)
        for (int u = u0; u < u1; ++u) s += patch[v * ps + u];
      return s / ((u1 - u0) * (v1 - v0));
    };

    bool border_ok = true;
    for (int r = 0; r < cells && border_ok; ++r)
      for (int c = 0; c < cells; ++c)
        if ((r == 0 || c == 0 || r == cells - 1 || c == cells - 1) && cell_mean(r, c) > thr)
          border_ok = false;
    if (!border_ok) continue;

    uint64_t word = 0;
    for (int r = 0; r < ms; ++r)
      for (int c = 0; c < ms; ++c)
        if (cell_mean(r + 1, c + 1) > thr) word |= 1ULL << (r * ms + c);

    int best_id = -1, best_rot = 0, best_dist = std::numeric_limits<int>::max();
    for (int id = 0; id < dict_size; ++id) {
      uint64_t rw = board_.dict.words[id];
      for (int k = 0; k < 4; ++k) {
        const int d = Dictionary::hamming(word, rw, bits);
        if (d < best_dist) {
          best_dist = d;
          best_id = id;
          best_rot = k;
        }
        rw = Dictionary::rotatedWord(rw, ms, 1);
      }
    }
    if (best_id < 0 || best_dist > cfg_.max_hamming_correction) continue;
    if (id_used[best_id]) continue;  // one physical marker per id per frame
    id_used[best_id] = 1;

    DetectedMarker m;
    m.id = static_cast<uint32_t>(best_id);
    // obs == canonical word rotated CW by best_rot, so canonical corner i sits
    // at image corner index (i + best_rot) mod 4.
    for (int i = 0; i < 4; ++i) m.corners[i] = q.corners[(i + best_rot) & 3];
    for (auto& c : m.corners)
      c = refineCornerSubpix(gray, width, height, stride, c, cfg_.corner_refinement_window,
                             cfg_.corner_refinement_iters);
    result.markers.push_back(m);
  }

  std::vector<const DetectedMarker*> by_index(dict_size, nullptr);
  for (const auto& m : result.markers) by_index[m.id] = &m;

  const int nx = board_.squares_x - 1, ny = board_.squares_y - 1;
  for (int y = 0; y < ny; ++y) {
    for (int x = 0; x < nx; ++x) {
      const int adj[4][2] = {{x, y}, {x + 1, y}, {x, y + 1}, {x + 1, y + 1}};
      Eigen::Vector2d acc = Eigen::Vector2d::Zero();
      int count = 0;
      for (const auto& sq : adj) {
        const int mid = board_.markerAt(sq[0], sq[1]);
        if (mid < 0 || mid >= dict_size || !by_index[mid]) continue;
        const bool right = (sq[0] == x), bottom = (sq[1] == y);
        const int ci = bottom ? (right ? 2 : 3) : (right ? 1 : 0);
        acc += by_index[mid]->corners[ci];
        ++count;
      }
      if (count == 0) continue;
      Eigen::Vector2d uv = acc / count;
      uv = refineCornerSubpix(gray, width, height, stride, uv,
                              cfg_.corner_refinement_window, cfg_.corner_refinement_iters);
      result.charuco_corners.push_back({static_cast<uint32_t>(y * nx + x), uv});
    }
  }
  return result;
}

}  // namespace mcc
