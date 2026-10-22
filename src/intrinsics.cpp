// Implementation of include/mcc/intrinsics.hpp — see the header for the
// pipeline overview. All Jacobians are analytic and consistent with the
// poseBoxplus convention in types.hpp (camera-frame perturbation).

#include "mcc/intrinsics.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

namespace mcc {
namespace {

struct FrameData {
  uint64_t frame_index = 0;
  std::vector<Eigen::Vector3d> pw;  // board-frame corner points (z = 0)
  std::vector<Eigen::Vector2d> uv;
  Eigen::Matrix3d H = Eigen::Matrix3d::Identity();
};

// Hartley isotropic normalization: zero centroid, mean distance sqrt(2).
bool hartleyNormalize(const std::vector<Eigen::Vector2d>& pts, Eigen::Matrix3d& T) {
  Eigen::Vector2d c = Eigen::Vector2d::Zero();
  for (const auto& p : pts) c += p;
  c /= static_cast<double>(pts.size());
  double mean_d = 0.0;
  for (const auto& p : pts) mean_d += (p - c).norm();
  mean_d /= static_cast<double>(pts.size());
  if (mean_d < 1e-12) return false;
  const double s = std::sqrt(2.0) / mean_d;
  T << s, 0.0, -s * c.x(), 0.0, s, -s * c.y(), 0.0, 0.0, 1.0;
  return true;
}

// Normalized DLT homography src -> dst (both 2D; src is the board plane).
bool homographyDlt(const std::vector<Eigen::Vector2d>& src,
                   const std::vector<Eigen::Vector2d>& dst,
                   Eigen::Matrix3d& H) {
  const int n = static_cast<int>(src.size());
  if (n < 4) return false;
  Eigen::Matrix3d Ts, Td;
  if (!hartleyNormalize(src, Ts) || !hartleyNormalize(dst, Td)) return false;
  Eigen::MatrixXd A(2 * n, 9);
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d X = (Ts * src[i].homogeneous()).head<2>();
    const Eigen::Vector2d u = (Td * dst[i].homogeneous()).head<2>();
    A.row(2 * i) << -X.x(), -X.y(), -1.0, 0.0, 0.0, 0.0, u.x() * X.x(),
        u.x() * X.y(), u.x();
    A.row(2 * i + 1) << 0.0, 0.0, 0.0, -X.x(), -X.y(), -1.0, u.y() * X.x(),
        u.y() * X.y(), u.y();
  }
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::VectorXd h = svd.matrixV().col(8);
  Eigen::Matrix3d Hn;
  Hn << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
  H = Td.inverse() * Hn * Ts;
  if (!H.allFinite() || std::abs(H(2, 2)) < 1e-12) return false;
  H /= H(2, 2);
  return true;
}

// Camera-from-board pose from a board->image homography and K (distortion
// ignored at this stage, refined later). Sign chosen so the board sits in
// front of the camera; R projected onto SO(3) via SVD polar decomposition.
bool poseFromHomography(const Eigen::Matrix3d& H, const Intrinsics& intr,
                        Pose& out) {
  if (intr.fx <= 0.0 || intr.fy <= 0.0) return false;
  Eigen::Matrix3d Kinv;
  Kinv << 1.0 / intr.fx, 0.0, -intr.cx / intr.fx, 0.0, 1.0 / intr.fy,
      -intr.cy / intr.fy, 0.0, 0.0, 1.0;
  const Eigen::Matrix3d M = Kinv * H;
  const double n1 = M.col(0).norm(), n2 = M.col(1).norm();
  if (n1 < 1e-12 || n2 < 1e-12) return false;
  const double lambda = 2.0 / (n1 + n2);
  Eigen::Vector3d r1 = lambda * M.col(0);
  Eigen::Vector3d r2 = lambda * M.col(1);
  Eigen::Vector3d t = lambda * M.col(2);
  if (t.z() < 0.0) {
    r1 = -r1;
    r2 = -r2;
    t = -t;
  }
  Eigen::Matrix3d R;
  R.col(0) = r1;
  R.col(1) = r2;
  R.col(2) = r1.cross(r2);
  const Eigen::JacobiSVD<Eigen::Matrix3d> svd(
      R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  const Eigen::Matrix3d Vt = svd.matrixV().transpose();
  if ((U * Vt).determinant() < 0.0) U.col(2) *= -1.0;
  out.q = Eigen::Quaterniond(U * Vt).normalized();
  out.t = t;
  return true;
}

// d(uv)/d(pc) for Intrinsics::project (Brown-Conrady, 5 coeffs).
Eigen::Matrix<double, 2, 3> projectJacPoint(const Intrinsics& intr,
                                            const Eigen::Vector3d& pc) {
  const double x = pc.x() / pc.z(), y = pc.y() / pc.z();
  const double r2 = x * x + y * y, r4 = r2 * r2, r6 = r4 * r2;
  const double k1 = intr.dist[0], k2 = intr.dist[1], p1 = intr.dist[2],
               p2 = intr.dist[3], k3 = intr.dist[4];
  const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
  // d(radial)/dx = 2x*g, d(radial)/dy = 2y*g.
  const double g = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
  const double dxd_dx = radial + 2.0 * x * x * g + 2.0 * p1 * y + 6.0 * p2 * x;
  const double dxd_dy = 2.0 * x * y * g + 2.0 * p1 * x + 2.0 * p2 * y;
  const double dyd_dy = radial + 2.0 * y * y * g + 6.0 * p1 * y + 2.0 * p2 * x;
  const double iz = 1.0 / pc.z();
  Eigen::Matrix<double, 2, 3> J;
  J(0, 0) = intr.fx * dxd_dx * iz;
  J(0, 1) = intr.fx * dxd_dy * iz;
  J(0, 2) = intr.fx * (-dxd_dx * x - dxd_dy * y) * iz;
  J(1, 0) = intr.fy * dxd_dy * iz;
  J(1, 1) = intr.fy * dyd_dy * iz;
  J(1, 2) = intr.fy * (-dxd_dy * x - dyd_dy * y) * iz;
  return J;
}

// d(uv)/d(delta) for poseBoxplus (right-composition, camera-frame delta):
// pc = q*pw + t with q' = q*dq, t' = t + q*dt gives, exactly,
// d(pc)/d(rot) = -R*[pw]_x, d(pc)/d(trans) = R, R = q.toRotationMatrix().
Eigen::Matrix<double, 2, 6> projectJacPose(const Intrinsics& intr,
                                           const Eigen::Vector3d& pc,
                                           const Eigen::Vector3d& pw,
                                           const Eigen::Matrix3d& R) {
  const Eigen::Matrix<double, 2, 3> Jp = projectJacPoint(intr, pc);
  Eigen::Matrix3d skew_pw;
  skew_pw << 0.0, -pw.z(), pw.y(), pw.z(), 0.0, -pw.x(), -pw.y(), pw.x(),
      0.0;
  Eigen::Matrix<double, 2, 6> J;
  J.leftCols<3>() = -Jp * R * skew_pw;
  J.rightCols<3>() = Jp * R;
  return J;
}

// d(uv)/d([fx, fy, cx, cy, k1, k2, p1, p2, k3]).
Eigen::Matrix<double, 2, 9> projectJacIntrinsics(const Intrinsics& intr,
                                                 const Eigen::Vector3d& pc) {
  const double x = pc.x() / pc.z(), y = pc.y() / pc.z();
  const double r2 = x * x + y * y, r4 = r2 * r2, r6 = r4 * r2;
  const double k1 = intr.dist[0], k2 = intr.dist[1], p1 = intr.dist[2],
               p2 = intr.dist[3], k3 = intr.dist[4];
  const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
  const double xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  const double yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
  Eigen::Matrix<double, 2, 9> J = Eigen::Matrix<double, 2, 9>::Zero();
  J(0, 0) = xd;
  J(0, 2) = 1.0;
  J(0, 4) = intr.fx * x * r2;
  J(0, 5) = intr.fx * x * r4;
  J(0, 6) = intr.fx * 2.0 * x * y;
  J(0, 7) = intr.fx * (r2 + 2.0 * x * x);
  J(0, 8) = intr.fx * x * r6;
  J(1, 1) = yd;
  J(1, 3) = 1.0;
  J(1, 4) = intr.fy * y * r2;
  J(1, 5) = intr.fy * y * r4;
  J(1, 6) = intr.fy * (r2 + 2.0 * y * y);
  J(1, 7) = intr.fy * 2.0 * x * y;
  J(1, 8) = intr.fy * y * r6;
  return J;
}

std::vector<FrameData> collectFrames(const CameraCache& cache,
                                     const CharucoBoard& board,
                                     int min_corners) {
  std::vector<FrameData> out;
  for (const auto& f : cache.frames) {
    if (static_cast<int>(f.corners.size()) < min_corners) continue;
    FrameData fd;
    fd.frame_index = f.frame_index;
    std::vector<Eigen::Vector2d> src, dst;
    for (const auto& c : f.corners) {
      Eigen::Vector3d p;
      if (!board.cornerPoint(c.id, p)) continue;
      fd.pw.push_back(p);
      fd.uv.push_back(c.uv);
      src.emplace_back(p.x(), p.y());
      dst.push_back(c.uv);
    }
    if (static_cast<int>(fd.pw.size()) < min_corners) continue;
    if (!homographyDlt(src, dst, fd.H)) continue;
    out.push_back(std::move(fd));
  }
  return out;
}

// Sum of per-frame corner bbox areas, clamped to the image area.
double cornerCoverage(const std::vector<FrameData>& frames, int w, int h) {
  if (w <= 0 || h <= 0) return 0.0;
  double area = 0.0;
  for (const auto& f : frames) {
    Eigen::Vector2d lo = f.uv.front(), hi = f.uv.front();
    for (const auto& u : f.uv) {
      lo = lo.cwiseMin(u);
      hi = hi.cwiseMax(u);
    }
    area += std::max(0.0, (hi.x() - lo.x()) * (hi.y() - lo.y()));
  }
  return std::min(1.0, area / (static_cast<double>(w) * h));
}

// Zhang closed form: nullspace of the v_ij constraint stack gives
// b = [B11, B12, B22, B13, B23, B33] with B = lambda * K^-T K^-1 (skew 0).
bool zhangClosedForm(const std::vector<FrameData>& frames, Intrinsics& out,
                     std::string& err) {
  const int n = static_cast<int>(frames.size());
  Eigen::MatrixXd V(2 * n, 6);
  const auto vij = [](const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    Eigen::Matrix<double, 6, 1> v;
    v << a.x() * b.x(), a.x() * b.y() + a.y() * b.x(), a.y() * b.y(),
        a.z() * b.x() + a.x() * b.z(), a.z() * b.y() + a.y() * b.z(),
        a.z() * b.z();
    return v;
  };
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d h1 = frames[i].H.col(0);
    const Eigen::Vector3d h2 = frames[i].H.col(1);
    V.row(2 * i) = vij(h1, h2).transpose();
    V.row(2 * i + 1) = (vij(h1, h1) - vij(h2, h2)).transpose();
  }
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(V, Eigen::ComputeFullV);
  const Eigen::VectorXd b = svd.matrixV().col(5);
  double B11 = b(0), B12 = b(1), B22 = b(2), B13 = b(3), B23 = b(4), B33 = b(5);
  if (B11 < 0.0) {  // B defined up to sign; want positive definite
    B11 = -B11;
    B12 = -B12;
    B22 = -B22;
    B13 = -B13;
    B23 = -B23;
    B33 = -B33;
  }
  const double det2 = B11 * B22 - B12 * B12;
  if (!(B11 > 0.0) || !(det2 > 0.0)) {
    err = "zhang closed form: image-of-absolute-conic not positive definite";
    return false;
  }
  const double v0 = (B12 * B13 - B11 * B23) / det2;
  const double lambda =
      B33 - (B13 * B13 + v0 * (B12 * B13 - B11 * B23)) / B11;
  if (!(lambda > 0.0)) {
    err = "zhang closed form: non-positive scale (degenerate views)";
    return false;
  }
  const double fx = std::sqrt(lambda / B11);
  const double fy = std::sqrt(lambda * B11 / det2);
  const double cx = -B13 * fx * fx / lambda;  // skew forced to zero
  const double cy = v0;
  if (!std::isfinite(fx) || !std::isfinite(fy) || !std::isfinite(cx) ||
      !std::isfinite(cy)) {
    err = "zhang closed form: non-finite intrinsics";
    return false;
  }
  out.fx = fx;
  out.fy = fy;
  out.cx = cx;
  out.cy = cy;
  return true;
}

Intrinsics applyIntrinsicsDelta(const Intrinsics& in, const Eigen::VectorXd& d,
                                bool fix_aspect) {
  Intrinsics out = in;
  if (fix_aspect) {
    out.fx = out.fy = in.fx + d[0];
    out.cx = in.cx + d[1];
    out.cy = in.cy + d[2];
    for (int k = 0; k < 5; ++k) out.dist[k] = in.dist[k] + d[3 + k];
  } else {
    out.fx = in.fx + d[0];
    out.fy = in.fy + d[1];
    out.cx = in.cx + d[2];
    out.cy = in.cy + d[3];
    for (int k = 0; k < 5; ++k) out.dist[k] = in.dist[k] + d[4 + k];
  }
  return out;
}

double reprojectionCost(const Intrinsics& intr,
                        const std::vector<Pose>& poses,
                        const std::vector<FrameData>& frames) {
  double cost = 0.0;
  for (size_t f = 0; f < frames.size(); ++f)
    for (size_t i = 0; i < frames[f].pw.size(); ++i)
      cost += (frames[f].uv[i] - intr.project(poses[f].apply(frames[f].pw[i])))
                  .squaredNorm();
  return cost;
}

// Marquardt LM over {intrinsics, distortion, all board poses}. Returns final
// per-residual RMS in pixels.
double refineLm(const IntrinsicsConfig& cfg, Intrinsics& intr,
                std::vector<Pose>& poses,
                const std::vector<FrameData>& frames) {
  const int w = cfg.fix_aspect_ratio ? 8 : 9;  // intrinsics block width
  const int F = static_cast<int>(frames.size());
  const int n = w + 6 * F;
  double m = 0.0;
  for (const auto& f : frames) m += 2.0 * f.pw.size();

  double lambda = 1e-3, nu = 2.0;
  double cost = reprojectionCost(intr, poses, frames);
  for (int it = 0; it < cfg.max_lm_iters; ++it) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
    cost = 0.0;
    for (int f = 0; f < F; ++f) {
      const int po = w + 6 * f;
      for (size_t i = 0; i < frames[f].pw.size(); ++i) {
        const Eigen::Vector3d pc = poses[f].apply(frames[f].pw[i]);
        const Eigen::Vector2d r =
            frames[f].uv[i] - intr.project(pc);
        cost += r.squaredNorm();
        const Eigen::Matrix<double, 2, 9> Ji9 = projectJacIntrinsics(intr, pc);
        Eigen::MatrixXd Jr_i(2, w);  // residual Jacobian = -J
        if (cfg.fix_aspect_ratio) {
          Jr_i.col(0) = -(Ji9.col(0) + Ji9.col(1));
          Jr_i.col(1) = -Ji9.col(2);
          Jr_i.col(2) = -Ji9.col(3);
          Jr_i.rightCols<5>() = -Ji9.rightCols<5>();
        } else {
          Jr_i = -Ji9;
        }
        const Eigen::Matrix<double, 2, 6> Jr_p =
            -projectJacPose(intr, pc, frames[f].pw[i], poses[f].R());
        H.block(0, 0, w, w) += Jr_i.transpose() * Jr_i;
        H.block(0, po, w, 6) += Jr_i.transpose() * Jr_p;
        H.block(po, 0, 6, w) += Jr_p.transpose() * Jr_i;
        H.block(po, po, 6, 6) += Jr_p.transpose() * Jr_p;
        g.head(w) += Jr_i.transpose() * r;
        g.segment<6>(po) += Jr_p.transpose() * r;
      }
    }
    const double prev_cost = cost;
    Eigen::VectorXd D = H.diagonal();
    for (int i = 0; i < n; ++i)
      if (D[i] < 1e-12) D[i] = 1e-12;

    bool accepted = false;
    for (int attempt = 0; attempt < 10; ++attempt) {
      Eigen::MatrixXd A = H;
      A.diagonal() += lambda * D;
      const Eigen::VectorXd delta = A.ldlt().solve(-g);
      if (!delta.allFinite()) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }
      const Intrinsics ti =
          applyIntrinsicsDelta(intr, delta.head(w), cfg.fix_aspect_ratio);
      if (ti.fx <= 0.0 || ti.fy <= 0.0) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }
      std::vector<Pose> tp(poses.size());
      for (int f = 0; f < F; ++f)
        tp[f] = poseBoxplus(poses[f], delta.segment<6>(w + 6 * f));
      const double new_cost = reprojectionCost(ti, tp, frames);
      const double predicted =
          delta.dot(lambda * (D.asDiagonal() * delta) - g);
      const double rho =
          (predicted > 0.0) ? (cost - new_cost) / predicted : -1.0;
      if (rho > 0.0 && std::isfinite(new_cost)) {
        intr = ti;
        poses = std::move(tp);
        lambda = std::max(
            1e-9, lambda * std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * rho - 1.0, 3)));
        nu = 2.0;
        cost = new_cost;
        accepted = true;
        break;
      }
      lambda *= nu;
      nu *= 2.0;
    }
    if (!accepted) break;
    if (prev_cost - cost < 1e-12 * std::max(1.0, prev_cost)) break;
  }
  return std::sqrt(cost / m);
}

}  // namespace

IntrinsicsResult calibrateIntrinsics(const CameraCache& cache,
                                     const CharucoBoard& board,
                                     const IntrinsicsConfig& cfg) {
  IntrinsicsResult res;
  std::vector<FrameData> frames =
      collectFrames(cache, board, cfg.min_corners_per_frame);
  const int need = std::max(cfg.min_frames, 3);
  if (static_cast<int>(frames.size()) < need) {
    res.failure_reason = "only " + std::to_string(frames.size()) +
                         " usable frames (need " + std::to_string(need) + ")";
    return res;
  }
  const double coverage =
      cornerCoverage(frames, cache.image_width, cache.image_height);
  if (coverage < cfg.min_coverage) {
    res.failure_reason = "corner coverage " + std::to_string(coverage) +
                         " below minimum " + std::to_string(cfg.min_coverage);
    return res;
  }
  Intrinsics intr;
  std::string err;
  if (!zhangClosedForm(frames, intr, err)) {
    res.failure_reason = err;
    return res;
  }
  if (cfg.fix_aspect_ratio) {
    const double f = 0.5 * (intr.fx + intr.fy);
    intr.fx = intr.fy = f;
  }
  std::vector<Pose> poses(frames.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    if (!poseFromHomography(frames[i].H, intr, poses[i])) {
      res.failure_reason = "homography decomposition failed on frame " +
                           std::to_string(frames[i].frame_index);
      return res;
    }
  }
  res.rms_px = refineLm(cfg, intr, poses, frames);
  res.intrinsics = intr;
  res.board_poses = std::move(poses);
  res.frame_indices.reserve(frames.size());
  for (const auto& f : frames) res.frame_indices.push_back(f.frame_index);
  res.ok = true;
  return res;
}

std::vector<IntrinsicsResult> calibrateAllIntrinsics(
    const std::vector<CameraCache>& caches, const CharucoBoard& board,
    const IntrinsicsConfig& cfg) {
  std::vector<IntrinsicsResult> out;
  out.reserve(caches.size());
  for (const auto& c : caches) out.push_back(calibrateIntrinsics(c, board, cfg));
  return out;
}

}  // namespace mcc
