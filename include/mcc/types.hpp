#pragma once

// Core geometric types for the multicam-calibration (mcc) library.
// Everything is plain Eigen; no OpenCV, no Ceres, no vendor SDK.

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace mcc {

// Pose semantics: camera-from-world rigid transform.
// A world-frame point pw maps into the camera frame as pc = q * pw + t.
// Composition a * b applies b first, then a (like matrix products).
struct Pose {
  Eigen::Quaterniond q{Eigen::Quaterniond::Identity()};
  Eigen::Vector3d t{Eigen::Vector3d::Zero()};

  Eigen::Matrix3d R() const { return q.normalized().toRotationMatrix(); }
  Eigen::Vector3d apply(const Eigen::Vector3d& pw) const { return q * pw + t; }
  Pose inverse() const {
    Eigen::Quaterniond qi = q.normalized().conjugate();
    return {qi, qi * (-t)};
  }
  Pose operator*(const Pose& b) const {
    Eigen::Quaterniond qa = q.normalized(), qb = b.q.normalized();
    return {qa * qb, qa * b.t + t};
  }
};

// Pinhole intrinsics with Brown-Conrady distortion (k1, k2, p1, p2, k3),
// coefficient order identical to OpenCV's 5-parameter model.
struct Intrinsics {
  double fx = 0.0, fy = 0.0, cx = 0.0, cy = 0.0;
  std::array<double, 5> dist{0.0, 0.0, 0.0, 0.0, 0.0};

  // Project a camera-frame point to pixels, with distortion applied.
  // Points at/behind the camera plane still return a finite value; callers
  // that need validity must check pc.z() > 0 themselves.
  Eigen::Vector2d project(const Eigen::Vector3d& pc) const {
    const double x = pc.x() / pc.z(), y = pc.y() / pc.z();
    const double r2 = x * x + y * y, r4 = r2 * r2, r6 = r4 * r2;
    const double radial = 1.0 + dist[0] * r2 + dist[1] * r4 + dist[4] * r6;
    const double xd = x * radial + 2.0 * dist[2] * x * y + dist[3] * (r2 + 2.0 * x * x);
    const double yd = y * radial + dist[2] * (r2 + 2.0 * y * y) + 2.0 * dist[3] * x * y;
    return {fx * xd + cx, fy * yd + cy};
  }

  Eigen::Vector2d projectUndistorted(const Eigen::Vector3d& pc) const {
    return {fx * (pc.x() / pc.z()) + cx, fy * (pc.y() / pc.z()) + cy};
  }
};

// Minimal Lie-algebra helpers (angle-axis / se(3) local parameterization).

inline Eigen::Vector3d quatToAngleAxis(const Eigen::Quaterniond& q_in) {
  Eigen::Quaterniond q = q_in.normalized();
  if (q.w() < 0.0) q.coeffs() *= -1.0;
  const double s = 2.0 * std::acos(std::clamp(q.w(), -1.0, 1.0));
  const double n = std::sqrt(std::max(1.0 - q.w() * q.w(), 1e-24));
  return q.vec() * (s / n);
}

inline Eigen::Quaterniond angleAxisToQuat(const Eigen::Vector3d& aa) {
  const double th = aa.norm();
  if (th < 1e-12) {
    // First-order expansion keeps derivatives finite at zero.
    return Eigen::Quaterniond(1.0, 0.5 * aa.x(), 0.5 * aa.y(), 0.5 * aa.z()).normalized();
  }
  return Eigen::Quaterniond(Eigen::AngleAxisd(th, aa / th));
}

// Local update: pose' = pose boxplus delta, delta = [rotation_aa; translation]
// applied in the camera frame (left-multiplied rotation, translation additive
// in camera frame before mapping to world).
inline Pose poseBoxplus(const Pose& p, const Eigen::Matrix<double, 6, 1>& delta) {
  const Eigen::Quaterniond dq = angleAxisToQuat(delta.head<3>());
  return {p.q.normalized() * dq, p.t + p.q.normalized() * delta.tail<3>()};
}

}  // namespace mcc
