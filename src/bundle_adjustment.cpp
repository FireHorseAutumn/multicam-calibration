// Global bundle adjustment: LM with analytic Jacobians, Huber IRLS
// weighting, Schur complement over per-frame board-pose blocks.
// See include/mcc/bundle_adjustment.hpp for the structure overview.

#include "mcc/bundle_adjustment.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <utility>
#include <vector>

#include "mcc/batched_solve.hpp"

namespace mcc {

namespace {

Eigen::Matrix3d skew(const Eigen::Vector3d& v) {
  Eigen::Matrix3d m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

// d(p.apply(x))/d(delta) under the poseBoxplus local parameterization of
// types.hpp (right-composition T' = T*dT: q' = q*dq, t' = t + q*dt):
//   p'.apply(x) = p.apply(x) + R*(-[x]_x*w + dt)  =>  J = [-R*[x]_x | R].
// The frequently quoted [-[pc]_x | I] belongs to the left-composition update
// (q' = dq*q, t' = dq*t + dt), which poseBoxplus does NOT implement; using it
// here would make the LM step inconsistent with the update. The pose p below
// is whichever factor is being perturbed, composed with everything upstream
// of it (for board blocks: camera_pose * board_pose).
Eigen::Matrix<double, 3, 6> pose_jacobian(const Pose& p, const Eigen::Vector3d& x) {
  Eigen::Matrix<double, 3, 6> J;
  const Eigen::Matrix3d R = p.R();
  J.block<3, 3>(0, 0) = -R * skew(x);
  J.block<3, 3>(0, 3) = R;
  return J;
}

// d(uv)/d(pc) for Intrinsics::project (Brown-Conrady).
Eigen::Matrix<double, 2, 3> projection_jacobian(const Intrinsics& K, const Eigen::Vector3d& pc) {
  const double z = std::abs(pc.z()) < 1e-12 ? (pc.z() < 0.0 ? -1e-12 : 1e-12) : pc.z();
  const double x = pc.x() / z, y = pc.y() / z;
  const double r2 = x * x + y * y;
  const double k1 = K.dist[0], k2 = K.dist[1], p1 = K.dist[2], p2 = K.dist[3], k3 = K.dist[4];
  const double radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2;
  const double a = 2.0 * k1 + 4.0 * k2 * r2 + 6.0 * k3 * r2 * r2;  // d(radial)/d(x) = a*x
  const double dxd_dx = radial + a * x * x + 2.0 * p1 * y + 6.0 * p2 * x;
  const double dxd_dy = a * x * y + 2.0 * p1 * x + 2.0 * p2 * y;
  const double dyd_dy = radial + a * y * y + 6.0 * p1 * y + 2.0 * p2 * x;
  Eigen::Matrix<double, 2, 3> J;
  J(0, 0) = K.fx * dxd_dx / z;
  J(0, 1) = K.fx * dxd_dy / z;
  J(0, 2) = -K.fx * (x * dxd_dx + y * dxd_dy) / z;
  J(1, 0) = K.fy * dxd_dy / z;  // dyd/dx == dxd/dy
  J(1, 1) = K.fy * dyd_dy / z;
  J(1, 2) = -K.fy * (x * dxd_dy + y * dyd_dy) / z;
  return J;
}

// d(uv)/d(theta), theta = [fx, fy, cx, cy, k1, k2, p1, p2, k3].
Eigen::Matrix<double, 2, 9> intrinsics_jacobian(const Intrinsics& K, const Eigen::Vector3d& pc) {
  const double z = std::abs(pc.z()) < 1e-12 ? (pc.z() < 0.0 ? -1e-12 : 1e-12) : pc.z();
  const double x = pc.x() / z, y = pc.y() / z;
  const double r2 = x * x + y * y, r4 = r2 * r2, r6 = r4 * r2;
  const double k1 = K.dist[0], k2 = K.dist[1], p1 = K.dist[2], p2 = K.dist[3], k3 = K.dist[4];
  const double radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
  const double xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x);
  const double yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y;
  Eigen::Matrix<double, 2, 9> J;
  J.setZero();
  J(0, 0) = xd;
  J(0, 2) = 1.0;
  J(1, 1) = yd;
  J(1, 3) = 1.0;
  J(0, 4) = K.fx * x * r2;
  J(0, 5) = K.fx * x * r4;
  J(0, 6) = K.fx * 2.0 * x * y;
  J(0, 7) = K.fx * (r2 + 2.0 * x * x);
  J(0, 8) = K.fx * x * r6;
  J(1, 4) = K.fy * y * r2;
  J(1, 5) = K.fy * y * r4;
  J(1, 6) = K.fy * (r2 + 2.0 * y * y);
  J(1, 7) = K.fy * 2.0 * x * y;
  J(1, 8) = K.fy * y * r6;
  return J;
}

double huber_sqrt_weight(double r, double delta) {
  const double a = std::abs(r);
  return a <= delta ? 1.0 : std::sqrt(delta / a);
}

double huber_rho(double r, double delta) {
  const double a = std::abs(r);
  return a <= delta ? 0.5 * r * r : delta * (a - 0.5 * delta);
}

// Iterative undistortion to normalized coordinates (fixed-point, 12 iters);
// used only for board-pose initialization.
Eigen::Vector2d undistort_normalized(const Intrinsics& K, const Eigen::Vector2d& uv) {
  const Eigen::Vector2d xd{(uv.x() - K.cx) / K.fx, (uv.y() - K.cy) / K.fy};
  Eigen::Vector2d x = xd;
  const auto& d = K.dist;
  for (int it = 0; it < 12; ++it) {
    const double r2 = x.squaredNorm();
    const double radial = 1.0 + d[0] * r2 + d[1] * r2 * r2 + d[4] * r2 * r2 * r2;
    const double tx = 2.0 * d[2] * x.x() * x.y() + d[3] * (r2 + 2.0 * x.x() * x.x());
    const double ty = d[2] * (r2 + 2.0 * x.y() * x.y()) + 2.0 * d[3] * x.x() * x.y();
    x = (xd - Eigen::Vector2d(tx, ty)) / radial;
  }
  return x;
}

// Planar pose init for one frame: normalized-DLT homography from board-plane
// points to undistorted normalized image coords, decomposed with unit
// intrinsics. `cam_pose` is the current camera-from-world estimate; the
// result is board-in-world (camera-0-from-board).
bool init_board_pose(const CharucoBoard& board, const Intrinsics& K, const Pose& cam_pose,
                     const FrameDetections& det, Pose& out) {
  std::vector<Eigen::Vector2d> src, dst;
  src.reserve(det.corners.size());
  dst.reserve(det.corners.size());
  for (const auto& c : det.corners) {
    Eigen::Vector3d pw;
    if (!board.cornerPoint(c.id, pw)) continue;
    src.emplace_back(pw.x(), pw.y());
    dst.push_back(undistort_normalized(K, c.uv));
  }
  const int m = static_cast<int>(src.size());
  if (m < 4) return false;

  auto normalizer = [](const std::vector<Eigen::Vector2d>& pts, Eigen::Matrix3d& T) {
    Eigen::Vector2d mean = Eigen::Vector2d::Zero();
    for (const auto& p : pts) mean += p;
    mean /= static_cast<double>(pts.size());
    double d = 0.0;
    for (const auto& p : pts) d += (p - mean).norm();
    const double s = d > 1e-12 ? std::sqrt(2.0) * static_cast<double>(pts.size()) / d : 1.0;
    T << s, 0.0, -s * mean.x(), 0.0, s, -s * mean.y(), 0.0, 0.0, 1.0;
  };
  Eigen::Matrix3d Ts, Td;
  normalizer(src, Ts);
  normalizer(dst, Td);

  Eigen::MatrixXd A(2 * m, 9);
  for (int i = 0; i < m; ++i) {
    const Eigen::Vector3d X = Ts * src[i].homogeneous();
    const Eigen::Vector3d x = Td * dst[i].homogeneous();
    A.row(2 * i) << 0.0, 0.0, 0.0, -X.x(), -X.y(), -X.z(), x.y() * X.x(), x.y() * X.y(),
        x.y() * X.z();
    A.row(2 * i + 1) << X.x(), X.y(), X.z(), 0.0, 0.0, 0.0, -x.x() * X.x(), -x.x() * X.y(),
        -x.x() * X.z();
  }
  const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  const Eigen::VectorXd hv = svd.matrixV().col(8);
  Eigen::Matrix3d Hn;
  Hn << hv(0), hv(1), hv(2), hv(3), hv(4), hv(5), hv(6), hv(7), hv(8);
  const Eigen::Matrix3d H = Td.inverse() * Hn * Ts;

  const double n1 = H.col(0).norm(), n2 = H.col(1).norm();
  if (n1 < 1e-12 || n2 < 1e-12) return false;
  const double s = 2.0 / (n1 + n2);
  Eigen::Vector3d r1 = s * H.col(0), r2 = s * H.col(1), t = s * H.col(2);
  if (t.z() < 0.0) {
    r1 = -r1;
    r2 = -r2;
    t = -t;
  }
  Eigen::Matrix3d R;
  R.col(0) = r1;
  R.col(1) = r2;
  R.col(2) = r1.cross(r2);
  const Eigen::JacobiSVD<Eigen::Matrix3d> rsvd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = rsvd.matrixU();
  Eigen::Matrix3d Ro = U * rsvd.matrixV().transpose();
  if (Ro.determinant() < 0.0) {
    U.col(2) *= -1.0;
    Ro = U * rsvd.matrixV().transpose();
  }
  const Pose t_cb{Eigen::Quaterniond(Ro), t};
  out = cam_pose.inverse() * t_cb;
  return out.t.allFinite() && out.q.coeffs().allFinite();
}

}  // namespace

BAResult BundleAdjuster::run(const std::vector<CameraCache>& caches,
                             const CharucoBoard& board,
                             const std::vector<Intrinsics>& intrinsics_init,
                             const std::vector<Pose>& camera_poses_init) const {
  BAResult result;
  const int ncams = static_cast<int>(caches.size());
  if (ncams == 0 || static_cast<int>(camera_poses_init.size()) != ncams ||
      static_cast<int>(intrinsics_init.size()) != ncams) {
    return result;
  }

  // Kept frames: sorted union of frame_index, >= 4 corners in some camera.
  std::map<uint64_t, int> max_corners;
  for (const auto& c : caches)
    for (const auto& f : c.frames)
      max_corners[f.frame_index] = std::max(max_corners[f.frame_index],
                                            static_cast<int>(f.corners.size()));
  std::vector<uint64_t> frame_ids;
  std::map<uint64_t, int> slot_of;
  for (const auto& [fi, mc] : max_corners)
    if (mc >= 4) {
      slot_of[fi] = static_cast<int>(frame_ids.size());
      frame_ids.push_back(fi);
    }
  const int nframes = static_cast<int>(frame_ids.size());

  struct Obs {
    int cam;
    Eigen::Vector2d uv;
    Eigen::Vector3d pw;
  };
  std::vector<std::vector<Obs>> obs_by_frame(nframes);
  std::vector<int> cam_obs_count(ncams, 0);
  std::vector<std::pair<int, const FrameDetections*>> frame_views(nframes, {-1, nullptr});
  for (int ci = 0; ci < ncams; ++ci) {
    for (const auto& f : caches[ci].frames) {
      const auto it = slot_of.find(f.frame_index);
      if (it == slot_of.end()) continue;
      const int slot = it->second;
      if (frame_views[slot].second == nullptr ||
          f.corners.size() > frame_views[slot].second->corners.size())
        frame_views[slot] = {ci, &f};
      for (const auto& co : f.corners) {
        Eigen::Vector3d pw;
        if (!board.cornerPoint(co.id, pw)) continue;
        obs_by_frame[slot].push_back({ci, co.uv, pw});
        ++cam_obs_count[ci];
      }
    }
  }
  int nobs = 0;
  for (const auto& v : obs_by_frame) nobs += static_cast<int>(v.size());
  if (nobs == 0) {
    result.camera_poses = camera_poses_init;
    result.intrinsics = intrinsics_init;
    result.converged = true;  // nothing to optimize
    return result;
  }

  // Board-pose initialization from the best view of each frame.
  std::vector<Pose> cur_boards(nframes);
  for (int s = 0; s < nframes; ++s) {
    const auto [ci, det] = frame_views[s];
    Pose p;
    if (det == nullptr || !init_board_pose(board, intrinsics_init[ci], camera_poses_init[ci],
                                           *det, p))
      p = Pose{Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.0, 0.0, 1.0)};
    cur_boards[s] = p;
  }
  std::vector<Pose> cur_cams = camera_poses_init;
  std::vector<Intrinsics> cur_intr = intrinsics_init;

  // Kept (reduced-system) blocks: active camera poses (cameras 1..N-1 with
  // observations), then active intrinsics blocks (any camera with
  // observations, if refinement is enabled). Camera 0's pose is the gauge.
  std::vector<int> cam_block(ncams, -1), intr_block(ncams, -1);
  int na = 0;
  for (int ci = 1; ci < ncams; ++ci)
    if (cam_obs_count[ci] > 0) cam_block[ci] = na++;
  int ni = 0;
  if (cfg_.refine_intrinsics)
    for (int ci = 0; ci < ncams; ++ci)
      if (cam_obs_count[ci] > 0) intr_block[ci] = ni++;
  const int kept_dim = 6 * na + 9 * ni;
  for (int ci = 0; ci < ncams; ++ci)
    if (cam_obs_count[ci] == 0 && cfg_.verbose)
      std::printf("mcc BA: camera %d (%s) has no observations; keeping its init pose\n", ci,
                  caches[ci].camera_name.c_str());

  // Per-frame list of kept blocks (offset, size) that observe it, plus each
  // block's row offset inside the frame's local coupling matrix E.
  struct FrameBlocks {
    std::vector<std::pair<int, int>> blocks;
    std::vector<int> local_row;
    int L = 0;
  };
  std::vector<FrameBlocks> fblocks(nframes);
  for (int s = 0; s < nframes; ++s) {
    std::map<int, int> offs;
    for (const auto& o : obs_by_frame[s]) {
      if (cam_block[o.cam] >= 0) offs[6 * cam_block[o.cam]] = 6;
      if (intr_block[o.cam] >= 0) offs[6 * na + 9 * intr_block[o.cam]] = 9;
    }
    int row = 0;
    for (const auto& [off, sz] : offs) {
      fblocks[s].blocks.push_back({off, sz});
      fblocks[s].local_row.push_back(row);
      row += sz;
    }
    fblocks[s].L = row;
  }

  const double huber = cfg_.huber_delta;
  auto eval = [&](const std::vector<Pose>& cams, const std::vector<Pose>& boards,
                  const std::vector<Intrinsics>& intr, double* raw_sum_sq) {
    double cost = 0.0, rss = 0.0;
    for (int s = 0; s < nframes; ++s)
      for (const auto& o : obs_by_frame[s]) {
        const Eigen::Vector3d pc = (cams[o.cam] * boards[s]).apply(o.pw);
        const Eigen::Vector2d r = o.uv - intr[o.cam].project(pc);
        rss += r.squaredNorm();
        cost += huber_rho(r.x(), huber) + huber_rho(r.y(), huber);
      }
    if (raw_sum_sq) *raw_sum_sq = rss;
    return cost;
  };

  Eigen::MatrixXd H;
  Eigen::VectorXd gb, Dk;
  std::vector<Eigen::Matrix<double, 6, 6>> B(nframes);
  std::vector<Eigen::Matrix<double, 6, 1>> bg(nframes), Df(nframes);
  std::vector<Eigen::MatrixXd> E(nframes);

  auto local_row = [&](int s, int off) {
    const auto& fb = fblocks[s];
    for (size_t k = 0; k < fb.blocks.size(); ++k)
      if (fb.blocks[k].first == off) return fb.local_row[k];
    return -1;
  };

  // Accumulate the undamped normal equations at the current state.
  // H/gb: kept system; B/bg: per-frame board blocks; E: kept-board coupling.
  auto accumulate = [&]() {
    H.setZero(kept_dim, kept_dim);
    gb.setZero(kept_dim);
    for (int s = 0; s < nframes; ++s) {
      B[s].setZero();
      bg[s].setZero();
      E[s].setZero(fblocks[s].L, 6);
      for (const auto& o : obs_by_frame[s]) {
        const Intrinsics& K = cur_intr[o.cam];
        const Pose tab = cur_cams[o.cam] * cur_boards[s];
        const Eigen::Vector3d pc = tab.apply(o.pw);
        const Eigen::Vector2d r = o.uv - K.project(pc);
        const double w0 = huber_sqrt_weight(r.x(), huber);
        const double w1 = huber_sqrt_weight(r.y(), huber);
        const Eigen::Vector2d rw(w0 * r.x(), w1 * r.y());
        const Eigen::Matrix<double, 2, 3> Jp = projection_jacobian(K, pc);

        Eigen::Matrix<double, 2, 6> Jb = Jp * pose_jacobian(tab, o.pw);
        Jb.row(0) *= w0;
        Jb.row(1) *= w1;
        B[s] += Jb.transpose() * Jb;
        bg[s] += Jb.transpose() * rw;

        const bool hc = cam_block[o.cam] >= 0;
        const bool hi = intr_block[o.cam] >= 0;
        Eigen::Matrix<double, 2, 6> Jc;
        Eigen::Matrix<double, 2, 9> Ji;
        if (hc) {
          Jc = Jp * pose_jacobian(cur_cams[o.cam], cur_boards[s].apply(o.pw));
          Jc.row(0) *= w0;
          Jc.row(1) *= w1;
          const int oc = 6 * cam_block[o.cam];
          H.block<6, 6>(oc, oc) += Jc.transpose() * Jc;
          gb.segment<6>(oc) += Jc.transpose() * rw;
          E[s].block<6, 6>(local_row(s, oc), 0) += Jc.transpose() * Jb;
        }
        if (hi) {
          Ji = intrinsics_jacobian(K, pc);
          Ji.row(0) *= w0;
          Ji.row(1) *= w1;
          const int oi = 6 * na + 9 * intr_block[o.cam];
          H.block(oi, oi, 9, 9) += Ji.transpose() * Ji;
          gb.segment(oi, 9) += Ji.transpose() * rw;
          E[s].block(local_row(s, oi), 0, 9, 6) += Ji.transpose() * Jb;
        }
        if (hc && hi) {
          const int oc = 6 * cam_block[o.cam];
          const int oi = 6 * na + 9 * intr_block[o.cam];
          H.block(oc, oi, 6, 9) += Jc.transpose() * Ji;
          H.block(oi, oc, 9, 6) += Ji.transpose() * Jc;
        }
      }
    }
  };

  double rss = 0.0;
  double cost = eval(cur_cams, cur_boards, cur_intr, &rss);
  result.initial_rms_px = std::sqrt(rss / (2.0 * nobs));
  double lambda = cfg_.init_lambda;
  int it = 0;
  bool converged = false;

  bool use_btl = cfg_.board_solve == BoardSolveBackend::btl_gpu;
  if (use_btl && !batchedSpdSolve6Available()) {
    std::fprintf(stderr,
                 "mcc BA: btl_gpu board solve requested but no CUDA device; "
                 "using eigen\n");
    use_btl = false;
  }

  // Damp, factor and solve every frame's 6x6 board block; fills
  // X[s] = (B_s + lambda*diag)^-1 E_s^T (6 x L_s) and xg[s] = same * bg_s.
  // Returns false if any block is not numerically SPD (btl reports per-system
  // info; the eigen branch checks finiteness) — the caller raises lambda and
  // retries, same as a failed reduced-system solve.
  auto solve_board_blocks = [&](std::vector<Eigen::MatrixXd>& X,
                                std::vector<Eigen::Matrix<double, 6, 1>>& xg) {
    if (use_btl) {
      // btl layout: full 6x6 row-major per system, one rhs per batch element
      // (adapter expands); rhs columns of frame s are the rows of E[s],
      // then bg[s].
      std::vector<double> a(nframes * 36), rhs_flat;
      std::vector<int> n_rhs(nframes);
      for (int s = 0; s < nframes; ++s) {
        Eigen::Matrix<double, 6, 6> Bd = B[s];
        for (int j = 0; j < 6; ++j) Bd(j, j) += lambda * Df[s](j);
        Eigen::Matrix<double, 6, 6> rm = Bd.transpose();  // to row-major
        std::copy_n(rm.data(), 36, a.begin() + s * 36);
        const int L = fblocks[s].L;
        n_rhs[s] = L + 1;
        for (int j = 0; j < L; ++j)
          for (int k = 0; k < 6; ++k) rhs_flat.push_back(E[s](j, k));
        for (int k = 0; k < 6; ++k) rhs_flat.push_back(bg[s](k));
      }
      std::vector<double> x_flat(rhs_flat.size());
      if (!batchedSpdSolve6(a.data(), nframes, n_rhs.data(), rhs_flat.data(),
                            x_flat.data()))
        return false;
      size_t off = 0;
      for (int s = 0; s < nframes; ++s) {
        const int L = fblocks[s].L;
        X[s].resize(6, L);
        for (int j = 0; j < L; ++j, off += 6)
          for (int k = 0; k < 6; ++k) X[s](k, j) = x_flat[off + k];
        for (int k = 0; k < 6; ++k) xg[s](k) = x_flat[off + k];
        off += 6;
      }
      return true;
    }
    for (int s = 0; s < nframes; ++s) {
      Eigen::Matrix<double, 6, 6> Bd = B[s];
      for (int j = 0; j < 6; ++j) Bd(j, j) += lambda * Df[s](j);
      const Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(Bd);
      X[s] = ldlt.solve(E[s].transpose());
      xg[s] = ldlt.solve(bg[s]);
      if (!X[s].allFinite() || !xg[s].allFinite()) return false;
    }
    return true;
  };

  while (it < cfg_.max_iters && !converged) {
    accumulate();
    double ginf = kept_dim > 0 ? gb.cwiseAbs().maxCoeff() : 0.0;
    for (int s = 0; s < nframes; ++s) ginf = std::max(ginf, bg[s].cwiseAbs().maxCoeff());
    if (ginf < 1e-10) {
      converged = true;
      break;
    }
    Dk = H.diagonal();
    for (int s = 0; s < nframes; ++s) Df[s] = B[s].diagonal();

    bool accepted = false;
    for (int retry = 0; retry < 20 && !accepted; ++retry) {
      // Marquardt damping on every block diagonal, added before elimination.
      Eigen::MatrixXd S = H;
      for (int i = 0; i < kept_dim; ++i) S(i, i) += lambda * Dk(i);
      Eigen::VectorXd rhs = gb;
      std::vector<Eigen::MatrixXd> X(nframes);  // B^-1 E^T per frame
      std::vector<Eigen::Matrix<double, 6, 1>> xg(nframes);
      bool ok = solve_board_blocks(X, xg);
      if (ok)
        for (int s = 0; s < nframes; ++s) {
          const auto& fb = fblocks[s];
          for (size_t a = 0; a < fb.blocks.size(); ++a) {
            const int ra = fb.local_row[a];
            const auto [oa, sa] = fb.blocks[a];
            rhs.segment(oa, sa) -= E[s].block(ra, 0, sa, 6) * xg[s];
            for (size_t b = 0; b < fb.blocks.size(); ++b) {
              const int rb = fb.local_row[b];
              const auto [ob, sb] = fb.blocks[b];
              S.block(oa, ob, sa, sb) -= E[s].block(ra, 0, sa, 6) * X[s].block(0, rb, 6, sb);
            }
          }
        }
      Eigen::VectorXd dk = Eigen::VectorXd::Zero(kept_dim);
      if (ok && kept_dim > 0) {
        const Eigen::LDLT<Eigen::MatrixXd> sl(S);
        dk = sl.solve(rhs);
        ok = sl.info() == Eigen::Success && dk.allFinite();
      }
      if (!ok) {
        lambda = std::min(lambda * 10.0, 1e9);
        continue;
      }

      // Back-substitute board poses; predicted reduction of the damped model
      // over the FULL parameter vector: 0.5 * d'*(b + lambda*D*d).
      double pred = 0.5 * dk.dot(gb + (lambda * Dk).cwiseProduct(dk));
      std::vector<Eigen::Matrix<double, 6, 1>> db(nframes);
      for (int s = 0; s < nframes; ++s) {
        Eigen::Matrix<double, 6, 1> v = xg[s];
        const auto& fb = fblocks[s];
        for (size_t a = 0; a < fb.blocks.size(); ++a) {
          const int ra = fb.local_row[a];
          const auto [oa, sa] = fb.blocks[a];
          v -= X[s].block(0, ra, 6, sa) * dk.segment(oa, sa);
        }
        db[s] = v;
        pred += 0.5 * v.dot(bg[s] + (lambda * Df[s]).cwiseProduct(v));
      }
      if (!(pred > 0.0)) {
        lambda = std::min(lambda * 10.0, 1e9);
        continue;
      }

      auto trial_cams = cur_cams;
      auto trial_boards = cur_boards;
      auto trial_intr = cur_intr;
      for (int ci = 1; ci < ncams; ++ci)
        if (cam_block[ci] >= 0)
          trial_cams[ci] = poseBoxplus(trial_cams[ci], dk.segment<6>(6 * cam_block[ci]));
      for (int ci = 0; ci < ncams; ++ci)
        if (intr_block[ci] >= 0) {
          const Eigen::Matrix<double, 9, 1> d = dk.segment<9>(6 * na + 9 * intr_block[ci]);
          Intrinsics& K = trial_intr[ci];
          K.fx += d[0];
          K.fy += d[1];
          K.cx += d[2];
          K.cy += d[3];
          for (int k = 0; k < 5; ++k) K.dist[k] += d[4 + k];
        }
      for (int s = 0; s < nframes; ++s)
        trial_boards[s] = poseBoxplus(trial_boards[s], db[s]);

      double new_rss = 0.0;
      const double new_cost = eval(trial_cams, trial_boards, trial_intr, &new_rss);
      if (!std::isfinite(new_cost)) {
        lambda = std::min(lambda * 10.0, 1e9);
        continue;
      }
      const double actual = cost - new_cost;
      const double gain = actual / pred;
      if (gain > cfg_.min_step_quality) {
        const double rel = actual / std::max(std::abs(cost), 1e-30);
        cur_cams = trial_cams;
        cur_boards = trial_boards;
        cur_intr = trial_intr;
        cost = new_cost;
        rss = new_rss;
        lambda = std::max(lambda * 0.1, 1e-9);
        accepted = true;
        ++it;
        if (cfg_.verbose)
          std::printf("mcc BA iter %2d: cost %.6f rms %.4f px lambda %.3e gain %.4f\n", it,
                      cost, std::sqrt(rss / (2.0 * nobs)), lambda, gain);
        if (rel < 1e-9 || actual < 1e-15) converged = true;
      } else {
        lambda = std::min(lambda * 10.0, 1e9);
      }
    }
    if (!accepted) break;  // lambda exhausted without a good step
  }

  result.camera_poses = cur_cams;
  result.board_poses = cur_boards;
  result.intrinsics = cur_intr;
  result.iterations = it;
  result.converged = converged;
  result.final_rms_px = std::sqrt(rss / (2.0 * nobs));
  return result;
}

}  // namespace mcc
