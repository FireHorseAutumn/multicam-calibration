// Implementation of include/mcc/pose_init.hpp — planar DLT + LM per-frame
// board poses, Markley-averaged pairwise edges, max-quality spanning tree.

#include "mcc/pose_init.hpp"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <unordered_map>

namespace mcc {
namespace {

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

// Normalized DLT homography src -> dst (src is the board plane).
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

// Camera-from-board pose from board->image homography and K (distortion
// ignored for the init; LM below includes it via Intrinsics::project).
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

struct UnionFind {
  std::vector<size_t> p;
  explicit UnionFind(size_t n) : p(n) { std::iota(p.begin(), p.end(), 0); }
  size_t find(size_t x) {
    while (p[x] != x) {
      p[x] = p[p[x]];
      x = p[x];
    }
    return x;
  }
  bool unite(size_t a, size_t b) {
    a = find(a);
    b = find(b);
    if (a == b) return false;
    p[b] = a;
    return true;
  }
};

}  // namespace

bool solveBoardPose(const FrameDetections& frame, const CharucoBoard& board,
                    const Intrinsics& K, Pose& camera_from_board,
                    double* rms_px) {
  std::vector<Eigen::Vector3d> pw;
  std::vector<Eigen::Vector2d> uv, src;
  for (const auto& c : frame.corners) {
    Eigen::Vector3d p;
    if (!board.cornerPoint(c.id, p)) continue;
    pw.push_back(p);
    uv.push_back(c.uv);
    src.emplace_back(p.x(), p.y());
  }
  if (pw.size() < 4) return false;
  Eigen::Matrix3d H;
  if (!homographyDlt(src, uv, H)) return false;
  if (!poseFromHomography(H, K, camera_from_board)) return false;

  double lambda = 1e-3, nu = 2.0;
  const auto costOf = [&](const Pose& p) {
    double c = 0.0;
    for (size_t i = 0; i < pw.size(); ++i)
      c += (uv[i] - K.project(p.apply(pw[i]))).squaredNorm();
    return c;
  };
  double cost = costOf(camera_from_board);
  for (int it = 0; it < 50; ++it) {
    Eigen::Matrix<double, 6, 6> Hn = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    cost = 0.0;
    const Eigen::Matrix3d R = camera_from_board.R();
    for (size_t i = 0; i < pw.size(); ++i) {
      const Eigen::Vector3d pc = camera_from_board.apply(pw[i]);
      const Eigen::Vector2d r = uv[i] - K.project(pc);
      cost += r.squaredNorm();
      const Eigen::Matrix<double, 2, 6> Jr = -projectJacPose(K, pc, pw[i], R);
      Hn += Jr.transpose() * Jr;
      g += Jr.transpose() * r;
    }
    const double prev_cost = cost;
    Eigen::Matrix<double, 6, 1> D = Hn.diagonal();
    for (int i = 0; i < 6; ++i)
      if (D[i] < 1e-12) D[i] = 1e-12;
    bool accepted = false;
    for (int attempt = 0; attempt < 10; ++attempt) {
      Eigen::Matrix<double, 6, 6> A = Hn;
      A.diagonal() += lambda * D;
      const Eigen::Matrix<double, 6, 1> delta = A.ldlt().solve(-g);
      if (!delta.allFinite()) {
        lambda *= nu;
        nu *= 2.0;
        continue;
      }
      const Pose tp = poseBoxplus(camera_from_board, delta);
      const double new_cost = costOf(tp);
      const double predicted =
          delta.dot(lambda * (D.asDiagonal() * delta) - g);
      const double rho =
          (predicted > 0.0) ? (cost - new_cost) / predicted : -1.0;
      if (rho > 0.0 && std::isfinite(new_cost)) {
        camera_from_board = tp;
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
  int front = 0;
  for (const auto& p : pw)
    if (camera_from_board.apply(p).z() > 0.0) ++front;
  if (front * 2 < static_cast<int>(pw.size())) return false;
  if (rms_px) *rms_px = std::sqrt(cost / (2.0 * pw.size()));
  return true;
}

PoseGraphInit initializeCameraPoses(const std::vector<CameraCache>& caches,
                                    const CharucoBoard& board,
                                    const std::vector<Intrinsics>& intrinsics,
                                    const PoseInitConfig& cfg) {
  PoseGraphInit out;
  const size_t nc = caches.size();
  out.camera_poses.assign(nc, Pose{});
  if (intrinsics.size() != nc) return out;  // uncalibrated input; no edges

  std::vector<std::unordered_map<uint64_t, size_t>> index(nc);
  for (size_t c = 0; c < nc; ++c)
    for (size_t i = 0; i < caches[c].frames.size(); ++i)
      index[c][caches[c].frames[i].frame_index] = i;

  for (size_t a = 0; a < nc; ++a) {
    if (intrinsics[a].fx <= 0.0) continue;
    for (size_t b = a + 1; b < nc; ++b) {
      if (intrinsics[b].fx <= 0.0) continue;
      std::vector<Eigen::Vector4d> qs;  // [x,y,z,w], sign-aligned
      std::vector<Eigen::Vector3d> ts;
      double rms_sum = 0.0;
      for (const auto& fa : caches[a].frames) {
        const auto itb = index[b].find(fa.frame_index);
        if (itb == index[b].end()) continue;
        const FrameDetections& fb = caches[b].frames[itb->second];
        if (static_cast<int>(commonCornerIds(fa, fb).size()) <
            cfg.min_common_corners)
          continue;
        Pose pa, pb;
        double ra = 0.0, rb = 0.0;
        if (!solveBoardPose(fa, board, intrinsics[a], pa, &ra)) continue;
        if (!solveBoardPose(fb, board, intrinsics[b], pb, &rb)) continue;
        const Pose rel = pa * pb.inverse();  // camera_a-from-camera_b
        Eigen::Vector4d c = rel.q.normalized().coeffs();
        if (!qs.empty() && c.dot(qs.front()) < 0.0) c = -c;
        qs.push_back(c);
        ts.push_back(rel.t);
        rms_sum += 0.5 * (ra + rb);
      }
      if (static_cast<int>(qs.size()) < cfg.min_common_frames) continue;
      // Markley quaternion average: top eigenvector of sum q q^T.
      Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
      for (const auto& q : qs) M += q * q.transpose();
      const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> es(M);
      const Eigen::Vector4d e = es.eigenvectors().col(3);
      Eigen::Vector3d tm = Eigen::Vector3d::Zero();
      for (const auto& t : ts) tm += t;
      tm /= static_cast<double>(ts.size());
      PairwiseEdge edge;
      edge.cam_a = a;
      edge.cam_b = b;
      edge.a_from_b = {Eigen::Quaterniond(e[3], e[0], e[1], e[2]).normalized(),
                       tm};
      edge.rms_px = rms_sum / static_cast<double>(qs.size());
      edge.common_frames = static_cast<int>(qs.size());
      out.edges.push_back(edge);
    }
  }

  // Kruskal spanning tree, lowest RMS first (tiebreak: more common frames).
  std::vector<size_t> order(out.edges.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](size_t i, size_t j) {
    if (out.edges[i].rms_px != out.edges[j].rms_px)
      return out.edges[i].rms_px < out.edges[j].rms_px;
    return out.edges[i].common_frames > out.edges[j].common_frames;
  });
  UnionFind uf(nc);
  std::vector<std::vector<std::pair<size_t, size_t>>> tree(nc);
  for (const size_t ei : order) {
    const auto& e = out.edges[ei];
    if (!uf.unite(e.cam_a, e.cam_b)) continue;
    tree[e.cam_a].emplace_back(e.cam_b, ei);
    tree[e.cam_b].emplace_back(e.cam_a, ei);
  }

  // Propagate absolute poses from camera 0. camera_poses[i] is
  // camera_i-from-world (world == camera 0 frame): p_i = P_i * p_w.
  // An edge satisfies p_a = a_from_b * p_b, so P_a = a_from_b * P_b and
  // hence P_b = a_from_b.inverse() * P_a.
  std::vector<char> visited(nc, 0);
  if (nc > 0) {
    visited[0] = 1;
    std::queue<size_t> queue;
    queue.push(0);
    while (!queue.empty()) {
      const size_t u = queue.front();
      queue.pop();
      for (const auto& [v, ei] : tree[u]) {
        if (visited[v]) continue;
        const PairwiseEdge& e = out.edges[ei];
        if (u == e.cam_a)
          out.camera_poses[v] = e.a_from_b.inverse() * out.camera_poses[u];
        else
          out.camera_poses[v] = e.a_from_b * out.camera_poses[u];
        visited[v] = 1;
        queue.push(v);
      }
    }
  }
  const size_t n_visited =
      static_cast<size_t>(std::count(visited.begin(), visited.end(), 1));
  out.connected = (n_visited == nc);
  return out;
}

}  // namespace mcc
