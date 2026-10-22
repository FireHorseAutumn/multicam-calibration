// COLMAP text export: cameras.txt, images.txt, points3D.txt.
// Our Pose is camera-from-world, which is exactly COLMAP's qvec/tvec
// convention, so poses are written directly as (qw qx qy qz, tx ty tz).

#include "mcc/colmap_export.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>

namespace mcc {

bool exportColmap(const std::string& out_dir,
                  const std::vector<CameraCache>& caches,
                  const std::vector<Intrinsics>& intrinsics,
                  const std::vector<Pose>& camera_poses,
                  std::string* error) {
  auto fail = [&](const std::string& msg) {
    if (error) *error = msg;
    return false;
  };
  const size_t n = caches.size();
  if (n == 0) return fail("exportColmap: empty camera cache list");
  if (intrinsics.size() != n) return fail("exportColmap: intrinsics/caches size mismatch");
  if (camera_poses.size() != n) return fail("exportColmap: camera_poses/caches size mismatch");

  std::error_code ec;
  std::filesystem::create_directories(out_dir, ec);
  if (ec) return fail("exportColmap: cannot create " + out_dir + ": " + ec.message());

  std::string notes;

  {
    std::ofstream f(std::filesystem::path(out_dir) / "cameras.txt");
    if (!f) return fail("exportColmap: cannot open cameras.txt for writing");
    f << std::setprecision(17);
    f << "# Camera list with one line of data per camera:\n"
         "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
      << "# Number of cameras: " << n << "\n";
    for (size_t i = 0; i < n; ++i) {
      const Intrinsics& K = intrinsics[i];
      f << i + 1 << " OPENCV " << caches[i].image_width << " " << caches[i].image_height
        << " " << K.fx << " " << K.fy << " " << K.cx << " " << K.cy << " " << K.dist[0] << " "
        << K.dist[1] << " " << K.dist[2] << " " << K.dist[3] << "\n";
      if (K.dist[4] != 0.0) {
        if (!notes.empty()) notes += "; ";
        notes += "camera " + std::to_string(i + 1) +
                 " has nonzero k3, dropped (OPENCV model has no k3)";
      }
    }
    if (!f) return fail("exportColmap: failed writing cameras.txt");
  }

  {
    size_t total = 0;
    for (const auto& c : caches) total += c.frames.size();
    std::ofstream f(std::filesystem::path(out_dir) / "images.txt");
    if (!f) return fail("exportColmap: cannot open images.txt for writing");
    f << std::setprecision(17);
    f << "# Image list with two lines of data per image:\n"
         "#   IMAGE_ID, QW QX QY QZ, TX TY TZ, CAMERA_ID, NAME\n"
         "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
      << "# Number of images: " << total << ", mean observations per image: 0\n";
    size_t image_id = 1;
    for (size_t i = 0; i < n; ++i) {
      const Eigen::Quaterniond q = camera_poses[i].q.normalized();
      const Eigen::Vector3d& t = camera_poses[i].t;
      for (const auto& fr : caches[i].frames) {
        f << image_id++ << " " << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
          << t.x() << " " << t.y() << " " << t.z() << " " << i + 1 << " "
          << caches[i].camera_name << "/" << fr.frame_index << ".png\n\n";
      }
    }
    if (!f) return fail("exportColmap: failed writing images.txt");
  }

  {
    std::ofstream f(std::filesystem::path(out_dir) / "points3D.txt");
    if (!f) return fail("exportColmap: cannot open points3D.txt for writing");
    f << "# 3D point list with one line of data per point:\n"
         "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
         "# Number of points: 0, mean track length: 0\n";
    if (!f) return fail("exportColmap: failed writing points3D.txt");
  }

  if (error) *error = notes;  // non-fatal notes only (e.g. dropped k3)
  return true;
}

}  // namespace mcc
