#pragma once

// COLMAP-format export (text): cameras.txt, images.txt, points3D.txt.
// Unlocks nerfstudio / 3DGS downstream tooling.
//
// Conventions:
//   - COLMAP camera model OPENCV (fx fy cx cy k1 k2 p1 p2) — our k3 is
//     dropped if nonzero (warning emitted); COLMAP's OPENCV has no k3.
//     FULL_OPENCV support can come with the schema pass.
//   - images.txt poses are world-from-camera (COLMAP stores camera-from-
//     world as qvec/tvec; we write exactly that and document it).
//   - images.txt is written per camera with the camera's cache frame list;
//     image names are "<camera_name>/<frame_index>.png" by convention —
//     the input adapters will own real filenames once they exist.
//   - points3D.txt is written empty-but-valid (triangulation is C2/C3).

#include <string>
#include <vector>

#include "mcc/corner_cache.hpp"
#include "mcc/types.hpp"

namespace mcc {

// Returns true on success; `error` receives a message otherwise.
bool exportColmap(const std::string& out_dir,
                  const std::vector<CameraCache>& caches,
                  const std::vector<Intrinsics>& intrinsics,
                  const std::vector<Pose>& camera_poses,
                  std::string* error = nullptr);

}  // namespace mcc
