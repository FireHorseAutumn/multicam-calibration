#pragma once

// Corner cache: the data contract between detection and calibration.
//
// INVARIANT (an archived-pipeline scar, kept deliberately): within every
// FrameDetections,
// `corners` is sorted ascending by `id`, with no duplicate ids. All
// producers must maintain this; consumers may assume it.
//
// Serialization is intentionally NOT defined here yet — the JSON schema is
// TBD (owner will supply it). The cache is an in-memory structure for now.

#include <cstdint>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace mcc {

struct CornerObs {
  uint32_t id;        // ChArUco corner id (see board.hpp)
  Eigen::Vector2d uv; // sub-pixel image position
};

struct FrameDetections {
  uint64_t frame_index = 0;
  double timestamp = 0.0;         // seconds; NaN if unknown
  std::vector<CornerObs> corners; // sorted ascending by id
};

struct CameraCache {
  std::string camera_name;
  int image_width = 0, image_height = 0;
  std::vector<FrameDetections> frames;
};

// Insert-or-reject helper maintaining the sorted-by-id invariant.
// Returns false if the id already exists.
bool insertCorner(FrameDetections& f, CornerObs obs);

// Merge two sorted corner lists (set union by id; `a` wins on collision).
std::vector<CornerObs> mergeCorners(const std::vector<CornerObs>& a,
                                    const std::vector<CornerObs>& b);

// The intersection of ids present in both frames (sorted ascending).
std::vector<uint32_t> commonCornerIds(const FrameDetections& a,
                                      const FrameDetections& b);

}  // namespace mcc
