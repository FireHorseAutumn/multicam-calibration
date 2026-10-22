#pragma once

// Own ChArUco detector — no OpenCV.
//
// Pipeline (CPU, Eigen; CUDA acceleration seams noted in the .cpp):
//   1. adaptive threshold via integral image
//   2. quad candidate extraction (Suzuki-Abe contour tracing on the
//      binarized image, polygon approximation, quad filter)
//   3. per-marker canonical warp + Otsu cell sampling
//   4. dictionary decode with rotation + Hamming-error correction
//   5. sub-pixel corner refinement (edge-direction constraint, the
//      standard cornerSubPix fixed-point iteration, implemented inline)
//   6. ChArUco corner interpolation from decoded marker neighborhoods
//
// Input is raw 8-bit grayscale. Color conversion belongs to the (future)
// input adapters.

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

#include "mcc/board.hpp"
#include "mcc/corner_cache.hpp"

namespace mcc {

struct DetectorConfig {
  int adaptive_window = 23;       // odd; block size for mean threshold
  double adaptive_offset = 7.0;   // constant subtracted from the mean
  double min_marker_perimeter = 40.0;    // pixels
  double max_marker_perimeter = 4000.0;  // pixels
  double polygonal_epsilon = 0.04;       // fraction of perimeter (Douglas-Peucker)
  int max_hamming_correction = 2;        // bits we allow ourselves to fix
  int corner_refinement_window = 5;      // half-window for sub-pixel refinement
  int corner_refinement_iters = 20;
  double min_corner_separation = 3.0;    // pixels; closer detections get culled
};

struct DetectedMarker {
  uint32_t id = 0;
  std::array<Eigen::Vector2d, 4> corners;  // canonical order: TL, TR, BR, BL
};

struct DetectionResult {
  std::vector<DetectedMarker> markers;
  std::vector<CornerObs> charuco_corners;  // sorted ascending by id
};

class CharucoDetector {
 public:
  explicit CharucoDetector(const CharucoBoard& board,
                           DetectorConfig cfg = DetectorConfig{});
  DetectionResult detect(const uint8_t* gray, int width, int height,
                         int stride) const;

 private:
  CharucoBoard board_;
  DetectorConfig cfg_;
};

}  // namespace mcc
