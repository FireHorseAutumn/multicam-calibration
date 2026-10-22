// Corner cache helpers; see include/mcc/corner_cache.hpp for the invariant
// (corners sorted ascending by id, no duplicates) these functions maintain.

#include "mcc/corner_cache.hpp"

#include <algorithm>

namespace mcc {

bool insertCorner(FrameDetections& f, CornerObs obs) {
  auto it = std::lower_bound(f.corners.begin(), f.corners.end(), obs.id,
                             [](const CornerObs& c, uint32_t id) { return c.id < id; });
  if (it != f.corners.end() && it->id == obs.id) return false;
  f.corners.insert(it, obs);
  return true;
}

std::vector<CornerObs> mergeCorners(const std::vector<CornerObs>& a,
                                    const std::vector<CornerObs>& b) {
  std::vector<CornerObs> out;
  out.reserve(a.size() + b.size());
  auto i = a.begin(), j = b.begin();
  while (i != a.end() && j != b.end()) {
    if (i->id < j->id) {
      out.push_back(*i++);
    } else if (j->id < i->id) {
      out.push_back(*j++);
    } else {
      out.push_back(*i);
      ++i;
      ++j;
    }
  }
  out.insert(out.end(), i, a.end());
  out.insert(out.end(), j, b.end());
  return out;
}

std::vector<uint32_t> commonCornerIds(const FrameDetections& a,
                                      const FrameDetections& b) {
  std::vector<uint32_t> out;
  auto i = a.corners.begin(), j = b.corners.begin();
  while (i != a.corners.end() && j != b.corners.end()) {
    if (i->id < j->id) {
      ++i;
    } else if (j->id < i->id) {
      ++j;
    } else {
      out.push_back(i->id);
      ++i;
      ++j;
    }
  }
  return out;
}

}  // namespace mcc
