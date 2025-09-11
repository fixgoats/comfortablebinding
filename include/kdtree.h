/* MIT License
 *
 * Copyright (c) 2017 gishi523
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef __KDTREE_H__
#define __KDTREE_H__

#include <algorithm>
#include <exception>
#include <functional>
#include <numeric>
#include <vector>

namespace kdt {
/** @brief k-d tree class.
 */
template <class PointT>
class KDTree {
public:
  /** @brief The constructors.
   */
  KDTree() : root_(nullptr) {};
  KDTree(const std::vector<PointT>& points) : root_(nullptr) { build(points); }

  /** @brief The destructor.
   */
  ~KDTree() { clear(); }

  /** @brief Re-builds k-d tree.
   */
  void build(const std::vector<PointT>& points) {
    clear();

    points_ = points;

    std::vector<int> indices(points.size());
    std::iota(std::begin(indices), std::end(indices), 0);

    root_ = buildRecursive(indices.data(), (int)points.size(), 0);
  }

  /** @brief Clears k-d tree.
   */
  void clear() {
    clearRecursive(root_);
    root_ = nullptr;
    points_.clear();
  }

  /** @brief Validates k-d tree.
   */
  bool validate() const {
    try {
      validateRecursive(root_, 0);
    } catch (const Exception&) {
      return false;
    }

    return true;
  }

  /** @brief Searches the nearest neighbor.
   */
  int nnSearch(const PointT& query, double* minDist = nullptr) const {
    int guess;
    double _minDist = std::numeric_limits<double>::max();

    nnSearchRecursive(query, root_, &guess, &_minDist);

    if (minDist)
      *minDist = _minDist;

    return guess;
  }

  /** @brief Searches k-nearest neighbors.
   */
  std::vector<int> knnSearch(const PointT& query, int k) const {
    KnnQueue queue(k);
    knnSearchRecursive(query, root_, queue, k);

    std::vector<int> indices(queue.size());
    for (size_t i = 0; i < queue.size(); i++)
      indices[i] = queue[i].second;

    return indices;
  }

  /** @brief Searches neighbors within radius from query.
   */
  std::vector<int> radiusSearch(const PointT& query, double radius) const {
    std::vector<int> indices;
    radiusSearchRecursive(query, root_, indices, radius);
    return indices;
  }

  /** @brief Searches neighbors within radius from axis.
   */
  std::vector<int> axisSearch(int axis, double radius) const {
    std::vector<int> indices;
    axisSearchRecursive(root_, indices, radius, axis);
    return indices;
  }

  /** @brief Searches neighbors within radius from query with metric dist_func.
   */
  template <class Func>
  std::vector<int> genRadiusSearch(const PointT& query, double radius,
                                   Func dist_func) const {
    std::vector<int> indices;
    genRadiusSearchRecursive(query, root_, indices, radius, dist_func);
    return indices;
  }

  /** @brief Searches neighbors within or at radius from query.
   */
  std::vector<int> radiusSearchInclusive(const PointT& query,
                                         double radius) const {
    std::vector<int> indices;
    radiusSearchInclusiveRecursive(query, root_, indices, radius);
    return indices;
  }

  /** @brief Searches neighbors within or at radius from axis.
   */
  std::vector<int> axisSearchInclusive(int axis, double radius) const {
    std::vector<int> indices;
    axisSearchInclusiveRecursive(root_, indices, radius, axis);
    return indices;
  }

  /** @brief Searches neighbors within or at radius from query with metric
   * dist_func.
   */
  template <class Func>
  std::vector<int> genRadiusSearchInclusive(const PointT& query, double radius,
                                            Func dist_func) const {
    std::vector<int> indices;
    genRadiusSearchInclusiveRecursive(query, root_, indices, radius, dist_func);
    return indices;
  }

  int axisFindMin(int ax) const { return axisFindMinRecursive(root_, ax); }
  int axisFindMax(int ax) const { return axisFindMaxRecursive(root_, ax); }

private:
  /** @brief k-d tree node.
   */
  struct Node {
    int idx;       //!< index to the original point
    Node* next[2]; //!< pointers to the child nodes
    int axis;      //!< dimension's axis

    Node() : idx(-1), axis(-1) { next[0] = next[1] = nullptr; }
  };

  /** @brief k-d tree exception.
   */
  class Exception : public std::exception {
    using std::exception::exception;
  };

  /** @brief Bounded priority queue.
   */
  template <class T, class Compare = std::less<T>>
  class BoundedPriorityQueue {
  public:
    BoundedPriorityQueue() = delete;
    BoundedPriorityQueue(size_t bound) : bound_(bound) {
      elements_.reserve(bound + 1);
    };

    void push(const T& val) {
      auto it = std::find_if(
          std::begin(elements_), std::end(elements_),
          [&](const T& element) { return Compare()(val, element); });
      elements_.insert(it, val);

      if (elements_.size() > bound_)
        elements_.resize(bound_);
    }

    const T& back() const { return elements_.back(); };
    const T& operator[](size_t index) const { return elements_[index]; }
    size_t size() const { return elements_.size(); }

  private:
    size_t bound_;
    std::vector<T> elements_;
  };

  /** @brief Priority queue of <distance, index> pair.
   */
  using KnnQueue = BoundedPriorityQueue<std::pair<double, int>>;

  /** @brief Builds k-d tree recursively.
   */
  Node* buildRecursive(int* indices, int npoints, int depth) {
    if (npoints <= 0)
      return nullptr;

    const int axis = depth % PointT::DIM;
    const int mid = (npoints - 1) / 2;

    std::nth_element(indices, indices + mid, indices + npoints,
                     [&](int lhs, int rhs) {
                       return points_[lhs][axis] < points_[rhs][axis];
                     });

    Node* node = new Node();
    node->idx = indices[mid];
    node->axis = axis;

    node->next[0] = buildRecursive(indices, mid, depth + 1);
    node->next[1] =
        buildRecursive(indices + mid + 1, npoints - mid - 1, depth + 1);

    return node;
  }

  /** @brief Clears k-d tree recursively.
   */
  void clearRecursive(Node* node) {
    if (node == nullptr)
      return;

    if (node->next[0])
      clearRecursive(node->next[0]);

    if (node->next[1])
      clearRecursive(node->next[1]);

    delete node;
  }

  /** @brief Validates k-d tree recursively.
   */
  void validateRecursive(const Node* node, int depth) const {
    if (node == nullptr)
      return;

    const int axis = node->axis;
    const Node* node0 = node->next[0];
    const Node* node1 = node->next[1];

    if (node0 && node1) {
      if (points_[node->idx][axis] < points_[node0->idx][axis])
        throw Exception();

      if (points_[node->idx][axis] > points_[node1->idx][axis])
        throw Exception();
    }

    if (node0)
      validateRecursive(node0, depth + 1);

    if (node1)
      validateRecursive(node1, depth + 1);
  }

  static double distance(const PointT& p, const PointT& q) {
    double dist = 0;
    for (size_t i = 0; i < PointT::DIM; i++)
      dist += (p[i] - q[i]) * (p[i] - q[i]);
    return sqrt(dist);
  }

  static double sqnorm(const PointT& p) {
    double norm = 0;
    for (size_t i = 0; i < PointT::DIM; i++)
      norm += p[i] * p[i];
    return norm;
  }

  static double norm(const PointT& p) { return sqrt(sqnorm(p)); }

  static double axisDistance(const PointT& p, int axis) {
    if constexpr (PointT::DIM == 2) {
      return abs(p[1 - axis]);
    } else {
      double sum = 0;
      for (size_t i = 0; i < PointT::DIM; i++) {
        if (i == (size_t)axis)
          continue;
        sum += p[i] * p[i];
      }
      return sqrt(sum);
    }
  }

  /** @brief Searches the nearest neighbor recursively.
   */
  void nnSearchRecursive(const PointT& query, const Node* node, int* guess,
                         double* minDist) const {
    if (node == nullptr)
      return;

    const PointT& train = points_[node->idx];

    const double dist = distance(query, train);
    if (dist < *minDist) {
      *minDist = dist;
      *guess = node->idx;
    }

    const int axis = node->axis;
    const int dir = query[axis] < train[axis] ? 0 : 1;
    nnSearchRecursive(query, node->next[dir], guess, minDist);

    const double diff = fabs(query[axis] - train[axis]);
    if (diff < *minDist)
      nnSearchRecursive(query, node->next[!dir], guess, minDist);
  }

  /** @brief Searches k-nearest neighbors recursively.
   */
  void knnSearchRecursive(const PointT& query, const Node* node,
                          KnnQueue& queue, int k) const {
    if (node == nullptr)
      return;

    const PointT& train = points_[node->idx];

    const double dist = distance(query, train);
    queue.push(std::make_pair(dist, node->idx));

    const int axis = node->axis;
    const int dir = query[axis] < train[axis] ? 0 : 1;
    knnSearchRecursive(query, node->next[dir], queue, k);

    const double diff = fabs(query[axis] - train[axis]);
    if ((int)queue.size() < k || diff < queue.back().first)
      knnSearchRecursive(query, node->next[!dir], queue, k);
  }

  /** @brief Searches neighbors within radius.
   */
  void radiusSearchRecursive(const PointT& query, const Node* node,
                             std::vector<int>& indices, double radius) const {
    if (node == nullptr)
      return;

    const PointT& train = points_[node->idx];

    const double dist = distance(query, train);
    if (dist < radius)
      indices.push_back(node->idx);

    const int axis = node->axis;
    const int dir = query[axis] < train[axis] ? 0 : 1;
    radiusSearchRecursive(query, node->next[dir], indices, radius);

    const double diff = fabs(query[axis] - train[axis]);
    if (diff < radius)
      radiusSearchRecursive(query, node->next[!dir], indices, radius);
  }

  void radiusSearchInclusiveRecursive(const PointT& query, const Node* node,
                                      std::vector<int>& indices,
                                      double radius) const {
    if (node == nullptr)
      return;

    const PointT& train = points_[node->idx];

    const double dist = distance(query, train);
    if (dist <= radius)
      indices.push_back(node->idx);

    const int axis = node->axis;
    const int dir = query[axis] < train[axis] ? 0 : 1;
    radiusSearchRecursive(query, node->next[dir], indices, radius);

    const double diff = fabs(query[axis] - train[axis]);
    if (diff <= radius)
      radiusSearchRecursive(query, node->next[!dir], indices, radius);
  }

  template <class Func>
  void genRadiusSearchRecursive(const PointT& query, const Node* node,
                                std::vector<int>& indices, double radius,
                                Func dist_func) const {
    if (node == nullptr)
      return;

    const PointT& train = points_[node->idx];

    const double dist = dist_func(query, train);
    if (dist < radius)
      indices.push_back(node->idx);

    const int axis = node->axis;
    const int dir = query[axis] < train[axis] ? 0 : 1;
    genRadiusSearchRecursive(query, node->next[dir], indices, radius,
                             dist_func);

    const double diff = fabs(query[axis] - train[axis]);
    if (diff < radius)
      genRadiusSearchRecursive(query, node->next[!dir], indices, radius,
                               dist_func);
  }

  template <class Func>
  void genRadiusSearchInclusiveRecursive(const PointT& query, const Node* node,
                                         std::vector<int>& indices,
                                         double radius, Func dist_func) const {
    if (node == nullptr)
      return;

    const PointT& train = points_[node->idx];

    const double dist = dist_func(query, train);
    if (dist <= radius)
      indices.push_back(node->idx);

    const int axis = node->axis;
    const int dir = query[axis] < train[axis] ? 0 : 1;
    genRadiusSearchRecursive(query, node->next[dir], indices, radius,
                             dist_func);

    const double diff = fabs(query[axis] - train[axis]);
    if (diff <= radius)
      genRadiusSearchRecursive(query, node->next[!dir], indices, radius,
                               dist_func);
  }

  /** @brief Searches points within distance from axis
   */
  void axisSearchRecursive(const Node* node, std::vector<int>& indices,
                           double distance, int ax) const {
    if (node == nullptr)
      return;

    const PointT& train = points_[node->idx];
    const double dist = axisDistance(train, ax);
    if (dist < distance)
      indices.push_back(node->idx);

    const int axis = node->axis;
    if (axis == ax) {
      axisSearchRecursive(node->next[0], indices, distance, ax);
      axisSearchRecursive(node->next[1], indices, distance, ax);
    } else {
      const int dir = 0 < train[axis] ? 0 : 1;
      axisSearchRecursive(node->next[dir], indices, distance, ax);

      const double diff = fabs(train[axis]);
      if (diff < distance)
        axisSearchRecursive(node->next[!dir], indices, distance, ax);
    }
  }

  void axisSearchInclusiveRecursive(const Node* node, std::vector<int>& indices,
                                    double distance, int ax) const {
    if (node == nullptr)
      return;

    const PointT& train = points_[node->idx];
    const double dist = axisDistance(train, ax);
    if (dist <= distance)
      indices.push_back(node->idx);

    const int axis = node->axis;
    if (axis == ax) {
      axisSearchInclusiveRecursive(node->next[0], indices, distance, ax);
      axisSearchInclusiveRecursive(node->next[1], indices, distance, ax);
    } else {
      const int dir = 0 < train[axis] ? 0 : 1;
      axisSearchInclusiveRecursive(node->next[dir], indices, distance, ax);

      const double diff = fabs(train[axis]);
      if (diff <= distance)
        axisSearchInclusiveRecursive(node->next[!dir], indices, distance, ax);
    }
  }

  /** @brief Finds point with lowest value along axis d. wtf þetta virkaði í svo
   * gott sem fyrstu atrennu!
   */
  int axisFindMinRecursive(const Node* node, int ax) const {
    if (!node->next[0] && !node->next[1])
      return node->idx;

    if (node->axis == ax) {
      if (node->next[0] == nullptr) {
        return node->idx;
      } else {
        return axisFindMinRecursive(node->next[0], ax);
      }
    } else {
      int one = node->idx;
      int two = node->idx;
      if (node->next[0]) {
        one = axisFindMinRecursive(node->next[0], ax);
      }
      if (node->next[1]) {
        two = axisFindMinRecursive(node->next[1], ax);
      }
      bool onelttwo = (points_[one][ax] <= points_[two][ax]);
      bool selfltone = (points_[node->idx][ax] <= points_[one][ax]);
      if (selfltone && onelttwo) {
        return node->idx;
      }
      if (selfltone) {
        return points_[node->idx][ax] <= points_[two][ax] ? node->idx : two;
      }
      if (onelttwo) {
        return one;
      }
      return two;
    }
  }

  /** @brief Finds point with highest value along axis d.
   */
  int axisFindMaxRecursive(const Node* node, int ax) const {
    if (!node->next[0] && !node->next[1])
      return node->idx;

    if (node->axis == ax) {
      if (node->next[1] == nullptr) {
        return node->idx;
      } else {
        return axisFindMaxRecursive(node->next[1], ax);
      }
    } else {
      int one = node->idx;
      int two = node->idx;
      if (node->next[0]) {
        one = axisFindMaxRecursive(node->next[0], ax);
      }
      if (node->next[1]) {
        two = axisFindMaxRecursive(node->next[1], ax);
      }
      bool onegttwo = (points_[one][ax] >= points_[two][ax]);
      bool selfgtone = (points_[node->idx][ax] >= points_[one][ax]);
      if (selfgtone && onegttwo) {
        return node->idx;
      }
      if (selfgtone) {
        return points_[node->idx][ax] >= points_[two][ax] ? node->idx : two;
      }
      if (onegttwo) {
        return one;
      }
      return two;
    }
  }

  Node* root_;                 //!< root node
  std::vector<PointT> points_; //!< points
};
} // namespace kdt

#endif // !__KDTREE_H__
