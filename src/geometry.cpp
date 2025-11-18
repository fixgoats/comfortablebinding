#include "geometry.h"
#include "io.h"
#include <iterator>

std::vector<Point> readPoints(const std::string& fname) {
  u32 m = 0;
  std::ifstream f(fname);
  if (!f.good()) {
    runtime_exc("File {} doesn't exist", fname);
  }
  std::vector<std::string> allLines{std::istream_iterator<Line>(f),
                                    std::istream_iterator<Line>()};
  m = allLines.size();
  // f.clear();
  // f.seekg(0, std::ios::beg);
  std::vector<Point> M(m);
  for (u32 j = 0; j < m; j++) {
    std::istringstream stream(allLines[j]);
    std::vector<double> v{std::istream_iterator<double>(stream),
                          std::istream_iterator<double>()};
    M[j] = {v[0], v[1], j};
  }
  return M;
}

std::vector<Point> extended_grid(const std::vector<Point>& base,
                                 const std::vector<int>& x_edge,
                                 const std::vector<int>& y_edge,
                                 const std::vector<int>& corner, double Lx,
                                 double Ly) {
  std::vector<Point> final_grid(base.size() + x_edge.size() + y_edge.size() +
                                corner.size());
  std::copy(base.cbegin(), base.cend(), final_grid.begin());
  size_t offset = base.size();
  for (size_t i = 0; i < x_edge.size(); i++) {
    Point p = base[x_edge[i]];
    p[1] += Ly;
    final_grid[offset + i] = p;
  }
  offset += x_edge.size();
  for (size_t i = 0; i < y_edge.size(); i++) {
    Point p = base[y_edge[i]];
    p[0] += Lx;
    final_grid[offset + i] = p;
  }
  offset += y_edge.size();
  for (size_t i = 0; i < corner.size(); i++) {
    Point p = base[corner[i]];
    p[0] += Lx;
    p[1] += Ly;
    final_grid[offset + i] = p;
  }
  return final_grid;
}
void standardise(std::vector<Point>& points) {
  kdt::KDTree<Point> kdtree(points);
  double minx = points[kdtree.axisFindMin(0)][0];
  double miny = points[kdtree.axisFindMin(1)][1];
  std::for_each(points.begin(), points.end(), [&](Point& p) {
    p[0] -= minx;
    p[1] -= miny;
  });
}
