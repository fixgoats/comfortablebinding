#include "geometry.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include "io.h"
#include "kdtree.h"
#include "raylib.h"
#include "spdlog/spdlog.h"
#include <cxxopts.hpp>

using Eigen::Vector2i;

s32 toScreenX(f64 r, f64 min, f64 max, s32 dim) {
  return (s32)(((r - min) / (max - min)) * (f64)dim);
}

s32 toScreenY(f64 r, f64 min, f64 max, s32 dim) {
  return (s32)(((r - max) / (min - max)) * (f64)dim);
}

void displayPointCouplings(const kdt::KDTree<Pt2>& kdtree,
                           const Eigen::MatrixX2i& couplings) {
  // kdt::KDTree<Pt2> kdtree(points);
  size_t xmin_idx = kdtree.axisFindMin(0);
  size_t xmax_idx = kdtree.axisFindMax(0);
  size_t ymin_idx = kdtree.axisFindMin(1);
  size_t ymax_idx = kdtree.axisFindMax(1);
  double xmin = kdtree.points_[xmin_idx][0];
  double xmax = kdtree.points_[xmax_idx][0];
  double ymin = kdtree.points_[ymin_idx][1];
  double ymax = kdtree.points_[ymax_idx][1];
  double exmin = xmin - 0.05 * (xmax - xmin);
  double eymin = ymin - 0.05 * (ymax - ymin);
  double exmax = xmax + 0.05 * (xmax - xmin);
  double eymax = ymax + 0.05 * (ymax - ymin);

  int width = 1080;
  int height = 1080;
  SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE |
                 FLAG_WINDOW_TRANSPARENT);
  InitWindow(width, height, "raylib test");
  SetTargetFPS(10);
  while (!WindowShouldClose()) {
    width = GetScreenWidth();
    height = GetScreenHeight();
    BeginDrawing();
    ClearBackground(WHITE);
    for (const auto& point : kdtree.points_) {
      DrawCircle(toScreenX(point[0], exmin, exmax, width),
                 toScreenY(point[1], eymin, eymax, height), 3.0f, RED);
    }
    for (const auto& nb : couplings.rowwise()) {
      auto startx = toScreenX(kdtree.points_[nb[0]][0], exmin, exmax, width);
      auto starty = toScreenY(kdtree.points_[nb[0]][1], eymin, eymax, height);
      auto endx = toScreenX(kdtree.points_[nb[1]][0], exmin, exmax, width);
      auto endy = toScreenY(kdtree.points_[nb[1]][1], eymin, eymax, height);
      DrawLine(startx, starty, endx, endy, BLUE);
    }
    EndDrawing();
    if (IsKeyPressed(KEY_P)) {
      TakeScreenshot("graph.png");
    }
  }
  CloseWindow();
}

int main(int argc, char* argv[]) {
  cxxopts::Options options(
      "couplings",
      "Create couplings between points.\n Input: File containing Nx2 HDF5 "
      "dataset called \"points,\" or Nx2 plain text file.\nOutput: HDF5 file "
      "containing Nx2 dataset \"points\" with the points, and Nx2 dataset "
      "\"couplings\" containing the indices of points in the first column and "
      "the indices of points they connect to in the second.");
  options.add_options()("points", "File containing points",
                        cxxopts::value<std::string>())(
      "r,radius", "Search radius", cxxopts::value<f64>())(
      "o,out", "Output file", cxxopts::value<std::string>())(
      "w,window", "View in window")("h,help", "Help");
  options.parse_positional({"points"});
  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
  if (result["h"].count() or !result["points"].count()) {
    std::cout << options.help() << std::endl;
    return 0;
  }
  if (result["points"].count()) {
    std::string fname = result["points"].as<std::string>();
    std::vector<Pt2> points;
    if (fname.substr(fname.find_last_of('.') + 1) == "h5") {
      HighFive::File pfile(fname, HighFive::File::ReadOnly);
      spdlog::debug("Read file {}", fname);
      auto space = pfile.getDataSet("points").getSpace();
      std::cout << space.getDimensions()[0] << '\n';
      points.resize(space.getDimensions()[0]);
      pfile.getDataSet("points").read_raw<f64>((f64*)points.data());
      // std::vector<f64> prePoints =
      //     file.getDataSet("points").read<std::vector<f64>>();
      spdlog::debug("Loaded points");
    } else {
      // Eigen::MatrixX2cd points = readEigen<f64>(fname);
      u32 m = 0;
      std::ifstream f(fname);
      if (!f.good()) {
        throw runtime_exc("File {} doesn't exist", fname);
      }
      std::vector<std::string> allLines{std::istream_iterator<Line>(f),
                                        std::istream_iterator<Line>()};
      m = allLines.size();
      spdlog::debug("Number of rows: {}", m);
      //  f.clear();
      //  f.seekg(0, std::ios::beg);
      points.resize(m);
      for (u64 j = 0; j < m; j++) {
        std::istringstream stream(allLines[j]);
        std::vector<double> v{std::istream_iterator<double>(stream),
                              std::istream_iterator<double>()};
        points[j] = {v[0], v[1]};
      }
    }
    kdt::KDTree<Pt2> kdtree(points);

    f64 radius = 0;
    switch (result["r"].count()) {
    case 0: {
      f64 maxDist = max_nn_dist(kdtree, points);
      f64 avgDist = avg_nn_dist(kdtree, points);
      std::cout << "Max distance between nearest neighbours: "
                << std::format("{}", maxDist);
      std::cout << "Average distance between nearest neighbours: "
                << std::format("{}", avgDist);
      break;
    }

    default: {
      radius = result["r"].as<f64>();
      break;
    }
    }
    Eigen::MatrixX2i couplings(points.size() * (points.size() - 1) / 2, 2);
    s64 tmp = 0;
    for (u64 i = 0; i < points.size(); ++i) {
      auto q = points[i];
      auto nbs = kdtree.radiusSearch(q, radius);
      for (const auto& idx : nbs) {
        if ((size_t)idx > i) {

          spdlog::debug("source: {}", i);
          spdlog::debug("dest: {}", idx);
          couplings(tmp, 0) = i;
          couplings(tmp, 1) = idx;
          ++tmp;
        }
      }
    }
    couplings.conservativeResize(tmp, 2);
    std::cout << points.size() << '\n';
    std::cout << "rows: " << couplings.rows() << '\n';
    std::cout << "cols: " << couplings.cols() << '\n';
    if (result["w"].count()) {
      displayPointCouplings(points, couplings);
    }
    std::string outfile = result["o"].as<std::string>();
    spdlog::debug("Writing to file {}", outfile);
    HighFive::File file(outfile, HighFive::File::Truncate);
    spdlog::debug("Creating dataset couplings");
    file.createDataSet("couplings", couplings);
    // auto couplingset = file.createDataSet<s64>("couplings", {(u64)tmp, 2});
    // couplingset.write_raw(couplings.data());
    std::array<size_t, 2> dims{points.size(), 2};
    spdlog::debug("Creating dataset points");
    auto pointSet =
        file.createDataSet<f64>("points", HighFive::DataSpace(dims));
    spdlog::debug("Writing points");
    pointSet.write_raw((f64*)points.data());
    return 0;
  }
  return 0;
}
