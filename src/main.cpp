#include "Eigen/Core"
#include "Eigen/Dense"
#include "H5Cpp.h"
#include "hermEigen.h"
#include "kdtree.h"
#include "lodepng.h"
#include "approximant.h"
#include "mathhelpers.h"
#include "typedefs.h"
#include "io.h"
#include "geometry.h"
#include "SDF.h"
#include <H5public.h>
#include <cxxopts.hpp>
#include <exception>
#include <iostream>
#include <limits>
#include <ostream>
#include <mdspan>
#include <ranges>
#include <sstream>
#include <string>
#include <toml++/toml.hpp>

using Eigen::MatrixX2d, Eigen::MatrixXd;

static Eigen::IOFormat defaultFormat(Eigen::StreamPrecision,
                                     Eigen::DontAlignCols, " ", "\n", "", "",
                                     "", "");

int writeh5wexc(hid_t dataset, hid_t type, hid_t fspace, const void* data) {
    return H5Dwrite(dataset, type, H5S_ALL, fspace, H5P_DEFAULT, data);
}

template <class D>
void saveEigen(const std::string& fname, const Eigen::MatrixBase<D>& x) {
  std::ofstream f(fname);
  f << x.format(defaultFormat);
  f.close();
}

template <class T>
Eigen::MatrixX<T> readEigenStream(const std::string& fname) {
  std::string line;
  std::ifstream f(fname);
  u32 m = 0;
  u32 n = 0;
  while (std::getline(f, line)) {
    if (n == 0) {
      auto splits = line | std::ranges::views::split(' ');
      for (const auto& _ : splits) {
        m += 1;
      }
    }
    n += 1;
  }
  f.clear();
  f.seekg(0, std::ios::beg);
  Eigen::MatrixX<T> M(n, m);
  u32 j = 0;
  u32 i = 0;
  M(i, j) >> f;
}

template <class T>
Eigen::MatrixX<T> readEigen(std::string fname) {
  u32 m = 0;
  u32 n = 0;
  std::ifstream f(fname);
  std::vector<std::string> allLines{std::istream_iterator<Line>(f),
                                    std::istream_iterator<Line>()};
  m = allLines.size();
  std::istringstream s(allLines[0]);
  std::vector<T> v{std::istream_iterator<T>(s), std::istream_iterator<T>()};
  n = v.size();
  // f.clear();
  // f.seekg(0, std::ios::beg);
  Eigen::MatrixX<T> M(m, n);
  u32 j = 0;
  for (const auto& line : allLines) {
    std::istringstream stream(line);
    std::vector<T> v{std::istream_iterator<T>(stream),
                     std::istream_iterator<T>()};
    u32 i = 0;
    for (const auto bleh : v) {
      M(j, i) = bleh;
      i += 1;
    }
    j += 1;
  }
  return M;
}

void pointsToPeriodicCouplings(std::vector<Point>& points, f64 rsq,
                               std::optional<double> lx,
                               std::optional<double> ly) {
  /* This function is meant to create couplings for approximants of
   * quasicrystals. The unit cell vectors are taken to be parallel to the x and
   * y axes. If lx and/or ly have values, those are taken to be the x/y lengths
   * of the unit cell, otherwise the lengths are estimated to be the x/y extents
   * of the approximant plus the average x/y separation between neighbouring
   * points.
   */
  standardise(points);
  kdt::KDTree<Point> kdtree(points);
  double maxx = points[kdtree.axisFindMax(0)][0];
  double maxy = points[kdtree.axisFindMax(1)][1];
  kdtree.build(points);
  double avg_nn_dist = 1.0;
  if (!lx.has_value() || !ly.has_value()) {
    if (points.size() > 1) {
      avg_nn_dist = avgNNDist(kdtree, points);
    }
  }
  double Lx = lx.value_or(maxx + avg_nn_dist);
  double Ly = ly.value_or(maxy + avg_nn_dist);
  std::cout << std::format("Lx is {}\n"
                           "Ly is {}\n"
                           "Average distance to next neighbour is: {}\n",
                           Lx, Ly, avg_nn_dist);
  double search_radius = std::sqrt(rsq);
  auto x_edge =
      kdtree.axisSearch(0, (1 + 1e-8) * (search_radius - avg_nn_dist));
  auto y_edge =
      kdtree.axisSearch(1, (1 + 1e-8) * (search_radius - avg_nn_dist));
  std::vector<int> xy_corner;
  // Estimate of how many points there are in the intersection of the edges.
  xy_corner.reserve((size_t)((double)(x_edge.size() * y_edge.size()) /
                             (double)points.size()));
  for (const auto xidx : x_edge) {
    for (const auto yidx : y_edge) {
      if (xidx == yidx) {
        xy_corner.push_back(xidx);
      }
    }
  }
  auto final_grid = extended_grid(points, x_edge, y_edge, xy_corner, Lx, Ly);
  kdtree.build(final_grid);
  std::vector<Neighbour> nb_info;
  for (size_t i = 0; i < points.size(); i++) {
    auto q = final_grid[i];
    auto nbs = kdtree.radiusSearch(q, search_radius);
    for (const auto idx : nbs) {
      if ((size_t)idx > i) {
        auto p = final_grid[idx];
        Vector2d d = {p[0] - q[0], p[1] - q[1]};
        nb_info.emplace_back(i, p.idx, d);
      }
    }
  }
  MatrixXcd hamiltonian = MatrixXcd::Zero(points.size(), points.size());
  constexpr u32 ksamples = 20;
  Vector2d dual1{2 * M_PI / Lx, 0};
  Vector2d dual2{0, 2 * M_PI / Ly};
  std::vector<double> energies(ksamples * ksamples * points.size());
  auto energy_view =
      std::mdspan(energies.data(), ksamples, ksamples, points.size());
  for (u32 j = 0; j < ksamples; j++) {
    double yfrac = (double)j / ksamples;
    for (u32 i = 0; i < ksamples; i++) {
      double xfrac = (double)i / ksamples;
      update_hamiltonian(
          hamiltonian, nb_info, xfrac * dual1 + yfrac * dual2,
          [](Vector2d) { return c64{-1, 0}; }, i | j);
      EigenSolution eigsol = hermitianEigenSolver(hamiltonian);
      for (size_t k = 0; k < points.size(); k++) {
        energy_view[i, j, k] = eigsol.D(k);
      }
    }
  }
  hsize_t dims[3] = {ksamples, ksamples, points.size()};
  H5::DataSpace space(3, dims);
  H5::H5File file("energies.h5", H5F_ACC_TRUNC);
  H5::DataSet dataset(
      file.createDataSet("aaa", H5::PredType::NATIVE_DOUBLE, space));
  dataset.write(energies.data(), H5::PredType::NATIVE_DOUBLE);
}

// array with space for at most N elements, but may have fewer
template <class T, size_t N>
struct MaxHeadroom {
  T data[N];
  size_t n;
  void push(T x) {
#ifndef NDEBUG
    assert(n < N);
#endif
    data[n] = x;
    n += 1;
  }
  void pop() {
#ifndef NDEBUG
    assert(n > 0);
#endif
    n -= 1;
  }
  constexpr T operator[](size_t i) const { return data[i]; }
  T& operator[](size_t i) { return data[i]; }
  T* begin() { return data; }

  T* end() { return data + n; }
  constexpr T* cbegin() noexcept { return data; }

  constexpr T* cend() noexcept { return data + n; }
};

double smallestNonZeroGap(const VectorXd& vals) {
  double min_gap = std::numeric_limits<double>::max();
  for (int i = 1; i < vals.size(); i++) {
    if (double gap = vals[i] - vals[i - 1]; gap > 1e-14 && gap < min_gap) {
      min_gap = gap;
    }
  }
  return min_gap;
}

struct SdfConf {
  // sharpness of Gaussian used to approximate Delta function
  double sharpening;
  // values below this will be removed from the Delta function.
  double cutoff;
  // In units of average nearest neighbour distance.
  double searchRadius;
  // if hasValue, do a dispersion relation with the line given by this range.
  std::optional<RangeConf<Vector2d>> DispKline;
  // Set Emin=Emax to automatically set E range
  std::optional<RangeConf<double>> DispE;
  std::optional<std::string> saveDiagonalisation;
  std::optional<std::string> useSavedDiag;
  std::optional<std::string> saveHamiltonian;
  // in units of "Brillouin zone", i.e. 1 = 2pi/a where a is a lattice constant.
  // Average nearest neighbour distance is used as a proxy for a
  RangeConf<double> SDFKx;
  RangeConf<double> SDFKy;
  // Set Emin=Emax to automatically set E range
  RangeConf<double> SDFE;
  // lattice point input (could possibly take special strings to do common
  // lattices)
  double fixed_e;
  std::string pointPath;
  // Output file name
  std::string H5Filename;
  // Generally what I want. Not any more expensive than computing DOS and the
  // rest can be obtained by slicing this dataset.
  bool doFullSDF;
  // Only set if you don't want to output a full sdf to disk. Will be ignored if
  // doFullSDF is set.
  bool doDOS;
  bool doEsection;
};

RangeConf<Vector2d> tblToVecRange(const toml::table& tbl) {
  toml::array start = *tbl["start"].as_array();
  toml::array end = *tbl["end"].as_array();
  return {{*start[0].as_floating_point(), *start[1].as_floating_point()}, {*end[0].as_floating_point(), *end[1].as_floating_point()}, tbl["n"].value_or<u64>(0)};
}

RangeConf<double> tblToRange(toml::table& tbl) {
  return {tbl["start"].value_or<double>(0.0), tbl["end"].value_or<double>(0.0),
          tbl["n"].value_or<u64>(0)};
}

#define SET_STRUCT_FIELD(c, tbl, key) \
  if (tbl.contains(#key))\
    c.key = *tbl[#key].value<decltype(c.key)>()

std::optional<SdfConf> tomlToSDFConf(std::string tomlPath) {
  toml::table tbl;
  try {
    tbl = toml::parse_file(tomlPath);
  } catch (const std::exception& err) {
    std::cerr << "Parsing file " << tomlPath  << " failed with exception: " << err.what() << '\n';
    return {};
  }
  SdfConf conf{};
  toml::table& preConf = *tbl["PreConf"].as_table();
  SET_STRUCT_FIELD(conf, preConf, sharpening);
  SET_STRUCT_FIELD(conf, preConf, cutoff);
  SET_STRUCT_FIELD(conf, preConf, searchRadius);
  SET_STRUCT_FIELD(conf, preConf, pointPath);
  SET_STRUCT_FIELD(conf, preConf, H5Filename);
  SET_STRUCT_FIELD(conf, preConf, doFullSDF);
  SET_STRUCT_FIELD(conf, preConf, doDOS);
  SET_STRUCT_FIELD(conf, preConf, doEsection);
  SET_STRUCT_FIELD(conf, preConf, fixed_e);
  if (tbl.contains("DispKline")) {
    conf.DispKline = tblToVecRange(*tbl["DispKline"].as_table());
    conf.DispE = tblToRange(*tbl["DispE"].as_table());
  }
  if (preConf.contains("saveDiagonalisation"))
    conf.saveDiagonalisation = preConf["saveDiagonalisation"].value<std::string>();
  if (preConf.contains("useSavedDiag"))
    conf.useSavedDiag = preConf["useSavedDiag"].value<std::string>();
  if (preConf.contains("saveHamiltonian"))
    conf.saveHamiltonian = preConf["saveHamiltonian"].value<std::string>();
  conf.SDFKx = tblToRange(*tbl["SDFKx"].as_table());
  conf.SDFKy = tblToRange(*tbl["SDFKy"].as_table());
  conf.SDFE = tblToRange(*tbl["SDFE"].as_table());
  return conf;
}
#undef SET_STRUCT_FIELD

EigenSolution pointsToDiagFormHamiltonian(const std::vector<Point>& points,
                                          const kdt::KDTree<Point>& kdtree,
                                          double rad) {
  auto H = pointsToFiniteHamiltonian(points, kdtree, rad);
  EigenSolution eig = hermitianEigenSolver(H);
  return eig;
}

template <size_t n>
void writeArray(std::string s, H5::H5File& file, void* data, hsize_t sizes[n]) {
  // std::array<hsize_t, sizeof(sizes)> dims{sizes};
  hid_t space = H5Screate_simple(n, sizes, nullptr);
  // hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  hid_t set = H5Dcreate2(file.getId(), s.c_str(), H5T_NATIVE_DOUBLE_g, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
      // file.createDataSet(s, H5::PredType::NATIVE_DOUBLE, space);
  hid_t res = H5Dwrite(set, H5T_NATIVE_DOUBLE_g, H5S_ALL, space, H5P_DEFAULT, data);
  if (res < 0) {
    std::cout << "Failed to write HDF5 file\n";
  } 
  // writeh5wexc(set, data, H5T_STD_B64LE_g);
}

bool file_exists(const std::string& name) {
  if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

int main(const int argc, const char* const* argv) {
  cxxopts::Options options("MyProgram", "bleh");
  options.add_options()("p,points", "File name", cxxopts::value<std::string>())(
      "c,config", "TOML configuration", cxxopts::value<std::string>())(
      "t,test", "Test whatever feature I'm working on rn.",
      cxxopts::value<std::string>());

  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return 1;
  }
  if (result["c"].count()) {
    std::string fname = result["c"].as<std::string>();
    SdfConf conf;
    if (auto opt = tomlToSDFConf(fname); opt.has_value()) {
      conf = opt.value();
    } else {
      return 1;
    }
    if (!(conf.DispKline.has_value() || conf.doEsection || conf.doDOS || conf.doFullSDF)) {
      std::cout << "No tasks selected.\n";
      return 0;
    }
    auto points = readPoints(conf.pointPath);
    kdt::KDTree kdtree(points);
    double a = avgNNDist(kdtree, points);
    EigenSolution eigsol;
    bool useSavedSucceeded = false;
    if (conf.useSavedDiag.has_value()) {
      const std::string& fname = conf.useSavedDiag.value();
      if (file_exists(fname)) {
        auto flist_id = H5Pcreate(H5P_FILE_ACCESS);
        std::cout << "Trying to open file" << fname << '\n';
        auto fid = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, flist_id);
        std::cout << "Trying to get dataset" << fname << '\n';
        auto did = H5Dopen2(fid, "D", H5P_DEFAULT);
        auto dspace = H5Dget_space(did);
        hsize_t dims[1];
        H5Sget_simple_extent_dims(dspace, dims, nullptr);
        eigsol.D.resize(dims[0]);
        eigsol.U.resize(dims[0], dims[0]);
        H5Sselect_all(dspace);
        
        H5Dread(did, H5T_NATIVE_DOUBLE, H5S_ALL, dspace, H5P_DEFAULT, eigsol.D.data());
        H5Sclose(dspace);
        H5Dclose(did);
        auto uid = H5Dopen2(fid, "U", H5P_DEFAULT);
        auto uspace = H5Dget_space(uid);
        H5Sselect_all(uspace);
        H5Dread(uid, H5T_NATIVE_DOUBLE, H5S_ALL, uspace, H5P_DEFAULT, eigsol.U.data());
        H5Sclose(uspace);
        H5Dclose(uid);
        H5Fclose(fid);
        H5Pclose(flist_id);
        useSavedSucceeded = true;
      }
    }
    if (!useSavedSucceeded) {
      MatrixXd H = pointsToFiniteHamiltonian(points, kdtree, conf.searchRadius * a);
      if (conf.saveHamiltonian.has_value()) {
        saveEigen(conf.saveHamiltonian.value(), H);
      }
      eigsol = hermitianEigenSolver(H);
    }
    if (conf.saveDiagonalisation.has_value() && !useSavedSucceeded) {
      H5::H5File file = H5Fcreate(conf.saveDiagonalisation.value().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      hsize_t sizes[2] = {static_cast<hsize_t>(eigsol.D.size()), static_cast<hsize_t>(2 * eigsol.D.size())};
      writeArray<1>("D", file, eigsol.D.data(), sizes);
      writeArray<2>("U", file, eigsol.U.data(), sizes);
      file.close();
    }
    H5::H5File file = H5Fcreate(conf.H5Filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file.getId() == H5I_INVALID_HID) {
      std::cerr << "Failed to create file " << conf.H5Filename << std::endl;
      return 1;
    }
    if (conf.doEsection) {
      auto UH = eigsol.U.adjoint();
      auto section = Esection(eigsol.D, UH, points, a, conf.SDFKx, conf.SDFKy, conf.fixed_e, conf.sharpening, conf.cutoff);
      hsize_t sizes[2] = {conf.SDFKx.n, conf.SDFKy.n};
      writeArray<2>("section", file, section.data(), sizes);
      double sdfBounds[5] = {conf.SDFKx.start, conf.SDFKx.end,
                                         conf.SDFKy.start, conf.SDFKy.end,
                                         conf.fixed_e};
      hsize_t boundsize[1] = {5};
      writeArray<1>("section_bounds", file, sdfBounds, boundsize);
    } else if (conf.doFullSDF) {
      auto UH = eigsol.U.adjoint();
      auto sdf =
          fullSDF(eigsol.D, UH, points, a, conf.SDFKx, conf.SDFKy, conf.SDFE, conf.sharpening, conf.cutoff);
      hsize_t sizes[3] = {conf.SDFKx.n, conf.SDFKy.n,
                 conf.SDFE.n};
      writeArray<3>("sdf", file, sdf.data(), sizes);
      double sdfBounds[6] = {conf.SDFKx.start, conf.SDFKx.end,
                                         conf.SDFKy.start, conf.SDFKy.end,
                                         conf.SDFE.start,  conf.SDFE.end};
      hsize_t boundsize[1] = {6};
      writeArray<1>("sdf_bounds", file, sdfBounds, boundsize);
    } else if (conf.doDOS) {
      auto UH = eigsol.U.adjoint();
      auto dos = DOS(eigsol.D, UH, points, a, conf.SDFKx, conf.SDFKy, conf.SDFE, conf.sharpening, conf.cutoff);
      writeArray<1>("dos", file, dos.data(), &conf.SDFE.n);
      double dosBounds[2] = {conf.SDFE.start, conf.SDFE.end};
      hsize_t boundsize[1] = {2};
      writeArray<1>("dos_bounds", file, dosBounds, boundsize); 
    }
    if (conf.DispKline.has_value()) {
      if (conf.DispE.has_value()) {
        auto kc = conf.DispKline.value();
        auto ec = conf.DispE.value();
        auto UH = eigsol.U.adjoint();
        auto dis = disp(eigsol.D, UH, points, a, kc, ec, conf.sharpening, conf.cutoff);
        hsize_t sizes[2] = {kc.n, ec.n};
        writeArray<2>("disp", file, dis.data(), sizes);
        std::array<double, 6> dispBounds = {kc.start[0], kc.start[1], kc.end[0],
                                            kc.end[1],   ec.start,    ec.end};
        hsize_t boundsizes[1] = {6};
        writeArray<1>("disp_bounds", file, dispBounds.data(), boundsizes);
      } else {
        std::cout << "Need to supply a non-zero number of energy samples\n";
      }
    }
  }
  if (result["t"].count()) {
    MatrixXd A = MatrixXd::Ones(3, 6);
    MatrixXd B = MatrixXd::Ones(6, 1);
    std::cout << A*B << '\n';
    // auto vec = readPoints(result["t"].as<std::string>());
    // constexpr size_t N = 40;
    // std::vector<Point> vec(N * N);
    // for (u32 i = 0; i < N; i++) {
    //   for (u32 j = 0; j < N; j++) {
    //     vec[N * i + j] = Point(i, j, N * i + j);
    //   }
    // }
    // pointsToSDFOnFile(vec, 1.0);
    return 0;
  }
}
