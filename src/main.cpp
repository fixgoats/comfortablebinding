#include "Eigen/Dense"
#include "H5Cpp.h"
#include "SDF.h"
#include "approximant.h"
#include "geometry.h"
#include "hermEigen.h"
#include "io.h"
#include "kdtree.h"
#include "vkcore.h"
// #include "lodepng.h"
#include "mathhelpers.h"
#include "typedefs.h"
#include <H5Fpublic.h>
#include <H5public.h>
#include <cstddef>
#include <cxxopts.hpp>
#include <exception>
#include <iostream>
#include <limits>
#include <mdspan>
#include <ostream>
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

struct PerConf {
  std::string fname;
  std::vector<Point> points;
  std::vector<Neighbour> nbs;
  std::optional<RangeConf<double>> kxrange;
  std::optional<RangeConf<double>> kyrange;
  std::vector<RangeConf<Vector2d>> kpath;
  Vector2d lat_vecs[2];
  Vector2d dual_vecs[2];
  bool do2D; // combine dispersion and dos, since dos follows from 2d disp
  bool doPath;

  // lat_vecs needs to be populated before running this. Should only be ran
  // once.
  void parseConnections(toml::array& arr) {
    nbs.reserve(arr.size() * 3); // decent guess for number of connections
    for (const auto& st : arr) {
      auto tbl = *st.as_table();
      size_t i = tbl["source"].value<size_t>().value();
      auto dests = *tbl["destinations"].as_array();
      for (const auto& dest : dests) {
        const auto t = *dest.as_array();
        s64 n0 = t[1].value<s64>().value();
        s64 n1 = t[2].value<s64>().value();
        size_t j = t[0].value<size_t>().value();
        Vector2d d = points[j].asVec() + n0 * lat_vecs[0] + n1 * lat_vecs[1] -
                     points[i].asVec();
        nbs.emplace_back(i, j, d);
      }
    }
    nbs.shrink_to_fit();
  }

  void makeDuals() {
    dual_vecs[0] = {
        2 * M_PI /
            (lat_vecs[0][0] - lat_vecs[0][1] * lat_vecs[1][0] / lat_vecs[1][1]),
        2 * M_PI /
            (lat_vecs[0][1] -
             lat_vecs[0][0] * lat_vecs[1][1] / lat_vecs[1][0])};
    dual_vecs[1] = {
        2 * M_PI /
            (lat_vecs[1][0] - lat_vecs[1][1] * lat_vecs[0][0] / lat_vecs[0][1]),
        2 * M_PI /
            (lat_vecs[1][1] -
             lat_vecs[1][0] * lat_vecs[0][1] / lat_vecs[0][0])};
  }
};

struct SdfConf {
  // sharpness of Gaussian used to approximate Delta function
  double sharpening;
  // values below this will be removed from the Delta function.
  double cutoff;
  // In units of average nearest neighbour distance.
  double searchRadius;
  // if hasValue, do a dispersion relation with the line given by this range.
  std::vector<RangeConf<Vector2d>> kpath;
  // Set Emin=Emax to automatically set E range
  std::optional<RangeConf<double>> dispE;
  std::optional<std::string> saveDiagonalisation;
  std::optional<std::string> useSavedDiag;
  std::optional<std::string> saveHamiltonian;
  // in units of "Brillouin zone", i.e. 1 = 2pi/a where a is a lattice constant.
  // Average nearest neighbour distance is used as a proxy for a
  RangeConf<double> sectionKx;
  RangeConf<double> sectionKy;
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
  bool doPath;
};

RangeConf<Vector2d> tblToVecRange(const toml::table& tbl) {
  toml::array start = *tbl["start"].as_array();
  toml::array end = *tbl["end"].as_array();
  return {{start[0].value<f64>().value(), start[1].value<f64>().value()},
          {end[0].value<f64>().value(), end[1].value<f64>().value()},
          tbl["n"].value_or<u64>(0)};
}

RangeConf<double> tblToRange(toml::table& tbl) {
  return {tbl["start"].value_or<double>(0.0), tbl["end"].value_or<double>(0.0),
          tbl["n"].value_or<u64>(0)};
}

template <class T>
std::vector<T> tarrayToVec(const toml::array& arr) {
  std::vector<T> tmp(arr.size());
  for (u64 i = 0; i < arr.size(); i++) {
    tmp[i] = arr[i].value<T>().value();
  }
  return tmp;
}

#define SET_STRUCT_FIELD(c, tbl, key)                                          \
  if (tbl.contains(#key))                                                      \
  c.key = *tbl[#key].value<decltype(c.key)>()

std::optional<PerConf> tomlToPerConf(std::string tomlPath) {
  toml::table tbl;
  try {
    tbl = toml::parse_file(tomlPath);
  } catch (const std::exception& err) {
    std::cerr << "Parsing file " << tomlPath
              << " failed with exception: " << err.what() << '\n';
    return {};
  }
  if (!tbl.contains("lattice"))
    return {};
  PerConf conf{};
  std::cout << "Got here\n";
  toml::table latticeTable = *tbl["lattice"].as_table();
  toml::table& fnameTable = *tbl["filename"].as_table();
  SET_STRUCT_FIELD(conf, fnameTable, fname);
  conf.do2D = tbl["calcs"]["do2D"].value_or(false);
  conf.doPath = tbl["calcs"]["doPath"].value_or(false);
  if (!conf.do2D & !conf.doPath) {
    std::cout << "No calculation specified." << std::endl;
    return {};
  }
  auto points = *latticeTable["points"].as_array();
  conf.points = [&]() {
    std::vector<Point> tmp;
    tmp.reserve(points.size());
    u32 i = 0;
    for (const auto& p : points) {
      auto x = *p.as_array();
      tmp.emplace_back(x[0].value<double>().value(),
                       x[1].value<double>().value(), i);
      ++i;
    }
    return tmp;
  }();
  auto lat_vecs = *latticeTable["lat_vecs"].as_array();
  auto v = *lat_vecs[0].as_array();
  conf.lat_vecs[0] = {v[0].value<double>().value(),
                      v[1].value<double>().value()};
  v = *lat_vecs[1].as_array();
  conf.lat_vecs[1] = {v[0].value<double>().value(),
                      v[1].value<double>().value()};
  conf.parseConnections(*latticeTable["connections"].as_array());
  if (conf.do2D) {
    try {
      toml::table gridSpec = *tbl["grid"].as_table();
      conf.kxrange = tblToRange(*gridSpec["kxrange"].as_table());
      conf.kyrange = tblToRange(*gridSpec["kyrange"].as_table());
    } catch (const std::exception& exc) {
      std::cerr << "2D calculation requested but no grid specified: "
                << exc.what() << std::endl;
      return {};
    }
  }
  if (conf.doPath) {
    try {
      toml::table pathSpec = *tbl["path"].as_table();
      toml::array ranges = *pathSpec["ranges"].as_array();
      for (const auto& range : ranges) {
        toml::table r = *range.as_table();
        conf.kpath.push_back(tblToVecRange(r));
      }
    } catch (const std::exception& exc) {
      std::cerr << "Path dispersion requested but no ranges specified: "
                << exc.what() << std::endl;
    }
  }
  return conf;
}

std::optional<SdfConf> tomlToSDFConf(std::string tomlPath) {
  toml::table tbl;
  try {
    tbl = toml::parse_file(tomlPath);
  } catch (const std::exception& err) {
    std::cerr << "Parsing file " << tomlPath
              << " failed with exception: " << err.what() << '\n';
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
  SET_STRUCT_FIELD(conf, preConf, doPath);
  SET_STRUCT_FIELD(conf, preConf, fixed_e);
  if (conf.doPath) {
    try {
      toml::table pathSpec = *tbl["path"].as_table();
      toml::array ranges = *pathSpec["ranges"].as_array();
      for (const auto& range : ranges) {
        toml::table r = *range.as_table();
        conf.kpath.push_back(tblToVecRange(r));
      }
      conf.dispE = tblToRange(*tbl["dispE"].as_table());
    } catch (const std::exception& exc) {
      std::cerr << "Path dispersion requested but no ranges specified: "
                << exc.what() << std::endl;
    }
  }

  if (preConf.contains("saveDiagonalisation"))
    conf.saveDiagonalisation =
        preConf["saveDiagonalisation"].value<std::string>();
  if (preConf.contains("useSavedDiag"))
    conf.useSavedDiag = preConf["useSavedDiag"].value<std::string>();
  if (preConf.contains("saveHamiltonian"))
    conf.saveHamiltonian = preConf["saveHamiltonian"].value<std::string>();
  conf.sectionKx = tblToRange(*tbl["sectionKx"].as_table());
  conf.sectionKy = tblToRange(*tbl["sectionKy"].as_table());
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
void writeArray(std::string s, hid_t fid, void* data, hsize_t sizes[n]) {
  // std::array<hsize_t, sizeof(sizes)> dims{sizes};
  hid_t space = H5Screate_simple(n, sizes, nullptr);
  // hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  hid_t set = H5Dcreate2(fid, s.c_str(), H5T_NATIVE_DOUBLE_g, space,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  // fid.createDataSet(s, H5::PredType::NATIVE_DOUBLE, space);
  hid_t res =
      H5Dwrite(set, H5T_NATIVE_DOUBLE_g, H5S_ALL, space, H5P_DEFAULT, data);
  if (res < 0) {
    std::cout << "Failed to write HDF5 fid\n";
  }
  // writeh5wexc(set, data, H5T_STD_B64LE_g);
}

template <size_t n>
void writeSingleArray(std::string s, hid_t fid, void* data, hsize_t sizes[n]) {
  // std::array<hsize_t, sizeof(sizes)> dims{sizes};
  hid_t space = H5Screate_simple(n, sizes, nullptr);
  // hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
  hid_t set = H5Dcreate2(fid, s.c_str(), H5T_NATIVE_FLOAT_g, space, H5P_DEFAULT,
                         H5P_DEFAULT, H5P_DEFAULT);
  // fid.createDataSet(s, H5::PredType::NATIVE_DOUBLE, space);
  hid_t res =
      H5Dwrite(set, H5T_NATIVE_FLOAT_g, H5S_ALL, space, H5P_DEFAULT, data);
  if (res < 0) {
    std::cout << "Failed to write HDF5 fid\n";
  }
  // writeh5wexc(set, data, H5T_STD_B64LE_g);
}

bool file_exists(const std::string& name) {
  if (FILE* file = fopen(name.c_str(), "r")) {
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
    if (!(conf.doPath || conf.doEsection || conf.doDOS || conf.doFullSDF)) {
      std::cout << "No tasks selected.\n";
      return 0;
    }

    std::vector<Point> points;
    if (conf.pointPath == "square") {
      points.resize(70 * 70);
      for (u32 i = 0; i < 70; i++) {
        for (u32 j = 0; j < 70; j++) {
          points[i * 70 + j] = {(double)i, (double)j, i * 70 + j};
        }
      }
    } else
      points = readPoints(conf.pointPath);

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

        H5Dread(did, H5T_NATIVE_DOUBLE, H5S_ALL, dspace, H5P_DEFAULT,
                eigsol.D.data());
        H5Sclose(dspace);
        H5Dclose(did);
        auto uid = H5Dopen2(fid, "U", H5P_DEFAULT);
        auto uspace = H5Dget_space(uid);
        H5Sselect_all(uspace);
        H5Dread(uid, H5T_NATIVE_DOUBLE, H5S_ALL, uspace, H5P_DEFAULT,
                eigsol.U.data());
        H5Sclose(uspace);
        H5Dclose(uid);
        H5Fclose(fid);
        H5Pclose(flist_id);
        useSavedSucceeded = true;
      }
    }
    if (!useSavedSucceeded) {
      MatrixXd H =
          pointsToFiniteHamiltonian(points, kdtree, conf.searchRadius * a);
      if (conf.saveHamiltonian.has_value()) {
        saveEigen(conf.saveHamiltonian.value(), H);
      }
      eigsol = hermitianEigenSolver(H);
    }
    if (conf.saveDiagonalisation.has_value() && !useSavedSucceeded) {
      hid_t file = H5Fcreate(conf.saveDiagonalisation.value().c_str(),
                             H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      hsize_t sizes[2] = {static_cast<hsize_t>(eigsol.D.size()),
                          static_cast<hsize_t>(2 * eigsol.D.size())};
      writeArray<1>("D", file, eigsol.D.data(), sizes);
      writeArray<2>("U", file, eigsol.U.data(), sizes);
      H5Fclose(file);
    }
    hid_t file = H5Fcreate(conf.H5Filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                           H5P_DEFAULT);
    if (file == H5I_INVALID_HID) {
      std::cerr << "Failed to create file " << conf.H5Filename << std::endl;
      return 1;
    }
    if (conf.doEsection) {
      auto UH = eigsol.U.adjoint();
      auto section =
          Esection(eigsol.D, UH, points, a, conf.sectionKx, conf.sectionKy,
                   conf.fixed_e, conf.sharpening, conf.cutoff);
      hsize_t sizes[2] = {conf.sectionKx.n, conf.sectionKy.n};
      writeArray<2>("section", file, section.data(), sizes);
      double sdfBounds[5] = {conf.sectionKx.start, conf.sectionKx.end,
                             conf.sectionKy.start, conf.sectionKy.end,
                             conf.fixed_e};
      hsize_t boundsize[1] = {5};
      writeArray<1>("section_bounds", file, sdfBounds, boundsize);
    } else if (conf.doFullSDF) {
      auto UH = eigsol.U.adjoint();
      auto sdf = fullSDF(eigsol.D, UH, points, a, conf.SDFKx, conf.SDFKy,
                         conf.SDFE, conf.sharpening, conf.cutoff);
      hsize_t sizes[3] = {conf.SDFKx.n, conf.SDFKy.n, conf.SDFE.n};
      writeArray<3>("sdf", file, sdf.data(), sizes);
      double sdfBounds[6] = {conf.SDFKx.start, conf.SDFKx.end,
                             conf.SDFKy.start, conf.SDFKy.end,
                             conf.SDFE.start,  conf.SDFE.end};
      hsize_t boundsize[1] = {6};
      writeArray<1>("sdf_bounds", file, sdfBounds, boundsize);
    }
    if (conf.doDOS) {
      auto UH = eigsol.U.adjoint();
      auto dos = DOS(eigsol.D, UH, points, a, conf.SDFKx, conf.SDFKy, conf.SDFE,
                     conf.sharpening, conf.cutoff);
      writeArray<1>("dos", file, dos.data(), &conf.SDFE.n);
      double dosBounds[2] = {conf.SDFE.start, conf.SDFE.end};
      hsize_t boundsize[1] = {2};
      writeArray<1>("dos_bounds", file, dosBounds, boundsize);
    }
    if (conf.doPath) {
      std::cout << "Doing path\n";
      if (conf.dispE.has_value()) {
        auto kc = conf.kpath;
        auto ec = conf.dispE.value();
        auto UH = eigsol.U.adjoint();
        auto dis =
            disp(eigsol.D, UH, points, a, kc, ec, conf.sharpening, conf.cutoff);
        u32 nsamples = 0;
        for (const auto& rc : kc) {
          nsamples += rc.n;
        }
        hsize_t sizes[2] = {nsamples, ec.n};
        writeArray<2>("disp", file, dis.data(), sizes);
        std::vector<f64> dispBounds;
        for (const auto& rc : kc) {
          dispBounds.insert(dispBounds.end(),
                            {rc.start[0], rc.start[1], rc.end[0], rc.end[1]});
        }
        dispBounds.insert(dispBounds.end(), {ec.start, ec.end});
        hsize_t boundsizes[1] = {dispBounds.size()};
        writeArray<1>("disp_bounds", file, dispBounds.data(), boundsizes);
      } else {
        std::cout << "Need to supply a non-zero number of energy samples\n";
      }
    }
    H5Fclose(file);
  }
  if (result["p"].count()) {
    std::string fname = result["p"].as<std::string>();
    PerConf conf;
    if (auto opt = tomlToPerConf(fname); opt.has_value()) {
      conf = opt.value();
    } else {
      std::cerr << "Couldn't parse file: " << fname << std::endl;
      return 1;
    }

    if (conf.do2D) {
      RangeConf<f64> kxrange = conf.kxrange.value();
      RangeConf<f64> kyrange = conf.kxrange.value();
      MatrixXcd hamiltonian =
          MatrixXcd::Zero(conf.points.size(), conf.points.size());
      std::vector<f64> energies(kxrange.n * kyrange.n * conf.points.size());
      auto energy_view = std::mdspan(energies.data(), kyrange.n, kxrange.n,
                                     conf.points.size());
#pragma omp parallel for
      for (u32 j = 0; j < kyrange.n; j++) {
        f64 ky = kyrange.ith(j);
        for (u32 i = 0; i < kxrange.n; i++) {
          f64 kx = kxrange.ith(i);
          update_hamiltonian(hamiltonian, conf.nbs, {kx, ky}, expCoupling,
                             i | j);
          EigenSolution eigsol = hermitianEigenSolver(hamiltonian);
          for (u32 k = 0; k < conf.points.size(); k++) {
            energy_view[j, i, k] = eigsol.D(k);
          }
        }
      }
#pragma omp barrier
      hid_t file = H5Fcreate(conf.fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                             H5P_DEFAULT);
      if (file == H5I_INVALID_HID) {
        std::cerr << "Failed to create file " << conf.fname << std::endl;
        return 1;
      }
      hsize_t sizes[3] = {kyrange.n, kxrange.n, conf.points.size()};
      writeArray<3>("energies", file, energies.data(), sizes);
      hsize_t boundsizes[1] = {4};
      writeArray<1>(
          "energies_bounds", file,
          (f64[4]){kxrange.start, kxrange.end, kyrange.start, kyrange.end},
          boundsizes);
      H5Fclose(file);
    }
    if (conf.doPath) {
      MatrixXcd H = MatrixXcd::Zero(conf.points.size(), conf.points.size());
      u32 npoints = 0;
      for (const auto& r : conf.kpath) {
        npoints += r.n;
      }
      std::vector<f64> energies;
      energies.reserve(npoints * conf.points.size());
      std::cout << "Number of path segments: " << conf.kpath.size() << '\n';
      bool reuse = false;
      for (const auto& r : conf.kpath) {
        for (u32 i = 0; i < r.n; i++) {
          Vector2d k = r.ith(i);
          update_hamiltonian(H, conf.nbs, k, expCoupling, reuse);
          EigenSolution eigsol = hermitianEigenSolver(H);
          for (const auto& e : eigsol.D) {
            energies.push_back(e);
          }
          reuse = true;
        }
      }
      hid_t file = H5Fcreate(conf.fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                             H5P_DEFAULT);
      if (file == H5I_INVALID_HID) {
        std::cerr << "Failed to create file " << "testing.h5" << std::endl;
        return 1;
      }
      std::cout << "Energy size should be: " << energies.size() << '\n';
      hsize_t sizes[2] = {npoints, conf.points.size()};
      writeArray<2>("disp", file, energies.data(), sizes);
      hsize_t boundsizes[1] = {conf.kpath.size() * 2 * 2};
      std::vector<Vector2d> bounds;
      bounds.reserve(conf.kpath.size());
      for (const auto& r : conf.kpath) {
        bounds.push_back(r.start);
        bounds.push_back(r.end);
      }
      writeArray<1>("disp_bounds", file, bounds.data(), boundsizes);
      H5Fclose(file);
    }
  }
  if (result["t"].count()) {
    Manager m(1024);
    std::string fname = result["t"].as<std::string>();
    SdfConf conf;
    if (auto opt = tomlToSDFConf(fname); opt.has_value()) {
      conf = opt.value();
    } else {
      return 1;
    }
    if (!(conf.doPath || conf.doEsection || conf.doDOS || conf.doFullSDF)) {
      std::cout << "No tasks selected.\n";
      return 0;
    }

    std::vector<Point> points;
    if (conf.pointPath == "square") {
      points.resize(70 * 70);
      for (u32 i = 0; i < 70; i++) {
        for (u32 j = 0; j < 70; j++) {
          points[i * 70 + j] = {(double)i, (double)j, i * 70 + j};
        }
      }
    } else
      points = readPoints(conf.pointPath);

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

        H5Dread(did, H5T_NATIVE_DOUBLE, H5S_ALL, dspace, H5P_DEFAULT,
                eigsol.D.data());
        H5Sclose(dspace);
        H5Dclose(did);
        auto uid = H5Dopen2(fid, "U", H5P_DEFAULT);
        auto uspace = H5Dget_space(uid);
        H5Sselect_all(uspace);
        H5Dread(uid, H5T_NATIVE_DOUBLE, H5S_ALL, uspace, H5P_DEFAULT,
                eigsol.U.data());
        H5Sclose(uspace);
        H5Dclose(uid);
        H5Fclose(fid);
        H5Pclose(flist_id);
        useSavedSucceeded = true;
      }
    }
    if (!useSavedSucceeded) {
      MatrixXd H =
          pointsToFiniteHamiltonian(points, kdtree, conf.searchRadius * a);
      if (conf.saveHamiltonian.has_value()) {
        saveEigen(conf.saveHamiltonian.value(), H);
      }
      eigsol = hermitianEigenSolver(H);
    }
    if (conf.saveDiagonalisation.has_value() && !useSavedSucceeded) {
      hid_t file = H5Fcreate(conf.saveDiagonalisation.value().c_str(),
                             H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
      hsize_t sizes[2] = {static_cast<hsize_t>(eigsol.D.size()),
                          static_cast<hsize_t>(2 * eigsol.D.size())};
      writeArray<1>("D", file, eigsol.D.data(), sizes);
      writeArray<2>("U", file, eigsol.U.data(), sizes);
      H5Fclose(file);
    }
    hid_t file = H5Fcreate(conf.H5Filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT,
                           H5P_DEFAULT);
    if (file == H5I_INVALID_HID) {
      std::cerr << "Failed to create file " << conf.H5Filename << std::endl;
      return 1;
    }
    if (conf.doEsection) {
      const auto UH = eigsol.U.adjoint();
      auto section = GPUEsection(m, eigsol.D, UH, points, a, conf.sectionKx,
                                 conf.sectionKy, conf.fixed_e, conf.sharpening,
                                 conf.cutoff);
      hsize_t sizes[2] = {conf.sectionKx.n, conf.sectionKy.n};
      writeSingleArray<2>("section", file, section.data(), sizes);
      double sdfBounds[5] = {conf.sectionKx.start, conf.sectionKx.end,
                             conf.sectionKy.start, conf.sectionKy.end,
                             conf.fixed_e};
      hsize_t boundsize[1] = {5};
      writeArray<1>("section_bounds", file, sdfBounds, boundsize);
    }
    H5Fclose(file);
    // auto alg = m.makeAlgorithm("Shaders/test.spv", {}, {&gpua, &gpub}, none);
    // auto buffer = m.beginRecord();
    // appendOp(buffer, alg, 1024, 1, 1);
    // buffer.end();
    // m.execute(buffer);
    // m.writeFromBuffer(gpub, b);
    // for (const auto& x : b) {
    //   std::cout << x << '\n';
    // }
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
