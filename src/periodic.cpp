#include "periodic.h"
#include "io.h"
#include <iostream>

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

#undef SET_STRUCT_FIELD

int doPeriodicModel(const PerConf& conf) {

  if (conf.do2D) {
    RangeConf<f64> kxrange = conf.kxrange.value();
    RangeConf<f64> kyrange = conf.kxrange.value();
    MatrixXcd hamiltonian =
        MatrixXcd::Zero(conf.points.size(), conf.points.size());
    std::vector<f64> energies(kxrange.n * kyrange.n * conf.points.size());
    auto energy_view =
        std::mdspan(energies.data(), kyrange.n, kxrange.n, conf.points.size());
#pragma omp parallel for
    for (u32 j = 0; j < kyrange.n; j++) {
      f64 ky = kyrange.ith(j);
      for (u32 i = 0; i < kxrange.n; i++) {
        f64 kx = kxrange.ith(i);
        update_hamiltonian(hamiltonian, conf.nbs, {kx, ky}, expCoupling, i | j);
        EigenSolution eigsol = hermitianEigenSolver(hamiltonian);
        for (u32 k = 0; k < conf.points.size(); k++) {
          energy_view[j, i, k] = eigsol.D(k);
        }
      }
    }
#pragma omp barrier
    H5File file(conf.fname.c_str());
    if (file == H5I_INVALID_HID) {
      std::cerr << "Failed to create file " << conf.fname << std::endl;
      return 1;
    }
    writeArray<3>("energies", *file, H5T_NATIVE_DOUBLE_g, energies.data(),
                  {kyrange.n, kxrange.n, conf.points.size()});
    writeArray<1>(
        "energies_bounds", *file, H5T_NATIVE_DOUBLE_g,
        (f64[4]){kxrange.start, kxrange.end, kyrange.start, kyrange.end}, {4});
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
    H5File file(conf.fname.c_str());
    if (file == H5I_INVALID_HID) {
      std::cerr << "Failed to create file " << "testing.h5" << std::endl;
      return 1;
    }
    std::cout << "Energy size should be: " << energies.size() << '\n';
    writeArray<2>("disp", file, H5T_NATIVE_DOUBLE_g, energies.data(),
                  {npoints, conf.points.size()});
    std::vector<Vector2d> bounds;
    bounds.reserve(conf.kpath.size());
    for (const auto& r : conf.kpath) {
      bounds.push_back(r.start);
      bounds.push_back(r.end);
    }
    writeArray<1>("disp_bounds", file, H5T_NATIVE_DOUBLE_g, bounds.data(),
                  {conf.kpath.size() * 2 * 2});
  }
  return 0;
}
