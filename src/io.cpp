#include "io.h"
#include <H5Fpublic.h>
#include <iostream>

std::optional<EigenSolution> loadDiag(std::string fname) {
  auto flist_id = H5Pcreate(H5P_FILE_ACCESS);
  std::cout << "Trying to open file" << fname << '\n';
  hid_t fid = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, flist_id);
  if (fid == H5I_INVALID_HID)
    return {};
  std::cout << "Trying to get dataset" << fname << '\n';
  auto did = H5Dopen2(fid, "D", H5P_DEFAULT);
  if (did == H5I_INVALID_HID)
    return {};
  auto dspace = H5Dget_space(did);
  if (dspace == H5I_INVALID_HID)
    return {};
  hsize_t dims[1];
  H5Sget_simple_extent_dims(dspace, dims, nullptr);
  EigenSolution eigsol{};
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
  return eigsol;
}

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
