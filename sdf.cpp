struct RangeConf {
  double min;
  double max;
  u64 n;
};

void SDF(const VectorXd& D, const MatrixXcd& UH,
         const std::vector<Point>& points, RangeConf kc, RangeConf ec) {
  const double dkx = (kc.max - kc.min) / (double)kc.n;
  const double de = (ec.max - ec.min) / (double)ec.n;
  const u32 its = kc.n / 10;
  std::vector<double> densities(kc.n * ec.n, 0);
  auto density_view = std::mdspan(densities.data(), ec.n, kc.n);
  for (size_t i = 0; i < kc.n; i++) {
    const double kx = kc.min + i * dkx;
    const VectorXcd k_vec = [&]() {
      VectorXcd tmp = VectorXcd::Zero(D.size());
      std::transform(points.begin(), points.end(), tmp.begin(), [&](Point p) {
        return (1. / sqrt(points.size())) * std::exp(c64{0, kx * p[0]});
      });
      tmp = UH * k_vec;
      return tmp;
    }();
    for (size_t j = 0; j < ec.n; j++) {
      const double e = ec.min + j * de;
      Eigen::VectorX<bool> del = Eigen::abs(D.array() - e).array() < 0.01;
      for (u32 k = 0; k < del.size(); k++) {
        if (del[k]) {
          /*density_view[j, i] += k_vec
                                    .dot((U(Eigen::indexing::all, k) *
                                          UH(k, Eigen::indexing::all)) *
                                         k_vec)
                                    .real();*/
          density_view[j, i] += std::norm(k_vec(k));
        }
      }
    }
    std::cout << '\n';
    if (i % its == 0) {
      std::cout << "█|";
    }
  }
  std::cout << "█]";
}
