#include "hermEigen.h"

EigenSolution hermitianEigenSolver(MatrixXcd H) {
  int n = H.rows();
  const char Nchar = 'V';
  const char Uchar = 'U';
  const char Achar = 'A';
  VectorXd eigReal(n);
  const int nb = 2;
  int lwork = (nb + 1) * n * n;
  double* work = new double[lwork];
  int lrwork = 24 * n;
  double* rwork = new double[lrwork];
  int liwork = 10 * n;
  int* iwork = new int[liwork];
  int info;
  double vl, vu;
  int il, iu;
  const char smin[13] = "Safe minimum";
  double abstol = dlamch_(smin);
  int M;
  MatrixXcd Z(n, n);
  int ldz = n;
  int* isuppz = new int[2 * n];
  zheevr_(&Nchar, &Achar, &Uchar, &n, (double*)H.data(), &n, &vl, &vu, &il, &iu,
          &abstol, &M, eigReal.data(), (double*)Z.data(), &ldz, isuppz, work,
          &lwork, rwork, &lrwork, iwork, &liwork, &info);
  delete[] isuppz;
  delete[] iwork;
  delete[] rwork;
  delete[] work;
  return {eigReal, Z};
}
