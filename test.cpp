#include <cblas.h>
#include <complex>
#include <fstream>
#include <iostream>
#include <mdspan>
#include <vector>

using namespace std;

// dgeev_ is a symbol in the LAPACK library files
extern "C" {
extern int zheevr_(char* JOBZ, char* RANGE, char* UPLO, int* N, double* A,
                   int* LDA, double* VL, double* VU, int* IL, int* IU,
                   double* ABSTOL, int* M, double* W, double* Z, int* LDZ,
                   int* ISUPPZ, double* WORK, int* LWORK, double* RWORK,
                   int* LRWORK, int* IWORK, int* LIWORK, int* INFO);
extern double dlamch_(char* cmach);
}

int main(int argc, char** argv) {

  // std::vector<std::complex<double>> data(n * m);
  // auto dataview = std::mdspan(data.data(), n, m);

  // read in a text file that contains a real matrix stored in column major
  // format but read it into row major format
  // ifstream fin(argv[1]);
  // if (!fin.is_open()) {
  //   cout << "Failed to open " << argv[1] << endl;
  //   return -1;
  // }
  // fin >> n >> m; // n is the number of rows, m the number of columns
  int n = 3;
  auto data = new double[2 * n * n];
  for (uint i = 0; i < 2 * n * n; i++) {
    data[i] = 0;
  }
  data[2] = 1.0;
  data[6] = 1.0;
  data[10] = 1.0;
  data[14] = 1.0;
  // for (int i = 0; i < n; i++) {
  //   data[2 * (i * n + i)] = i;
  //   data[2 * (i * n + i) + 1] = 0;
  //   for (int j = i + 1; j < n; j++) {
  //     data[2 * (i * n + j)] = i;
  //     data[2 * (i * n + j) + 1] = j;
  //     data[2 * (j * n + i)] = i;
  //     data[2 * (j * n + i) + 1] = -j;
  //   }
  // }
  // cout << "--- Matrix before: ---" << endl;
  // for (int i = 0; i < n; i++) {
  //   for (int j = 0; j < n; j++) {
  //     cout << "(" << data[2 * (j * n + i)] << ", " << data[2 * (j * n + i) +
  //     1]
  //          << ") ";
  //   }
  //   std::cout << '\n';
  // }
  // cout << endl;

  // if (fin.fail() || fin.eof()) {
  //   cout << "Error while reading " << argv[1] << endl;
  //   return -1;
  // }
  // fin.close();

  // // check that matrix is square
  // if (n != m) {
  //   cout << "Matrix is not square" << endl;
  //   return -1;
  // }

  // allocate data
  // for (uint i = 0; i < n; i++) {
  //   for (uint j = i; j < m; j++) {
  //     dataview[i, j] = {(double)i, (double)j};
  //     dataview[j, i] = {(double)i, -((double)j)};
  //   }
  // }
  char Nchar = 'V';
  char Uchar = 'U';
  char Achar = 'A';
  double* eigReal = new double[n];
  int nb = 2;
  int lwork = (nb + 1) * n * n;
  double* work = new double[lwork];
  int lrwork = 24 * n;
  double* rwork = new double[lrwork];
  int liwork = 10 * n;
  int* iwork = new int[liwork];
  int info;
  double vl, vu;
  int il, iu;
  char smin[13] = "Safe minimum";
  double abstol = 0; // dlamch_(smin);
  int M;
  double* Z = new double[2 * n * n];
  int ldz = n;
  int* isuppz = new int[2 * n];

  // calculate eigenvalues using the DGEEV subroutine
  zheevr_(&Nchar, &Achar, &Uchar, &n, data, &n, &vl, &vu, &il, &iu, &abstol, &M,
          eigReal, Z, &ldz, isuppz, work, &lwork, rwork, &lrwork, iwork,
          &liwork, &info);

  // check for errors
  if (info != 0) {
    cout << "Error: dgeev returned error code " << info << endl;
    return -1;
  }

  // output eigenvalues to stdout
  cout << "--- Eigenvalues ---" << endl;
  for (int i = 0; i < n; i++) {
    cout << eigReal[i] << "\n";
  }
  cout << endl;

  cout << "--- Eigenvectors ---" << endl;
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      cout << "(" << Z[2 * (j * n + i)] << ", " << Z[2 * (j * n + i) + 1]
           << ") ";
    }
    std::cout << '\n';
  }
  cout << endl;

  // deallocate
  delete[] data;
  delete[] Z;
  delete[] isuppz;
  delete[] eigReal;
  delete[] work;
  delete[] rwork;
  delete[] iwork;

  return 0;
}
