#pragma once
#include "Eigen/Dense"

using Eigen::MatrixXcd, Eigen::VectorXd;

struct EigenSolution {
  VectorXd D;
  MatrixXcd U;
};

extern "C" {
extern int zheevr_(const char* JOBZ, const char* RANGE, const char* UPLO,
                   int* N, double* A, int* LDA, double* VL, double* VU, int* IL,
                   int* IU, double* ABSTOL, int* M, double* W, double* Z,
                   int* LDZ, int* ISUPPZ, double* WORK, int* LWORK,
                   double* RWORK, int* LRWORK, int* IWORK, int* LIWORK,
                   int* INFO);
extern double dlamch_(const char* cmach);
}

EigenSolution hermitianEigenSolver(MatrixXcd H);
