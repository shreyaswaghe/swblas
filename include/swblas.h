#ifndef _SWBLAS_H
#define _SWBLAS_H

#include <cstddef>

namespace swblas {

// axpy
void cblas_daxpy(size_t n, double alpha, double* x, size_t incx, double* y,
				 size_t incy);
void cblas_saxpy(size_t n, float alpha, float* x, size_t incx, float* y,
				 size_t incy);

// scal
void cblas_dscal(size_t n, double alpha, double* x, size_t incx);
void cblas_sscal(size_t n, float alpha, float* x, size_t incx);

// dot
double cblas_ddot(size_t n, double* x, size_t incx, double* y, size_t incy);
float cblas_sdot(size_t n, float* x, size_t incx, float* y, size_t incy);

// nrm2
double cblas_dnrm2(size_t n, double* x, size_t incx);
float cblas_snrm2(size_t n, float* x, size_t incx);

// asum
double cblas_dasum(size_t n, double* x, size_t incx);
float cblas_sasum(size_t n, float* x, size_t incx);

//
// LEVEL 2 ROUTINES
//

namespace impl {

void cblas_dgemvCN(size_t m, size_t n, double alpha, const double* a,
				   size_t lda, const double* x, size_t incx, double beta,
				   double* y, size_t incy);

void cblas_dgemvCT(size_t m, size_t n, double alpha, const double* a,
				   size_t lda, const double* x, size_t incx, double beta,
				   double* y, size_t incy);

}  // namespace impl

void cblas_dgemv(char fmt, char trans, size_t m, size_t n, double alpha,
				 const double* a, size_t lda, const double* x, size_t incx,
				 double beta, double* y, size_t incy);

}  // namespace swblas

#endif
