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

}  // namespace swblas

#endif
