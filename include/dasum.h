#pragma once
#include "blas_internal.h"
#include "swblas.h"

double cblas_dasum_naive(size_t n, double* x, size_t incx);

double cblas_dasum_unroll2(size_t n, double* x, size_t incx);

double cblas_dasum_unroll4(size_t n, double* x, size_t incx);

double cblas_dasum_unroll8(size_t n, double* x, size_t incx);

double cblas_dasum_sse(size_t n, double* x, size_t incx);

double cblas_dasum_sse_unroll2(size_t n, double* x, size_t incx);

double cblas_dasum_sse_unroll4(size_t n, double* x, size_t incx);

double cblas_dasum_sse_unroll8(size_t n, double* x, size_t incx);
