#pragma once
#include "blas_internal.h"
#include "swblas.h"

double cblas_ddot_full_naive(size_t n, double* __restrict x, size_t incx,
							 double* __restrict y, size_t incy);

double cblas_ddot_unroll2(size_t n, double* __restrict x, size_t incx,
						  double* __restrict y, size_t incy);

double cblas_ddot_unroll4(size_t n, double* __restrict x, size_t incx,
						  double* __restrict y, size_t incy);

double cblas_ddot_unroll2_parith(size_t n, double* __restrict x, size_t incx,
								 double* __restrict y, size_t incy);

double cblas_ddot_unroll4_parith(size_t n, double* __restrict x, size_t incx,
								 double* __restrict y, size_t incy);

double cblas_ddot_assume_aligned(size_t n, double* __restrict x, size_t incx,
								 double* __restrict y, size_t incy);

double cblas_ddot_assume_aligned_unroll2(size_t n, double* __restrict x,
										 size_t incx, double* __restrict y,
										 size_t incy);

double cblas_ddot_sse(size_t n, double* __restrict x, size_t /*incx = 1*/,
					  double* __restrict y, size_t /*incy = 1*/);

double cblas_ddot_sse_unroll2(size_t n, double* __restrict x,
							  size_t /*incx = 1*/, double* __restrict y,
							  size_t /*incy = 1*/);

double cblas_ddot_sse_unroll4(size_t n, double* __restrict x,
							  size_t /*incx = 1*/, double* __restrict y,
							  size_t /*incy = 1*/);

double cblas_ddot_copy_and_sse(size_t n, double* __restrict x, size_t incx,
							   double* __restrict y, size_t incy);
