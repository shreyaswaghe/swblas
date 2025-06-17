#pragma once

#include "blas_internal.h"
#include "swblas.h"

void cblas_daxpy_full_naive(size_t n, double alpha, double* __restrict x,
							size_t incx, double* __restrict y, size_t incy);

void cblas_daxpy_unroll2(size_t n, double alpha, double* __restrict x,
						 size_t incx, double* __restrict y, size_t incy);

void cblas_daxpy_unroll4(size_t n, double alpha, double* __restrict x,
						 size_t incx, double* __restrict y, size_t incy);

void cblas_daxpy_unroll8(size_t n, double alpha, double* __restrict x,
						 size_t incx, double* __restrict y, size_t incy);

void cblas_daxpy_unroll2_uncoupled(size_t n, double alpha, double* __restrict x,
								   size_t incx, double* __restrict y,
								   size_t incy);

void cblas_daxpy_unroll4_uncoupled(size_t n, double alpha, double* __restrict x,
								   size_t incx, double* __restrict y,
								   size_t incy);

void cblas_daxpy_unroll8_uncoupled(size_t n, double alpha, double* __restrict x,
								   size_t incx, double* __restrict y,
								   size_t incy);

void cblas_daxpy_assume_aligned(size_t n, double alpha, double* __restrict x,
								size_t incx, double* __restrict y, size_t incy);

void cblas_daxpy_sse(size_t n, double alpha, double* __restrict x,
					 size_t /*incx = 1*/, double* __restrict y,
					 size_t /*incy = 1 */);

void cblas_daxpy_sse_unroll2(size_t n, double alpha, double* __restrict x,
							 size_t /*incx = 1*/, double* __restrict y,
							 size_t /*incy = 1 */);

void cblas_daxpy_sse_unroll4(size_t n, double alpha, double* __restrict x,
							 size_t /*incx = 1*/, double* __restrict y,
							 size_t /*incy = 1 */);
