#pragma once
#include "blas_internal.h"
#include "swblas.h"

float cblas_sdot_full_naive(size_t n, float* __restrict x, size_t incx,
							float* __restrict y, size_t incy);

float cblas_sdot_unroll2(size_t n, float* __restrict x, size_t incx,
						 float* __restrict y, size_t incy);

float cblas_sdot_unroll4(size_t n, float* __restrict x, size_t incx,
						 float* __restrict y, size_t incy);

float cblas_sdot_unroll8(size_t n, float* __restrict x, size_t incx,
						 float* __restrict y, size_t incy);

float cblas_sdot_assume_aligned(size_t n, float* __restrict x, size_t incx,
								float* __restrict y, size_t incy);

float cblas_sdot_assume_aligned_unroll2(size_t n, float* __restrict x,
										size_t incx, float* __restrict y,
										size_t incy);

float cblas_sdot_sse(size_t n, float* __restrict x, size_t /*incx = 1*/,
					 float* __restrict y, size_t /*incy = 1*/);

float cblas_sdot_sse_unroll2(size_t n, float* __restrict x, size_t /*incx = 1*/,
							 float* __restrict y, size_t /*incy = 1*/);

float cblas_sdot_sse_unroll4(size_t n, float* __restrict x, size_t /*incx = 1*/,
							 float* __restrict y, size_t /*incy = 1*/);

float cblas_sdot_sse_unroll8(size_t n, float* __restrict x, size_t /*incx = 1*/,
							 float* __restrict y, size_t /*incy = 1*/);
