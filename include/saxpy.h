#pragma once

#include "blas_internal.h"
#include "swblas.h"

void cblas_saxpy_full_naive(size_t n, float alpha, float* __restrict x,
							size_t incx, float* __restrict y, size_t incy);

void cblas_saxpy_unroll2(size_t n, float alpha, float* __restrict x,
						 size_t incx, float* __restrict y, size_t incy);

void cblas_saxpy_unroll4(size_t n, float alpha, float* __restrict x,
						 size_t incx, float* __restrict y, size_t incy);

void cblas_saxpy_unroll8(size_t n, float alpha, float* __restrict x,
						 size_t incx, float* __restrict y, size_t incy);

void cblas_saxpy_unroll16(size_t n, float alpha, float* __restrict x,
						  size_t incx, float* __restrict y, size_t incy);

void cblas_saxpy_assume_aligned(size_t n, float alpha, float* __restrict x,
								size_t incx, float* __restrict y, size_t incy);

void cblas_saxpy_sse(size_t n, float alpha, float* __restrict x,
					 size_t /*incx = 1*/, float* __restrict y,
					 size_t /*incy = 1 */);

void cblas_saxpy_sse_unroll2(size_t n, float alpha, float* __restrict x,
							 size_t /*incx = 1*/, float* __restrict y,
							 size_t /*incy = 1 */);

void cblas_saxpy_sse_unroll4(size_t n, float alpha, float* __restrict x,
							 size_t /*incx = 1*/, float* __restrict y,
							 size_t /*incy = 1 */);
