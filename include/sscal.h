#pragma once

#include "blas_internal.h"
#include "swblas.h"

void cblas_sscal_full_naive(size_t n, float alpha, float* __restrict x,
							size_t incx);

void cblas_sscal_unroll2(size_t n, float alpha, float* __restrict x,
						 size_t incx);

void cblas_sscal_unroll4(size_t n, float alpha, float* __restrict x,
						 size_t incx);

void cblas_sscal_unroll8(size_t n, float alpha, float* __restrict x,
						 size_t incx);

void cblas_sscal_assume_aligned(size_t n, float alpha, float* __restrict x,
								size_t incx);

void cblas_sscal_sse(size_t n, float alpha, float* __restrict x,
					 size_t /*incx = 1*/);

void cblas_sscal_sse_unroll2(size_t n, float alpha, float* __restrict x,
							 size_t /*incx = 1*/);

void cblas_sscal_sse_unroll4(size_t n, float alpha, float* __restrict x,
							 size_t /*incx = 1*/);
