#pragma once

#include "blas_internal.h"
#include "swblas.h"

void cblas_dscal_full_naive(size_t n, double alpha, double* __restrict x,
							size_t incx);

void cblas_dscal_unroll2(size_t n, double alpha, double* __restrict x,
						 size_t incx);

void cblas_dscal_unroll4(size_t n, double alpha, double* __restrict x,
						 size_t incx);

void cblas_dscal_unroll8(size_t n, double alpha, double* __restrict x,
						 size_t incx);

void cblas_dscal_assume_aligned(size_t n, double alpha, double* __restrict x,
								size_t incx);

void cblas_dscal_sse(size_t n, double alpha, double* __restrict x,
					 size_t /*incx = 1*/);

void cblas_dscal_sse_unroll2(size_t n, double alpha, double* __restrict x,
							 size_t /*incx = 1*/);

void cblas_dscal_sse_unroll4(size_t n, double alpha, double* __restrict x,
							 size_t /*incx = 1*/);
