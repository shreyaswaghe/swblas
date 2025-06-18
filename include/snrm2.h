#pragma once
#include "blas_internal.h"
#include "swblas.h"

float cblas_snrm2_naive(size_t n, float* x, size_t incx);

float cblas_snrm2_unroll2(size_t n, float* x, size_t incx);

float cblas_snrm2_unroll2_parith(size_t n, float* x, size_t incx);

float cblas_snrm2_unroll4(size_t n, float* x, size_t incx);

float cblas_snrm2_unroll8(size_t n, float* x, size_t incx);

float cblas_snrm2_unroll16(size_t n, float* x, size_t incx);

float cblas_snrm2_sse(size_t n, float* x, size_t incx);

float cblas_snrm2_sse_unroll2(size_t n, float* x, size_t incx);

float cblas_snrm2_sse_unroll4(size_t n, float* x, size_t incx);

float cblas_snrm2_sse_unroll8(size_t n, float* x, size_t incx);
