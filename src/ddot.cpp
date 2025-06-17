#include <cstdint>

#include "../include/ddot.h"
#include "swblas.h"

double cblas_ddot_full_naive(size_t n, double* __restrict x, size_t incx,
							 double* __restrict y, size_t incy) {
	double res = 0.0;
	size_t i = 0;
	for (; i < n; i++) {
		res += x[incx * i] * y[incy * i];
	}
	return res;
}

double cblas_ddot_unroll2(size_t n, double* __restrict x, size_t incx,
						  double* __restrict y, size_t incy) {
	double res1 = 0.0, res2 = 0.0;
	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		res1 += x[incx * i] * y[incy * i];
		res2 += x[incx * (i + 1)] * y[incy * (i + 1)];
	}
	res1 += res2;
	for (; i < n; i++) {
		res1 += x[incx * i] * y[incy * i];
	}
	return res1;
}

double cblas_ddot_unroll4(size_t n, double* __restrict x, size_t incx,
						  double* __restrict y, size_t incy) {
	double res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		res1 += x[incx * i] * y[incy * i];
		res2 += x[incx * (i + 1)] * y[incy * (i + 1)];
		res3 += x[incx * (i + 2)] * y[incy * (i + 2)];
		res4 += x[incx * (i + 3)] * y[incy * (i + 3)];
	}
	res1 += res2 + res3 + res4;
	for (; i < n; i++) {
		res1 += x[incx * i] * y[incy * i];
	}
	return res1;
}

double cblas_ddot_unroll2_parith(size_t n, double* __restrict x, size_t incx,
								 double* __restrict y, size_t incy) {
	double res = 0.0;
	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		res += x[0] * y[0];
		res += x[incx] * y[incy];
		x += 2 * incx;
		y += 2 * incy;
	}
	for (; i < n; i++) {
		res += x[incx * i] * y[incy * i];
	}
	return res;
}

double cblas_ddot_unroll4_parith(size_t n, double* __restrict x, size_t incx,
								 double* __restrict y, size_t incy) {
	double res = 0.0;
	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		res += x[0] * y[0];
		res += x[incx] * y[incy];
		res += x[incx * 2] * y[incy * 2];
		res += x[incx * 3] * y[incy * 3];

		x += 4 * incx;
		y += 4 * incy;
	}
	for (; i < n; i++) {
		res += x[incx * i] * y[incy * i];
	}
	return res;
}

double cblas_ddot_assume_aligned(size_t n, double* __restrict x, size_t incx,
								 double* __restrict y, size_t incy) {
	double res = 0.0;
	size_t i = 0;

	x = (double*)__builtin_assume_aligned(x, 16);
	y = (double*)__builtin_assume_aligned(y, 16);
	for (; i < n; i++) {
		res += x[incx * i] * y[incy * i];
	}
	return res;
}

double cblas_ddot_assume_aligned_unroll2(size_t n, double* __restrict x,
										 size_t incx, double* __restrict y,
										 size_t incy) {
	double res = 0.0;
	size_t i = 0;

	x = (double*)__builtin_assume_aligned(x, 16);
	y = (double*)__builtin_assume_aligned(y, 16);
	for (; i <= n - 2; i += 2) {
		res += x[incx * i] * y[incy * i];
		res += x[incx * (i + 1)] * y[incy * (i + 1)];
	}
	for (; i < n; i++) {
		res += x[incx * i] * y[incy * i];
	}
	return res;
}

double cblas_ddot_sse(size_t n, double* __restrict x, size_t /*incx = 1*/,
					  double* __restrict y, size_t /*incy = 1*/) {
	__m128d res = _mm_setzero_pd();
	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		__m128d xx = _mm_load_pd(x + i);
		__m128d yy = _mm_load_pd(y + i);
		__m128d xxyy = _mm_mul_pd(xx, yy);
		res = _mm_add_pd(xxyy, res);
	}
	__m128 tmp = _mm_add_sd(res, _mm_unpackhi_pd(res, res));
	double rs;
	_mm_store_sd(&rs, tmp);
	for (; i < n; i++) {
		rs += x[i] * y[i];
	}
	return rs;
}

double cblas_ddot_sse_unroll2(size_t n, double* __restrict x,
							  size_t /*incx = 1*/, double* __restrict y,
							  size_t /*incy = 1*/) {
	__m128d r1 = _mm_setzero_pd();
	__m128d r2 = _mm_setzero_pd();
	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		__m128d xx1 = _mm_load_pd(x + i);
		__m128d yy1 = _mm_load_pd(y + i);
		__m128d xxyy1 = _mm_mul_pd(xx1, yy1);
		r1 = _mm_add_pd(xxyy1, r1);

		__m128d xx2 = _mm_load_pd(x + i + 2);
		__m128d yy2 = _mm_load_pd(y + i + 2);
		__m128d xxyy2 = _mm_mul_pd(xx2, yy2);
		r2 = _mm_add_pd(xxyy2, r2);
	}
	__m128 tmp1 = _mm_add_sd(r1, _mm_unpackhi_pd(r1, r1));
	__m128 tmp2 = _mm_add_sd(r2, _mm_unpackhi_pd(r2, r2));
	double rs1, rs2;
	_mm_store_sd(&rs1, tmp1);
	_mm_store_sd(&rs2, tmp2);
	rs1 += rs2;
	for (; i < n; i++) {
		rs1 += x[i] * y[i];
	}
	return rs1;
}

double cblas_ddot_sse_unroll4(size_t n, double* __restrict x,
							  size_t /*incx = 1*/, double* __restrict y,
							  size_t /*incy = 1*/) {
	__m128d r1 = _mm_setzero_pd();
	__m128d r2 = _mm_setzero_pd();
	__m128d r3 = _mm_setzero_pd();
	__m128d r4 = _mm_setzero_pd();
	double rs1, rs2, rs3, rs4;
	size_t i = 0;
	for (; i <= n - 8; i += 8) {
		r1 = _mm_add_pd(r1, _mm_mul_pd(_mm_load_pd(x + i), _mm_load_pd(y + i)));
		r2 = _mm_add_pd(
			r2, _mm_mul_pd(_mm_load_pd(x + i + 2), _mm_load_pd(y + i + 2)));
		r3 = _mm_add_pd(
			r3, _mm_mul_pd(_mm_load_pd(x + i + 4), _mm_load_pd(y + i + 4)));
		r4 = _mm_add_pd(
			r4, _mm_mul_pd(_mm_load_pd(x + i + 6), _mm_load_pd(y + i + 6)));
	}
	__m128 tmp1 = _mm_add_sd(r1, _mm_unpackhi_pd(r1, r1));
	__m128 tmp2 = _mm_add_sd(r2, _mm_unpackhi_pd(r2, r2));
	__m128 tmp3 = _mm_add_sd(r3, _mm_unpackhi_pd(r3, r3));
	__m128 tmp4 = _mm_add_sd(r4, _mm_unpackhi_pd(r4, r4));
	_mm_store_sd(&rs1, tmp1);
	_mm_store_sd(&rs2, tmp2);
	_mm_store_sd(&rs3, tmp3);
	_mm_store_sd(&rs4, tmp4);
	rs1 += rs2;
	rs1 += rs3;
	rs1 += rs4;
	for (; i < n; i++) {
		rs1 += x[i] * y[i];
	}
	return rs1;
}

double cblas_ddot_copy_and_sse(size_t n, double* __restrict x, size_t incx,
							   double* __restrict y, size_t incy) {
	double *xx = x, *yy = y;
	alignas(16) double y_copy[n];
	alignas(16) double x_copy[n];
	if (incx != 1 || (uint64_t)x % 16) {
		for (size_t i = 0; i < n; i++) x_copy[i] = x[incx * i];
		xx = x_copy;
	}
	if (incy != 1 || (uint64_t)y % 16) {
		for (size_t i = 0; i < n; i++) y_copy[i] = y[incy * i];
		yy = y_copy;
	}
	return cblas_ddot_sse_unroll4(n, xx, 1, yy, 1);
}

namespace swblas {

double cblas_ddot(size_t n, double* x, size_t incx, double* y, size_t incy) {
	if (n == 0) return 0.0;

	bool unit_strided = (incx | incy) == 1;
	if (unit_strided && (uintptr_t)x % 16 == 0 && (uintptr_t)y % 16 == 0) {
		return cblas_ddot_sse_unroll4(n, x, 1, y, 1);
	} else {
		return cblas_ddot_unroll4(n, x, incx, y, incy);
	}
}

}  // namespace swblas
