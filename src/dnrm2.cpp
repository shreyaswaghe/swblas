#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../include/dnrm2.h"
#include "sse2neon.h"
#include "swblas.h"

double cblas_dnrm2_naive(size_t n, double* x, size_t incx) {
	double res = 0.0;
	size_t i = 0;
	for (; i < n; i++) {
		double xi = x[i * incx];
		res += xi * xi;
	}
	return sqrt(res);
}

double cblas_dnrm2_unroll2(size_t n, double* x, size_t incx) {
	double res1 = 0.0, res2 = 0.0;
	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		double xi1 = x[i * incx];
		double xi2 = x[(i + 1) * incx];
		res1 += xi1 * xi1;
		res2 += xi2 * xi2;
	}
	res1 += res2;
	for (; i < n; i++) {
		double xi = x[i * incx];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

double cblas_dnrm2_unroll4(size_t n, double* x, size_t incx) {
	double res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		double xi1 = x[i * incx];
		double xi2 = x[(i + 1) * incx];
		double xi3 = x[(i + 2) * incx];
		double xi4 = x[(i + 3) * incx];
		res1 += xi1 * xi1;
		res2 += xi2 * xi2;
		res3 += xi3 * xi3;
		res4 += xi4 * xi4;
	}
	res1 += res2 + res3 + res4;
	for (; i < n; i++) {
		double xi = x[i * incx];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

double cblas_dnrm2_unroll8(size_t n, double* x, size_t incx) {
	double res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
	double res5 = 0.0, res6 = 0.0, res7 = 0.0, res8 = 0.0;
	size_t i = 0;
	for (; i <= n - 8; i += 8) {
		double xi1 = x[i * incx];
		double xi2 = x[(i + 1) * incx];
		double xi3 = x[(i + 2) * incx];
		double xi4 = x[(i + 3) * incx];
		double xi5 = x[(i + 4) * incx];
		double xi6 = x[(i + 5) * incx];
		double xi7 = x[(i + 6) * incx];
		double xi8 = x[(i + 7) * incx];
		res1 += xi1 * xi1;
		res2 += xi2 * xi2;
		res3 += xi3 * xi3;
		res4 += xi4 * xi4;
		res5 += xi5 * xi5;
		res6 += xi6 * xi6;
		res7 += xi7 * xi7;
		res8 += xi8 * xi8;
	}
	res1 += res2 + res3 + res4;
	res1 += res5 + res6 + res7 + res8;
	for (; i < n; i++) {
		double xi = x[i * incx];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

double cblas_dnrm2_sse(size_t n, double* x, size_t /*incx = 1*/) {
	__m128d res = _mm_setzero_pd();
	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		__m128d y = _mm_load_pd(x);
		y = _mm_mul_pd(y, y);
		res = _mm_add_pd(res, y);
		x += 2;
	}
	double res1;
	res = _mm_hadd_pd(res, res);
	_mm_store_sd(&res1, res);
	for (; i < n; i++) {
		double xi = x[i];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

double cblas_dnrm2_sse_unroll2(size_t n, double* x, size_t /*incx = 1*/) {
	__m128d r1 = _mm_setzero_pd();
	__m128d r2 = _mm_setzero_pd();
	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		__m128d y1 = _mm_load_pd(x);
		__m128d y2 = _mm_load_pd(x + 2);
		y1 = _mm_mul_pd(y1, y1);
		y2 = _mm_mul_pd(y2, y2);
		r1 = _mm_add_pd(r1, y1);
		r2 = _mm_add_pd(r2, y2);
		x += 4;
	}
	double res1;
	r1 = _mm_add_pd(r1, r2);
	r1 = _mm_hadd_pd(r1, r1);
	_mm_store_sd(&res1, r1);
	for (; i < n; i++) {
		double xi = x[i];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

double cblas_dnrm2_sse_unroll4(size_t n, double* x, size_t /*incx = 1*/) {
	__m128d r1 = _mm_setzero_pd();
	__m128d r2 = _mm_setzero_pd();
	__m128d r3 = _mm_setzero_pd();
	__m128d r4 = _mm_setzero_pd();
	size_t i = 0;
	for (; i <= n - 8; i += 8) {
		__m128d y1 = _mm_load_pd(x);
		__m128d y2 = _mm_load_pd(x + 2);
		__m128d y3 = _mm_load_pd(x + 4);
		__m128d y4 = _mm_load_pd(x + 6);
		y1 = _mm_mul_pd(y1, y1);
		y2 = _mm_mul_pd(y2, y2);
		y3 = _mm_mul_pd(y3, y3);
		y4 = _mm_mul_pd(y4, y4);
		r1 = _mm_add_pd(r1, y1);
		r2 = _mm_add_pd(r2, y2);
		r3 = _mm_add_pd(r3, y3);
		r4 = _mm_add_pd(r4, y4);
		x += 8;
	}
	double res1;
	r1 = _mm_add_pd(r1, r2);
	r1 = _mm_add_pd(r1, r3);
	r1 = _mm_add_pd(r1, r4);
	r1 = _mm_hadd_pd(r1, r1);
	_mm_store_sd(&res1, r1);
	for (; i < n; i++) {
		double xi = x[i];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

double cblas_dnrm2_sse_unroll8(size_t n, double* x, size_t incx) {
	__m128d r1 = _mm_setzero_pd();
	__m128d r2 = _mm_setzero_pd();
	__m128d r3 = _mm_setzero_pd();
	__m128d r4 = _mm_setzero_pd();
	__m128d r5 = _mm_setzero_pd();
	__m128d r6 = _mm_setzero_pd();
	__m128d r7 = _mm_setzero_pd();
	__m128d r8 = _mm_setzero_pd();
	size_t i = 0;
	for (; i <= n - 16; i += 16) {
		__m128d y1 = _mm_load_pd(x);
		__m128d y2 = _mm_load_pd(x + 2);
		__m128d y3 = _mm_load_pd(x + 4);
		__m128d y4 = _mm_load_pd(x + 6);
		__m128d y5 = _mm_load_pd(x + 8);
		__m128d y6 = _mm_load_pd(x + 10);
		__m128d y7 = _mm_load_pd(x + 12);
		__m128d y8 = _mm_load_pd(x + 14);
		y1 = _mm_mul_pd(y1, y1);
		y2 = _mm_mul_pd(y2, y2);
		y3 = _mm_mul_pd(y3, y3);
		y4 = _mm_mul_pd(y4, y4);
		y5 = _mm_mul_pd(y5, y5);
		y6 = _mm_mul_pd(y6, y6);
		y7 = _mm_mul_pd(y7, y7);
		y8 = _mm_mul_pd(y8, y8);
		r1 = _mm_add_pd(r1, y1);
		r2 = _mm_add_pd(r2, y2);
		r3 = _mm_add_pd(r3, y3);
		r4 = _mm_add_pd(r4, y4);
		r5 = _mm_add_pd(r5, y5);
		r6 = _mm_add_pd(r6, y6);
		r7 = _mm_add_pd(r7, y7);
		r8 = _mm_add_pd(r8, y8);
		x += 16;
	}
	double res1;
	r1 = _mm_add_pd(r1, r2);
	r1 = _mm_add_pd(r1, r3);
	r1 = _mm_add_pd(r1, r4);
	r1 = _mm_add_pd(r1, r5);
	r1 = _mm_add_pd(r1, r6);
	r1 = _mm_add_pd(r1, r7);
	r1 = _mm_add_pd(r1, r8);
	r1 = _mm_hadd_pd(r1, r1);
	_mm_store_sd(&res1, r1);
	for (; i < n; i++) {
		double xi = x[i];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

namespace swblas {

double cblas_dnrm2(size_t n, double* x, size_t incx) {
	constexpr size_t thresh = 4096;
	if (n == 0) return 0.0;
	bool unit_strided = incx == 1;
	if (unit_strided && (uintptr_t)x % 16 == 0) {
		if (n <= thresh)
			return cblas_dnrm2_sse_unroll2(n, x, 1);
		else
			return cblas_dnrm2_sse_unroll8(n, x, 1);
	} else
		return cblas_dnrm2_unroll8(n, x, incx);
}

}  // namespace swblas
