#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../include/snrm2.h"
#include "sse2neon.h"
#include "swblas.h"

float cblas_snrm2_naive(size_t n, float* x, size_t incx) {
	float res = 0.0;
	size_t i = 0;
	for (; i < n; i++) {
		float xi = x[i * incx];
		res += xi * xi;
	}
	return sqrt(res);
}

float cblas_snrm2_unroll2(size_t n, float* x, size_t incx) {
	float res1 = 0.0, res2 = 0.0;
	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		float xi1 = x[i * incx];
		float xi2 = x[(i + 1) * incx];
		res1 += xi1 * xi1;
		res2 += xi2 * xi2;
	}
	res1 += res2;
	for (; i < n; i++) {
		float xi = x[i * incx];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

float cblas_snrm2_unroll2_parith(size_t n, float* x, size_t incx) {
	float res1 = 0.0, res2 = 0.0;
	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		float xi1 = x[0];
		float xi2 = x[1 * incx];
		res1 += xi1 * xi1;
		res2 += xi2 * xi2;
		x += 2 * incx;
	}
	res1 += res2;
	for (; i < n; i++) {
		float xi = x[i * incx];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

float cblas_snrm2_unroll4(size_t n, float* x, size_t incx) {
	float res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		float xi1 = x[i * incx];
		float xi2 = x[(i + 1) * incx];
		float xi3 = x[(i + 2) * incx];
		float xi4 = x[(i + 3) * incx];
		res1 += xi1 * xi1;
		res2 += xi2 * xi2;
		res3 += xi3 * xi3;
		res4 += xi4 * xi4;
	}
	res1 += res2 + res3 + res4;
	for (; i < n; i++) {
		float xi = x[i * incx];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

float cblas_snrm2_unroll8(size_t n, float* x, size_t incx) {
	float res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
	float res5 = 0.0, res6 = 0.0, res7 = 0.0, res8 = 0.0;
	size_t i = 0;
	for (; i <= n - 8; i += 8) {
		float xi1 = x[0];
		float xi2 = x[1 * incx];
		float xi3 = x[2 * incx];
		float xi4 = x[3 * incx];
		float xi5 = x[4 * incx];
		float xi6 = x[5 * incx];
		float xi7 = x[6 * incx];
		float xi8 = x[7 * incx];
		res1 += xi1 * xi1;
		res2 += xi2 * xi2;
		res3 += xi3 * xi3;
		res4 += xi4 * xi4;
		res5 += xi5 * xi5;
		res6 += xi6 * xi6;
		res7 += xi7 * xi7;
		res8 += xi8 * xi8;
		x += 8 * incx;
	}
	res1 += res2 + res3 + res4;
	res1 += res5 + res6 + res7 + res8;
	for (; i < n; i++) {
		float xi = x[i * incx];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

float cblas_snrm2_unroll16(size_t n, float* x, size_t incx) {
	float res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
	float res5 = 0.0, res6 = 0.0, res7 = 0.0, res8 = 0.0;
	float res9 = 0.0, res10 = 0.0, res11 = 0.0, res12 = 0.0;
	float res13 = 0.0, res14 = 0.0, res15 = 0.0, res16 = 0.0;
	size_t i = 0;
	for (; i <= n - 16; i += 16) {
		float xi1 = x[0];
		float xi2 = x[1 * incx];
		float xi3 = x[2 * incx];
		float xi4 = x[3 * incx];
		float xi5 = x[4 * incx];
		float xi6 = x[5 * incx];
		float xi7 = x[6 * incx];
		float xi8 = x[7 * incx];
		float xi9 = x[8 * incx];
		float xi10 = x[9 * incx];
		float xi11 = x[10 * incx];
		float xi12 = x[11 * incx];
		float xi13 = x[12 * incx];
		float xi14 = x[13 * incx];
		float xi15 = x[14 * incx];
		float xi16 = x[15 * incx];

		res1 += xi1 * xi1;
		res2 += xi2 * xi2;
		res3 += xi3 * xi3;
		res4 += xi4 * xi4;
		res5 += xi5 * xi5;
		res6 += xi6 * xi6;
		res7 += xi7 * xi7;
		res8 += xi8 * xi8;
		res9 += xi9 * xi9;
		res10 += xi10 * xi10;
		res11 += xi11 * xi11;
		res12 += xi12 * xi12;
		res13 += xi13 * xi13;
		res14 += xi14 * xi14;
		res15 += xi15 * xi15;
		res16 += xi16 * xi16;

		x += 16 * incx;
	}
	res1 += res2 + res3 + res4;
	res1 += res5 + res6 + res7 + res8;
	res1 += res9 + res10 + res11 + res12;
	res1 += res13 + res14 + res15 + res16;
	for (; i < n; i++) {
		float xi = x[i * incx];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

float cblas_snrm2_sse(size_t n, float* x, size_t /*incx = 1*/) {
	__m128 res = _mm_setzero_ps();
	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		__m128 y = _mm_load_ps(x);
		y = _mm_mul_ps(y, y);
		res = _mm_add_ps(res, y);
		x += 4;
	}
	float res1;
	res = _mm_hadd_ps(res, res);
	res = _mm_hadd_ps(res, res);
	_mm_store_ss(&res1, res);
	for (; i < n; i++) {
		float xi = x[i];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

float cblas_snrm2_sse_unroll2(size_t n, float* x, size_t /*incx = 1*/) {
	__m128 r1 = _mm_setzero_ps();
	__m128 r2 = _mm_setzero_ps();
	size_t i = 0;
	for (; i <= n - 8; i += 8) {
		__m128 y1 = _mm_load_ps(x);
		__m128 y2 = _mm_load_ps(x + 4);
		y1 = _mm_mul_ps(y1, y1);
		y2 = _mm_mul_ps(y2, y2);
		r1 = _mm_add_ps(r1, y1);
		r2 = _mm_add_ps(r2, y2);
		x += 8;
	}
	float res1;
	r1 = _mm_add_ps(r1, r2);
	r1 = _mm_hadd_ps(r1, r1);
	r1 = _mm_hadd_ps(r1, r1);
	_mm_store_ss(&res1, r1);
	for (; i < n; i++) {
		float xi = x[i];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

float cblas_snrm2_sse_unroll4(size_t n, float* x, size_t /*incx = 1*/) {
	__m128 r1 = _mm_setzero_ps();
	__m128 r2 = _mm_setzero_ps();
	__m128 r3 = _mm_setzero_ps();
	__m128 r4 = _mm_setzero_ps();
	size_t i = 0;
	for (; i <= n - 16; i += 16) {
		__m128 y1 = _mm_load_ps(x);
		__m128 y2 = _mm_load_ps(x + 4);
		__m128 y3 = _mm_load_ps(x + 8);
		__m128 y4 = _mm_load_ps(x + 12);
		y1 = _mm_mul_ps(y1, y1);
		y2 = _mm_mul_ps(y2, y2);
		y3 = _mm_mul_ps(y3, y3);
		y4 = _mm_mul_ps(y4, y4);
		r1 = _mm_add_ps(r1, y1);
		r2 = _mm_add_ps(r2, y2);
		r3 = _mm_add_ps(r3, y3);
		r4 = _mm_add_ps(r4, y4);
		x += 16;
	}
	float res1;
	r1 = _mm_add_ps(r1, r2);
	r1 = _mm_add_ps(r1, r3);
	r1 = _mm_add_ps(r1, r4);
	r1 = _mm_hadd_ps(r1, r1);
	r1 = _mm_hadd_ps(r1, r1);
	_mm_store_ss(&res1, r1);
	for (; i < n; i++) {
		float xi = x[i];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

float cblas_snrm2_sse_unroll8(size_t n, float* x, size_t incx) {
	__m128 r1 = _mm_setzero_ps();
	__m128 r2 = _mm_setzero_ps();
	__m128 r3 = _mm_setzero_ps();
	__m128 r4 = _mm_setzero_ps();
	__m128 r5 = _mm_setzero_ps();
	__m128 r6 = _mm_setzero_ps();
	__m128 r7 = _mm_setzero_ps();
	__m128 r8 = _mm_setzero_ps();
	size_t i = 0;
	for (; i <= n - 32; i += 32) {
		__m128 y1 = _mm_load_ps(x);
		__m128 y2 = _mm_load_ps(x + 4);
		__m128 y3 = _mm_load_ps(x + 8);
		__m128 y4 = _mm_load_ps(x + 12);
		__m128 y5 = _mm_load_ps(x + 16);
		__m128 y6 = _mm_load_ps(x + 20);
		__m128 y7 = _mm_load_ps(x + 24);
		__m128 y8 = _mm_load_ps(x + 28);
		y1 = _mm_mul_ps(y1, y1);
		y2 = _mm_mul_ps(y2, y2);
		y3 = _mm_mul_ps(y3, y3);
		y4 = _mm_mul_ps(y4, y4);
		y5 = _mm_mul_ps(y5, y5);
		y6 = _mm_mul_ps(y6, y6);
		y7 = _mm_mul_ps(y7, y7);
		y8 = _mm_mul_ps(y8, y8);
		r1 = _mm_add_ps(r1, y1);
		r2 = _mm_add_ps(r2, y2);
		r3 = _mm_add_ps(r3, y3);
		r4 = _mm_add_ps(r4, y4);
		r5 = _mm_add_ps(r5, y5);
		r6 = _mm_add_ps(r6, y6);
		r7 = _mm_add_ps(r7, y7);
		r8 = _mm_add_ps(r8, y8);
		x += 32;
	}
	float res1;
	r1 = _mm_add_ps(r1, r2);
	r1 = _mm_add_ps(r1, r3);
	r1 = _mm_add_ps(r1, r4);
	r1 = _mm_add_ps(r1, r5);
	r1 = _mm_add_ps(r1, r6);
	r1 = _mm_add_ps(r1, r7);
	r1 = _mm_add_ps(r1, r8);
	r1 = _mm_hadd_ps(r1, r1);
	r1 = _mm_hadd_ps(r1, r1);
	_mm_store_ss(&res1, r1);
	for (; i < n; i++) {
		float xi = x[i];
		res1 += xi * xi;
	}
	return sqrt(res1);
}

namespace swblas {

float cblas_snrm2(size_t n, float* x, size_t incx) {
	constexpr size_t thresh = 4096;
	if (n == 0) return 0.0;
	bool unit_strided = incx == 1;
	if (unit_strided && (uintptr_t)x % 16 == 0) {
		return cblas_snrm2_sse_unroll8(n, x, 1);
	} else {
		return cblas_snrm2_unroll2(n, x, incx);
	}
}

}  // namespace swblas
