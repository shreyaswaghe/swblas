#include <cmath>
#include <cstddef>
#include <cstdint>

#include "../include/dasum.h"
#include "sse2neon.h"
#include "swblas.h"

inline double fabs(double x) {
	union {
		double d;
		uint64_t u;
	} val;
	val.d = x;
	val.u &= 0x7FFFFFFFFFFFFFFF;
	return val.d;
}

double cblas_dasum_naive(size_t n, double* x, size_t incx) {
	double res = 0.0;
	size_t i = 0;
	for (; i < n; i++) {
		double xi = x[i * incx];
		res += fabs(xi);
	}
	return res;
}

double cblas_dasum_unroll2(size_t n, double* x, size_t incx) {
	const double* xx = x;
	double res1 = 0.0, res2 = 0.0;
	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		double xi1 = xx[0];
		double xi2 = xx[1 * incx];
		res1 += fabs(xi1);
		res2 += fabs(xi2);
		xx += 2 * incx;
	}
	res1 += res2;
	for (; i < n; i++) {
		double xi = x[i * incx];
		res1 += fabs(xi);
	}
	return res1;
}

double cblas_dasum_unroll4(size_t n, double* x, size_t incx) {
	const double* xx = x;
	double res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		double xi1 = xx[0];
		double xi2 = xx[1 * incx];
		double xi3 = xx[2 * incx];
		double xi4 = xx[3 * incx];
		res1 += fabs(xi1);
		res2 += fabs(xi2);
		res3 += fabs(xi3);
		res4 += fabs(xi4);
		xx += 4 * incx;
	}
	res1 += res2 + res3 + res4;
	for (; i < n; i++) {
		double xi = x[i * incx];
		res1 += fabs(xi);
	}
	return res1;
}

double cblas_dasum_unroll8(size_t n, double* x, size_t incx) {
	double res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
	double res5 = 0.0, res6 = 0.0, res7 = 0.0, res8 = 0.0;
	size_t i = 0;
	const double* xx = x;
	for (; i <= n - 8; i += 8) {
		double xi1 = xx[0];
		double xi2 = xx[1 * incx];
		double xi3 = xx[2 * incx];
		double xi4 = xx[3 * incx];
		double xi5 = xx[4 * incx];
		double xi6 = xx[5 * incx];
		double xi7 = xx[6 * incx];
		double xi8 = xx[7 * incx];
		res1 += fabs(xi1);
		res2 += fabs(xi2);
		res3 += fabs(xi3);
		res4 += fabs(xi4);
		res5 += fabs(xi5);
		res6 += fabs(xi6);
		res7 += fabs(xi7);
		res8 += fabs(xi8);
		xx += 8 * incx;
	}
	res1 += res2 + res3 + res4;
	res1 += res5 + res6 + res7 + res8;
	for (; i < n; i++) {
		double xi = x[i * incx];
		res1 += fabs(xi);
	}
	return res1;
}

double cblas_dasum_sse(size_t n, double* x, size_t /*incx = 1*/) {
	__m128d res = _mm_setzero_pd();
	__m128d abser = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));

	const double* xx = x;
	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		__m128d y = _mm_load_pd(xx);
		y = _mm_and_pd(y, abser);
		res = _mm_add_pd(res, y);
		xx += 2;
	}
	double res1;
	res = _mm_hadd_pd(res, res);
	_mm_store_sd(&res1, res);
	for (; i < n; i++) {
		double xi = x[i];
		res1 += fabs(xi);
	}
	return res1;
}

double cblas_dasum_sse_unroll2(size_t n, double* x, size_t /*incx = 1*/) {
	const double* xx = x;
	__m128d r1 = _mm_setzero_pd();
	__m128d r2 = _mm_setzero_pd();

	__m128d abser = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));

	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		__m128d y1 = _mm_load_pd(xx);
		__m128d y2 = _mm_load_pd(xx + 2);
		y1 = _mm_and_pd(y1, abser);
		y2 = _mm_and_pd(y2, abser);
		r1 = _mm_add_pd(r1, y1);
		r2 = _mm_add_pd(r2, y2);
		xx += 4;
	}
	double res1;
	r1 = _mm_add_pd(r1, r2);
	r1 = _mm_hadd_pd(r1, r1);
	_mm_store_sd(&res1, r1);
	for (; i < n; i++) {
		double xi = x[i];
		res1 += fabs(xi);
	}
	return res1;
}

double cblas_dasum_sse_unroll4(size_t n, double* x, size_t /*incx = 1*/) {
	const double* xx = x;
	__m128d r1 = _mm_setzero_pd();
	__m128d r2 = _mm_setzero_pd();
	__m128d r3 = _mm_setzero_pd();
	__m128d r4 = _mm_setzero_pd();

	__m128d abser = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));

	size_t i = 0;
	for (; i <= n - 8; i += 8) {
		__m128d y1 = _mm_load_pd(xx);
		__m128d y2 = _mm_load_pd(xx + 2);
		__m128d y3 = _mm_load_pd(xx + 4);
		__m128d y4 = _mm_load_pd(xx + 6);
		y1 = _mm_and_pd(y1, abser);
		y2 = _mm_and_pd(y2, abser);
		y3 = _mm_and_pd(y3, abser);
		y4 = _mm_and_pd(y4, abser);
		r1 = _mm_add_pd(r1, y1);
		r2 = _mm_add_pd(r2, y2);
		r3 = _mm_add_pd(r3, y3);
		r4 = _mm_add_pd(r4, y4);
		xx += 8;
	}
	double res1;
	r1 = _mm_add_pd(r1, r2);
	r1 = _mm_add_pd(r1, r3);
	r1 = _mm_add_pd(r1, r4);
	r1 = _mm_hadd_pd(r1, r1);
	_mm_store_sd(&res1, r1);
	for (; i < n; i++) {
		double xi = x[i];
		res1 += fabs(xi);
	}
	return res1;
}

double cblas_dasum_sse_unroll8(size_t n, double* x, size_t incx) {
	const double* xx = x;
	__m128d r1 = _mm_setzero_pd();
	__m128d r2 = _mm_setzero_pd();
	__m128d r3 = _mm_setzero_pd();
	__m128d r4 = _mm_setzero_pd();
	__m128d r5 = _mm_setzero_pd();
	__m128d r6 = _mm_setzero_pd();
	__m128d r7 = _mm_setzero_pd();
	__m128d r8 = _mm_setzero_pd();

	__m128d abser = _mm_castsi128_pd(_mm_set1_epi64x(0x7FFFFFFFFFFFFFFF));
	size_t i = 0;
	for (; i <= n - 16; i += 16) {
		__m128d y1 = _mm_load_pd(xx);
		__m128d y2 = _mm_load_pd(xx + 2);
		__m128d y3 = _mm_load_pd(xx + 4);
		__m128d y4 = _mm_load_pd(xx + 6);
		__m128d y5 = _mm_load_pd(xx + 8);
		__m128d y6 = _mm_load_pd(xx + 10);
		__m128d y7 = _mm_load_pd(xx + 12);
		__m128d y8 = _mm_load_pd(xx + 14);
		y1 = _mm_and_pd(y1, abser);
		y2 = _mm_and_pd(y2, abser);
		y3 = _mm_and_pd(y3, abser);
		y4 = _mm_and_pd(y4, abser);
		y5 = _mm_and_pd(y5, abser);
		y6 = _mm_and_pd(y6, abser);
		y7 = _mm_and_pd(y7, abser);
		y8 = _mm_and_pd(y8, abser);
		r1 = _mm_add_pd(r1, y1);
		r2 = _mm_add_pd(r2, y2);
		r3 = _mm_add_pd(r3, y3);
		r4 = _mm_add_pd(r4, y4);
		r5 = _mm_add_pd(r5, y5);
		r6 = _mm_add_pd(r6, y6);
		r7 = _mm_add_pd(r7, y7);
		r8 = _mm_add_pd(r8, y8);
		xx += 16;
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
		res1 += fabs(xi);
	}
	return res1;
}

namespace swblas {

double cblas_dasum(size_t n, double* x, size_t incx) {
	constexpr size_t thresh = 4096;
	if (n == 0) return 0.0;
	bool unit_strided = incx == 1;
	if (unit_strided && (uintptr_t)x % 16 == 0) {
		return cblas_dasum_sse_unroll8(n, x, 1);
	} else {
		return cblas_dasum_unroll8(n, x, incx);
	}
}

}  // namespace swblas
