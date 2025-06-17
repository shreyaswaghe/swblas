#include <cstddef>
#include <cstdint>

#include "../include/sdot.h"
#include "sse2neon.h"
#include "swblas.h"

float cblas_sdot_full_naive(size_t n, float* __restrict x, size_t incx,
							float* __restrict y, size_t incy) {
	float res = 0.0;
	size_t i = 0;
	for (; i < n; i++) {
		res += x[incx * i] * y[incy * i];
	}
	return res;
}

float cblas_sdot_unroll2(size_t n, float* __restrict x, size_t incx,
						 float* __restrict y, size_t incy) {
	float res1 = 0.0, res2 = 0.0;
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

float cblas_sdot_unroll4(size_t n, float* __restrict x, size_t incx,
						 float* __restrict y, size_t incy) {
	float res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0;
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

float cblas_sdot_unroll8(size_t n, float* __restrict x, size_t incx,
						 float* __restrict y, size_t incy) {
	float res1 = 0.0, res2 = 0.0, res3 = 0.0, res4 = 0.0, res5 = 0.0,
		  res6 = 0.0, res7 = 0.0, res8 = 0.0;
	size_t i = 0;
	for (; i <= n - 8; i += 8) {
		res1 += x[incx * i] * y[incy * i];
		res2 += x[incx * (i + 1)] * y[incy * (i + 1)];
		res3 += x[incx * (i + 2)] * y[incy * (i + 2)];
		res4 += x[incx * (i + 3)] * y[incy * (i + 3)];
		res5 += x[incx * (i + 4)] * y[incy * (i + 4)];
		res6 += x[incx * (i + 5)] * y[incy * (i + 5)];
		res7 += x[incx * (i + 6)] * y[incy * (i + 6)];
		res8 += x[incx * (i + 7)] * y[incy * (i + 7)];
	}
	res1 += res2 + res3 + res4;
	res1 += res5 + res6 + res7 + res8;
	for (; i < n; i++) {
		res1 += x[incx * i] * y[incy * i];
	}
	return res1;
}

float cblas_sdot_assume_aligned(size_t n, float* __restrict x, size_t incx,
								float* __restrict y, size_t incy) {
	float res = 0.0;
	size_t i = 0;

	x = (float*)__builtin_assume_aligned(x, 16);
	y = (float*)__builtin_assume_aligned(y, 16);
	for (; i < n; i++) {
		res += x[incx * i] * y[incy * i];
	}
	return res;
}

float cblas_sdot_assume_aligned_unroll2(size_t n, float* __restrict x,
										size_t incx, float* __restrict y,
										size_t incy) {
	float res1 = 0.0, res2 = 0.0;
	size_t i = 0;

	x = (float*)__builtin_assume_aligned(x, 16);
	y = (float*)__builtin_assume_aligned(y, 16);
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

float cblas_sdot_sse(size_t n, float* __restrict x, size_t /*incx = 1*/,
					 float* __restrict y, size_t /*incy = 1*/) {
	__m128 res = _mm_setzero_ps();
	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		__m128 xx = _mm_load_ps(x + i);
		__m128 yy = _mm_load_ps(y + i);
		__m128 xxyy = _mm_mul_ps(xx, yy);
		res = _mm_add_ps(xxyy, res);
	}
	__m128 tmp = _mm_hadd_ps(res, res);
	tmp = _mm_hadd_ps(tmp, tmp);
	float rs;
	_mm_store_ss(&rs, tmp);
	for (; i < n; i++) {
		rs += x[i] * y[i];
	}
	return rs;
}

float cblas_sdot_sse_unroll2(size_t n, float* __restrict x, size_t /*incx = 1*/,
							 float* __restrict y, size_t /*incy = 1*/) {
	__m128 r1 = _mm_setzero_ps();
	__m128 r2 = _mm_setzero_ps();
	size_t i = 0;
	for (; i <= n - 8; i += 8) {
		__m128 xx1 = _mm_load_ps(x + i);
		__m128 yy1 = _mm_load_ps(y + i);
		__m128 xxyy1 = _mm_mul_ps(xx1, yy1);
		r1 = _mm_add_ps(xxyy1, r1);

		__m128 xx2 = _mm_load_ps(x + i + 4);
		__m128 yy2 = _mm_load_ps(y + i + 4);
		__m128 xxyy2 = _mm_mul_ps(xx2, yy2);
		r2 = _mm_add_ps(xxyy2, r2);
	}
	__m128 tmp1 = _mm_hadd_ps(r1, r1);
	tmp1 = _mm_hadd_ps(tmp1, tmp1);
	__m128 tmp2 = _mm_hadd_ps(r2, r2);
	tmp2 = _mm_hadd_ps(tmp2, tmp2);

	float rs1, rs2;
	_mm_store_ss(&rs1, tmp1);
	_mm_store_ss(&rs2, tmp2);
	rs1 += rs2;
	for (; i < n; i++) {
		rs1 += x[i] * y[i];
	}
	return rs1;
}

float cblas_sdot_sse_unroll4(size_t n, float* __restrict x, size_t /*incx = 1*/,
							 float* __restrict y, size_t /*incy = 1*/) {
	__m128 r1 = _mm_setzero_ps();
	__m128 r2 = _mm_setzero_ps();
	__m128 r3 = _mm_setzero_ps();
	__m128 r4 = _mm_setzero_ps();
	float rs1, rs2, rs3, rs4;
	size_t i = 0;
	for (; i <= n - 16; i += 16) {
		r1 = _mm_add_ps(r1, _mm_mul_ps(_mm_load_ps(x + i), _mm_load_ps(y + i)));
		r2 = _mm_add_ps(
			r2, _mm_mul_ps(_mm_load_ps(x + i + 4), _mm_load_ps(y + i + 4)));
		r3 = _mm_add_ps(
			r3, _mm_mul_ps(_mm_load_ps(x + i + 8), _mm_load_ps(y + i + 8)));
		r4 = _mm_add_ps(
			r4, _mm_mul_ps(_mm_load_ps(x + i + 12), _mm_load_ps(y + i + 12)));
	}
	__m128 tmp1 = _mm_hadd_ps(r1, r1);
	tmp1 = _mm_hadd_ps(tmp1, tmp1);
	__m128 tmp2 = _mm_hadd_ps(r2, r2);
	tmp2 = _mm_hadd_ps(tmp2, tmp2);
	__m128 tmp3 = _mm_hadd_ps(r3, r3);
	tmp3 = _mm_hadd_ps(tmp3, tmp3);
	__m128 tmp4 = _mm_hadd_ps(r4, r4);
	tmp4 = _mm_hadd_ps(tmp4, tmp4);

	_mm_store_ss(&rs1, tmp1);
	_mm_store_ss(&rs2, tmp2);
	_mm_store_ss(&rs3, tmp3);
	_mm_store_ss(&rs4, tmp4);
	rs1 += rs2;
	rs1 += rs3;
	rs1 += rs4;
	for (; i < n; i++) {
		rs1 += x[i] * y[i];
	}
	return rs1;
}

float cblas_sdot_sse_unroll8(size_t n, float* __restrict x, size_t /*incx = 1*/,
							 float* __restrict y, size_t /*incy = 1*/) {
	__m128 r1 = _mm_setzero_ps();
	__m128 r2 = _mm_setzero_ps();
	__m128 r3 = _mm_setzero_ps();
	__m128 r4 = _mm_setzero_ps();
	__m128 r5 = _mm_setzero_ps();
	__m128 r6 = _mm_setzero_ps();
	__m128 r7 = _mm_setzero_ps();
	__m128 r8 = _mm_setzero_ps();
	float rs1, rs2, rs3, rs4, rs5, rs6, rs7, rs8;
	size_t i = 0;
	for (; i <= n - 32; i += 32) {
		r1 = _mm_add_ps(r1, _mm_mul_ps(_mm_load_ps(x + i), _mm_load_ps(y + i)));
		r2 = _mm_add_ps(
			r2, _mm_mul_ps(_mm_load_ps(x + i + 4), _mm_load_ps(y + i + 4)));
		r3 = _mm_add_ps(
			r3, _mm_mul_ps(_mm_load_ps(x + i + 8), _mm_load_ps(y + i + 8)));
		r4 = _mm_add_ps(
			r4, _mm_mul_ps(_mm_load_ps(x + i + 12), _mm_load_ps(y + i + 12)));
		r5 = _mm_add_ps(
			r5, _mm_mul_ps(_mm_load_ps(x + i + 16), _mm_load_ps(y + i + 16)));
		r6 = _mm_add_ps(
			r6, _mm_mul_ps(_mm_load_ps(x + i + 20), _mm_load_ps(y + i + 20)));
		r7 = _mm_add_ps(
			r7, _mm_mul_ps(_mm_load_ps(x + i + 24), _mm_load_ps(y + i + 24)));
		r8 = _mm_add_ps(
			r8, _mm_mul_ps(_mm_load_ps(x + i + 28), _mm_load_ps(y + i + 28)));
	}
	__m128 tmp1 = _mm_hadd_ps(r1, r1);
	tmp1 = _mm_hadd_ps(tmp1, tmp1);
	__m128 tmp2 = _mm_hadd_ps(r2, r2);
	tmp2 = _mm_hadd_ps(tmp2, tmp2);
	__m128 tmp3 = _mm_hadd_ps(r3, r3);
	tmp3 = _mm_hadd_ps(tmp3, tmp3);
	__m128 tmp4 = _mm_hadd_ps(r4, r4);
	tmp4 = _mm_hadd_ps(tmp4, tmp4);
	__m128 tmp5 = _mm_hadd_ps(r5, r5);
	tmp5 = _mm_hadd_ps(tmp5, tmp5);
	__m128 tmp6 = _mm_hadd_ps(r6, r6);
	tmp6 = _mm_hadd_ps(tmp6, tmp6);
	__m128 tmp7 = _mm_hadd_ps(r7, r7);
	tmp7 = _mm_hadd_ps(tmp7, tmp7);
	__m128 tmp8 = _mm_hadd_ps(r8, r8);
	tmp8 = _mm_hadd_ps(tmp8, tmp8);

	_mm_store_ss(&rs1, tmp1);
	_mm_store_ss(&rs2, tmp2);
	_mm_store_ss(&rs3, tmp3);
	_mm_store_ss(&rs4, tmp4);
	_mm_store_ss(&rs5, tmp5);
	_mm_store_ss(&rs6, tmp6);
	_mm_store_ss(&rs7, tmp7);
	_mm_store_ss(&rs8, tmp8);
	rs1 += rs2;
	rs1 += rs3;
	rs1 += rs4;
	rs1 += rs5;
	rs1 += rs6;
	rs1 += rs7;
	rs1 += rs8;
	for (; i < n; i++) {
		rs1 += x[i] * y[i];
	}
	return rs1;
}

namespace swblas {

float cblas_sdot(size_t n, float* x, size_t incx, float* y, size_t incy) {
	if (n == 0) return 0.0;

	if (incx == 1 && incy == 1) {
		if ((uintptr_t)x % 16 == 0 && (uintptr_t)y % 16 == 0)
			return cblas_sdot_sse_unroll8(n, x, 1, y, 1);
		else
			return cblas_sdot_unroll8(n, x, 1, y, 1);
	} else {
		return cblas_sdot_unroll8(n, x, 1, y, 1);
	}
}

}  // namespace swblas
