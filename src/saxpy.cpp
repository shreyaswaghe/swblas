#include <cstddef>

#include "../include/saxpy.h"
#include "sse2neon.h"

void cblas_saxpy_full_naive(size_t n, float alpha, float* __restrict x,
							size_t incx, float* __restrict y, size_t incy) {
	for (size_t i = 0; i < n; i++) y[incy * i] += alpha * x[incx * i];
}

void cblas_saxpy_unroll2(size_t n, float alpha, float* __restrict x,
						 size_t incx, float* __restrict y, size_t incy) {
	size_t nn = n & -2;
	for (size_t i = 0; i < nn; i += 2) {
		y[incy * i] += alpha * x[incx * i];
		y[incy * (i + 1)] += alpha * x[incx * (i + 1)];
	}
	size_t rem = n - nn;
	for (size_t i = nn; i < n; i++) {
		y[incy * i] += alpha * x[incx * i];
	}
}

void cblas_saxpy_unroll4(size_t n, float alpha, float* __restrict x,
						 size_t incx, float* __restrict y, size_t incy) {
	size_t nn = n & -4;
	for (size_t i = 0; i < nn; i += 4) {
		y[incy * i] += alpha * x[incx * i];
		y[incy * (i + 1)] += alpha * x[incx * (i + 1)];
		y[incy * (i + 2)] += alpha * x[incx * (i + 2)];
		y[incy * (i + 3)] += alpha * x[incx * (i + 3)];
	}
	size_t rem = n - nn;
	for (size_t i = nn; i < n; i++) {
		y[incy * i] += alpha * x[incx * i];
	}
}

void cblas_saxpy_unroll8(size_t n, float alpha, float* __restrict x,
						 size_t incx, float* __restrict y, size_t incy) {
	size_t nn = n & -8;
	for (size_t i = 0; i < nn; i += 8) {
		y[incy * i] += alpha * x[incx * i];
		y[incy * (i + 1)] += alpha * x[incx * (i + 1)];
		y[incy * (i + 2)] += alpha * x[incx * (i + 2)];
		y[incy * (i + 3)] += alpha * x[incx * (i + 3)];
		y[incy * (i + 4)] += alpha * x[incx * (i + 4)];
		y[incy * (i + 5)] += alpha * x[incx * (i + 5)];
		y[incy * (i + 6)] += alpha * x[incx * (i + 6)];
		y[incy * (i + 7)] += alpha * x[incx * (i + 7)];
	}
	size_t rem = n - nn;
	for (size_t i = nn; i < n; i++) {
		y[incy * i] += alpha * x[incx * i];
	}
}

void cblas_saxpy_unroll16(size_t n, float alpha, float* __restrict x,
						  size_t incx, float* __restrict y, size_t incy) {
	size_t nn = n & -8;
	for (size_t i = 0; i < nn; i += 8) {
		y[incy * i] += alpha * x[incx * i];
		y[incy * (i + 1)] += alpha * x[incx * (i + 1)];
		y[incy * (i + 2)] += alpha * x[incx * (i + 2)];
		y[incy * (i + 3)] += alpha * x[incx * (i + 3)];
		y[incy * (i + 4)] += alpha * x[incx * (i + 4)];
		y[incy * (i + 5)] += alpha * x[incx * (i + 5)];
		y[incy * (i + 6)] += alpha * x[incx * (i + 6)];
		y[incy * (i + 7)] += alpha * x[incx * (i + 7)];
		y[incy * (i + 8)] += alpha * x[incx * (i + 8)];
		y[incy * (i + 9)] += alpha * x[incx * (i + 9)];
		y[incy * (i + 10)] += alpha * x[incx * (i + 10)];
		y[incy * (i + 11)] += alpha * x[incx * (i + 11)];
		y[incy * (i + 12)] += alpha * x[incx * (i + 12)];
		y[incy * (i + 13)] += alpha * x[incx * (i + 13)];
		y[incy * (i + 14)] += alpha * x[incx * (i + 14)];
		y[incy * (i + 15)] += alpha * x[incx * (i + 15)];
	}
	size_t rem = n - nn;
	for (size_t i = nn; i < n; i++) {
		y[incy * i] += alpha * x[incx * i];
	}
}

void cblas_saxpy_assume_aligned(size_t n, float alpha, float* __restrict x,
								size_t incx, float* __restrict y, size_t incy) {
	float* x_align = (float*)__builtin_assume_aligned(x, 16);
	float* y_align = (float*)__builtin_assume_aligned(y, 16);
	for (size_t i = 0; i < n; i++)
		y_align[incy * i] += alpha * x_align[incx * i];
}

void cblas_saxpy_sse(size_t n, float alpha, float* __restrict x,
					 size_t /*incx = 1*/, float* __restrict y,
					 size_t /*incy = 1 */) {
	size_t incx = 1, incy = 1;

	float* x_align = (float*)__builtin_assume_aligned(x, 16);
	float* y_align = (float*)__builtin_assume_aligned(y, 16);

	__m128 a = _mm_load_ps1(&alpha);

	size_t i;
	for (i = 0; i <= n - 4; i += 4) {
		// load 2 floats into 128 bit registers
		__m128 x1 = _mm_load_ps(x_align + i);
		__m128 x2 = _mm_load_ps(y_align + i);
		__m128 x3 = _mm_mul_ps(a, x1);	 // mul a * x
		__m128 x4 = _mm_add_ps(x2, x3);	 // add y + a * x
		_mm_store_ps(y_align + i, x4);	 // y <- y + a * x
	}
	for (; i < n; i++) {
		y_align[i] += alpha * x_align[i];
	}
}

void cblas_saxpy_sse_unroll2(size_t n, float alpha, float* __restrict x,
							 size_t /*incx = 1*/, float* __restrict y,
							 size_t /*incy = 1 */) {
	size_t incx = 1, incy = 1;

	float* x_align = (float*)__builtin_assume_aligned(x, 16);
	float* y_align = (float*)__builtin_assume_aligned(y, 16);

	__m128 a = _mm_load_ps1(&alpha);

	size_t i;
	for (i = 0; i <= n - 8; i += 8) {
		__m128 x1 = _mm_load_ps(x_align + i);
		__m128 y1 = _mm_load_ps(y_align + i);
		__m128 x2 = _mm_load_ps(x_align + i + 4);
		__m128 y2 = _mm_load_ps(y_align + i + 4);

		_mm_store_ps(y_align + i, _mm_add_ps(y1, _mm_mul_ps(a, x1)));
		_mm_store_ps(y_align + i + 4, _mm_add_ps(y2, _mm_mul_ps(a, x2)));
	}
	for (; i < n; i++) {
		y_align[i] += alpha * x_align[i];
	}
}

void cblas_saxpy_sse_unroll4(size_t n, float alpha, float* __restrict x,
							 size_t /*incx = 1*/, float* __restrict y,
							 size_t /*incy = 1 */) {
	size_t incx = 1, incy = 1;

	float* x_align = (float*)__builtin_assume_aligned(x, 16);
	float* y_align = (float*)__builtin_assume_aligned(y, 16);

	__m128 a = _mm_load_ps1(&alpha);

	size_t i;
	for (i = 0; i <= n - 16; i += 16) {
		__m128 x1 = _mm_load_ps(x_align + i);
		__m128 y1 = _mm_load_ps(y_align + i);
		__m128 x2 = _mm_load_ps(x_align + i + 4);
		__m128 y2 = _mm_load_ps(y_align + i + 4);
		__m128 x3 = _mm_load_ps(x_align + i + 8);
		__m128 y3 = _mm_load_ps(y_align + i + 8);
		__m128 x4 = _mm_load_ps(x_align + i + 16);
		__m128 y4 = _mm_load_ps(y_align + i + 16);

		_mm_store_ps(y_align + i, _mm_add_ps(y1, _mm_mul_ps(a, x1)));
		_mm_store_ps(y_align + i + 4, _mm_add_ps(y2, _mm_mul_ps(a, x2)));
		_mm_store_ps(y_align + i + 8, _mm_add_ps(y3, _mm_mul_ps(a, x3)));
		_mm_store_ps(y_align + i + 12, _mm_add_ps(y4, _mm_mul_ps(a, x4)));
	}
	for (; i < n; i++) {
		y_align[i] += alpha * x_align[i];
	}
}

namespace swblas {
void cblas_saxpy(size_t n, float alpha, float* x, size_t incx, float* y,
				 size_t incy) {
	if (n == 0) return;
	if (incx == 1 && incy == 1)
		cblas_saxpy_sse_unroll4(n, alpha, x, incx, y, incy);
	else
		cblas_saxpy_full_naive(n, alpha, x, incx, y, incy);
}
};	// namespace swblas
