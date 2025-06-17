#include <cstddef>

#include "../include/daxpy.h"
#include "sse2neon.h"

void cblas_daxpy_full_naive(size_t n, double alpha, double* __restrict x,
							size_t incx, double* __restrict y, size_t incy) {
	for (size_t i = 0; i < n; i++) y[incy * i] += alpha * x[incx * i];
}

void cblas_daxpy_unroll2(size_t n, double alpha, double* __restrict x,
						 size_t incx, double* __restrict y, size_t incy) {
	size_t nn = n & -2;
	for (size_t i = 0; i < nn; i += 2) {
		y[0] += alpha * x[0];
		y[incy * 1] += alpha * x[incx * 1];
		y += 2 * incy;
		x += 2 * incx;
	}
	size_t rem = n - nn;
	for (size_t i = nn; i < n; i++) {
		y[incy * i] += alpha * x[incx * i];
	}
}

void cblas_daxpy_unroll4(size_t n, double alpha, double* __restrict x,
						 size_t incx, double* __restrict y, size_t incy) {
	size_t nn = n & -4;
	for (size_t i = 0; i < nn; i += 4) {
		y[0] += alpha * x[0];
		y[incy * 1] += alpha * x[incx * 1];
		y[incy * 2] += alpha * x[incx * 2];
		y[incy * 3] += alpha * x[incx * 3];
		y += 4 * incy;
		x += 4 * incx;
	}
	size_t rem = n - nn;
	for (size_t i = nn; i < n; i++) {
		y[incy * i] += alpha * x[incx * i];
	}
}

void cblas_daxpy_unroll8(size_t n, double alpha, double* __restrict x,
						 size_t incx, double* __restrict y, size_t incy) {
	size_t nn = n & -8;
	for (size_t i = 0; i < nn; i += 8) {
		y[0] += alpha * x[0];
		y[incy * 1] += alpha * x[incx * 1];
		y[incy * 2] += alpha * x[incx * 2];
		y[incy * 3] += alpha * x[incx * 3];
		y[incy * 4] += alpha * x[incx * 4];
		y[incy * 5] += alpha * x[incx * 5];
		y[incy * 6] += alpha * x[incx * 6];
		y[incy * 7] += alpha * x[incx * 7];
		y += 8 * incy;
		x += 8 * incx;
	}
	size_t rem = n - nn;
	for (size_t i = nn; i < n; i++) {
		y[incy * i] += alpha * x[incx * i];
	}
}

void cblas_daxpy_unroll2_uncoupled(size_t n, double alpha, double* __restrict x,
								   size_t incx, double* __restrict y,
								   size_t incy) {
	size_t nn = n & -2;
	for (size_t i = 0; i < nn; i += 2) {
		double yax0 = y[0] + alpha * x[0];
		double yax1 = y[incy] + alpha * x[incx];
		y[0] = yax0;
		y[incy * 1] = yax1;
		y += 2 * incy;
		x += 2 * incx;
	}
	size_t rem = n - nn;
	for (size_t i = nn; i < n; i++) {
		y[incy * i] += alpha * x[incx * i];
	}
}

void cblas_daxpy_unroll4_uncoupled(size_t n, double alpha, double* __restrict x,
								   size_t incx, double* __restrict y,
								   size_t incy) {
	size_t nn = n & -4;
	for (size_t i = 0; i < nn; i += 4) {
		double yax0 = y[0] + alpha * x[0];
		double yax1 = y[incy] + alpha * x[incx];
		double yax2 = y[incy * 2] + alpha * x[incx * 2];
		double yax3 = y[incy * 3] + alpha * x[incx * 3];

		y[0] = yax0;
		y[incy * 1] = yax1;
		y[incy * 2] = yax2;
		y[incy * 3] = yax3;

		y += 4 * incy;
		x += 4 * incx;
	}
	size_t rem = n - nn;
	for (size_t i = nn; i < n; i++) {
		y[incy * i] += alpha * x[incx * i];
	}
}

void cblas_daxpy_unroll8_uncoupled(size_t n, double alpha, double* __restrict x,
								   size_t incx, double* __restrict y,
								   size_t incy) {
	size_t nn = n & -8;
	for (size_t i = 0; i < nn; i += 8) {
		double yax0 = y[0] + alpha * x[0];
		double yax1 = y[incy] + alpha * x[incx];
		double yax2 = y[incy * 2] + alpha * x[incx * 2];
		double yax3 = y[incy * 3] + alpha * x[incx * 3];
		double yax4 = y[incy * 4] + alpha * x[incx * 4];
		double yax5 = y[incy * 5] + alpha * x[incx * 5];
		double yax6 = y[incy * 6] + alpha * x[incx * 6];
		double yax7 = y[incy * 7] + alpha * x[incx * 7];

		y[0] = yax0;
		y[incy * 1] = yax1;
		y[incy * 2] = yax2;
		y[incy * 3] = yax3;
		y[incy * 4] = yax4;
		y[incy * 5] = yax5;
		y[incy * 6] = yax6;
		y[incy * 7] = yax7;

		y += 8 * incy;
		x += 8 * incx;
	}
	size_t rem = n - nn;
	for (size_t i = nn; i < n; i++) {
		y[incy * i] += alpha * x[incx * i];
	}
}

void cblas_daxpy_assume_aligned(size_t n, double alpha, double* __restrict x,
								size_t incx, double* __restrict y,
								size_t incy) {
	double* x_align = (double*)__builtin_assume_aligned(x, 16);
	double* y_align = (double*)__builtin_assume_aligned(y, 16);
	for (size_t i = 0; i < n; i++)
		y_align[incy * i] += alpha * x_align[incx * i];
}

void cblas_daxpy_sse(size_t n, double alpha, double* __restrict x,
					 size_t /*incx = 1*/, double* __restrict y,
					 size_t /*incy = 1 */) {
	size_t incx = 1, incy = 1;

	double* x_align = (double*)__builtin_assume_aligned(x, 16);
	double* y_align = (double*)__builtin_assume_aligned(y, 16);

	__m128d a = _mm_load_pd1(&alpha);

	size_t ndiv2 = n / 2;
	for (size_t i = 0; i < ndiv2; i++) {
		// load 2 doubles into 128 bit registers
		__m128d x1 = _mm_load_pd(x_align);
		__m128d x2 = _mm_load_pd(y_align);
		x1 = _mm_mul_pd(a, x1);		 // mul a * x
		x2 = _mm_add_pd(x2, x1);	 // add y + a * x
		_mm_stream_pd(y_align, x2);	 // y <- y + a * x
		x_align += 2;
		y_align += 2;
	}
	for (size_t i = ndiv2 * 2; i < n; i++) {
		y_align[i] += alpha * x_align[i];
	}
}

void cblas_daxpy_sse_unroll2(size_t n, double alpha, double* __restrict x,
							 size_t /*incx = 1*/, double* __restrict y,
							 size_t /*incy = 1 */) {
	size_t incx = 1, incy = 1;

	double* x_align = (double*)__builtin_assume_aligned(x, 16);
	double* y_align = (double*)__builtin_assume_aligned(y, 16);

	__m128d a = _mm_load_pd1(&alpha);

	size_t i;
	for (i = 0; i <= n - 4; i += 4) {
		__m128d x1 = _mm_load_pd(x_align);
		__m128d y1 = _mm_load_pd(y_align);
		__m128d x2 = _mm_load_pd(x_align + 2);
		__m128d y2 = _mm_load_pd(y_align + 2);

		_mm_stream_pd(y_align, _mm_add_pd(y1, _mm_mul_pd(a, x1)));
		_mm_stream_pd(y_align + 2, _mm_add_pd(y2, _mm_mul_pd(a, x2)));
		y_align += 4;
		x_align += 4;
	}
	for (; i < n; i++) {
		y_align[i] += alpha * x_align[i];
	}
}

void cblas_daxpy_sse_unroll4(size_t n, double alpha, double* __restrict x,
							 size_t /*incx = 1*/, double* __restrict y,
							 size_t /*incy = 1 */) {
	size_t incx = 1, incy = 1;

	double* x_align = (double*)__builtin_assume_aligned(x, 16);
	double* y_align = (double*)__builtin_assume_aligned(y, 16);

	__m128d a = _mm_load_pd1(&alpha);

	size_t i;
	for (i = 0; i <= n - 8; i += 8) {
		__m128d x1 = _mm_load_pd(x_align);
		__m128d y1 = _mm_load_pd(y_align);
		__m128d x2 = _mm_load_pd(x_align + 2);
		__m128d y2 = _mm_load_pd(y_align + 2);
		__m128d x3 = _mm_load_pd(x_align + 4);
		__m128d y3 = _mm_load_pd(y_align + 4);
		__m128d x4 = _mm_load_pd(x_align + 6);
		__m128d y4 = _mm_load_pd(y_align + 6);

		_mm_stream_pd(y_align, _mm_add_pd(y1, _mm_mul_pd(a, x1)));
		_mm_stream_pd(y_align + 2, _mm_add_pd(y2, _mm_mul_pd(a, x2)));
		_mm_stream_pd(y_align + 4, _mm_add_pd(y3, _mm_mul_pd(a, x3)));
		_mm_stream_pd(y_align + 6, _mm_add_pd(y4, _mm_mul_pd(a, x4)));

		y_align += 8;
		x_align += 8;
	}
	for (; i < n; i++) {
		y_align[i] += alpha * x_align[i];
	}
}

namespace swblas {
void cblas_daxpy(size_t n, double alpha, double* x, size_t incx, double* y,
				 size_t incy) {
	if (n == 0) return;
	if (incx == 1 && incy == 1)
		cblas_daxpy_sse_unroll4(n, alpha, x, incx, y, incy);
	else
		cblas_daxpy_full_naive(n, alpha, x, incx, y, incy);
}
};	// namespace swblas
