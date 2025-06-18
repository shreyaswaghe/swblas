#include <cstddef>

#include "../include/dscal.h"
#include "sse2neon.h"

void cblas_dscal_full_naive(size_t n, double alpha, double* __restrict x,
							size_t incx) {
	size_t i = 0;
	for (; i < n; i++) {
		x[i * incx] *= alpha;
	}
}

void cblas_dscal_unroll2(size_t n, double alpha, double* __restrict x,
						 size_t incx) {
	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		x[i * incx] *= alpha;
		x[(i + 1) * incx] *= alpha;
	}
	for (; i < n; i++) {
		x[i * incx] *= alpha;
	}
}

void cblas_dscal_unroll4(size_t n, double alpha, double* __restrict x,
						 size_t incx) {
	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		x[i * incx] *= alpha;
		x[(i + 1) * incx] *= alpha;
		x[(i + 2) * incx] *= alpha;
		x[(i + 3) * incx] *= alpha;
	}
	for (; i < n; i++) {
		x[i * incx] *= alpha;
	}
}

void cblas_dscal_unroll8(size_t n, double alpha, double* __restrict x,
						 size_t incx) {
	size_t i = 0;
	for (; i <= n - 8; i += 8) {
		x[i * incx] *= alpha;
		x[(i + 1) * incx] *= alpha;
		x[(i + 2) * incx] *= alpha;
		x[(i + 3) * incx] *= alpha;
		x[(i + 4) * incx] *= alpha;
		x[(i + 5) * incx] *= alpha;
		x[(i + 6) * incx] *= alpha;
		x[(i + 7) * incx] *= alpha;
	}
	for (; i < n; i++) {
		x[i * incx] *= alpha;
	}
};

void cblas_dscal_assume_aligned(size_t n, double alpha, double* __restrict x,
								size_t incx) {
	x = (double*)__builtin_assume_aligned(x, 16);
	size_t i = 0;
	for (; i < n; i++) {
		x[i * incx] *= alpha;
	}
};

void cblas_dscal_sse(size_t n, double alpha, double* __restrict x,
					 size_t /*incx = 1*/) {
	x = (double*)__builtin_assume_aligned(x, 16);
	size_t i = 0;
	__m128d a = _mm_set_pd1(alpha);
	for (; i < n - 2; i += 2) {
		__m128d x1 = _mm_load_pd(x + i);
		_mm_store_pd(x + i, _mm_mul_pd(a, x1));
	}
	for (; i < n; i++) {
		x[i] *= alpha;
	}
}

void cblas_dscal_sse_unroll2(size_t n, double alpha, double* __restrict x,
							 size_t /*incx = 1*/) {
	x = (double*)__builtin_assume_aligned(x, 16);
	size_t i = 0;
	__m128d a = _mm_set_pd1(alpha);
	for (; i < n - 4; i += 4) {
		__m128d x1 = _mm_load_pd(x + i);
		__m128d x2 = _mm_load_pd(x + i + 2);
		_mm_store_pd(x + i, _mm_mul_pd(a, x1));
		_mm_store_pd(x + i + 2, _mm_mul_pd(a, x1));
	}
	for (; i < n; i++) {
		x[i] *= alpha;
	}
};

void cblas_dscal_sse_unroll4(size_t n, double alpha, double* __restrict x,
							 size_t /*incx = 1*/) {
	x = (double*)__builtin_assume_aligned(x, 16);
	size_t i = 0;
	__m128d a = _mm_set_pd1(alpha);
	for (; i < n - 8; i += 8) {
		__m128d x1 = _mm_load_pd(x + i);
		__m128d x2 = _mm_load_pd(x + i + 2);
		__m128d x3 = _mm_load_pd(x + i + 4);
		__m128d x4 = _mm_load_pd(x + i + 6);
		_mm_store_pd(x + i, _mm_mul_pd(a, x1));
		_mm_store_pd(x + i + 2, _mm_mul_pd(a, x2));
		_mm_store_pd(x + i + 4, _mm_mul_pd(a, x3));
		_mm_store_pd(x + i + 6, _mm_mul_pd(a, x4));
	}
	for (; i < n; i++) {
		x[i] *= alpha;
	}
};

void cblas_dscal_neon(size_t n, double alpha, double* __restrict x, size_t) {
	float64x2_t a_vec =
		vdupq_n_f64(alpha);	 // Broadcast alpha to a NEON register
	size_t i = 0;

	// Process 2 doubles (16 bytes) at a time
	for (; i + 1 < n; i += 2) {
		float64x2_t x_vec = vld1q_f64(&x[i]);		// Load 2 doubles
		float64x2_t res = vmulq_f64(x_vec, a_vec);	// Multiply
		vst1q_f64(&x[i], res);						// Store result
	}

	// Tail case for odd-length vectors
	for (; i < n; ++i) {
		x[i] *= alpha;
	}
}

namespace swblas {

void cblas_dscal(size_t n, double alpha, double* x, size_t incx) {
	if (n == 0) return;
	if (incx == 1) cblas_dscal_sse_unroll2(n, alpha, x, incx);
	cblas_dscal_full_naive(n, alpha, x, incx);
}

}  // namespace swblas
