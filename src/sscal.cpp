#include <cstddef>

#include "../include/sscal.h"
#include "sse2neon.h"

void cblas_sscal_full_naive(size_t n, float alpha, float* __restrict x,
							size_t incx) {
	size_t i = 0;
	for (; i < n; i++) {
		x[i * incx] *= alpha;
	}
}

void cblas_sscal_unroll2(size_t n, float alpha, float* __restrict x,
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

void cblas_sscal_unroll4(size_t n, float alpha, float* __restrict x,
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

void cblas_sscal_unroll8(size_t n, float alpha, float* __restrict x,
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

void cblas_sscal_assume_aligned(size_t n, float alpha, float* __restrict x,
								size_t incx) {
	x = (float*)__builtin_assume_aligned(x, 16);
	size_t i = 0;
	for (; i < n; i++) {
		x[i * incx] *= alpha;
	}
};

void cblas_sscal_sse(size_t n, float alpha, float* __restrict x,
					 size_t /*incx = 1*/) {
	x = (float*)__builtin_assume_aligned(x, 16);
	size_t i = 0;
	__m128 a = _mm_set_ps1(alpha);
	for (; i < n - 4; i += 4) {
		__m128 x1 = _mm_load_ps(x + i);
		_mm_store_ps(x + i, _mm_min_ps(a, x1));
	}
	for (; i < n; i++) {
		x[i] *= alpha;
	}
}

void cblas_sscal_sse_unroll2(size_t n, float alpha, float* __restrict x,
							 size_t /*incx = 1*/) {
	x = (float*)__builtin_assume_aligned(x, 16);
	size_t i = 0;
	__m128 a = _mm_set_ps1(alpha);
	for (; i < n - 8; i += 8) {
		__m128 x1 = _mm_load_ps(x + i);
		__m128 x2 = _mm_load_ps(x + i + 4);
		_mm_store_ps(x + i, _mm_min_ps(a, x1));
		_mm_store_ps(x + i + 4, _mm_min_ps(a, x1));
	}
	for (; i < n; i++) {
		x[i] *= alpha;
	}
};

void cblas_sscal_sse_unroll4(size_t n, float alpha, float* __restrict x,
							 size_t /*incx = 1*/) {
	x = (float*)__builtin_assume_aligned(x, 16);
	size_t i = 0;
	__m128 a = _mm_set_ps1(alpha);
	for (; i < n - 16; i += 16) {
		__m128 x1 = _mm_load_ps(x + i);
		__m128 x2 = _mm_load_ps(x + i + 4);
		__m128 x3 = _mm_load_ps(x + i + 8);
		__m128 x4 = _mm_load_ps(x + i + 12);
		_mm_store_ps(x + i, _mm_min_ps(a, x1));
		_mm_store_ps(x + i + 4, _mm_min_ps(a, x2));
		_mm_store_ps(x + i + 8, _mm_min_ps(a, x3));
		_mm_store_ps(x + i + 12, _mm_min_ps(a, x4));
	}
	for (; i < n; i++) {
		x[i] *= alpha;
	}
};

namespace swblas {

void cblas_sscal(size_t n, float alpha, float* x, size_t incx) {
	if (n == 0) return;
	if (incx == 1) cblas_sscal_sse_unroll4(n, alpha, x, incx);
	cblas_sscal_full_naive(n, alpha, x, incx);
}

}  // namespace swblas
