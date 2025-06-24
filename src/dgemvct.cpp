#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "../include/dgemvct.h"
#include "../include/swblas.h"
#include "sse2neon.h"

void cblas_dgemv(char fmt, char trans, size_t m, size_t n, double alpha,
				 const double* a, size_t lda, const double* x, size_t incx,
				 double beta, double* y, size_t incy) {}

inline double hadd_register(__m128d reg) {
	reg = _mm_add_sd(reg, _mm_shuffle_pd(reg, reg, 0b01));
	double res;
	_mm_store_sd(&res, reg);
	return res;
}

void cblas_dgemvCT_naive(size_t m, size_t n, double alpha, const double* a,
						 size_t lda, const double* x, size_t incx, double beta,
						 double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i < m; i++) {
		const double* ai = a + i * lda;

		double res = 0.0;
		size_t j = 0;
		for (; j < n; j++) {
			const double xj = x[j * incx];
			res += ai[j] * xj;
		}
		y[i * incy] += alpha * res;
	}
}

void cblas_dgemvCT_unrollinner2(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i < m; i++) {
		const double* ai = a + i * lda;

		double res[2] = {0.0, 0.0};
		size_t j = 0;
		for (; j <= n - 2; j += 2) {
			res[0] += ai[j] * x[j * incx];
			res[1] += ai[j + 1] * x[(j + 1) * incx];
		}
		for (; j < n; j++) {
			const double xj = x[j * incx];
			res[0] += ai[j] * xj;
		}
		y[i * incy] += alpha * (res[0] + res[1]);
	}
}

void cblas_dgemvCT_unrollinner4(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		double res[4] = {0.0, 0.0, 0.0, 0.0};
		size_t j = 0;
		for (; j <= m - 4; j += 4) {
			res[0] += ai[j] * x[j * incx];
			res[1] += ai[j + 1] * x[(j + 1) * incx];
			res[2] += ai[j + 2] * x[(j + 2) * incx];
			res[3] += ai[j + 3] * x[(j + 3) * incx];
		}
		for (; j < m; j++) {
			const double xj = x[j * incx];
			res[0] += ai[j] * xj;
		}
		y[i * incy] += alpha * (res[0] + res[1] + res[2] + res[3]);
	}
}

void cblas_dgemvCT_unrollinner8(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		double res[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		size_t j = 0;
		for (; j <= m - 8; j += 8) {
			res[0] += ai[j] * x[j * incx];
			res[1] += ai[j + 1] * x[(j + 1) * incx];
			res[2] += ai[j + 2] * x[(j + 2) * incx];
			res[3] += ai[j + 3] * x[(j + 3) * incx];
			res[4] += ai[j + 4] * x[(j + 4) * incx];
			res[5] += ai[j + 5] * x[(j + 5) * incx];
			res[6] += ai[j + 6] * x[(j + 6) * incx];
			res[7] += ai[j + 7] * x[(j + 7) * incx];
		}
		for (; j < m; j++) {
			const double xj = x[j * incx];
			res[0] += ai[j] * xj;
		}
		y[i * incy] += alpha * (res[0] + res[1] + res[2] + res[3] + res[4] +
								res[5] + res[6] + res[7]);
	}
}

void cblas_dgemvCT_sseinner(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		__m128d res = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 2; j += 2) {
			__m128d a0, x0;
			a0 = _mm_loadu_pd(ai + j);
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
			}
			res = _mm_add_pd(res, _mm_mul_pd(a0, x0));
		}
		double cleanup = 0.0;
		for (; j < m; j++) {
			cleanup += ai[j] * x[j * incx];
		}
		alignas(16) double yy0[2];
		_mm_store_pd(yy0, res);
		y[i * incy] += alpha * (cleanup + yy0[0] + yy0[1]);
	}
}

void cblas_dgemvCT_sseinner2(size_t m, size_t n, double alpha, const double* a,
							 size_t lda, const double* x, size_t incx,
							 double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 4; j += 4) {
			__m128d a0, x0, a1, x1;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
			}

			a0 = _mm_loadu_pd(ai + j);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			a1 = _mm_loadu_pd(ai + j + 2);
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x1));
		}
		double cleanup = 0.0;
		for (; j < m; j++) {
			cleanup += ai[j] * x[j * incx];
		}
		alignas(16) double yy0[4];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup + yy0[0] + yy0[1] + yy0[2] + yy0[3]);
	}
}

void cblas_dgemvCT_sseinner4(size_t m, size_t n, double alpha, const double* a,
							 size_t lda, const double* x, size_t incx,
							 double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 8; j += 8) {
			__m128d a0, a1, x0, x1, x2, x3;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
				x2 = _mm_loadu_pd(x + j + 4);
				x3 = _mm_loadu_pd(x + j + 6);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
				x2 = _mm_set_pd(x[(j + 5) * incx], x[(j + 4) * incx]);
				x3 = _mm_set_pd(x[(j + 7) * incx], x[(j + 6) * incx]);
			}

			a0 = _mm_loadu_pd(ai + j);
			a1 = _mm_loadu_pd(ai + j + 2);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x1));

			a0 = _mm_loadu_pd(ai + j + 4);
			a1 = _mm_loadu_pd(ai + j + 6);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x2));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x3));
		}
		double cleanup = 0.0;
		for (; j < m; j++) {
			cleanup += ai[j] * x[j * incx];
		}
		alignas(16) double yy0[4];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup + yy0[0] + yy0[1] + yy0[2] + yy0[3]);
	}
}

void cblas_dgemvCT_strideC2(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i <= n - 2; i += 2) {
		const double* ai0 = a + i * lda;
		const double* ai1 = a + (i + 1) * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 8; j += 8) {
			__m128d a0, a1, x0, x1;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
			}

			a0 = _mm_loadu_pd(ai0 + j);
			a1 = _mm_loadu_pd(ai1 + j);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai0 + j + 2);
			a1 = _mm_loadu_pd(ai0 + j + 2);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x0));
		}
		double cleanup[2] = {0.0, 0.0};
		for (; j < m; j++) {
			cleanup[0] += ai0[j] * x[j * incx];
			cleanup[1] += ai1[j] * x[j * incx];
		}
		alignas(16) double yy0[4];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup[0] + yy0[0] + yy0[1]);
		y[(i + 1) * incy] += alpha * (cleanup[1] + yy0[2] + yy0[3]);
	}
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 8; j += 8) {
			__m128d a0, a1, x0, x1, x2, x3;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
				x2 = _mm_loadu_pd(x + j + 4);
				x3 = _mm_loadu_pd(x + j + 6);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
				x2 = _mm_set_pd(x[(j + 5) * incx], x[(j + 4) * incx]);
				x3 = _mm_set_pd(x[(j + 7) * incx], x[(j + 6) * incx]);
			}

			a0 = _mm_loadu_pd(ai + j);
			a1 = _mm_loadu_pd(ai + j + 2);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x1));

			a0 = _mm_loadu_pd(ai + j + 4);
			a1 = _mm_loadu_pd(ai + j + 6);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x2));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x3));
		}
		double cleanup = 0.0;
		for (; j < m; j++) {
			cleanup += ai[j] * x[j * incx];
		}
		alignas(16) double yy0[4];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup + yy0[0] + yy0[1] + yy0[2] + yy0[3]);
	}
}

void cblas_dgemvCT_strideC4(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		const double* ai0 = a + i * lda;
		const double* ai1 = a + (i + 1) * lda;
		const double* ai2 = a + (i + 2) * lda;
		const double* ai3 = a + (i + 3) * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		__m128d res2 = _mm_setzero_pd();
		__m128d res3 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 2; j += 2) {
			__m128d a0, a1, x0;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
			}

			a0 = _mm_loadu_pd(ai0 + j);
			a1 = _mm_loadu_pd(ai1 + j);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai2 + j);
			a1 = _mm_loadu_pd(ai3 + j);
			res2 = _mm_add_pd(res2, _mm_mul_pd(a0, x0));
			res3 = _mm_add_pd(res3, _mm_mul_pd(a1, x0));
		}
		double cleanup[4] = {0.0, 0.0, 0.0, 0.0};
		for (; j < m; j++) {
			cleanup[0] += ai0[j] * x[j * incx];
			cleanup[1] += ai1[j] * x[j * incx];
			cleanup[2] += ai2[j] * x[j * incx];
			cleanup[3] += ai3[j] * x[j * incx];
		}
		alignas(16) double yy0[8];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		_mm_store_pd(yy0 + 4, res2);
		_mm_store_pd(yy0 + 6, res3);
		y[i * incy] += alpha * (cleanup[0] + yy0[0] + yy0[1]);
		y[(i + 1) * incy] += alpha * (cleanup[1] + yy0[2] + yy0[3]);
		y[(i + 2) * incy] += alpha * (cleanup[2] + yy0[4] + yy0[5]);
		y[(i + 3) * incy] += alpha * (cleanup[3] + yy0[6] + yy0[7]);
	}
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 8; j += 8) {
			__m128d a0, a1, x0, x1, x2, x3;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
				x2 = _mm_loadu_pd(x + j + 4);
				x3 = _mm_loadu_pd(x + j + 6);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
				x2 = _mm_set_pd(x[(j + 5) * incx], x[(j + 4) * incx]);
				x3 = _mm_set_pd(x[(j + 7) * incx], x[(j + 6) * incx]);
			}

			a0 = _mm_loadu_pd(ai + j);
			a1 = _mm_loadu_pd(ai + j + 2);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x1));

			a0 = _mm_loadu_pd(ai + j + 4);
			a1 = _mm_loadu_pd(ai + j + 6);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x2));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x3));
		}
		double cleanup = 0.0;
		for (; j < m; j++) {
			cleanup += ai[j] * x[j * incx];
		}
		alignas(16) double yy0[4];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup + yy0[0] + yy0[1] + yy0[2] + yy0[3]);
	}
}

void cblas_dgemvCT_strideC4R2(size_t m, size_t n, double alpha, const double* a,
							  size_t lda, const double* x, size_t incx,
							  double beta, double* y, size_t incy) {
	swblas::cblas_dscal(n, beta, y, incy);

	size_t i = 0;
	for (; i <= n - 4; i += 4) {
		const double* ai0 = a + i * lda;
		const double* ai1 = a + (i + 1) * lda;
		const double* ai2 = a + (i + 2) * lda;
		const double* ai3 = a + (i + 3) * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		__m128d res2 = _mm_setzero_pd();
		__m128d res3 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 4; j += 4) {
			__m128d a0, a1, a2, a3, x0, x1;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
			}

			a0 = _mm_loadu_pd(ai0 + j);
			a1 = _mm_loadu_pd(ai1 + j);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai0 + j + 2);
			a1 = _mm_loadu_pd(ai1 + j + 2);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x1));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x1));

			a0 = _mm_loadu_pd(ai2 + j);
			a1 = _mm_loadu_pd(ai3 + j);
			res2 = _mm_add_pd(res2, _mm_mul_pd(a0, x0));
			res3 = _mm_add_pd(res3, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai2 + j + 2);
			a1 = _mm_loadu_pd(ai3 + j + 2);
			res2 = _mm_add_pd(res2, _mm_mul_pd(a0, x1));
			res3 = _mm_add_pd(res3, _mm_mul_pd(a1, x1));
		}
		double cleanup[4] = {0.0, 0.0, 0.0, 0.0};
		for (; j < m; j++) {
			cleanup[0] += ai0[j] * x[j * incx];
			cleanup[1] += ai1[j] * x[j * incx];
			cleanup[2] += ai2[j] * x[j * incx];
			cleanup[3] += ai3[j] * x[j * incx];
		}
		alignas(16) double yy0[4];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup[0] + yy0[0] + yy0[1]);
		y[(i + 1) * incy] += alpha * (cleanup[1] + yy0[2] + yy0[3]);
		_mm_store_pd(yy0, res2);
		_mm_store_pd(yy0 + 2, res3);
		y[(i + 2) * incy] += alpha * (cleanup[2] + yy0[0] + yy0[1]);
		y[(i + 3) * incy] += alpha * (cleanup[3] + yy0[2] + yy0[3]);
	}
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 8; j += 8) {
			__m128d a0, a1, x0, x1, x2, x3;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
				x2 = _mm_loadu_pd(x + j + 4);
				x3 = _mm_loadu_pd(x + j + 6);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
				x2 = _mm_set_pd(x[(j + 5) * incx], x[(j + 4) * incx]);
				x3 = _mm_set_pd(x[(j + 7) * incx], x[(j + 6) * incx]);
			}

			a0 = _mm_loadu_pd(ai + j);
			a1 = _mm_loadu_pd(ai + j + 2);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x1));

			a0 = _mm_loadu_pd(ai + j + 4);
			a1 = _mm_loadu_pd(ai + j + 6);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x2));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x3));
		}
		double cleanup = 0.0;
		for (; j < m; j++) {
			cleanup += ai[j] * x[j * incx];
		}
		alignas(16) double yy0[4];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup + yy0[0] + yy0[1] + yy0[2] + yy0[3]);
	}
}

void cblas_dgemvCT_strideC6(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i <= n - 6; i += 6) {
		const double* ai0 = a + i * lda;
		const double* ai1 = a + (i + 1) * lda;
		const double* ai2 = a + (i + 2) * lda;
		const double* ai3 = a + (i + 3) * lda;
		const double* ai4 = a + (i + 4) * lda;
		const double* ai5 = a + (i + 5) * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		__m128d res2 = _mm_setzero_pd();
		__m128d res3 = _mm_setzero_pd();
		__m128d res4 = _mm_setzero_pd();
		__m128d res5 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 2; j += 2) {
			__m128d a0, a1, x0;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
			}

			a0 = _mm_loadu_pd(ai0 + j);
			a1 = _mm_loadu_pd(ai1 + j);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai2 + j);
			a1 = _mm_loadu_pd(ai3 + j);
			res2 = _mm_add_pd(res2, _mm_mul_pd(a0, x0));
			res3 = _mm_add_pd(res3, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai4 + j);
			a1 = _mm_loadu_pd(ai5 + j);
			res4 = _mm_add_pd(res4, _mm_mul_pd(a0, x0));
			res5 = _mm_add_pd(res5, _mm_mul_pd(a1, x0));
		}
		double cleanup[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		for (; j < m; j++) {
			cleanup[0] += ai0[j] * x[j * incx];
			cleanup[1] += ai1[j] * x[j * incx];
			cleanup[2] += ai2[j] * x[j * incx];
			cleanup[3] += ai3[j] * x[j * incx];
			cleanup[4] += ai4[j] * x[j * incx];
			cleanup[5] += ai5[j] * x[j * incx];
		}
		alignas(16) double yy0[4];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup[0] + yy0[0] + yy0[1]);
		y[(i + 1) * incy] += alpha * (cleanup[1] + yy0[2] + yy0[3]);
		_mm_store_pd(yy0, res2);
		_mm_store_pd(yy0 + 2, res3);
		y[(i + 2) * incy] += alpha * (cleanup[2] + yy0[0] + yy0[1]);
		y[(i + 3) * incy] += alpha * (cleanup[3] + yy0[2] + yy0[3]);
		_mm_store_pd(yy0, res4);
		_mm_store_pd(yy0 + 2, res5);
		y[(i + 4) * incy] += alpha * (cleanup[4] + yy0[0] + yy0[1]);
		y[(i + 5) * incy] += alpha * (cleanup[5] + yy0[2] + yy0[3]);
	}
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 8; j += 8) {
			__m128d a0, a1, x0, x1, x2, x3;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
				x2 = _mm_loadu_pd(x + j + 4);
				x3 = _mm_loadu_pd(x + j + 6);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
				x2 = _mm_set_pd(x[(j + 5) * incx], x[(j + 4) * incx]);
				x3 = _mm_set_pd(x[(j + 7) * incx], x[(j + 6) * incx]);
			}

			a0 = _mm_loadu_pd(ai + j);
			a1 = _mm_loadu_pd(ai + j + 2);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x1));

			a0 = _mm_loadu_pd(ai + j + 4);
			a1 = _mm_loadu_pd(ai + j + 6);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x2));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x3));
		}
		double cleanup = 0.0;
		for (; j < m; j++) {
			cleanup += ai[j] * x[j * incx];
		}
		alignas(16) double yy0[4] = {0.0, 0.0, 0.0, 0.0};
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup + yy0[0] + yy0[1] + yy0[2] + yy0[3]);
	}
}

void cblas_dgemvCT_strideC6R2(size_t m, size_t n, double alpha, const double* a,
							  size_t lda, const double* x, size_t incx,
							  double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i <= n - 6; i += 6) {
		const double* ai0 = a + i * lda;
		const double* ai1 = a + (i + 1) * lda;
		const double* ai2 = a + (i + 2) * lda;
		const double* ai3 = a + (i + 3) * lda;
		const double* ai4 = a + (i + 4) * lda;
		const double* ai5 = a + (i + 5) * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		__m128d res2 = _mm_setzero_pd();
		__m128d res3 = _mm_setzero_pd();
		__m128d res4 = _mm_setzero_pd();
		__m128d res5 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 4; j += 4) {
			__m128d a0, a1, x0, x1;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
			}

			a0 = _mm_loadu_pd(ai0 + j);
			a1 = _mm_loadu_pd(ai1 + j);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai0 + j + 2);
			a1 = _mm_loadu_pd(ai1 + j + 2);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x1));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x1));

			a0 = _mm_loadu_pd(ai2 + j);
			a1 = _mm_loadu_pd(ai3 + j);
			res2 = _mm_add_pd(res2, _mm_mul_pd(a0, x0));
			res3 = _mm_add_pd(res3, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai2 + j + 2);
			a1 = _mm_loadu_pd(ai3 + j + 2);
			res2 = _mm_add_pd(res2, _mm_mul_pd(a0, x1));
			res3 = _mm_add_pd(res3, _mm_mul_pd(a1, x1));

			a0 = _mm_loadu_pd(ai4 + j);
			a1 = _mm_loadu_pd(ai5 + j);
			res4 = _mm_add_pd(res4, _mm_mul_pd(a0, x0));
			res5 = _mm_add_pd(res5, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai4 + j + 2);
			a1 = _mm_loadu_pd(ai5 + j + 2);
			res4 = _mm_add_pd(res4, _mm_mul_pd(a0, x1));
			res5 = _mm_add_pd(res5, _mm_mul_pd(a1, x1));
		}
		double cleanup[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		for (; j < m; j++) {
			cleanup[0] += ai0[j] * x[j * incx];
			cleanup[1] += ai1[j] * x[j * incx];
			cleanup[2] += ai2[j] * x[j * incx];
			cleanup[3] += ai3[j] * x[j * incx];
			cleanup[4] += ai4[j] * x[j * incx];
			cleanup[5] += ai5[j] * x[j * incx];
		}
		alignas(16) double yy0[4];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup[0] + yy0[0] + yy0[1]);
		y[(i + 1) * incy] += alpha * (cleanup[1] + yy0[2] + yy0[3]);
		_mm_store_pd(yy0, res2);
		_mm_store_pd(yy0 + 2, res3);
		y[(i + 2) * incy] += alpha * (cleanup[2] + yy0[0] + yy0[1]);
		y[(i + 3) * incy] += alpha * (cleanup[3] + yy0[2] + yy0[3]);
		_mm_store_pd(yy0, res4);
		_mm_store_pd(yy0 + 2, res5);
		y[(i + 4) * incy] += alpha * (cleanup[4] + yy0[0] + yy0[1]);
		y[(i + 5) * incy] += alpha * (cleanup[5] + yy0[2] + yy0[3]);
	}
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 8; j += 8) {
			__m128d a0, a1, x0, x1, x2, x3;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
				x2 = _mm_loadu_pd(x + j + 4);
				x3 = _mm_loadu_pd(x + j + 6);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
				x2 = _mm_set_pd(x[(j + 5) * incx], x[(j + 4) * incx]);
				x3 = _mm_set_pd(x[(j + 7) * incx], x[(j + 6) * incx]);
			}

			a0 = _mm_loadu_pd(ai + j);
			a1 = _mm_loadu_pd(ai + j + 2);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x1));

			a0 = _mm_loadu_pd(ai + j + 4);
			a1 = _mm_loadu_pd(ai + j + 6);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x2));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x3));
		}
		double cleanup = 0.0;
		for (; j < m; j++) {
			cleanup += ai[j] * x[j * incx];
		}
		alignas(16) double yy0[4] = {0.0, 0.0, 0.0, 0.0};
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup + yy0[0] + yy0[1] + yy0[2] + yy0[3]);
	}
}

void cblas_dgemvCT_strideC8(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t i = 0;
	for (; i <= n - 8; i += 8) {
		const double* ai0 = a + i * lda;
		const double* ai1 = a + (i + 1) * lda;
		const double* ai2 = a + (i + 2) * lda;
		const double* ai3 = a + (i + 3) * lda;
		const double* ai4 = a + (i + 4) * lda;
		const double* ai5 = a + (i + 5) * lda;
		const double* ai6 = a + (i + 6) * lda;
		const double* ai7 = a + (i + 7) * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		__m128d res2 = _mm_setzero_pd();
		__m128d res3 = _mm_setzero_pd();
		__m128d res4 = _mm_setzero_pd();
		__m128d res5 = _mm_setzero_pd();
		__m128d res6 = _mm_setzero_pd();
		__m128d res7 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 2; j += 2) {
			__m128d a0, a1, x0;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
			}

			a0 = _mm_loadu_pd(ai0 + j);
			a1 = _mm_loadu_pd(ai1 + j);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai2 + j);
			a1 = _mm_loadu_pd(ai3 + j);
			res2 = _mm_add_pd(res2, _mm_mul_pd(a0, x0));
			res3 = _mm_add_pd(res3, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai4 + j);
			a1 = _mm_loadu_pd(ai5 + j);
			res4 = _mm_add_pd(res4, _mm_mul_pd(a0, x0));
			res5 = _mm_add_pd(res5, _mm_mul_pd(a1, x0));

			a0 = _mm_loadu_pd(ai6 + j);
			a1 = _mm_loadu_pd(ai7 + j);
			res6 = _mm_add_pd(res6, _mm_mul_pd(a0, x0));
			res7 = _mm_add_pd(res7, _mm_mul_pd(a1, x0));
		}
		double cleanup[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
		for (; j < m; j++) {
			cleanup[0] += ai0[j] * x[j * incx];
			cleanup[1] += ai1[j] * x[j * incx];
			cleanup[2] += ai2[j] * x[j * incx];
			cleanup[3] += ai3[j] * x[j * incx];
			cleanup[4] += ai4[j] * x[j * incx];
			cleanup[5] += ai5[j] * x[j * incx];
			cleanup[6] += ai5[j] * x[j * incx];
			cleanup[7] += ai5[j] * x[j * incx];
		}
		alignas(16) double yy0[4];
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup[0] + yy0[0] + yy0[1]);
		y[(i + 1) * incy] += alpha * (cleanup[1] + yy0[2] + yy0[3]);
		_mm_store_pd(yy0, res2);
		_mm_store_pd(yy0 + 2, res3);
		y[(i + 2) * incy] += alpha * (cleanup[2] + yy0[0] + yy0[1]);
		y[(i + 3) * incy] += alpha * (cleanup[3] + yy0[2] + yy0[3]);
		_mm_store_pd(yy0, res4);
		_mm_store_pd(yy0 + 2, res5);
		y[(i + 4) * incy] += alpha * (cleanup[4] + yy0[0] + yy0[1]);
		y[(i + 5) * incy] += alpha * (cleanup[5] + yy0[2] + yy0[3]);

		_mm_store_pd(yy0, res6);
		_mm_store_pd(yy0 + 2, res7);
		y[(i + 6) * incy] += alpha * (cleanup[6] + yy0[0] + yy0[1]);
		y[(i + 7) * incy] += alpha * (cleanup[7] + yy0[2] + yy0[3]);
	}
	for (; i < n; i++) {
		const double* ai = a + i * lda;

		__m128d res0 = _mm_setzero_pd();
		__m128d res1 = _mm_setzero_pd();
		size_t j = 0;
		for (; j <= m - 8; j += 8) {
			__m128d a0, a1, x0, x1, x2, x3;
			if (incx == 1) {
				x0 = _mm_loadu_pd(x + j);
				x1 = _mm_loadu_pd(x + j + 2);
				x2 = _mm_loadu_pd(x + j + 4);
				x3 = _mm_loadu_pd(x + j + 6);
			} else {
				x0 = _mm_set_pd(x[(j + 1) * incx], x[j * incx]);
				x1 = _mm_set_pd(x[(j + 3) * incx], x[(j + 2) * incx]);
				x2 = _mm_set_pd(x[(j + 5) * incx], x[(j + 4) * incx]);
				x3 = _mm_set_pd(x[(j + 7) * incx], x[(j + 6) * incx]);
			}

			a0 = _mm_loadu_pd(ai + j);
			a1 = _mm_loadu_pd(ai + j + 2);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x0));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x1));

			a0 = _mm_loadu_pd(ai + j + 4);
			a1 = _mm_loadu_pd(ai + j + 6);
			res0 = _mm_add_pd(res0, _mm_mul_pd(a0, x2));
			res1 = _mm_add_pd(res1, _mm_mul_pd(a1, x3));
		}
		double cleanup = 0.0;
		for (; j < m; j++) {
			cleanup += ai[j] * x[j * incx];
		}
		alignas(16) double yy0[4] = {0.0, 0.0, 0.0, 0.0};
		_mm_store_pd(yy0, res0);
		_mm_store_pd(yy0 + 2, res1);
		y[i * incy] += alpha * (cleanup + yy0[0] + yy0[1] + yy0[2] + yy0[3]);
	}
}

void cblas_dgemvCT_block32(size_t m, size_t n, double alpha, const double* a,
						   size_t lda, const double* x, size_t incx,
						   double beta, double* y, size_t incy) {
	constexpr size_t L1block = 32;
	constexpr size_t L2block = 512;

	swblas::cblas_dscal(n, beta, y, incy);

	for (size_t ii = 0; ii < m; ii += L2block) {
		size_t iend = std::min(ii + L2block, m);

		for (size_t jj = 0; jj < n; jj += L1block) {
			size_t jend = std::min(jj + L1block, n);

			cblas_dgemvCT_strideC6(iend - ii, jend - jj, alpha,
								   a + ii + jj * lda, lda, x + ii * incx, incx,
								   1.0, y + jj * incy, incy);
		}
	}
}

void cblas_dgemvCT_block64(size_t m, size_t n, double alpha, const double* a,
						   size_t lda, const double* x, size_t incx,
						   double beta, double* y, size_t incy) {
	constexpr size_t L1block = 64;
	constexpr size_t L2block = 512;

	swblas::cblas_dscal(m, beta, y, incy);
	for (size_t ii = 0; ii < m; ii += L2block) {
		size_t iend = std::min(ii + L2block, m);

		for (size_t jj = 0; jj < n; jj += L1block) {
			size_t jend = std::min(jj + L1block, n);

			cblas_dgemvCT_strideC6(iend - ii, jend - jj, alpha,
								   a + ii + jj * lda, lda, x + ii * incx, incx,
								   1.0, y + jj * incy, incy);
		}
	}
}

void cblas_dgemvCT_block128(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	constexpr size_t L1block = 128;
	constexpr size_t L2block = 512;

	swblas::cblas_dscal(m, beta, y, incy);

	for (size_t ii = 0; ii < m; ii += L2block) {
		size_t iend = std::min(ii + L2block, m);

		for (size_t jj = 0; jj < n; jj += L1block) {
			size_t jend = std::min(jj + L1block, n);

			cblas_dgemvCT_strideC6(iend - ii, jend - jj, alpha,
								   a + ii + jj * lda, lda, x + ii * incx, incx,
								   1.0, y + jj * incy, incy);
		}
	}
}

void cblas_dgemvCT_block256(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	constexpr size_t L1block = 256;
	constexpr size_t L2block = 512;

	swblas::cblas_dscal(m, beta, y, incy);

	for (size_t ii = 0; ii < m; ii += L2block) {
		size_t iend = std::min(ii + L2block, m);

		for (size_t jj = 0; jj < n; jj += L1block) {
			size_t jend = std::min(jj + L1block, n);

			cblas_dgemvCT_strideC6(iend - ii, jend - jj, alpha,
								   a + ii + jj * lda, lda, x + ii * incx, incx,
								   1.0, y + jj * incy, incy);
		}
	}
}

void cblas_dgemvCT_block512(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	constexpr size_t L1block = 512;
	constexpr size_t L2block = 512;

	swblas::cblas_dscal(m, beta, y, incy);

	for (size_t ii = 0; ii < m; ii += L2block) {
		size_t iend = std::min(ii + L2block, m);

		for (size_t jj = 0; jj < n; jj += L1block) {
			size_t jend = std::min(jj + L1block, n);

			cblas_dgemvCT_strideC6(iend - ii, jend - jj, alpha,
								   a + ii + jj * lda, lda, x + ii * incx, incx,
								   1.0, y + jj * incy, incy);
		}
	}
}

namespace swblas {

namespace impl {

void cblas_dgemvCT(size_t m, size_t n, double alpha, const double* a,
				   size_t lda, const double* x, size_t incx, double beta,
				   double* y, size_t incy) {
	char mode = 0;
	if (incx == 1 && incy == 1) {
		mode = 1;
	} else if (incx == 1) {
		mode = 2;
	} else if (incy == 1) {
		mode = 3;
	} else {
		mode = 4;
	}
	if (m <= 4 or n <= 4) {
		mode = 5;
	}

	switch (mode) {
		case 1:
		case 2:
		case 3:
		case 4:
			cblas_dgemvCT_strideC6R2(m, n, alpha, a, lda, x, incx, beta, y,
									 incy);
			break;
		case 5:
			cblas_dgemvCT_naive(m, n, alpha, a, lda, x, incx, beta, y, incy);
			break;
	}
}

}  // namespace impl

}  // namespace swblas
