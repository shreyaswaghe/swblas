#include <sys/_types/_u_int8_t.h>

#include <algorithm>
#include <cstddef>

#include "../include/dgemvcn.h"
#include "../include/swblas.h"
#include "sse2neon.h"

void cblas_dgemv(char fmt, char trans, size_t m, size_t n, double alpha,
				 const double* a, size_t lda, const double* x, size_t incx,
				 double beta, double* y, size_t incy) {}

void cblas_dgemvCN_naive(size_t m, size_t n, double alpha, const double* a,
						 size_t lda, const double* x, size_t incx, double beta,
						 double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);
	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		size_t i = 0;
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_unrollinner2(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);
	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		size_t i = 0;
		for (; i < (m & -2); i += 2) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;

			yi[0] += alpha * aij[0] * xj;
			yi[1 * incy] += alpha * aij[1] * xj;
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

#define PREFETCH_DISTANCE 128

void cblas_dgemvCN_unr2_prefetch(size_t m, size_t n, double alpha,
								 const double* a, size_t lda, const double* x,
								 size_t incx, double beta, double* y,
								 size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);
	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		size_t i = 0;
		for (; i < (m & -2); i += 2) {
			if (i + PREFETCH_DISTANCE < m && incy <= 4) {
				__builtin_prefetch(a0j + PREFETCH_DISTANCE, 0, 3);
				__builtin_prefetch(y + PREFETCH_DISTANCE * incy, 1, 3);
			}
			const double* aij = a0j + i;
			double* yi = y + i * incy;

			yi[0] += alpha * aij[0] * xj;
			yi[1 * incy] += alpha * aij[1] * xj;
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_unrollinner4(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		size_t i = 0;
		for (; i < (m & -4); i += 4) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;

			yi[0] += alpha * aij[0] * xj;
			yi[1 * incy] += alpha * aij[1] * xj;
			yi[2 * incy] += alpha * aij[2] * xj;
			yi[3 * incy] += alpha * aij[3] * xj;
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_unrollinner8(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		size_t i = 0;
		for (; i < (m & -8); i += 8) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;

			yi[0] += alpha * aij[0] * xj;
			yi[1 * incy] += alpha * aij[1] * xj;
			yi[2 * incy] += alpha * aij[2] * xj;
			yi[3 * incy] += alpha * aij[3] * xj;
			yi[4 * incy] += alpha * aij[4] * xj;
			yi[5 * incy] += alpha * aij[5] * xj;
			yi[6 * incy] += alpha * aij[6] * xj;
			yi[7 * incy] += alpha * aij[7] * xj;
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_sseinner(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t /*incx*/,
							double beta, double* y, size_t /*incy*/) {
	size_t incx = 1, incy = 1;

	swblas::cblas_dscal(m, beta, y, incy);

	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		__m128d alph = _mm_set1_pd(alpha);
		__m128d axj = _mm_mul_pd(alph, _mm_set1_pd(xj));

		size_t i = 0;
		for (; i < (m & -2); i += 2) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;

			__m128d aa = _mm_load_pd(aij);
			__m128d yy = _mm_load_pd(yi);

			_mm_store_pd(yi, _mm_add_pd(yy, _mm_mul_pd(aa, axj)));
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_sseinner_nodscal(size_t m, size_t n, double alpha,
									const double* a, size_t lda,
									const double* x, size_t /*incx*/,
									double beta, double* y, size_t /*incy*/) {
	size_t incx = 1, incy = 1;
	size_t j = 0;

	swblas::cblas_dscal(m, beta, y, incy);
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		__m128d axj = _mm_set1_pd(alpha * xj);
		size_t i = 0;
		for (; i < (m & -2); i += 2) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;

			__m128d aa = _mm_load_pd(aij);
			__m128d yy = _mm_load_pd(yi);
			_mm_store_pd(yi, _mm_add_pd(yy, _mm_mul_pd(aa, axj)));
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_sseinner2(size_t m, size_t n, double alpha, const double* a,
							 size_t lda, const double* x, size_t /*incx*/,
							 double beta, double* y, size_t /*incy*/) {
	size_t incx = 1, incy = 1;

	swblas::cblas_dscal(m, beta, y, incy);

	x = (const double*)__builtin_assume_aligned(x, 16);
	y = (double*)__builtin_assume_aligned(y, 16);

	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		__m128d axj = _mm_set1_pd(alpha * xj);

		size_t i = 0;
		for (; i < (m & -4); i += 4) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;

			__m128d aa1 = _mm_load_pd(aij);
			__m128d yy1 = _mm_load_pd(yi);
			_mm_store_pd(yi, _mm_add_pd(yy1, _mm_mul_pd(aa1, axj)));

			__m128d aa2 = _mm_load_pd(aij + 2);
			__m128d yy2 = _mm_load_pd(yi + 2);
			_mm_store_pd(yi + 2, _mm_add_pd(yy2, _mm_mul_pd(aa2, axj)));
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_sseinner4(size_t m, size_t n, double alpha, const double* a,
							 size_t lda, const double* x, size_t /*incx*/,
							 double beta, double* y, size_t /*incy*/) {
	size_t incx = 1, incy = 1;

	swblas::cblas_dscal(m, beta, y, incy);

	x = (const double*)__builtin_assume_aligned(x, 16);
	y = (double*)__builtin_assume_aligned(y, 16);

	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		__m128d axj = _mm_set1_pd(alpha * xj);

		size_t i = 0;
		for (; i < (m & -8); i += 8) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;

			__m128d aa1 = _mm_load_pd(aij);
			__m128d yy1 = _mm_load_pd(yi);
			_mm_store_pd(yi, _mm_add_pd(yy1, _mm_mul_pd(aa1, axj)));

			__m128d aa2 = _mm_load_pd(aij + 2);
			__m128d yy2 = _mm_load_pd(yi + 2);
			_mm_store_pd(yi + 2, _mm_add_pd(yy2, _mm_mul_pd(aa2, axj)));

			__m128d aa3 = _mm_load_pd(aij + 4);
			__m128d yy3 = _mm_load_pd(yi + 4);
			_mm_store_pd(yi + 4, _mm_add_pd(yy3, _mm_mul_pd(aa3, axj)));

			__m128d aa4 = _mm_load_pd(aij + 6);
			__m128d yy4 = _mm_load_pd(yi + 6);
			_mm_store_pd(yi + 6, _mm_add_pd(yy4, _mm_mul_pd(aa4, axj)));
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_sseinner2_nodscal(size_t m, size_t n, double alpha,
									 const double* a, size_t lda,
									 const double* x, size_t /*incx*/,
									 double beta, double* y, size_t /*incy*/) {
	size_t incx = 1, incy = 1;

	size_t j = 0;

	__m128d bet = _mm_set1_pd(beta);
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		__m128d axj = _mm_set1_pd(alpha * xj);

		size_t i = 0;
		for (; i < (m & -4); i += 4) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;

			__m128d aa1 = _mm_load_pd(aij);
			__m128d yy1 = _mm_load_pd(yi);
			__m128d byy1 = _mm_mul_pd(yy1, bet);
			_mm_store_pd(yi, _mm_add_pd(byy1, _mm_mul_pd(aa1, axj)));

			__m128d aa2 = _mm_load_pd(aij + 2);
			__m128d yy2 = _mm_load_pd(yi + 2);
			__m128d byy2 = _mm_mul_pd(yy2, bet);
			_mm_store_pd(yi + 2, _mm_add_pd(byy2, _mm_mul_pd(aa2, axj)));
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_sseinner_incx1(size_t m, size_t n, double alpha,
								  const double* a, size_t lda, const double* x,
								  size_t /*incx*/, double beta, double* y,
								  size_t incy) {
	size_t incx = 1;
	swblas::cblas_dscal(m, beta, y, incy);

	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		__m128d alph = _mm_set1_pd(alpha);
		__m128d axj = _mm_mul_pd(alph, _mm_set1_pd(xj));

		size_t i = 0;
		for (; i < (m & -2); i += 2) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;

			__m128d aa = _mm_load_pd(aij);
			__m128d yy = _mm_setr_pd(yi[0], yi[incy]);

			alignas(16) double res[2];
			_mm_store_pd(res, _mm_add_pd(yy, _mm_mul_pd(aa, axj)));
			yi[incy] = res[1];
			yi[0] = res[0];
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_sseinner2_incx1(size_t m, size_t n, double alpha,
								   const double* a, size_t lda, const double* x,
								   size_t /*incx*/, double beta, double* y,
								   size_t incy) {
	size_t incx = 1;

	swblas::cblas_dscal(m, beta, y, incy);

	x = (const double*)__builtin_assume_aligned(x, 16);
	y = (double*)__builtin_assume_aligned(y, 16);

	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		__m128d axj = _mm_set1_pd(alpha * xj);

		size_t i = 0;
		for (; i < (m & -4); i += 4) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;
			alignas(16) double res[4];

			__m128d aa1 = _mm_load_pd(aij);
			__m128d yy1 = _mm_setr_pd(yi[0], yi[incy]);
			_mm_store_pd(res, _mm_add_pd(yy1, _mm_mul_pd(aa1, axj)));

			__m128d aa2 = _mm_load_pd(aij + 2);
			__m128d yy2 = _mm_setr_pd(yi[2 * incy], yi[3 * incy]);
			_mm_store_pd(res + 2, _mm_add_pd(yy2, _mm_mul_pd(aa2, axj)));

			yi[0] = res[0];
			yi[incy] = res[1];
			yi[2 * incy] = res[2];
			yi[3 * incy] = res[3];
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_sseinner4_incx1(size_t m, size_t n, double alpha,
								   const double* a, size_t lda, const double* x,
								   size_t /*incx*/, double beta, double* y,
								   size_t incy) {
	size_t incx = 1;

	swblas::cblas_dscal(m, beta, y, incy);

	x = (const double*)__builtin_assume_aligned(x, 16);
	y = (double*)__builtin_assume_aligned(y, 16);

	size_t j = 0;
	for (; j < n; j++) {
		size_t idx = j * lda;
		const double* a0j = a + idx;
		const double xj = x[j * incx];

		__m128d axj = _mm_set1_pd(alpha * xj);

		size_t i = 0;
		for (; i < (m & -8); i += 8) {
			const double* aij = a0j + i;
			double* yi = y + i * incy;
			alignas(16) double res[8];

			__m128d aa1 = _mm_load_pd(aij);
			__m128d yy1 = _mm_setr_pd(yi[0], yi[incy]);
			_mm_store_pd(res, _mm_add_pd(yy1, _mm_mul_pd(aa1, axj)));

			__m128d aa2 = _mm_load_pd(aij + 2);
			__m128d yy2 = _mm_setr_pd(yi[2 * incy], yi[3 * incy]);
			_mm_store_pd(res + 2, _mm_add_pd(yy2, _mm_mul_pd(aa2, axj)));

			__m128d aa3 = _mm_load_pd(aij + 4);
			__m128d yy3 = _mm_setr_pd(yi[4 * incy], yi[5 * incy]);
			_mm_store_pd(res + 4, _mm_add_pd(yy3, _mm_mul_pd(aa3, axj)));

			__m128d aa4 = _mm_load_pd(aij + 6);
			__m128d yy4 = _mm_setr_pd(yi[6 * incy], yi[7 * incy]);
			_mm_store_pd(res + 6, _mm_add_pd(yy4, _mm_mul_pd(aa4, axj)));

			yi[0] = res[0];
			yi[incy] = res[1];
			yi[2 * incy] = res[2];
			yi[3 * incy] = res[3];
			yi[4 * incy] = res[4];
			yi[5 * incy] = res[5];
			yi[6 * incy] = res[6];
			yi[7 * incy] = res[7];
		}
		for (; i < m; i++) {
			const double aij = a0j[i];
			y[i * incy] += alpha * aij * xj;
		}
	}
}

void cblas_dgemvCN_strideC4(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t j = 0;
	constexpr size_t jjump = 4;
	for (; j < (n & -jjump); j += jjump) {
		const double* xj = x + j * incx;
		const double* ai0j0 = a + j * lda;
		const double* ai0j1 = a + (j + 1) * lda;
		const double* ai0j2 = a + (j + 2) * lda;
		const double* ai0j3 = a + (j + 3) * lda;

		__m128d x0 = _mm_set1_pd(alpha * xj[0]);
		__m128d x1 = _mm_set1_pd(alpha * xj[incx]);
		__m128d x2 = _mm_set1_pd(alpha * xj[2 * incx]);
		__m128d x3 = _mm_set1_pd(alpha * xj[3 * incx]);

		size_t i = 0;
		for (; i < (m & -2); i += 2) {
			__m128d a00 = _mm_loadu_pd(ai0j0 + i);
			__m128d a01 = _mm_loadu_pd(ai0j1 + i);
			__m128d a02 = _mm_loadu_pd(ai0j2 + i);
			__m128d a03 = _mm_loadu_pd(ai0j3 + i);

			a00 = _mm_mul_pd(x0, a00);
			a01 = _mm_mul_pd(x1, a01);
			a02 = _mm_mul_pd(x2, a02);
			a03 = _mm_mul_pd(x3, a03);

			a00 = _mm_add_pd(a00, a01);
			a02 = _mm_add_pd(a02, a03);
			a00 = _mm_add_pd(a00, a02);

			alignas(16) double yy[2];
			_mm_store_pd(yy, a00);
			y[i * incy] += yy[0];
			y[(i + 1) * incy] += yy[1];
		}
		for (; i < m; i++) {
			y[i * incy] += alpha * ai0j0[i] * xj[0];
			y[i * incy] += alpha * ai0j1[i] * xj[1 * incx];
			y[i * incy] += alpha * ai0j2[i] * xj[2 * incx];
			y[i * incy] += alpha * ai0j3[i] * xj[3 * incx];
		}
	}
	for (; j < n; j++) {
		const double* xj = x + j * incx;
		const double* ai0j0 = a + j * lda;

		__m128d x0 = _mm_set1_pd(alpha * xj[0]);
		size_t i = 0;
		for (; i < (m & -2); i += 2) {
			__m128d a00 = _mm_loadu_pd(ai0j0 + i);
			a00 = _mm_mul_pd(x0, a00);

			alignas(16) double yy[2];
			_mm_store_pd(yy, a00);
			y[i * incy] += yy[0];
			y[(i + 1) * incy] += yy[1];
		}
		for (; i < m; i++) {
			y[i * incy] += alpha * ai0j0[i] * xj[0];
		}
	}
}

void cblas_dgemvCN_strideC6(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);

	size_t j = 0;
	constexpr size_t jjump = 6;
	for (; j <= n - jjump; j += jjump) {
		const double* xj = x + j * incx;
		const double* ai0j0 = a + j * lda;
		const double* ai0j1 = a + (j + 1) * lda;
		const double* ai0j2 = a + (j + 2) * lda;
		const double* ai0j3 = a + (j + 3) * lda;
		const double* ai0j4 = a + (j + 4) * lda;
		const double* ai0j5 = a + (j + 5) * lda;

		__m128d x0 = _mm_set1_pd(alpha * xj[0]);
		__m128d x1 = _mm_set1_pd(alpha * xj[incx]);
		__m128d x2 = _mm_set1_pd(alpha * xj[2 * incx]);
		__m128d x3 = _mm_set1_pd(alpha * xj[3 * incx]);
		__m128d x4 = _mm_set1_pd(alpha * xj[4 * incx]);
		__m128d x5 = _mm_set1_pd(alpha * xj[5 * incx]);

		size_t i = 0;
		for (; i < (m & -2); i += 2) {
			__m128d a00 = _mm_loadu_pd(ai0j0 + i);
			__m128d a01 = _mm_loadu_pd(ai0j1 + i);
			__m128d a02 = _mm_loadu_pd(ai0j2 + i);
			__m128d a03 = _mm_loadu_pd(ai0j3 + i);
			__m128d a04 = _mm_loadu_pd(ai0j4 + i);
			__m128d a05 = _mm_loadu_pd(ai0j5 + i);

			a00 = _mm_mul_pd(x0, a00);
			a01 = _mm_mul_pd(x1, a01);
			a02 = _mm_mul_pd(x2, a02);
			a03 = _mm_mul_pd(x3, a03);
			a04 = _mm_mul_pd(x4, a04);
			a05 = _mm_mul_pd(x5, a05);

			a00 = _mm_add_pd(a00, a01);
			a02 = _mm_add_pd(a02, a03);
			a04 = _mm_add_pd(a04, a05);
			a00 = _mm_add_pd(a00, a02);
			a00 = _mm_add_pd(a00, a04);

			alignas(16) double yy[2];
			_mm_store_pd(yy, a00);
			y[i * incy] += yy[0];
			y[(i + 1) * incy] += yy[1];
		}
		for (; i < m; i++) {
			y[i * incy] += alpha * ai0j0[i] * xj[0];
			y[i * incy] += alpha * ai0j1[i] * xj[1 * incx];
			y[i * incy] += alpha * ai0j2[i] * xj[2 * incx];
			y[i * incy] += alpha * ai0j3[i] * xj[3 * incx];
			y[i * incy] += alpha * ai0j4[i] * xj[4 * incx];
			y[i * incy] += alpha * ai0j5[i] * xj[5 * incx];
		}
	}
	for (; j < n; j++) {
		const double* xj = x + j * incx;
		const double* ai0j0 = a + j * lda;

		__m128d x0 = _mm_set1_pd(alpha * xj[0]);

		size_t i = 0;
		for (; i < (m & -2); i += 2) {
			__m128d a00 = _mm_loadu_pd(ai0j0 + i);
			a00 = _mm_mul_pd(x0, a00);

			alignas(16) double yy[2];
			_mm_store_pd(yy, a00);
			y[i * incy] += yy[0];
			y[(i + 1) * incy] += yy[1];
		}
		for (; i < m; i++) {
			y[i * incy] += alpha * ai0j0[i] * xj[0];
		}
	}
}

void cblas_dgemvCN_block32(size_t m, size_t n, double alpha, const double* a,
						   size_t lda, const double* x, size_t incx,
						   double beta, double* y, size_t incy) {
	constexpr size_t L1block = 32;
	constexpr size_t L2block = 512;

	swblas::cblas_dscal(m, beta, y, incy);
	for (size_t jj = 0; jj < n; jj += L2block) {
		size_t jend = std::min(jj + L2block, n);

		for (size_t ii = 0; ii < m; ii += L1block) {
			size_t iend = std::min(ii + L1block, m);

			cblas_dgemvCN_strideC8(iend - ii, jend - jj, alpha,
								   a + ii + jj * lda, lda, x + jj * incx, incx,
								   1.0, y + ii * incy, incy);
		}
	}
}

void cblas_dgemvCN_strideC8(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	swblas::cblas_dscal(m, beta, y, incy);
	size_t j = 0;
	constexpr size_t jjump = 6;
	for (; j <= n - jjump; j += jjump) {
		const double* xj = x + j * incx;
		const double* ai0j0 = a + j * lda;
		const double* ai0j1 = a + (j + 1) * lda;
		const double* ai0j2 = a + (j + 2) * lda;
		const double* ai0j3 = a + (j + 3) * lda;
		const double* ai0j4 = a + (j + 4) * lda;
		const double* ai0j5 = a + (j + 5) * lda;
		__m128d x0 = _mm_set1_pd(alpha * xj[0]);
		__m128d x1 = _mm_set1_pd(alpha * xj[incx]);
		__m128d x2 = _mm_set1_pd(alpha * xj[2 * incx]);
		__m128d x3 = _mm_set1_pd(alpha * xj[3 * incx]);
		__m128d x4 = _mm_set1_pd(alpha * xj[4 * incx]);
		__m128d x5 = _mm_set1_pd(alpha * xj[5 * incx]);
		size_t i = 0;
		for (; i < (m & -4); i += 4) {
			__m128d y0, y1;
			if (incy == 1) {
				y0 = _mm_loadu_pd(y + i);
				y1 = _mm_loadu_pd(y + i + 2);
			} else {
				y0 = _mm_set_pd(y[(i + 1) * incy], y[i * incy]);
				y1 = _mm_set_pd(y[(i + 3) * incy], y[(i + 2) * incy]);
			}
			__m128d a0 = _mm_loadu_pd(ai0j0 + i);
			__m128d a1 = _mm_loadu_pd(ai0j0 + i + 2);
			y0 = _mm_add_pd(y0, _mm_mul_pd(x0, a0));
			y1 = _mm_add_pd(y1, _mm_mul_pd(x0, a1));

			a0 = _mm_loadu_pd(ai0j1 + i);
			a1 = _mm_loadu_pd(ai0j1 + i + 2);
			y0 = _mm_add_pd(y0, _mm_mul_pd(x1, a0));
			y1 = _mm_add_pd(y1, _mm_mul_pd(x1, a1));

			a0 = _mm_loadu_pd(ai0j2 + i);
			a1 = _mm_loadu_pd(ai0j2 + i + 2);
			y0 = _mm_add_pd(y0, _mm_mul_pd(x2, a0));
			y1 = _mm_add_pd(y1, _mm_mul_pd(x2, a1));

			a0 = _mm_loadu_pd(ai0j3 + i);
			a1 = _mm_loadu_pd(ai0j3 + i + 2);
			y0 = _mm_add_pd(y0, _mm_mul_pd(x3, a0));
			y1 = _mm_add_pd(y1, _mm_mul_pd(x3, a1));

			a0 = _mm_loadu_pd(ai0j4 + i);
			a1 = _mm_loadu_pd(ai0j4 + i + 2);
			y0 = _mm_add_pd(y0, _mm_mul_pd(x4, a0));
			y1 = _mm_add_pd(y1, _mm_mul_pd(x4, a1));

			a0 = _mm_loadu_pd(ai0j5 + i);
			a1 = _mm_loadu_pd(ai0j5 + i + 2);
			y0 = _mm_add_pd(y0, _mm_mul_pd(x5, a0));
			y1 = _mm_add_pd(y1, _mm_mul_pd(x5, a1));

			if (incy == 1) {
				_mm_storeu_pd(y + i, y0);
				_mm_storeu_pd(y + i + 2, y1);
			} else {
				alignas(16) double yy[4];
				_mm_store_pd(yy, y0);
				_mm_store_pd(yy + 2, y1);
				y[i * incy] = yy[0];
				y[(i + 1) * incy] = yy[1];
				y[(i + 2) * incy] = yy[2];
				y[(i + 3) * incy] = yy[3];
			}
		}
		for (; i < m; i++) {
			y[i * incy] +=
				alpha * (ai0j0[i] * xj[0] + ai0j1[i] * xj[incx] +
						 ai0j2[i] * xj[2 * incx] + ai0j3[i] * xj[3 * incx] +
						 ai0j4[i] * xj[4 * incx] + ai0j5[i] * xj[5 * incx]);
		}
	}
	for (; j < n; j++) {
		const double* xj = x + j * incx;
		const double* ai0j0 = a + j * lda;
		__m128d x0 = _mm_set1_pd(alpha * xj[0]);
		size_t i = 0;
		for (; i < (m & -4); i += 4) {
			__m128d y0, y1, a0, a1;
			if (incy == 1) {
				y0 = _mm_loadu_pd(y + i);
				y1 = _mm_loadu_pd(y + i + 2);
			} else {
				y0 = _mm_set_pd(y[(i + 1) * incy], y[i * incy]);
				y1 = _mm_set_pd(y[(i + 3) * incy], y[(i + 2) * incy]);
			}
			a0 = _mm_loadu_pd(ai0j0 + i);
			a1 = _mm_loadu_pd(ai0j0 + i + 2);
			y0 = _mm_add_pd(y0, _mm_mul_pd(x0, a0));
			y1 = _mm_add_pd(y1, _mm_mul_pd(x0, a1));

			if (incy == 1) {
				_mm_storeu_pd(y + i, y0);
				_mm_storeu_pd(y + i + 2, y1);
			} else {
				alignas(16) double yy[4];
				_mm_store_pd(yy, y0);
				_mm_store_pd(yy + 2, y1);
				y[i * incy] = yy[0];
				y[(i + 1) * incy] = yy[1];
				y[(i + 2) * incy] = yy[2];
				y[(i + 3) * incy] = yy[3];
			}
		}
		for (; i < m; i++) {
			y[i * incy] += alpha * ai0j0[i] * xj[0];
		}
	}
}

void cblas_dgemvCN_block64(size_t m, size_t n, double alpha, const double* a,
						   size_t lda, const double* x, size_t incx,
						   double beta, double* y, size_t incy) {
	constexpr size_t L1block = 64;
	constexpr size_t L2block = 512;

	swblas::cblas_dscal(m, beta, y, incy);
	for (size_t jj = 0; jj < n; jj += L2block) {
		size_t jend = std::min(jj + L2block, n);

		for (size_t ii = 0; ii < m; ii += L1block) {
			size_t iend = std::min(ii + L1block, m);

			cblas_dgemvCN_strideC8(iend - ii, jend - jj, alpha,
								   a + ii + jj * lda, lda, x + jj * incx, incx,
								   1.0, y + ii * incy, incy);
		}
	}
}

void cblas_dgemvCN_block64_v2(size_t m, size_t n, double alpha, const double* a,
							  size_t lda, const double* x, size_t incx,
							  double beta, double* y, size_t incy) {
	constexpr size_t L1block = 64;
	constexpr size_t L2block = 1024;

	swblas::cblas_dscal(m, beta, y, incy);
	for (size_t jj = 0; jj < n; jj += L2block) {
		size_t jend = std::min(jj + L2block, n);

		for (size_t ii = 0; ii < m; ii += L1block) {
			size_t iend = std::min(ii + L1block, m);

			cblas_dgemvCN_strideC8(iend - ii, jend - jj, alpha,
								   a + ii + jj * lda, lda, x + jj * incx, incx,
								   1.0, y + ii * incy, incy);
		}
	}
}

void cblas_dgemvCN_block128(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	constexpr size_t L1block = 128;
	constexpr size_t L2block = 512;

	swblas::cblas_dscal(m, beta, y, incy);
	for (size_t jj = 0; jj < n; jj += L2block) {
		size_t jend = std::min(jj + L2block, n);

		for (size_t ii = 0; ii < m; ii += L1block) {
			size_t iend = std::min(ii + L1block, m);

			cblas_dgemvCN_strideC8(iend - ii, jend - jj, alpha,
								   a + ii + jj * lda, lda, x + jj * incx, incx,
								   1.0, y + ii * incy, incy);
		}
	}
}

void cblas_dgemvCN_block256(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	constexpr size_t L1block = 256;
	constexpr size_t L2block = 512;

	swblas::cblas_dscal(m, beta, y, incy);
	for (size_t jj = 0; jj < n; jj += L2block) {
		size_t jend = std::min(jj + L2block, n);

		for (size_t ii = 0; ii < m; ii += L1block) {
			size_t iend = std::min(ii + L1block, m);

			cblas_dgemvCN_strideC8(iend - ii, jend - jj, alpha,
								   a + ii + jj * lda, lda, x + jj * incx, incx,
								   1.0, y + ii * incy, incy);
		}
	}
}

void cblas_dgemvCN_block512(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy) {
	constexpr size_t L1block = 512;
	constexpr size_t L2block = 512;

	swblas::cblas_dscal(m, beta, y, incy);
	for (size_t jj = 0; jj < n; jj += L2block) {
		size_t jend = std::min(jj + L2block, n);

		for (size_t ii = 0; ii < m; ii += L1block) {
			size_t iend = std::min(ii + L1block, m);

			cblas_dgemvCN_naive(iend - ii, jend - jj, alpha, a + ii + jj * lda,
								lda, x + jj * incx, incx, 1.0, y + ii * incy,
								incy);
		}
	}
}

namespace swblas {

namespace impl {

void cblas_dgemvCN(size_t m, size_t n, double alpha, const double* a,
				   size_t lda, const double* x, size_t incx, double beta,
				   double* y, size_t incy) {
	constexpr size_t MAX_SINGLE_SIZE = 2048;
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

	switch (mode) {
		case 1:
			if (m * n <= MAX_SINGLE_SIZE * MAX_SINGLE_SIZE) {
				cblas_dgemvCN_block256(m, n, alpha, a, lda, x, incx, beta, y,
									   incy);
			} else {
				cblas_dgemvCN_strideC8(m, n, alpha, a, lda, x, incx, beta, y,
									   incy);
			}
			break;
		default:
			cblas_dgemvCN_strideC8(m, n, alpha, a, lda, x, incx, beta, y, incy);
			break;
	}
}

}  // namespace impl

}  // namespace swblas
