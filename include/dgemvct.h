#include "swblas.h"

void cblas_dgemvCT_naive(size_t m, size_t n, double alpha, const double* a,
						 size_t lda, const double* x, size_t incx, double beta,
						 double* y, size_t incy);

void cblas_dgemvCT_unrollinner2(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy);

void cblas_dgemvCT_unrollinner4(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy);

void cblas_dgemvCT_unrollinner8(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy);

void cblas_dgemvCT_sseinner(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t /*incx*/,
							double beta, double* y, size_t /*incy*/);

void cblas_dgemvCT_sseinner2(size_t m, size_t n, double alpha, const double* a,
							 size_t lda, const double* x, size_t /*incx*/,
							 double beta, double* y, size_t /*incy*/);

void cblas_dgemvCT_sseinner4(size_t m, size_t n, double alpha, const double* a,
							 size_t lda, const double* x, size_t /*incx*/,
							 double beta, double* y, size_t /*incy*/);

void cblas_dgemvCT_strideC4(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);

void cblas_dgemvCT_strideC4R2(size_t m, size_t n, double alpha, const double* a,
							  size_t lda, const double* x, size_t incx,
							  double beta, double* y, size_t incy);

void cblas_dgemvCT_strideC6(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);

void cblas_dgemvCT_strideC6R2(size_t m, size_t n, double alpha, const double* a,
							  size_t lda, const double* x, size_t incx,
							  double beta, double* y, size_t incy);

void cblas_dgemvCT_strideC8(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);

void cblas_dgemvCT_block32(size_t m, size_t n, double alpha, const double* a,
						   size_t lda, const double* x, size_t incx,
						   double beta, double* y, size_t incy);

void cblas_dgemvCT_block64(size_t m, size_t n, double alpha, const double* a,
						   size_t lda, const double* x, size_t incx,
						   double beta, double* y, size_t incy);

void cblas_dgemvCT_block128(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);

void cblas_dgemvCT_block256(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);

void cblas_dgemvCT_block512(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);
