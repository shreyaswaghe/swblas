#include "swblas.h"

void cblas_dgemvCN_naive(size_t m, size_t n, double alpha, const double* a,
						 size_t lda, const double* x, size_t incx, double beta,
						 double* y, size_t incy);

void cblas_dgemvCN_unrollinner2(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy);

void cblas_dgemvCN_unr2_prefetch(size_t m, size_t n, double alpha,
								 const double* a, size_t lda, const double* x,
								 size_t incx, double beta, double* y,
								 size_t incy);

void cblas_dgemvCN_unrollinner4(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy);

void cblas_dgemvCN_unrollinner8(size_t m, size_t n, double alpha,
								const double* a, size_t lda, const double* x,
								size_t incx, double beta, double* y,
								size_t incy);

void cblas_dgemvCN_sseinner(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t /*incx*/,
							double beta, double* y, size_t /*incy*/);

void cblas_dgemvCN_sseinner_nodscal(size_t m, size_t n, double alpha,
									const double* a, size_t lda,
									const double* x, size_t /*incx*/,
									double beta, double* y, size_t /*incy*/);

void cblas_dgemvCN_sseinner2(size_t m, size_t n, double alpha, const double* a,
							 size_t lda, const double* x, size_t /*incx*/,
							 double beta, double* y, size_t /*incy*/);

void cblas_dgemvCN_sseinner2_nodscal(size_t m, size_t n, double alpha,
									 const double* a, size_t lda,
									 const double* x, size_t /*incx*/,
									 double beta, double* y, size_t /*incy*/);

void cblas_dgemvCN_sseinner4(size_t m, size_t n, double alpha, const double* a,
							 size_t lda, const double* x, size_t /*incx*/,
							 double beta, double* y, size_t /*incy*/);

void cblas_dgemvCN_sseinner_incx1(size_t m, size_t n, double alpha,
								  const double* a, size_t lda, const double* x,
								  size_t /*incx*/, double beta, double* y,
								  size_t incy);

void cblas_dgemvCN_sseinner2_incx1(size_t m, size_t n, double alpha,
								   const double* a, size_t lda, const double* x,
								   size_t /*incx*/, double beta, double* y,
								   size_t incy);

void cblas_dgemvCN_sseinner4_incx1(size_t m, size_t n, double alpha,
								   const double* a, size_t lda, const double* x,
								   size_t /*incx*/, double beta, double* y,
								   size_t incy);

void cblas_dgemvCN_strideC4(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);

void cblas_dgemvCN_strideC8(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);

void cblas_dgemvCN_block32(size_t m, size_t n, double alpha, const double* a,
						   size_t lda, const double* x, size_t incx,
						   double beta, double* y, size_t incy);

void cblas_dgemvCN_block64(size_t m, size_t n, double alpha, const double* a,
						   size_t lda, const double* x, size_t incx,
						   double beta, double* y, size_t incy);

void cblas_dgemvCN_block64_v2(size_t m, size_t n, double alpha, const double* a,
							  size_t lda, const double* x, size_t incx,
							  double beta, double* y, size_t incy);

void cblas_dgemvCN_block128(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);

void cblas_dgemvCN_block256(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);

void cblas_dgemvCN_block512(size_t m, size_t n, double alpha, const double* a,
							size_t lda, const double* x, size_t incx,
							double beta, double* y, size_t incy);
