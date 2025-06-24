#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../include/dgemvct.h"
#include "armpl.h"

void armpl_dgemv_wrapper(size_t m, size_t n, double alpha, const double* a,
						 size_t lda, const double* x, size_t incx, double beta,
						 double* y, size_t incy) {
	cblas_dgemv(CblasColMajor, CblasTrans, m, n, alpha, a, lda, x, incx, beta,
				y, incy);
}

using DgemvFunc = void (*)(size_t, size_t, double, const double*, size_t,
						   const double*, size_t, double, double*, size_t);

double benchmark_ms(DgemvFunc func, size_t m, size_t n, double alpha,
					const double* a, size_t lda, const double* x, size_t incx,
					double beta, double* y, size_t incy) {
	auto start = std::chrono::high_resolution_clock::now();
	func(m, n, alpha, a, lda, x, incx, beta, y, incy);
	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

bool verify_dgemv(DgemvFunc test_func, size_t m, size_t n, double alpha,
				  const double* a, size_t lda, const double* x, size_t incx,
				  double beta, const double* y_orig, size_t incy,
				  double tolerance = 1e-10) {
	constexpr std::size_t kAlignment = 64;
	size_t y_size = n * incy;  // Note: output size is n for transpose
	size_t y_bytes =
		((y_size * sizeof(double) + kAlignment - 1) / kAlignment) * kAlignment;

	double* y_ref =
		static_cast<double*>(std::aligned_alloc(kAlignment, y_bytes));
	double* y_test =
		static_cast<double*>(std::aligned_alloc(kAlignment, y_bytes));

	std::memcpy(y_ref, y_orig, y_size * sizeof(double));
	std::memcpy(y_test, y_orig, y_size * sizeof(double));

	armpl_dgemv_wrapper(m, n, alpha, a, lda, x, incx, beta, y_ref, incy);
	test_func(m, n, alpha, a, lda, x, incx, beta, y_test, incy);

	bool passed = true;
	double max_error = 0.0;
	size_t max_error_idx = 0;

	for (size_t i = 0; i < n; ++i) {  // Note: loop over n for transpose
		size_t idx = i * incy;
		double error = std::abs(y_ref[idx] - y_test[idx]);
		double rel_error = error / (std::abs(y_ref[idx]) + 1e-15);

		if (error > max_error) {
			max_error = error;
			max_error_idx = i;
		}

		if (error > tolerance && rel_error > tolerance) {
			passed = false;
			break;
		}
	}

	if (!passed) {
		std::cout << "    VERIFICATION FAILED!\n";
		std::cout << "    Max absolute error: " << std::scientific << max_error
				  << " at index " << max_error_idx << "\n";
		std::cout << "    Reference: " << y_ref[max_error_idx * incy]
				  << ", Test: " << y_test[max_error_idx * incy] << "\n";
	}

	std::free(y_ref);
	std::free(y_test);
	return passed;
}

int main() {
	const std::vector<std::pair<size_t, size_t>> matrix_sizes = {
		{32, 32},	  {64, 64},		{128, 128},	  {512, 512},	 {1024, 1024},
		{2048, 2048}, {4096, 4096}, {8192, 8192}, {16384, 16384}};
	const double alpha = 2.5;
	const double beta = 1.5;
	const int runs = 20;

	struct {
		const char* name;
		DgemvFunc func;
		bool needs_unit_incx;
		bool needs_unit_incy;
	} functions[] = {
		{"ARMPL", armpl_dgemv_wrapper, false, false},
		{"Naive", cblas_dgemvCT_naive, false, false},
		{"UnrollI2", cblas_dgemvCT_unrollinner2, false, false},
		{"UnrollI4", cblas_dgemvCT_unrollinner4, false, false},
		{"UnrollI8", cblas_dgemvCT_unrollinner8, false, false},
		{"BlockC4", cblas_dgemvCT_strideC4, false, false},
		{"BlockC4R2", cblas_dgemvCT_strideC4R2, false, false},
		{"BlockC6", cblas_dgemvCT_strideC6, false, false},
		{"BlockC6R2", cblas_dgemvCT_strideC6R2, false, false},
		{"BlockC8", cblas_dgemvCT_strideC8, false, false},
		{"SSE", cblas_dgemvCT_sseinner, false, false},
		{"SSE2", cblas_dgemvCT_sseinner2, false, false},
		{"SSE4", cblas_dgemvCT_sseinner4, false, false},
		{"Block32", cblas_dgemvCT_block32, false, false},
		{"Block64", cblas_dgemvCT_block64, false, false},
		{"Block128", cblas_dgemvCT_block128, false, false},
		{"Block256", cblas_dgemvCT_block256, false, false},
		{"Block512", cblas_dgemvCT_block256, false, false}};  // Your new func

	struct {
		const char* name;
		size_t incx, incy;
	} stride_tests[] = {{"Unit stride (incx=1, incy=1)", 1, 1},
						{"Strided X (incx=2, incy=1)", 2, 1},
						{"Strided Y (incx=1, incy=2)", 1, 2},
						{"Both strided (incx=2, incy=2)", 2, 2}};

	std::mt19937 gen(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	for (auto& size : matrix_sizes) {
		size_t m = size.first;
		size_t n = size.second;

		std::cout << "\n=== Matrix size: " << m << "x" << n
				  << " (Transpose) ===\n";

		for (auto& stride : stride_tests) {
			std::cout << "\n" << stride.name << "\n";
			std::cout << std::setw(12) << "Function" << std::setw(12)
					  << "Time (ms)" << std::setw(12) << "GFLOP/s"
					  << std::setw(10) << "Speedup" << std::setw(12)
					  << "Verify\n";
			std::cout << std::string(58, '-') << "\n";

			size_t x_size = m * stride.incx;  // Note: x size is m for transpose
			size_t y_size = n * stride.incy;  // Note: y size is n for transpose
			size_t a_size = m * n;			  // lda = m for column major
			size_t lda = m;

			constexpr std::size_t kAlignment = 64;

			size_t a_bytes =
				((a_size * sizeof(double) + kAlignment - 1) / kAlignment) *
				kAlignment;
			size_t x_bytes =
				((x_size * sizeof(double) + kAlignment - 1) / kAlignment) *
				kAlignment;
			size_t y_bytes =
				((y_size * sizeof(double) + kAlignment - 1) / kAlignment) *
				kAlignment;

			double* a =
				static_cast<double*>(std::aligned_alloc(kAlignment, a_bytes));
			double* x =
				static_cast<double*>(std::aligned_alloc(kAlignment, x_bytes));
			double* y_orig =
				static_cast<double*>(std::aligned_alloc(kAlignment, y_bytes));
			double* y_work =
				static_cast<double*>(std::aligned_alloc(kAlignment, y_bytes));

			for (size_t i = 0; i < a_size; ++i) a[i] = dist(gen);
			for (size_t i = 0; i < x_size; ++i) x[i] = dist(gen);
			for (size_t i = 0; i < y_size; ++i) y_orig[i] = dist(gen);

			double baseline = 0;
			bool verification_failed = false;

			for (auto& f : functions) {
				if ((f.needs_unit_incx && stride.incx != 1) ||
					(f.needs_unit_incy && stride.incy != 1)) {
					std::cout << std::setw(12) << f.name << std::setw(12)
							  << "N/A" << std::setw(12) << "N/A"
							  << std::setw(10) << "N/A" << std::setw(12)
							  << "N/A\n";
					continue;
				}

				bool verified = true;
				if (strcmp(f.name, "ARMPL") != 0) {
					verified =
						verify_dgemv(f.func, m, n, alpha, a, lda, x,
									 stride.incx, beta, y_orig, stride.incy);
					if (!verified) {
						verification_failed = true;
					}
				}

				std::vector<double> times;
				for (int i = 0; i < runs; ++i) {
					std::memcpy(y_work, y_orig, y_size * sizeof(double));

					times.push_back(benchmark_ms(f.func, m, n, alpha, a, lda, x,
												 stride.incx, beta, y_work,
												 stride.incy));
				}

				std::sort(times.begin(), times.end());
				double time_ms = times[runs / 2];

				double gflops = (2.0 * m * n) / (time_ms * 1e6);

				if (baseline == 0) baseline = time_ms;

				std::cout << std::setw(12) << f.name << std::setw(12)
						  << std::fixed << std::setprecision(6) << time_ms
						  << std::setw(12) << gflops << std::setw(10)
						  << baseline / time_ms << "x" << std::setw(12)
						  << (verified ? "PASS" : "FAIL") << "\n";
			}

			std::free(a);
			std::free(x);
			std::free(y_orig);
			std::free(y_work);

			if (verification_failed) {
				std::cout << "\n*** WARNING: Some implementations failed "
							 "verification! ***\n";
			}
		}
	}

	return 0;
}
