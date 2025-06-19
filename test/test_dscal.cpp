#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../include/dscal.h"
#include "armpl.h"

void armpl_dscal_wrapper(size_t n, double alpha, double* x, size_t incx) {
	cblas_dscal(n, alpha, x, incx);
}

using DscalFunc = void (*)(size_t, double, double*, size_t);

double benchmark_ms(DscalFunc func, size_t n, double alpha, double* x,
					size_t incx) {
	auto start = std::chrono::high_resolution_clock::now();
	func(n, alpha, x, incx);
	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

bool verify_dscal(DscalFunc test_func, size_t n, double alpha,
				  const double* x_orig, size_t incx, double tolerance = 1e-12) {
	// Allocate memory for reference and test results
	constexpr std::size_t kAlignment = 64;
	size_t x_size = n * incx;
	size_t x_bytes =
		((x_size * sizeof(double) + kAlignment - 1) / kAlignment) * kAlignment;

	double* x_ref =
		static_cast<double*>(std::aligned_alloc(kAlignment, x_bytes));
	double* x_test =
		static_cast<double*>(std::aligned_alloc(kAlignment, x_bytes));

	// Copy original x vector
	std::memcpy(x_ref, x_orig, x_size * sizeof(double));
	std::memcpy(x_test, x_orig, x_size * sizeof(double));

	// Compute reference result using ARMPL
	armpl_dscal_wrapper(n, alpha, x_ref, incx);

	// Compute test result
	test_func(n, alpha, x_test, incx);

	// Compare results
	bool passed = true;
	double max_error = 0.0;
	size_t max_error_idx = 0;

	for (size_t i = 0; i < n; ++i) {
		size_t idx = i * incx;
		double error = std::abs(x_ref[idx] - x_test[idx]);
		double rel_error =
			error / (std::abs(x_ref[idx]) + 1e-15);	 // Avoid division by zero

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
		std::cout << "    Reference: " << x_ref[max_error_idx * incx]
				  << ", Test: " << x_test[max_error_idx * incx] << "\n";
	}

	std::free(x_ref);
	std::free(x_test);
	return passed;
}

int main() {
	const std::vector<size_t> vector_sizes = {100, 1000, 10000, 100000,
											  1000000};
	const double alpha = 2.5;
	const int runs = 5;

	struct {
		const char* name;
		DscalFunc func;
		bool needs_unit_stride;
	} functions[] = {
		{"ARMPL", armpl_dscal_wrapper, false},
		{"Naive", cblas_dscal_full_naive, false},
		{"Unroll2", cblas_dscal_unroll2, false},
		{"Unroll4", cblas_dscal_unroll4, false},
		{"Unroll8", cblas_dscal_unroll8, false},
		{"Aligned", cblas_dscal_assume_aligned, false},
		{"SSE", reinterpret_cast<DscalFunc>(cblas_dscal_sse), true},
		{"Neon", reinterpret_cast<DscalFunc>(cblas_dscal_neon), true},
		{"SSE UnR2", reinterpret_cast<DscalFunc>(cblas_dscal_sse_unroll2),
		 true},
		{"SSE UnR4", reinterpret_cast<DscalFunc>(cblas_dscal_sse_unroll4),
		 true}};

	struct {
		const char* name;
		size_t incx;
	} stride_tests[] = {{"Unit stride (incx=1)", 1}, {"Strided (incx=2)", 2}};

	std::mt19937 gen(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	for (auto& size : vector_sizes) {
		size_t n = size;

		std::cout << "\n=== Vector size: " << n << " elements ===\n";

		for (auto& stride : stride_tests) {
			std::cout << "\n" << stride.name << "\n";
			std::cout << std::setw(12) << "Function" << std::setw(12)
					  << "Time (ms)" << std::setw(12) << "GFLOP/s"
					  << std::setw(10) << "Speedup" << std::setw(12)
					  << "Verify\n";
			std::cout << std::string(58, '-') << "\n";

			// Calculate required array sizes with stride
			size_t x_size = n * stride.incx;

			constexpr std::size_t kAlignment =
				64;	 // Increased for better cache alignment

			// Calculate aligned sizes
			size_t x_bytes =
				((x_size * sizeof(double) + kAlignment - 1) / kAlignment) *
				kAlignment;

			// Allocate aligned memory
			double* x_orig =
				static_cast<double*>(std::aligned_alloc(kAlignment, x_bytes));
			double* x_work =
				static_cast<double*>(std::aligned_alloc(kAlignment, x_bytes));

			// Initialize array
			for (size_t i = 0; i < x_size; ++i) x_orig[i] = dist(gen);

			double baseline = 0;
			bool verification_failed = false;

			for (auto& f : functions) {
				// Skip functions that need unit stride when using strided
				// access
				if (f.needs_unit_stride && stride.incx != 1) {
					std::cout << std::setw(12) << f.name << std::setw(12)
							  << "N/A" << std::setw(12) << "N/A"
							  << std::setw(10) << "N/A" << std::setw(12)
							  << "N/A\n";
					continue;
				}

				// Verify correctness (skip ARMPL as it's our reference)
				bool verified = true;
				if (strcmp(f.name, "ARMPL") != 0) {
					verified =
						verify_dscal(f.func, n, alpha, x_orig, stride.incx);
					if (!verified) {
						verification_failed = true;
					}
				}

				std::vector<double> times;
				for (int i = 0; i < runs; ++i) {
					// Copy original x vector for each run
					std::memcpy(x_work, x_orig, x_size * sizeof(double));

					times.push_back(
						benchmark_ms(f.func, n, alpha, x_work, stride.incx));
				}

				std::sort(times.begin(), times.end());
				double time_ms = times[runs / 2];  // Median time

				// DSCAL: x = alpha * x
				// Operations: n multiplications
				double gflops = (1.0 * n) / (time_ms * 1e6);

				if (baseline == 0) baseline = time_ms;

				std::cout << std::setw(12) << f.name << std::setw(12)
						  << std::fixed << std::setprecision(2) << time_ms
						  << std::setw(12) << gflops << std::setw(10)
						  << baseline / time_ms << "x" << std::setw(12)
						  << (verified ? "PASS" : "FAIL") << "\n";
			}

			// Free allocated memory
			std::free(x_orig);
			std::free(x_work);

			if (verification_failed) {
				std::cout << "\n*** WARNING: Some implementations failed "
							 "verification! ***\n";
			}
		}
	}

	return 0;
}
