#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../include/daxpy.h"
#include "armpl.h"

void armpl_daxpy_wrapper(size_t n, double alpha, double* x, size_t incx,
						 double* y, size_t incy) {
	cblas_daxpy(n, alpha, x, incx, y, incy);
}
using DaxpyFunc = void (*)(size_t, double, double*, size_t, double*, size_t);

double benchmark_ms(DaxpyFunc func, size_t n, double alpha, double* x,
					double* y, size_t incx = 1, size_t incy = 1) {
	auto start = std::chrono::high_resolution_clock::now();
	func(n, alpha, x, incx, y, incy);
	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
	const std::vector<size_t> sizes = {10000, 100000, 1000000};
	const double alpha = 2.5;
	const int runs = 5;

	struct {
		const char* name;
		DaxpyFunc func;
		bool needs_unit_stride;
	} functions[] = {
		{"ARMPL", armpl_daxpy_wrapper, false},
		{"Naive", cblas_daxpy_full_naive, false},
		{"Unroll2", cblas_daxpy_unroll2, false},
		{"Unroll4", cblas_daxpy_unroll4, false},
		{"Unroll8", cblas_daxpy_unroll8, false},
		{"Unroll2Un", cblas_daxpy_unroll2_uncoupled, false},
		{"Unroll4Un", cblas_daxpy_unroll4_uncoupled, false},
		{"Unroll8Un", cblas_daxpy_unroll8_uncoupled, false},
		{"Aligned", cblas_daxpy_assume_aligned, false},
		{"SSE", reinterpret_cast<DaxpyFunc>(cblas_daxpy_sse), true},
		{"SSE UnR2", reinterpret_cast<DaxpyFunc>(cblas_daxpy_sse_unroll2),
		 true},
		{"SSE UnR4", reinterpret_cast<DaxpyFunc>(cblas_daxpy_sse_unroll4),
		 true}};

	struct {
		const char* name;
		size_t incx, incy;
	} stride_tests[] = {{"Unit stride (incx=1, incy=1)", 1, 1},
						{"Strided X (incx=2, incy=1)", 2, 1},
						{"Strided Y (incx=1, incy=2)", 1, 2},
						{"Both strided (incx=2, incy=2)", 2, 2}};

	std::mt19937 gen(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	for (size_t n : sizes) {
		std::cout << "\n=== Vector size: " << n << " elements ===\n";

		for (auto& stride : stride_tests) {
			std::cout << "\n" << stride.name << "\n";
			std::cout << std::setw(10) << "Function" << std::setw(12)
					  << "Time (ms)" << std::setw(12) << "GFLOP/s"
					  << std::setw(10) << "Speedup\n";
			std::cout << std::string(44, '-') << "\n";

			// Allocate arrays with space for stride
			size_t x_size = n * stride.incx;
			size_t y_size = n * stride.incy;

			constexpr std::size_t kAlignment = 16;
			size_t x_bytes =
				((x_size * sizeof(double) + kAlignment - 1) / kAlignment) *
				kAlignment;
			size_t y_bytes =
				((y_size * sizeof(double) + kAlignment - 1) / kAlignment) *
				kAlignment;

			double* x =
				static_cast<double*>(std::aligned_alloc(kAlignment, x_bytes));
			double* y_orig =
				static_cast<double*>(std::aligned_alloc(kAlignment, y_bytes));

			// Initialize arrays
			for (size_t i = 0; i < x_size; ++i) x[i] = dist(gen);
			for (size_t i = 0; i < y_size; ++i) y_orig[i] = dist(gen);

			double baseline = 0;
			for (auto& f : functions) {
				// Skip SSE for non-unit stride
				if (f.needs_unit_stride &&
					(stride.incx != 1 || stride.incy != 1)) {
					std::cout << std::setw(10) << f.name << std::setw(12)
							  << "N/A" << std::setw(12) << "N/A"
							  << std::setw(10) << "N/A\n";
					continue;
				}

				std::vector<double> times;
				for (int i = 0; i < runs; ++i) {
					double* y = y_orig;	 // Reset
					times.push_back(benchmark_ms(f.func, n, alpha, x, y,
												 stride.incx, stride.incy));
				}

				std::sort(times.begin(), times.end());
				double time_ms = times[runs / 2];  // Median
				double gflops = (2.0 * n) / (time_ms * 1e6);

				if (baseline == 0) baseline = time_ms;

				std::cout << std::setw(10) << f.name << std::setw(12)
						  << std::fixed << std::setprecision(2) << time_ms
						  << std::setw(12) << gflops << std::setw(10)
						  << baseline / time_ms << "x\n";
			}
			std::free(x);
			std::free(y_orig);
		}
	}

	return 0;
}
