#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../include/ddot.h"  // Replace with your actual path
#include "armpl.h"

double armpl_ddot_wrapper(size_t n, double* x, size_t incx, double* y,
						  size_t incy) {
	return cblas_ddot(n, x, incx, y, incy);
}

using DdotFunc = double (*)(size_t, double*, size_t, double*, size_t);

double benchmark_ms(DdotFunc func, size_t n, double* x, double* y,
					size_t incx = 1, size_t incy = 1,
					double* result = nullptr) {
	auto start = std::chrono::high_resolution_clock::now();
	double res = func(n, x, incx, y, incy);
	auto end = std::chrono::high_resolution_clock::now();
	if (result) *result = res;
	return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
	const std::vector<size_t> sizes = {100, 10000, 100000, 1000000};
	const int runs = 5;

	struct {
		const char* name;
		DdotFunc func;
		bool needs_unit_stride;
	} functions[] = {{"ARMPL", armpl_ddot_wrapper, false},
					 {"Naive", cblas_ddot_full_naive, false},
					 {"Unroll2", cblas_ddot_unroll2, false},
					 {"Unroll4", cblas_ddot_unroll4, false},
					 {"Un2 Parith", cblas_ddot_unroll2_parith, false},
					 {"Un4 Parith", cblas_ddot_unroll4_parith, false},
					 {"Aligned", cblas_ddot_assume_aligned, false},
					 {"AlignedU2", cblas_ddot_assume_aligned_unroll2, false},
					 {"CopySSE", cblas_ddot_copy_and_sse, false},
					 {"SSE", cblas_ddot_sse, true},
					 {"SSE UnR2", cblas_ddot_sse_unroll2, true},
					 {"SSE UnR4", cblas_ddot_sse_unroll4, true}};

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
			double* y =
				static_cast<double*>(std::aligned_alloc(kAlignment, y_bytes));

			for (size_t i = 0; i < x_size; ++i) x[i] = dist(gen);
			for (size_t i = 0; i < y_size; ++i) y[i] = dist(gen);

			// Measure all function times
			std::vector<std::tuple<std::string, double, double, double>>
				results;
			double armpl_time = -1.0;

			for (auto& f : functions) {
				if (f.needs_unit_stride &&
					(stride.incx != 1 || stride.incy != 1)) {
					results.emplace_back(f.name, -1.0, -1.0, 0.0);
					continue;
				}

				std::vector<double> times;
				double result = 0;
				for (int i = 0; i < runs; ++i) {
					double temp_result = 0;
					double time = benchmark_ms(f.func, n, x, y, stride.incx,
											   stride.incy, &temp_result);
					if (i == runs / 2) result = temp_result;
					times.push_back(time);
				}
				std::sort(times.begin(), times.end());
				double time_ms = times[runs / 2];
				double gflops = (2.0 * n) / (time_ms * 1e6);

				if (std::strcmp(f.name, "ARMPL") == 0) armpl_time = time_ms;

				results.emplace_back(f.name, time_ms, gflops, result);
			}

			// Print table
			std::cout << std::setw(12) << "Function" << std::setw(12)
					  << "Time (ms)" << std::setw(12) << "GFLOP/s"
					  << std::setw(12) << "Speedup" << std::setw(16)
					  << "Dot Product\n";
			std::cout << std::string(64, '-') << "\n";

			for (auto& [name, time_ms, gflops, result] : results) {
				if (time_ms < 0.0) {
					std::cout << std::setw(12) << name << std::setw(12) << "N/A"
							  << std::setw(12) << "N/A" << std::setw(12)
							  << "N/A" << std::setw(16) << "N/A\n";
				} else {
					double speedup = armpl_time / time_ms;
					std::cout << std::setw(12) << name << std::setw(12)
							  << std::fixed << std::setprecision(2) << time_ms
							  << std::setw(12) << gflops << std::setw(11)
							  << std::fixed << std::setprecision(2) << speedup
							  << "x" << std::setw(16) << std::scientific
							  << std::setprecision(6) << result << "\n";
				}
			}

			std::free(x);
			std::free(y);
		}
	}

	return 0;
}
