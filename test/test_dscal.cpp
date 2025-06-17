
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
					size_t incx = 1) {
	auto start = std::chrono::high_resolution_clock::now();
	func(n, alpha, x, incx);
	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
	const std::vector<size_t> sizes = {10000, 100000, 1000000};
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

	for (size_t n : sizes) {
		std::cout << "\n=== Vector size: " << n << " elements ===\n";

		for (auto& stride : stride_tests) {
			std::cout << "\n" << stride.name << "\n";
			std::cout << std::setw(10) << "Function" << std::setw(12)
					  << "Time (ms)" << std::setw(12) << "GFLOP/s"
					  << std::setw(10) << "Speedup\n";
			std::cout << std::string(44, '-') << "\n";

			size_t x_size = n * stride.incx;

			constexpr std::size_t kAlignment = 16;
			size_t x_bytes =
				((x_size * sizeof(double) + kAlignment - 1) / kAlignment) *
				kAlignment;

			double* x_orig =
				static_cast<double*>(std::aligned_alloc(kAlignment, x_bytes));

			for (size_t i = 0; i < x_size; ++i) x_orig[i] = dist(gen);

			double baseline = 0;
			for (auto& f : functions) {
				if (f.needs_unit_stride && stride.incx != 1) {
					std::cout << std::setw(10) << f.name << std::setw(12)
							  << "N/A" << std::setw(12) << "N/A"
							  << std::setw(10) << "N/A\n";
					continue;
				}

				std::vector<double> times;
				for (int i = 0; i < runs; ++i) {
					double* x = x_orig;	 // Reset
					times.push_back(
						benchmark_ms(f.func, n, alpha, x, stride.incx));
				}

				std::sort(times.begin(), times.end());
				double time_ms = times[runs / 2];  // Median
				double gflops =
					(1.0 * n) / (time_ms * 1e6);  // 1 FLOP per element

				if (baseline == 0) baseline = time_ms;

				std::cout << std::setw(10) << f.name << std::setw(12)
						  << std::fixed << std::setprecision(2) << time_ms
						  << std::setw(12) << gflops << std::setw(10)
						  << baseline / time_ms << "x\n";
			}
			std::free(x_orig);
		}
	}

	return 0;
}
