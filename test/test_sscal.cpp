#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "../include/sscal.h"
#include "armpl.h"

void armpl_sscal_wrapper(size_t n, float alpha, float* x, size_t incx) {
	cblas_sscal(n, alpha, x, incx);
}
using SscalFunc = void (*)(size_t, float, float*, size_t);

float benchmark_ms(SscalFunc func, size_t n, float alpha, float* x,
				   size_t incx = 1) {
	auto start = std::chrono::high_resolution_clock::now();
	func(n, alpha, x, incx);
	auto end = std::chrono::high_resolution_clock::now();
	return std::chrono::duration<float, std::milli>(end - start).count();
}

int main() {
	const std::vector<size_t> sizes = {10000, 100000, 1000000};
	const float alpha = 2.5;
	const int runs = 5;

	struct {
		const char* name;
		SscalFunc func;
		bool needs_unit_stride;
	} functions[] = {
		{"ARMPL", armpl_sscal_wrapper, false},
		{"Naive", cblas_sscal_full_naive, false},
		{"Unroll2", cblas_sscal_unroll2, false},
		{"Unroll4", cblas_sscal_unroll4, false},
		{"Unroll8", cblas_sscal_unroll8, false},
		{"Aligned", cblas_sscal_assume_aligned, false},
		{"SSE", reinterpret_cast<SscalFunc>(cblas_sscal_sse), true},
		{"SSE UnR2", reinterpret_cast<SscalFunc>(cblas_sscal_sse_unroll2),
		 true},
		{"SSE UnR4", reinterpret_cast<SscalFunc>(cblas_sscal_sse_unroll4),
		 true}};

	struct {
		const char* name;
		size_t incx;
	} stride_tests[] = {{"Unit stride (incx=1)", 1}, {"Strided (incx=2)", 2}};

	std::mt19937 gen(42);
	std::uniform_real_distribution<float> dist(-1.0, 1.0);

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
				((x_size * sizeof(float) + kAlignment - 1) / kAlignment) *
				kAlignment;

			float* x_orig =
				static_cast<float*>(std::aligned_alloc(kAlignment, x_bytes));

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
					float* x = x_orig;	// Reset
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
