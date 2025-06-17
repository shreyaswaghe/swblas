#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <ratio>
#include <vector>

#include "../include/dnrm2.h"  // Replace with your actual path
#include "armpl.h"

double armpl_dnrm2_wrapper(size_t n, double* x, size_t incx) {
	return cblas_dnrm2(n, x, incx);
}

using Dnrm2Func = double (*)(size_t, double*, size_t);

double benchmark_ms(Dnrm2Func func, size_t n, double* x, size_t incx = 1,
					double* result = nullptr) {
	auto start = std::chrono::high_resolution_clock::now();
	double res = func(n, x, incx);
	auto end = std::chrono::high_resolution_clock::now();
	if (result) *result = res;
	return std::chrono::duration<double, std::micro>(end - start).count() /
		   1000.0;
}

int main() {
	const std::vector<size_t> sizes = {100, 10000, 100000, 1000000};
	const int runs = 10;

	struct {
		const char* name;
		Dnrm2Func func;
		bool needs_unit_stride;
	} functions[] = {{"ARMPL", armpl_dnrm2_wrapper, false},
					 {"Naive", cblas_dnrm2_naive, false},
					 {"Unroll2", cblas_dnrm2_unroll2, false},
					 {"Unroll4", cblas_dnrm2_unroll4, false},
					 {"Unroll8", cblas_dnrm2_unroll8, false},
					 {"SSE", cblas_dnrm2_sse, true},
					 {"SSE UnR2", cblas_dnrm2_sse_unroll2, true},
					 {"SSE UnR4", cblas_dnrm2_sse_unroll4, true},
					 {"SSE UnR8", cblas_dnrm2_sse_unroll8, true}};

	struct {
		const char* name;
		size_t incx;
	} stride_tests[] = {{"Unit stride (incx=1)", 1}, {"Strided X (incx=2)", 2}};

	std::mt19937 gen(42);
	std::uniform_real_distribution<double> dist(-1.0, 1.0);

	for (size_t n : sizes) {
		std::cout << "\n=== Vector size: " << n << " elements ===\n";

		for (auto& stride : stride_tests) {
			std::cout << "\n" << stride.name << "\n";

			// Allocate arrays with space for stride
			size_t x_size = n * stride.incx;

			constexpr std::size_t kAlignment = 16;
			size_t x_bytes =
				((x_size * sizeof(double) + kAlignment - 1) / kAlignment) *
				kAlignment;

			double* x =
				static_cast<double*>(std::aligned_alloc(kAlignment, x_bytes));

			for (size_t i = 0; i < x_size; ++i) x[i] = dist(gen);

			// Measure all function times
			std::vector<std::tuple<std::string, double, double, double>>
				results;
			double armpl_time = -1.0;

			for (auto& f : functions) {
				if (f.needs_unit_stride && stride.incx != 1) {
					results.emplace_back(f.name, -1.0, -1.0, 0.0);
					continue;
				}

				std::vector<double> times;
				double result = 0;
				for (int i = 0; i < runs; ++i) {
					double temp_result = 0;
					double time =
						benchmark_ms(f.func, n, x, stride.incx, &temp_result);
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
					  << "Norm\n";
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
		}
	}

	return 0;
}
