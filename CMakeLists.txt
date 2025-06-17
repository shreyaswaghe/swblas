cmake_minimum_required(VERSION 3.16)
project(
  SWBLAS
  VERSION 1.0.0
  LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -ffast-math")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra")

# Find OpenMP for parallelization
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
else()
  message(
    WARNING "OpenMP not found - parallel implementations will be disabled")
endif()

# Source files
#
file(GLOB BLAS_SOURCES CONFIGURE_DEPENDS src/*.cpp)
set(BLAS_HEADERS include/swblas.h include/blas_internal.h)

# Create the BLAS library
add_library(swblas STATIC ${BLAS_SOURCES})

# Include directories
target_include_directories(
  swblas PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
  target_link_libraries(swblas PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(swblas PUBLIC MYBLAS_HAVE_OPENMP)
endif()

# Benchmarking executable add_executable(blas_benchmark benchmark/benchmark.cpp
# benchmark/timing.cpp)

# target_link_libraries(blas_benchmark PRIVATE swblas)
# target_include_directories(blas_benchmark PRIVATE benchmark)

# Testing executable add_executable(blas_test test/test_level1.cpp
# test/test_level2.cpp test/test_level3.cpp test/test_main.cpp)

# target_link_libraries(blas_test PRIVATE swblas)
# target_include_directories(blas_test PRIVATE test)

# Installation
install(
  TARGETS swblas
  EXPORT SWBLASTargets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(FILES ${BLAS_HEADERS} DESTINATION include)

# Enable testing
enable_testing()
add_subdirectory(test)
# add_test(NAME blas_tests COMMAND blas_test)
