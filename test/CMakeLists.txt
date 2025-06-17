file(GLOB TEST_SOURCES CONFIGURE_DEPENDS *.cpp)

set(ARMPL_ROOT "/opt/arm/armpl_24.10_flang-new_clang_19")
set(ARMPL_INCLUDE_DIR "${ARMPL_ROOT}/include")
set(ARMPL_LIB_DIR "${ARMPL_ROOT}/lib")

# Locate ARMPL libraries
find_library(
  ARMPL_BLAS_LIB
  NAMES armpl
  PATHS ${ARMPL_LIB_DIR})
find_library(
  ARMPL_LAPACK_LIB
  NAMES armpl
  PATHS ${ARMPL_LIB_DIR})

# Ensure ARMPL is found
if(ARMPL_BLAS_LIB AND ARMPL_LAPACK_LIB)
  message(STATUS "Found ARMPL BLAS: ${ARMPL_BLAS_LIB}")
  message(STATUS "Found ARMPL LAPACK: ${ARMPL_LAPACK_LIB}")
  message(STATUS "Found ARMPL AMATH: ${ARMPL_AMATH_LIB}")
endif()
#
# Path to the wrapper script
set(WRAPPER_SCRIPT ${CMAKE_CURRENT_SOURCE_DIR}/test_run_and_redir_output.sh)

foreach(test_src ${TEST_SOURCES})
  get_filename_component(test_name ${test_src} NAME_WE)

  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE swblas ${ARMPL_BLAS_LIB}
                                             ${ARMPL_LAPACK_LIB})
  target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/include
                                                  ${ARMPL_INCLUDE_DIR})

  add_test(NAME ${test_name} COMMAND ${WRAPPER_SCRIPT} ${test_name}
                                     $<TARGET_FILE:${test_name}>)
endforeach()
