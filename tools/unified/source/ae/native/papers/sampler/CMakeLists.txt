cmake_minimum_required(VERSION 3.18)
project(digit_sampler_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(DiGiTSamplerCUDA MODULE digit_sampler_cuda.cu)
target_compile_options(
  DiGiTSamplerCUDA
  PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;--expt-relaxed-constexpr>
)
set_target_properties(
  DiGiTSamplerCUDA
  PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    LIBRARY_OUTPUT_DIRECTORY "${DIGIT_EXTENSION_OUTPUT}"
)

if(NOT DIGIT_EXTENSION_OUTPUT)
  message(FATAL_ERROR "Set DIGIT_EXTENSION_OUTPUT to the selected runtime module directory")
endif()
