cmake_minimum_required(VERSION 3.18)
project(DiGiTOutputCUDA LANGUAGES CXX CUDA)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(DiGiTOutputCUDA output_fusion.cu)
set_target_properties(DiGiTOutputCUDA PROPERTIES CUDA_STANDARD 17 CXX_STANDARD 17
    CUDA_ARCHITECTURES 89 LIBRARY_OUTPUT_DIRECTORY "${DIGIT_EXTENSION_OUTPUT}")
target_compile_options(DiGiTOutputCUDA PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)

if(NOT DIGIT_EXTENSION_OUTPUT)
  message(FATAL_ERROR "Set DIGIT_EXTENSION_OUTPUT to the selected runtime module directory")
endif()
