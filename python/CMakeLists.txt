cmake_minimum_required(VERSION 3.0)

project(cuFilters_python)
add_subdirectory(pybind11)
pybind11_add_module(cuda_filters cuda_filters.cpp)
target_link_libraries(cuda_filters PRIVATE medianFilter3D morphology fillhole)
