cmake_minimum_required(VERSION 3.0)

FIND_PACKAGE(CUDA REQUIRED)

cuda_add_library(medianFilter3D
  medianFilter3D.cu medianFilter3D.h
  medianFilter3D_3.cu medianFilter3D_3.cuh
  medianFilter3D_5.cu medianFilter3D_5.cuh
  medianFilter3D_7.cu medianFilter3D_7.cuh
  averageResample3D.cu averageResample3D.h
  tempTextureObject.cpp tempTextureObject.h
  )
cuda_add_executable(median
  main.cpp
  )
target_link_libraries(median medianFilter3D)
cuda_add_executable(median_benchmark
  median_benchmark.cpp
  )
target_link_libraries(median_benchmark medianFilter3D ${CUDA_curand_LIBRARY})
