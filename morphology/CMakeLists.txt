cmake_minimum_required(VERSION 3.0)

FIND_PACKAGE(CUDA REQUIRED)

cuda_add_library(morphology
  morphology.cu morphology.h
  scopedTextureObject.h
  minmax_monoid.h
  )

cuda_add_library(fillhole
  fillhole.cu fillhole.h
  )

target_link_libraries(fillhole morphology)

cuda_add_executable(fillhole_m
  main.cpp
  )

target_link_libraries(fillhole_m morphology fillhole)
