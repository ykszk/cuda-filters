cmake_minimum_required(VERSION 3.0)

project(cuFilters)

FIND_PACKAGE(CUDA REQUIRED)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
INCLUDE(SetCompilerFlags)

if (NOT WIN32)
	set(CMAKE_CXX_STANDARD 11)
	set(CMAKE_CXX_STANDARD_REQUIRED ON)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
	set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
endif()

add_subdirectory(median)
include_directories(median)
add_subdirectory(morphology)
include_directories(morphology)

include_directories(${CUDA_TOOLKIT_INCLUDE})
# # shared library
# add_library(cuFilters SHARED cuFilters.h cuFilters.cpp)
# target_link_libraries(cuFilters medianFilter3D)
# static library
add_library(cuFilters-static STATIC cuFilters.h cuFilters.cpp)
target_link_libraries(cuFilters-static medianFilter3D)
SET_TARGET_PROPERTIES(cuFilters-static PROPERTIES OUTPUT_NAME "cuFilters")
SET_TARGET_PROPERTIES(cuFilters-static PROPERTIES PREFIX "lib")

# for building mex
configure_file(mex/static.bat.in mex/static.bat)
configure_file(mex/medianFilter3D.cpp mex/medianFilter3D.cpp COPYONLY)
configure_file(mex/medianFilter3D.m mex/medianFilter3D.m COPYONLY)

if (BUILD_PYTHON_BINDING)
  add_subdirectory(python)
endif()
