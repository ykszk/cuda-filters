# Configure nvcc compiler flags
set(TARGET_CUDA_ARCHS 61 CACHE STRING "Target architecture(s) for the cuda compilation. Use \";\" as a delimiter.")
unset(CUDA_NVCC_FLAGS CACHE)
FOREACH(ARCH IN ITEMS ${TARGET_CUDA_ARCHS})
  string(REPLACE "ARCH" ${ARCH}  FLAGS " -gencode arch=compute_ARCH,code=compute_ARCH")
  string(CONCAT CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${FLAGS})
ENDFOREACH(ARCH)

#disable warning C4819 (Compile warning about character encoding)
if(MSVC)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} " -Xcompiler \"/wd 4819\"")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4819" )
endif(MSVC)

message(STATUS "CUDA_NVCC_FLAGS " ${CUDA_NVCC_FLAGS})
