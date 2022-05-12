#ifndef MEDIANFILTER3D_7_CUH
#define MEDIANFILTER3D_7_CUH
#include <cuda_runtime.h>

template <typename PixelType>
__global__ void median_filter_kernel_7_2pix(cudaTextureObject_t input, PixelType * output, int3 dims);
template <typename PixelType>
__global__ void median_filter_kernel_7(cudaTextureObject_t input, PixelType * output, int3 dims, int3 offset);


#endif /* MEDIANFILTER3D_7_CUH */