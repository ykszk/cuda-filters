#ifndef MEDIANFILTER3D_5_CUH
#define MEDIANFILTER3D_5_CUH
#include <cuda_runtime.h>

template <typename PixelType>
__global__ void median_filter_kernel_5_2pix(cudaTextureObject_t input, PixelType * output, int3 dims);
template <typename PixelType>
__global__ void median_filter_kernel_5(cudaTextureObject_t input, PixelType * output, int3 dims, int3 offset);

#endif /* MEDIANFILTER3D_5_CUH */