#ifndef MEDIANFILTER3D_3_CUH
#define MEDIANFILTER3D_3_CUH
#include <cuda_runtime.h>

template <typename PixelType>
__global__ void median_filter_kernel_3(cudaTextureObject_t input, PixelType * output, int3 dims, int3 offset);
template <typename PixelType>
__global__ void median_filter_kernel_3_2pix(cudaTextureObject_t input, PixelType * output, int3 dims);

#endif /* MEDIANFILTER3D_3_CUH */