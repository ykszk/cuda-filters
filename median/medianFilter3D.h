#ifndef MEDIANFILTER3D_H
#define MEDIANFILTER3D_H
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

template <typename PixelType>
void medianFilter3D(const PixelType* h_input, PixelType* h_output, const int3 &dims, int filter_size, cudaTextureAddressMode addressMode);
template <typename PixelType>
void medianFilter3D(thrust::device_ptr<PixelType> d_input, thrust::device_ptr<PixelType> d_output, const int3 &dims, int filter_size, cudaTextureAddressMode addressMode);

#endif /* MEDIANFILTER3D_H */
