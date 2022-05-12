#ifndef AVERAGERESAMPLE3D_H
#define AVERAGERESAMPLE3D_H

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>

void averageResample3D(const float* h_input, float* h_output, const int3 &dims, int size);
//void averageResample3D(thrust::device_ptr<float> d_input, thrust::device_ptr<float> d_output, const int3 &dims, int filter_size, cudaTextureAddressMode addressMode);


#endif /* AVERAGERESAMPLE3D_H */
