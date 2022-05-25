#include "medianFilter3D_3.cuh"
#include <device_launch_parameters.h>
#include "minmax.cuh"

//Fine-tuned high-speed implementation of a GPU-based median filter
//https://books.google.co.jp/books?hl=ja&lr=&id=C1_SBQAAQBAJ&oi=fnd&pg=PA31&ots=NHmypIV79d&sig=GNTYzW7wP6LiIRLLo-9qP3_Wd_k#v=onepage&q&f=false
template <typename PixelType>
__global__ void median_filter_kernel_3(cudaTextureObject_t input, PixelType * output, int3 dims, int3 offset)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x + offset.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y + offset.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z + offset.z;
  if (x >= dims.x || y >= dims.y || z >= dims.z) {
    return;
  }
  PixelType regs[15];
  ///load first 15 elements
  // 9 elements
  int k = -1;
#pragma unroll
  for (int j = -1; j <= 1; ++j) {
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
      regs[(j + 1) * 3 + (i + 1)] = tex3D<PixelType>(input, x + i, y + j, z + k);
    }
  }
  // 6 elements
  k = 0;
#pragma unroll
  for (int j = -1; j <= 0; ++j) {
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
      regs[9 + (j + 1) * 3 + (i + 1)] = tex3D<PixelType>(input, x + i, y + j, z + k);
    }
  }

  ///load new elements and reduce
  minmax(regs, 15);
  int step = 1;
  // 3 elements
  k = 0;
#pragma unroll
  for (int j = 1; j <= 1; ++j) {
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
      regs[14] = tex3D<PixelType>(input, x + i, y + j, z + k);
      minmax(regs + step, 15 - step);
      ++step;
    }
  }
  // 9 elements
  k = 1;
#pragma unroll
  for (int j = -1; j <= 1; ++j) {
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
      regs[14] = tex3D<PixelType>(input, x + i, y + j, z + k);
      minmax(regs + step, 15 - step);
      ++step;
    }
  }
  output[z*dims.x*dims.y + y*dims.x + x] = regs[13];
}

///3x3x3, output 2 voxels per thread
template <typename PixelType>
__global__ void median_filter_kernel_3_2pix(cudaTextureObject_t input, PixelType * output, int3 dims)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = 2 * (blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= dims.x || y >= dims.y || z >= dims.z-1) {
    return;
  }
  PixelType a[15];
  PixelType b[11];
  /// first common 15 elements
  // common 9 elements
  int k = 0;
#pragma unroll
  for (int j = -1; j <= 1; ++j) {
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
      a[(j + 1) * 3 + (i + 1)] = tex3D<PixelType>(input, x + i, y + j, z + k);
    }
  }
  // common 6 elements
  k = 1;
#pragma unroll
  for (int j = -1; j <= 0; ++j) {
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
      a[9 + (j + 1) * 3 + (i + 1)] = tex3D<PixelType>(input, x + i, y + j, z + k);
    }
  }

  ///load new common elements and reduce
  minmax(a, 15);
  int step = 1;
  // common 3 elements
  k = 1;
#pragma unroll
  for (int j = 1; j <= 1; ++j) {
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
      a[14] = tex3D<PixelType>(input, x + i, y + j, z + k);
      minmax(a + step, 15 - step);
      ++step;
    }
  }

  //copy
#pragma unroll
  for (int i = 0; i < 10; ++i) {
    b[i] = a[i+4];
  }

  ///load new elements and reduce for a
  k = -1;
#pragma unroll
  for (int j = -1; j <= 1; ++j) {
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
      a[14] = tex3D<PixelType>(input, x + i, y + j, z + k);
      minmax(a + step, 15 - step);
      ++step;
    }
  }
  output[z*dims.x*dims.y + y*dims.x + x] = a[13];

  ///load new elements and reduce for b
  step = 0;
  k = 2;
#pragma unroll
  for (int j = -1; j <= 1; ++j) {
#pragma unroll
    for (int i = -1; i <= 1; ++i) {
      b[10] = tex3D<PixelType>(input, x + i, y + j, z + k);
      minmax(b + step, 11 - step);
      ++step;
    }
  }
  output[(z+1)*dims.x*dims.y + y*dims.x + x] = b[9];
}

#define instantiate(TYPE) \
template __global__ void median_filter_kernel_3<TYPE>(cudaTextureObject_t, TYPE *, int3, int3); \
template __global__ void median_filter_kernel_3_2pix<TYPE>(cudaTextureObject_t, TYPE *, int3);

#include <cstdint>
instantiate(float)
instantiate(uint8_t)
instantiate(int16_t)
