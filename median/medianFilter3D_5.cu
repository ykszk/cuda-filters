#include "medianFilter3D_5.cuh"
#include <device_launch_parameters.h>
#include "minmax.cuh"
#include "minmax_64.cuh"
#include "sharedmem.cuh"

template <typename PixelType>
__global__ void median_filter_kernel_5_2pix(cudaTextureObject_t input, PixelType * output, int3 dims)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = 2 * (blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= dims.x || y >= dims.y || z >= dims.z-1) {
    return;
  }
  constexpr int buf_size = 26;
  SharedMemory<PixelType> shared;
  auto shared_buf = shared.getPointer();
  PixelType *buf = shared_buf + (buf_size+1) * (threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x);
  PixelType regs[38]; //registers shared by lower part (a) and upper part (b)
  PixelType *regs_p = regs;
  int idx = 1;
  ///load first common 64 elements
  // common 50 elements
#pragma unroll
  for (int k = -1; k <= 0; ++k) {
#pragma unroll
    for (int j = -2; j <= 2; ++j) {
#pragma unroll
      for (int i = -2; i <= 2; ++i) {
        if (idx < buf_size) {
          buf[idx] = tex3D<PixelType>(input, x + i, y + j, z + k);
          ++idx;
        } else {
          regs_p[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
          ++regs_p;
        }
      }
    }
  }
  // common 10 elements
#pragma unroll
  for (int k = 1; k <= 1; ++k) {
#pragma unroll
    for (int j = -2; j <= -1; ++j) {
#pragma unroll
      for (int i = -2; i <= 2; ++i) {
        regs_p[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        ++regs_p;
      }
    }
  }
  // common 3 elements
#pragma unroll
  for (int k = 1; k <= 1; ++k) {
#pragma unroll
    for (int j = 0; j <= 0; ++j) {
#pragma unroll
      for (int i = -2; i <= 0; ++i) {
        regs_p[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        ++regs_p;
      }
    }
  }

  idx = 64;
  ///load new common elements and reduce
  // common 2 element
#pragma unroll
  for (int k = 1; k <= 1; ++k) {
#pragma unroll
    for (int j = 0; j <= 0; ++j) {
#pragma unroll
      for (int i = 1; i <= 2; ++i) {
        buf[0] = tex3D<PixelType>(input, x+i, y+j, z+k);
        minmax_64(buf, buf_size);
        minmax_64(regs, idx - buf_size);
        mm2(buf, regs);
        mm2(buf+buf_size-1, regs + idx-buf_size-1);
        --idx;
      }
    }
  }
  // common 10 elements
#pragma unroll
  for (int k = 1; k <= 1; ++k) {
#pragma unroll
    for (int j = 1; j <= 2; ++j) {
#pragma unroll
      for (int i = -2; i <= 2; ++i) {
        buf[0] = tex3D<PixelType>(input, x+i, y+j, z+k);
        minmax_64(buf, buf_size);
        minmax_64(regs, idx - buf_size);
        mm2(buf, regs);
        mm2(buf+buf_size-1, regs + idx-buf_size-1);
        --idx;
      }
    }
  }
  // common 25 elements
#pragma unroll
  for (int k = 2; k <= 2; ++k) {
#pragma unroll
    for (int j = -2; j <= 2; ++j) {
#pragma unroll
      for (int i = -2; i <= 2; ++i) {
        buf[0] = tex3D<PixelType>(input, x+i, y+j, z+k);
        minmax_64(buf, buf_size);
        minmax_64(regs, idx - buf_size);
        mm2(buf, regs);
        mm2(buf+buf_size-1, regs + idx-buf_size-1);
        --idx;
      }
    }
  }
 
  buf[buf_size] = regs[0];
  //copy for b
  for (int i = 1; i < 27; ++i) {
    regs[i] = buf[i];
  }

  ///load individual elements and reduce
  // 25 elements for a
#pragma unroll
  for (int k = -2; k <= -2; ++k) {
#pragma unroll
    for (int j = -2; j <= 2; ++j) {
#pragma unroll
      for (int i = -2; i <= 2; ++i) {
        buf[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        minmax_64(buf, idx);
        --idx;
      }
    }
  }
  output[z*dims.x*dims.y + y*dims.x + x] = buf[1];
  idx = 27;
  // 25 elements for b 
#pragma unroll
  for (int k = 3; k <= 3; ++k) {
#pragma unroll
    for (int j = -2; j <= 2; ++j) {
#pragma unroll
      for (int i = -2; i <= 2; ++i) {
        regs[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        minmax_64(regs, idx);
        --idx;
      }
    }
  }
  output[(z + 1)*dims.x*dims.y + y*dims.x + x] = regs[1];
}

template <typename PixelType>
__global__ void median_filter_kernel_5(cudaTextureObject_t input, PixelType * output, int3 dims, int3 offset)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x + offset.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y + offset.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z + offset.z;
  if (x >= dims.x || y >= dims.y || z >= dims.z) {
    return;
  }
  PixelType regs[64];
  ///load first 64 elements
  // 50 elements
#pragma unroll
  for (int k = -2; k <= -1; ++k) {
#pragma unroll
    for (int j = -2; j <= 2; ++j) {
#pragma unroll
      for (int i = -2; i <= 2; ++i) {
        regs[(k + 2) * 25 + (j + 2) * 5 + (i + 2)] = tex3D<PixelType>(input, x + i, y + j, z + k);
      }
    }
  }
  // 10 elements
#pragma unroll
  for (int k = 0; k <= 0; ++k) {
#pragma unroll
    for (int j = -2; j <= -1; ++j) {
#pragma unroll
      for (int i = -2; i <= 2; ++i) {
        regs[(k + 2) * 25 + (j + 2) * 5 + (i + 2)] = tex3D<PixelType>(input, x + i, y + j, z + k);
      }
    }
  }
  // 4 elements
#pragma unroll
  for (int k = 0; k <= 0; ++k) {
#pragma unroll
    for (int j = 0; j <= 0; ++j) {
#pragma unroll
      for (int i = -2; i <= 1; ++i) {
        regs[(k + 2) * 25 + (j + 2) * 5 + (i + 2)] = tex3D<PixelType>(input, x + i, y + j, z + k);
      }
    }
  }
  ///load new elements and reduce
  minmax_64(regs, 64);
  int step = 1;
  // 1 element
#pragma unroll
  for (int k = 0; k <= 0; ++k) {
#pragma unroll
    for (int j = 0; j <= 0; ++j) {
#pragma unroll
      for (int i = 2; i <= 2; ++i) {
        regs[63] = tex3D<PixelType>(input, x + i, y + j, z + k);
        minmax_64(regs + step, 64 - step);
        ++step;
      }
    }
  }
  // 10 elements
#pragma unroll
  for (int k = 0; k <= 0; ++k) {
#pragma unroll
    for (int j = 1; j <= 2; ++j) {
#pragma unroll
      for (int i = -2; i <= 2; ++i) {
        regs[63] = tex3D<PixelType>(input, x + i, y + j, z + k);
        minmax_64(regs + step, 64 - step);
        ++step;
      }
    }
  }
  // 50 elements
#pragma unroll
  for (int k = 1; k <= 2; ++k) {
#pragma unroll
    for (int j = -2; j <= 2; ++j) {
#pragma unroll
      for (int i = -2; i <= 2; ++i) {
        regs[63] = tex3D<PixelType>(input, x + i, y + j, z + k);
        minmax_64(regs + step, 64 - step);
        ++step;
      }
    }
  }
  output[z*dims.x*dims.y + y*dims.x + x] = regs[62];
}


#define instantiate(TYPE) \
template __global__ void median_filter_kernel_5<TYPE>(cudaTextureObject_t, TYPE *, int3, int3); \
template __global__ void median_filter_kernel_5_2pix<TYPE>(cudaTextureObject_t, TYPE *, int3);

#include <cstdint>
instantiate(float)
instantiate(uint8_t)
instantiate(int16_t)
