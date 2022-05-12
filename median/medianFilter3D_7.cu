#include "medianFilter3D_7.cuh"
#include <device_launch_parameters.h>
#include "minmax.cuh"
#include "minmax_64.cuh"
#include "minmax_87.cuh"
#include "sharedmem.cuh"

template <typename PixelType>
__global__ void median_filter_kernel_7(cudaTextureObject_t input, PixelType * output, int3 dims, int3 offset)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x + offset.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y + offset.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z + offset.z;
  if (x >= dims.x || y >= dims.y || z >= dims.z) {
    return;
  }
  constexpr int buf_size = 86;
  SharedMemory<PixelType> shared;
  auto shared_buf = shared.getPointer();
  PixelType *buf = shared_buf + (buf_size+1) * (threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x);
  PixelType buf_l[87];
  PixelType *buf_l_p = buf_l;
  int idx = 1;
  // 3 slices (= 147)
  for (int k = -3; k <= -1; ++k) {
    for (int j = -3; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        if (idx < buf_size) {
          buf[idx] = tex3D<PixelType>(input, x + i, y + j, z + k);
        } else {
          buf_l_p[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
          ++buf_l_p;
        }
        ++idx;
      }
    }
  }
  /// 4th
  // common 21 elements
  for (int k = 0; k <= 0; ++k) {
    for (int j = -3; j <= -1; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf_l_p[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        ++buf_l_p;
      }
    }
  }
  // common 4 elements
  for (int k = 0; k <= 0; ++k) {
    for (int j = 0; j <= 0; ++j) {
      for (int i = -3; i <= 0; ++i) {
        buf_l_p[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        ++buf_l_p;
      }
    }
  }
  idx = 173;
  /// start reduction
  // common 3 elements
  for (int k = 0; k <= 0; ++k) {
    for (int j = 0; j <= 0; ++j) {
      for (int i = 1; i <= 3; ++i) {
        buf[0] = tex3D<PixelType>(input, x+i, y+j, z+k);
        minmax_87(buf, buf_size);
        minmax_87(buf_l, idx - buf_size);
        mm2(buf, buf_l);
        mm2(buf+buf_size-1, buf_l + idx-buf_size-1);
        --idx;
      }
    }
  }
  // 21 elements
  for (int k = 0; k <= 0; ++k) {
    for (int j = 1; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf[0] = tex3D<PixelType>(input, x+i, y+j, z+k);
        minmax_87(buf, buf_size);
        minmax_87(buf_l, idx - buf_size);
        mm2(buf, buf_l);
        mm2(buf+buf_size-1, buf_l + idx-buf_size-1);
        --idx;
      }
    }
  }
  // 1 slice (= 49)
  for (int k = 1; k <= 1; ++k) {
    for (int j = -3; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf[0] = tex3D<PixelType>(input, x+i, y+j, z+k);
        minmax_87(buf, buf_size);
        minmax_87(buf_l, idx - buf_size);
        mm2(buf, buf_l);
        mm2(buf+buf_size-1, buf_l + idx-buf_size-1);
        --idx;
      }
    }
  }
  // 1 slice
  for (int k = 2; k <= 2; ++k) {
    for (int j = -3; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        if (idx > (buf_size + 1)) {
          minmax_87(buf, buf_size);
          minmax_87(buf_l, idx - buf_size);
          mm2(buf, buf_l);
          mm2(buf + buf_size - 1, buf_l + idx - buf_size - 1);
          if (idx == buf_size + 2) {
            buf[buf_size] = buf_l[0];
          }
        } else {
          minmax_87(buf, idx);
        }
        --idx;
      }
    }
  }
  // 1 slice
  for (int k = 3; k <= 3; ++k) {
    for (int j = -3; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        minmax_64(buf, idx);
        --idx;
      }
    }
  }
  output[z*dims.x*dims.y + y*dims.x + x] = buf[1];
}

template <typename PixelType>
__global__ void median_filter_kernel_7_2pix(cudaTextureObject_t input, PixelType * output, int3 dims)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = 2 * (blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= dims.x || y >= dims.y || z >= dims.z-1) {
    return;
  }
  constexpr int buf_size = 86;
  SharedMemory<PixelType> shared;
  auto shared_buf = shared.getPointer();
  PixelType *buf = shared_buf + (buf_size+1) * (threadIdx.z*blockDim.y*blockDim.x + threadIdx.y*blockDim.x + threadIdx.x);
  PixelType buf_l[87];
  PixelType *buf_l_p = buf_l;
  int idx = 1;
  /// first common 3 slices
  // common 3 slices (= 147)
  for (int k = -2; k <= 0; ++k) {
    for (int j = -3; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        if (idx < buf_size) {
          buf[idx] = tex3D<PixelType>(input, x + i, y + j, z + k);
        } else {
          buf_l_p[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
          ++buf_l_p;
        }
        ++idx;
      }
    }
  }
  /// 4th common slice
  // common 21 elements
  for (int k = 1; k <= 1; ++k) {
    for (int j = -3; j <= -1; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf_l_p[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        ++buf_l_p;
      }
    }
  }
  // common 4 elements
  for (int k = 1; k <= 1; ++k) {
    for (int j = 0; j <= 0; ++j) {
      for (int i = -3; i <= 0; ++i) {
        buf_l_p[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        ++buf_l_p;
      }
    }
  }
  idx = 173;
  /// start reduction
  // common 3 elements
  for (int k = 1; k <= 1; ++k) {
    for (int j = 0; j <= 0; ++j) {
      for (int i = 1; i <= 3; ++i) {
        buf[0] = tex3D<PixelType>(input, x+i, y+j, z+k);
        minmax_87(buf, buf_size);
        minmax_87(buf_l, idx - buf_size);
        mm2(buf, buf_l);
        mm2(buf+buf_size-1, buf_l + idx-buf_size-1);
        --idx;
      }
    }
  }
  // 21 elements
  for (int k = 1; k <= 1; ++k) {
    for (int j = 1; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf[0] = tex3D<PixelType>(input, x+i, y+j, z+k);
        minmax_87(buf, buf_size);
        minmax_87(buf_l, idx - buf_size);
        mm2(buf, buf_l);
        mm2(buf+buf_size-1, buf_l + idx-buf_size-1);
        --idx;
      }
    }
  }
  /// 5th common slice
  // 49 elements
  for (int k = 2; k <= 2; ++k) {
    for (int j = -3; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf[0] = tex3D<PixelType>(input, x+i, y+j, z+k);
        minmax_87(buf, buf_size);
        minmax_87(buf_l, idx - buf_size);
        mm2(buf, buf_l);
        mm2(buf+buf_size-1, buf_l + idx-buf_size-1);
        --idx;
      }
    }
  }
  /// 6th common slice
  // 49 elements
  for (int k = 3; k <= 3; ++k) {
    for (int j = -3; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        if (idx > (buf_size + 1)) {
          minmax_87(buf, buf_size);
          minmax_87(buf_l, idx - buf_size);
          mm2(buf, buf_l);
          mm2(buf + buf_size - 1, buf_l + idx - buf_size - 1);
          if (idx == buf_size + 2) {
            buf[buf_size] = buf_l[0];
          }
        } else {
          minmax_87(buf, idx);
        }
        --idx;
      }
    }
  }

  ///copy
  // idx == 51
  //PixelType buf_b[51];
  PixelType *buf_b = buf_l;
  //PixelType *buf_b = buf + 52;
  for (int i = 1; i < 51; ++i) {
    buf_b[i] = buf[i];
  }

  /// 1st slice for pixel a
  for (int k = -3; k <= -3; ++k) {
    for (int j = -3; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        minmax_64(buf, idx);
        --idx;
      }
    }
  }
  output[z*dims.x*dims.y + y*dims.x + x] = buf[1];
  /// the last slice for pixel b
  idx = 51;
  for (int k = 4; k <= 4; ++k) {
    for (int j = -3; j <= 3; ++j) {
      for (int i = -3; i <= 3; ++i) {
        buf_b[0] = tex3D<PixelType>(input, x + i, y + j, z + k);
        minmax_64(buf_b, idx);
        --idx;
      }
    }
  }
  output[(z + 1)*dims.x*dims.y + y*dims.x + x] = buf_b[1];
}

#define instantiate(TYPE) \
template __global__ void median_filter_kernel_7<TYPE>(cudaTextureObject_t, TYPE *, int3, int3); \
template __global__ void median_filter_kernel_7_2pix<TYPE>(cudaTextureObject_t, TYPE *, int3);

#include <cstdint>
instantiate(float)
instantiate(uint8_t)
instantiate(int16_t)
