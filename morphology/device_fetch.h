#ifndef DEVICE_FETCH_H
#define DEVICE_FETCH_H
#include "util.h"


// template for fetching textureor or accessing device pointer
template <typename InputType, typename T>
struct fetch2D {
  static __device__ T fetch(InputType input, int x, int y, const int2 &dims)
  {
    static_assert(AlwaysFalse<T>(), "this function must be specialized.");
    return 1;
  }
};
// specialization for fetching texture
template <typename T>
struct fetch2D<cudaTextureObject_t, T>
{
  static __device__ T fetch(cudaTextureObject_t input, int x, int y, const int2 &dims) {
    return tex2D<T>(input, x, y);
  }
};
// specialization for accessing device pointer
// outside value is always 0
template <typename T>
struct fetch2D<T*, T>
{
  static __device__ T fetch(T *input, int x, int y, const int2 &dims) {
    if (x < 0 || y < 0 || x >= dims.x || y >= dims.y) {
      return 0;
    }
    else {
      return input[y*dims.x + x];
    }
  }
};

// template for fetching textureor or accessing device pointer
template <typename InputType, typename T>
struct fetch2D_layered {
  static __device__ T fetch(InputType input, int x, int y, int z, const int3 &dims)
  {
    static_assert(AlwaysFalse<T>(), "this function must be specialized.");
    return 1;
  }
};
// specialization for accessing device pointer
// outside value is always 0
template <typename T>
struct fetch2D_layered<T*, T>
{
  static __device__ T fetch(T *input, int x, int y, int z, const int3 &dims) {
    if (x < 0 || y < 0 || x >= dims.x || y >= dims.y) {
      return 0;
    }
    else {
      return input[z*dims.x*dims.y + y*dims.x + x];
    }
  }
};
#endif /* DEVICE_FETCH_H */

