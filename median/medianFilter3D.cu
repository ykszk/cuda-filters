#include "medianFilter3D.h"
#include "tempTextureObject.h"
#include <thrust/device_vector.h>
#include "medianFilter3D_3.cuh"
#include "medianFilter3D_5.cuh"
#include "medianFilter3D_7.cuh"

namespace
{
  template <typename PixelType>
  void medianFilter3D(cudaTextureObject_t tex, PixelType* d_output, const int3& dims, int filter_size)
  {
    if (filter_size == 3) {
      const dim3 blockSize(8, 8, 8);
      const dim3 gridSize((dims.x + blockSize.x - 1) / blockSize.x, (dims.y + blockSize.y - 1) / blockSize.y, (dims.z + (2*blockSize.z) - 1) / (2*blockSize.z));
      median_filter_kernel_3_2pix << <gridSize, blockSize >> > (tex, d_output, dims);
    } else if (filter_size == 5) {
      const dim3 blockSize(8, 8, 4);
      const dim3 gridSize((dims.x + blockSize.x - 1) / blockSize.x, (dims.y + blockSize.y - 1) / blockSize.y, (dims.z + (2*blockSize.z) - 1) / (2*blockSize.z));
      median_filter_kernel_5_2pix << <gridSize, blockSize, sizeof(PixelType) * 27 * (blockSize.x*blockSize.y*blockSize.z)  >> > (tex, d_output, dims);
    } else if (filter_size == 7) {
      cudaDeviceSetCacheConfig(cudaFuncCache::cudaFuncCachePreferShared);
      const dim3 blockSize(4, 4, 4);
      const dim3 gridSize((dims.x + blockSize.x - 1) / blockSize.x, (dims.y + blockSize.y - 1) / blockSize.y, (dims.z + (2*blockSize.z) - 1) / (2*blockSize.z));
      median_filter_kernel_7_2pix << <gridSize, blockSize, sizeof(PixelType) * 87 * (blockSize.x*blockSize.y*blockSize.z) >> > (tex, d_output, dims);
    } else {
      throw std::invalid_argument("Unsupported filter size.");
    }
    cudaThreadSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
      throw std::runtime_error(cudaGetErrorString(error));
    }
    if (dims.z % 2 != 0) { //last slice
      const dim3 blockSize(16, 16);
      const dim3 gridSize((dims.x + blockSize.x - 1) / blockSize.x, (dims.y + blockSize.y - 1) / blockSize.y);
      int3 offset = make_int3(0,0,dims.z-1);
      if (filter_size == 3) {
        median_filter_kernel_3<< <gridSize, blockSize >> > (tex, d_output, dims, offset);
      }
      else if (filter_size == 5) {
        median_filter_kernel_5<< <gridSize, blockSize >> > (tex, d_output, dims, offset);
      }
      else if (filter_size == 7) {
        const dim3 blockSize(8,8);
        const dim3 gridSize((dims.x + blockSize.x - 1) / blockSize.x, (dims.y + blockSize.y - 1) / blockSize.y);
        median_filter_kernel_7<< <gridSize, blockSize, sizeof(PixelType) * 87 * (blockSize.x*blockSize.y*blockSize.z) >> > (tex, d_output, dims, offset);
      }
      cudaError_t error = cudaGetLastError();
      if (error != cudaSuccess)
      {
        throw std::runtime_error(cudaGetErrorString(error));
      }
    }
  }
}

#include "timer.h"
template <typename PixelType>
void medianFilter3D(const PixelType * h_input, PixelType * h_output, const int3 &dims, int filter_size, cudaTextureAddressMode addressMode)
{
  cudaPointerAttributes attr;
  auto err = cudaPointerGetAttributes(&attr, h_output);
  if (err == cudaErrorInvalidValue || attr.type!=cudaMemoryTypeManaged) { //standard host memory of unmanaged memory
    cudaGetLastError(); //reset last error
    size_t image_size = dims.x*dims.y*dims.z;
    thrust::device_vector<PixelType> output(image_size);
    tempTextureObject<PixelType> input(h_input, dims, cudaMemcpyHostToDevice, cudaArrayDefault, addressMode, cudaFilterModePoint);
    medianFilter3D(input.get(), output.data().get(), dims, filter_size);
    cudaMemcpy(h_output, output.data().get(), image_size * sizeof(PixelType), cudaMemcpyDeviceToHost);
  } else {
    tempTextureObject<PixelType> input(h_input, dims, cudaMemcpyHostToDevice, cudaArrayDefault, addressMode, cudaFilterModePoint);
    medianFilter3D(input.get(), h_output, dims, filter_size);
  }
}

template <typename PixelType>
void medianFilter3D(thrust::device_ptr<PixelType> d_input, thrust::device_ptr<PixelType> d_output, const int3 &dims, int filter_size, cudaTextureAddressMode addressMode)
{
  tempTextureObject<PixelType> input(d_input.get(), dims, cudaMemcpyDeviceToDevice, cudaArrayDefault, addressMode, cudaFilterModePoint);
  medianFilter3D(input.get(), d_output.get(), dims, filter_size);
}

#define instantiate(TYPE) \
template void medianFilter3D(const TYPE * h_input, TYPE * h_output, const int3 &dims, int filter_size, cudaTextureAddressMode addressMode); \
template void medianFilter3D(thrust::device_ptr<TYPE> d_input, thrust::device_ptr<TYPE> d_output, const int3 &dims, int filter_size, cudaTextureAddressMode addressMode);

#include <cstdint>
instantiate(float)
instantiate(uint8_t)
instantiate(int16_t)

