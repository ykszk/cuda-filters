#include "averageResample3D.h"
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>

namespace
{
  __global__ void average_resample_kernel(float *input, float *output, int3 dims, int3 output_dims, int resample_size)
  {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    x *= resample_size;
    y *= resample_size;
    z *= resample_size;
    if (x >= dims.x || y >= dims.y || z >= dims.z) {
      return;
    }
    float result = 0;
    for (int k = 0; k < resample_size; ++k) {
      for (int j = 0; j < resample_size; ++j) {
        for (int i = 0; i < resample_size; ++i) {
          result += input[(z + k)*dims.x*dims.y + (y + j)*dims.x + x + i];
        }
      }
    }
    output[z*output_dims.x*output_dims.y/resample_size + y*output_dims.y/resample_size + x/resample_size] = result / (resample_size*resample_size*resample_size);
  }

  void averageResample3D(float* d_input, float* d_output, const int3 &dims, const int3 &output_dims, int resample_size)
  {
    const dim3 blockSize(8, 8, 8);
    const dim3 gridSize((dims.x + blockSize.x - 1) / blockSize.x / resample_size, (dims.y + blockSize.y - 1) / blockSize.y / resample_size, (dims.z + blockSize.z - 1) / blockSize.z / resample_size);
    average_resample_kernel<< <gridSize, blockSize >> > (d_input, d_output, dims, output_dims, resample_size);
  }
}

void averageResample3D(const float* h_input, float* h_output, const int3 &dims, int resample_size)
{
  size_t image_size = dims.x*dims.y*dims.z;
  thrust::device_vector<float> input(h_input, h_input+image_size);
  thrust::device_vector<float> output(image_size/(resample_size*resample_size*resample_size));
  auto out_dims = make_int3(dims.x / resample_size, dims.y / resample_size, dims.z / resample_size);
  averageResample3D(input.data().get(), output.data().get(), dims, out_dims, resample_size);
  cudaMemcpy(h_output, output.data().get(), out_dims.x*out_dims.y*out_dims.z * sizeof(float), cudaMemcpyDeviceToHost);
}
