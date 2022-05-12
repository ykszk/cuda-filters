#include "morphology.h"
#include <memory>
#include <vector>
#include "scopedTextureObject.h"
#include "cuda_safe_call.h"
#include "minmax_monoid.h"
#include "device_fetch.h"

#define CUDA_CHECK_LAST_ERROR \
auto err = cudaGetLastError();\
if (err != cudaSuccess) {\
  throw(std::runtime_error(std::string(cudaGetErrorString(err)) + AT));\
}

#define PROD2(v) v.x*v.y
#define PROD3(v) v.x*v.y*v.z

template <typename T>
std::shared_ptr<T> make_shared_device(size_t size)
{
  T *ptr;
  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc(&ptr, sizeof(T)*size));
  std::shared_ptr<T> sptr(ptr, [](T* ptr) {cudaFree(ptr); });
  return sptr;
}


template <typename T, typename Monoid, typename InputType>
__global__ void reduce_2D(InputType input, T * output, int2 dims, const int *footprint, int2 footprint_dims)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dims.x || y >= dims.y) {
    return;
  }
  x -= footprint_dims.x / 2;
  y -= footprint_dims.y / 2;
  auto value = Monoid::identity_element();
  for (int j = 0; j < footprint_dims.y; ++j) {
    for (int i = 0; i < footprint_dims.x; ++i) {
      if (footprint[j*footprint_dims.x + i]) {
        value = Monoid::op(value, fetch2D<InputType, T>::fetch(input, x + i, y + j, dims));
      }
    }
  }
  x += footprint_dims.x / 2;
  y += footprint_dims.y / 2;
  output[y*dims.x + x] = value;
}

template <typename T, typename InputType>
void operate2D(MOP op, InputType tex, T* d_output, const int2 &dims, const int *d_footprint, const int2 &footprint_dims)
{
  const dim3 blockSize(8, 8);
  const dim3 gridSize((dims.x + blockSize.x - 1) / blockSize.x, (dims.y + blockSize.y - 1) / blockSize.y);
  if (op == MOP::Dilation) {
    reduce_2D<T, MaxMonoid<T>, InputType> << <gridSize, blockSize >> > (tex, d_output, dims, d_footprint, footprint_dims);
  } else {
    reduce_2D<T, MinMonoid<T>, InputType> << <gridSize, blockSize >> > (tex, d_output, dims, d_footprint, footprint_dims);
  }
  cudaDeviceSynchronize();
  CUDA_CHECK_LAST_ERROR;
}

template <typename T, typename Monoid>
__global__ void reduce_3D(cudaTextureObject_t input, T * output, int3 dims, const int *footprint, int3 footprint_dims)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= dims.x || y >= dims.y || z >= dims.z) {
    return;
  }
  x -= footprint_dims.x / 2;
  y -= footprint_dims.y / 2;
  z -= footprint_dims.z / 2;
  auto value = Monoid::identity_element();
  for (int k = 0; k < footprint_dims.z; ++k) {
    for (int j = 0; j < footprint_dims.y; ++j) {
      for (int i = 0; i < footprint_dims.x; ++i) {
        if (footprint[k*footprint_dims.x*footprint_dims.y + j * footprint_dims.x + i]) {
          value = Monoid::op(value, tex3D<T>(input, x + i, y + j, z + k));
        }
      }
    }
  }
  x += footprint_dims.x / 2;
  y += footprint_dims.y / 2;
  z += footprint_dims.z / 2;
  output[z*dims.x*dims.y + y*dims.x + x] = value;
}
template <typename T>
void operate3D(MOP op, cudaTextureObject_t tex, T* d_output, const int3 &dims, const int *d_footprint, const int3 &footprint_dims)
{
  const dim3 blockSize(8, 8, 8);
  const dim3 gridSize((dims.x + blockSize.x - 1) / blockSize.x, (dims.y + blockSize.y - 1) / blockSize.y, (dims.z + blockSize.z - 1) / blockSize.z);
  if (op == MOP::Dilation) {
    reduce_3D<T, MaxMonoid<T>> << <gridSize, blockSize >> > (tex, d_output, dims, d_footprint, footprint_dims);
  } else {
    reduce_3D<T, MinMonoid<T>> << <gridSize, blockSize >> > (tex, d_output, dims, d_footprint, footprint_dims);
  }
  cudaDeviceSynchronize();
  CUDA_CHECK_LAST_ERROR;
}

void flip(const int *input, int *output, const int2 &dims)
{
  for (int j = 0; j < dims.y; ++j) {
    for (int i = 0; i < dims.x; ++i) {
      output[j*dims.x + i] = input[(dims.y-j-1)*dims.x + (dims.x-i-1)];
    }
  }
}

template<typename T>
void operate2D(const T * h_input, T * h_output, const int2 & dims, const int *h_footprint, const int2 &footprint_dims, const std::vector<MOP> &ops, cudaTextureAddressMode addressMode)
{
  auto footprint_size = PROD2(footprint_dims);
  std::vector<int> reflected(footprint_size);
  flip(h_footprint, reflected.data(), footprint_dims);
  auto d_footprint = make_shared_device<int>(footprint_size);
  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_footprint.get(), reflected.data(), sizeof(int)*footprint_size, cudaMemcpyHostToDevice));
  scopedTextureObject2D<T> input(h_input, dims, cudaMemcpyHostToDevice, cudaArrayDefault, addressMode, cudaFilterModePoint);
  size_t image_size = PROD2(dims);
  auto output = make_shared_device<T>(image_size);
  for (int i = 0; i < ops.size(); ++i) {
    operate2D<T, cudaTextureObject_t>(ops[i], input.get(), output.get(), dims, d_footprint.get(), footprint_dims);
    if (i < ops.size() - 1) {
      input.updateArray(output.get(), cudaMemcpyDeviceToDevice);
    }
  }
  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(h_output, output.get(), image_size * sizeof(T), cudaMemcpyDeviceToHost));
}

template<typename T>
void dilate2D(const T * h_input, T * h_output, const int2 & dims, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode)
{
  std::vector<MOP> ops(n_iter, MOP::Dilation);
  operate2D<T>(h_input, h_output, dims, h_footprint, footprint_dims, ops, addressMode);
}
template<typename T>
void erode2D(const T * h_input, T * h_output, const int2 & dims, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode)
{
  std::vector<MOP> ops(n_iter, MOP::Erosion);
  operate2D<T>(h_input, h_output, dims, h_footprint, footprint_dims, ops, addressMode);
}
template<typename T>
void close2D(const T * h_input, T * h_output, const int2 & dims, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode)
{
  std::vector<MOP> ops(n_iter, MOP::Dilation);
  ops.reserve(2 * n_iter);
  for (int i = 0; i < n_iter; ++i) {
    ops.push_back(MOP::Erosion);
  }
  operate2D<T>(h_input, h_output, dims, h_footprint, footprint_dims, ops, addressMode);
}
template<typename T>
void open2D(const T * h_input, T * h_output, const int2 & dims, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode)
{
  std::vector<MOP> ops(n_iter, MOP::Erosion);
  ops.reserve(2 * n_iter);
  for (int i = 0; i < n_iter; ++i) {
    ops.push_back(MOP::Dilation);
  }
  operate2D<T>(h_input, h_output, dims, h_footprint, footprint_dims, ops, addressMode);
}

void flip(const int *input, int *output, const int3 &dims)
{
  for (int k = 0; k < dims.z; ++k) {
    for (int j = 0; j < dims.y; ++j) {
      for (int i = 0; i < dims.x; ++i) {
        output[k*dims.x*dims.y + j*dims.x + i] = input[(dims.z - k - 1)*dims.x*dims.y + (dims.y - j - 1)*dims.x + (dims.x - i - 1)];
      }
    }
  }
}
template<typename T>
void operate3D(const T * h_input, T * h_output, const int3 & dims, const int *h_footprint, const int3 &footprint_dims, const std::vector<MOP> &ops, cudaTextureAddressMode addressMode)
{
  auto footprint_size = PROD3(footprint_dims);
  std::vector<int> reflected(footprint_size);
  flip(h_footprint, reflected.data(), footprint_dims);
  auto d_footprint = make_shared_device<int>(footprint_size);
  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_footprint.get(), reflected.data(), sizeof(int)*footprint_size, cudaMemcpyHostToDevice));
  scopedTextureObject<T> input(h_input, dims, cudaMemcpyHostToDevice, cudaArrayDefault, addressMode, cudaFilterModePoint);
  size_t image_size = PROD3(dims);
  auto output = make_shared_device<T>(image_size);
  for (int i = 0; i < ops.size(); ++i) {
    operate3D<T>(ops[i], input.get(), output.get(), dims, d_footprint.get(), footprint_dims);
    if (i < ops.size() - 1) {
      input.updateArray(output.get(), cudaMemcpyDeviceToDevice);
    }
  }
  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(h_output, output.get(), image_size * sizeof(T), cudaMemcpyDeviceToHost));
}
template<typename T>
void dilate3D(const T * h_input, T * h_output, const int3 & dims, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode)
{
  std::vector<MOP> ops(n_iter, MOP::Dilation);
  operate3D<T>(h_input, h_output, dims, h_footprint, footprint_dims, ops, addressMode);
}
template<typename T>
void erode3D(const T * h_input, T * h_output, const int3 & dims, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode)
{
  std::vector<MOP> ops(n_iter, MOP::Erosion);
  operate3D<T>(h_input, h_output, dims, h_footprint, footprint_dims, ops, addressMode);
}
template<typename T>
void close3D(const T * h_input, T * h_output, const int3 & dims, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode)
{
  std::vector<MOP> ops(n_iter, MOP::Dilation);
  ops.reserve(2 * n_iter);
  for (int i = 0; i < n_iter; ++i) {
    ops.push_back(MOP::Erosion);
  }
  operate3D<T>(h_input, h_output, dims, h_footprint, footprint_dims, ops, addressMode);
}
template<typename T>
void open3D(const T * h_input, T * h_output, const int3 & dims, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode addressMode)
{
  std::vector<MOP> ops(n_iter, MOP::Erosion);
  ops.reserve(2 * n_iter);
  for (int i = 0; i < n_iter; ++i) {
    ops.push_back(MOP::Dilation);
  }
  operate3D<T>(h_input, h_output, dims, h_footprint, footprint_dims, ops, addressMode);
}

#define instantiate(TYPE) \
template void dilate2D<TYPE>(const TYPE*, TYPE*, const int2&, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode); \
template void erode2D<TYPE>(const TYPE*, TYPE*, const int2&, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode); \
template void close2D<TYPE>(const TYPE*, TYPE*, const int2&, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode); \
template void open2D<TYPE>(const TYPE*, TYPE*, const int2&, const int *h_footprint, const int2 &footprint_dims, int n_iter, cudaTextureAddressMode); \
template void dilate3D<TYPE>(const TYPE*, TYPE*, const int3&, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode); \
template void erode3D<TYPE>(const TYPE*, TYPE*, const int3&, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode); \
template void close3D<TYPE>(const TYPE*, TYPE*, const int3&, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode); \
template void open3D<TYPE>(const TYPE*, TYPE*, const int3&, const int *h_footprint, const int3 &footprint_dims, int n_iter, cudaTextureAddressMode); \
template void operate2D<TYPE, TYPE*>(MOP op, TYPE* tex, TYPE* d_output, const int2 &dims, const int *d_footprint, const int2 &footprint_dims); \

//template void operate2D<uint8_t, uint8_t*>(MOP op, uint8_t* tex, uint8_t* d_output, const int2 &dims, const int *d_footprint, const int2 &footprint_dims);

instantiate(int8_t)
instantiate(int16_t)
instantiate(int32_t)
//instantiate(int64_t)
instantiate(uint8_t)
instantiate(uint16_t)
instantiate(uint32_t)
//instantiate(uint64_t)
instantiate(float)
//instantiate(double)
