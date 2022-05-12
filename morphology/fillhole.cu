#include "morphology.h"
#include "fillhole.h"
#include "scopedTextureObject.h"
#include <vector>
#include <algorithm>
#include <numeric>

#include <iostream>

template <typename T>
std::shared_ptr<T> make_shared_device(size_t size)
{
  T *ptr;
  CUDA_SAFE_CALL_NO_SYNC(cudaMalloc(&ptr, sizeof(T)*size));
  std::shared_ptr<T> sptr(ptr, [](T* ptr) {cudaFree(ptr); });
  return sptr;
}
template <typename T>
std::shared_ptr<T> make_shared_managed(size_t size)
{
  T *ptr;
  CUDA_SAFE_CALL_NO_SYNC(cudaMallocHost(&ptr, sizeof(T)*size));
  std::shared_ptr<T> sptr(ptr, [](T* ptr) {cudaFreeHost(ptr); });
  return sptr;
}

#include <thrust/transform.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

template <typename T>
struct masking : public thrust::binary_function<T, T, T>
{
  __host__ __device__
    float operator()(T x, T m) { return m ? 1 : x; }
};
template <typename T>
struct not_equal : public thrust::binary_function<T, T, T>
{
  __host__ __device__
    float operator()(T x, T y) { return x != y; }
};
template <typename T>
struct logical_or : public thrust::binary_function<T, T, T>
{
  __host__ __device__
    float operator()(T x, T y) { return max(x, y); }
};

__global__ void mark_unchanged(uint8_t *input, uint8_t *mask, uint8_t *prev, uint8_t *masked, uint8_t *unchanged, size_t size)
{
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= size) {
    return;
  }
  uint8_t masked_input = mask[i] ? 1 : input[i];
  masked[i] = masked_input;
  unchanged[i] = prev[i] != masked_input;
}

__global__ void check_if_changed(uint8_t *input, uint8_t *output, size_t size, size_t num)
{
  size_t i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i >= num) {
    return;
  }
  auto row = input + i * size;
  for (int x = 0; x < size; ++x) {
    if (row[x] > 0) {
      output[i] = 1;
      return;
    }
  }
  output[i] = 0;
}

#define UCDP(p) thrust::device_ptr<uint8_t>(p)
#define TO_PTR(v) thrust::raw_pointer_cast(v.data())

template <typename T>
void copy_thru_managed(T* dst, const T *src, T *buff, size_t size, cudaMemcpyKind kind)
{
  if (kind == cudaMemcpyHostToDevice) {
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(buff, src, size * sizeof(T), cudaMemcpyHostToHost));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(dst, buff, size * sizeof(T), cudaMemcpyHostToDevice));
  } else {
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(buff, src, size * sizeof(T), cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(dst, buff, size * sizeof(T), cudaMemcpyHostToHost));
  }
}
void fillhole2D(const uint8_t * h_input, uint8_t * h_output, const int2 & dims)
{
  auto image_size = dims.x * dims.y;
  auto footprint_dims = make_int2(3, 3);
  int footprint_size = footprint_dims.x*footprint_dims.y;
  std::vector<int> footprint(footprint_size, 0);
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      int x = i - 1, y = j - 1;
      if (abs(x) + abs(y) <= 1) {
        footprint[j*footprint_dims.x + i] = 1;
      }
    }
  }
  int threads = 512;
  int blocks = (image_size + threads - 1) / threads;
  auto m_buffer = make_shared_managed<uint8_t>(image_size);
  thrust::device_vector<uint8_t> d_mask(image_size);
  copy_thru_managed<uint8_t>(TO_PTR(d_mask), h_input, m_buffer.get(), image_size, cudaMemcpyHostToDevice);
  auto d_footprint = make_shared_device<int>(footprint_size);
  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_footprint.get(), footprint.data(), sizeof(int)*footprint_size, cudaMemcpyHostToDevice));
  thrust::device_vector<uint8_t> d_input(image_size);
 // CUDA_SAFE_CALL_NO_SYNC(cudaMemset(thrust::raw_pointer_cast(d_input.data()), 1, image_size));
  //init << <blocks, threads >> > (TO_PTR(d_input), image_size);
  CUDA_SAFE_CALL_NO_SYNC(cudaMemset(TO_PTR(d_input), 1, image_size * sizeof(uint8_t)));
  thrust::device_vector<uint8_t> output(image_size);
  thrust::device_vector<uint8_t> buffer(image_size);
  int count = 0;
  while (true)
  {
    operate2D<uint8_t, uint8_t*>(MOP::Erosion, thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(output.data()), dims, d_footprint.get(), footprint_dims);
    mark_unchanged << <blocks, threads >> > (TO_PTR(output), TO_PTR(d_mask), TO_PTR(d_input), TO_PTR(buffer), TO_PTR(output), image_size);
//    thrust::transform(output.begin(), output.end(), d_mask.begin(), buffer.begin(), masking<uint8_t>());
//    thrust::transform(buffer.begin(), buffer.end(), d_input.begin(), output.begin(), not_equal<uint8_t>()); // pixels that's been changed will have value 1 otherwise 0
    bool changed = thrust::reduce(output.begin(), output.end(), 0, logical_or<uint8_t>());
    if (changed) {
      thrust::copy(buffer.begin(), buffer.end(), d_input.begin());
    } else {
      break;
    }
  }
//  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(h_output, thrust::raw_pointer_cast(d_input.data()), image_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
  copy_thru_managed(h_output, TO_PTR(d_input), m_buffer.get(), image_size, cudaMemcpyDeviceToHost);
}

#include "minmax_monoid.h"
#include "device_fetch.h"
#include "reduce.h"
template <typename T, typename Monoid, typename InputType>
__global__ void reduce_2D(InputType input, T * output, int2 dims)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dims.x || y >= dims.y) {
    return;
  }
  //if (x > count && y > count && x < dims.x-count-1 && y < dims.y-count-1)
  //{
  //  output[y*dims.x + x] = 1;
  //  return;
  //}
  //x -= footprint_dims.x / 2;
  //y -= footprint_dims.y / 2;
  //auto value = Monoid::identity_element();
  //for (int j = 0; j < footprint_dims.y; ++j) {
  //  for (int i = 0; i < footprint_dims.x; ++i) {
  //    if (footprint[j*footprint_dims.x + i]) {
  //      value = Monoid::op(value, fetch2D<InputType, T>::fetch(input, x + i, y + j, dims));
  //    }
  //  }
  //}
  //x += footprint_dims.x / 2;
  //y += footprint_dims.y / 2;
  auto value = Monoid::identity_element();
  value = Monoid::op(value, fetch2D<InputType, T>::fetch(input, x, y-1, dims));
  value = Monoid::op(value, fetch2D<InputType, T>::fetch(input, x-1, y, dims));
  value = Monoid::op(value, fetch2D<InputType, T>::fetch(input, x, y, dims));
  value = Monoid::op(value, fetch2D<InputType, T>::fetch(input, x+1, y, dims));
  value = Monoid::op(value, fetch2D<InputType, T>::fetch(input, x, y+1, dims));
  output[y*dims.x + x] = value;
}

template <typename T, typename Monoid, typename InputType>
__global__ void reduce_2D(InputType input, T * output, int3 dims)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  if (x >= dims.x || y >= dims.y || z >= dims.z) {
    return;
  }
  auto value = Monoid::identity_element();
  value = Monoid::op(value, fetch2D_layered<InputType, T>::fetch(input, x, y-1, z, dims));
  value = Monoid::op(value, fetch2D_layered<InputType, T>::fetch(input, x-1, y, z, dims));
  value = Monoid::op(value, fetch2D_layered<InputType, T>::fetch(input, x, y, z, dims));
  value = Monoid::op(value, fetch2D_layered<InputType, T>::fetch(input, x+1, y, z, dims));
  value = Monoid::op(value, fetch2D_layered<InputType, T>::fetch(input, x, y+1, z, dims));
  output[y*dims.x + x] = value;
}
#define CUDA_CHECK_LAST_ERROR \
auto err = cudaGetLastError();\
if (err != cudaSuccess) {\
  throw(std::runtime_error(std::string(cudaGetErrorString(err)) + AT));\
}
void fillhole2D_consecutive(const uint8_t * h_input_volume, uint8_t * h_output_volume, const int3 & dims3d)
{
  auto dims = make_int2(dims3d.x, dims3d.y);
  auto image_size = dims.x * dims.y;
  auto volume_size = dims3d.x * dims3d.y * dims3d.z;
  int threads = 512;
  int blocks = (image_size + threads - 1) / threads;

  auto m_buffer = make_shared_managed<uint8_t>(volume_size);
  auto d_changed = make_shared_device<uint8_t>(1024);
  typedef uint8_t T;
  auto d_input = make_shared_device<T>(volume_size);
  auto d_mask = make_shared_device<T>(volume_size);
  auto d_output = make_shared_device<T>(volume_size);
  auto d_buffer = make_shared_device<T>(volume_size);
  auto m_changed = make_shared_managed<size_t>(dims3d.z);
  const dim3 blockSize(8, 8, 8);
  const dim3 gridSize((dims.x + blockSize.x - 1) / blockSize.x, (dims.y + blockSize.y - 1) / blockSize.y, (dims3d.z + blockSize.z - 1) / blockSize.z);

  copy_thru_managed<uint8_t>(d_mask.get(), h_input_volume, m_buffer.get(), volume_size, cudaMemcpyHostToDevice);
  CUDA_SAFE_CALL_NO_SYNC(cudaMemset(d_input.get(), 1, volume_size * sizeof(T)));
  //init << <blocks, threads >> > (d_input.get(), volume_size);
  int count = 0;
  std::cout << "blocks" << (dims3d.z + threads - 1) / threads << " thraeds " << threads << std::endl;
  std::cout << dims3d.z << std::endl;
  while (true)
  {
    ++count;
    reduce_2D<T, MinMonoid<T>, T*> << <gridSize, blockSize >> > (d_input.get(), d_output.get(), dims);
    mark_unchanged << <blocks, threads >> > (d_output.get(), d_mask.get(), d_input.get(), d_buffer.get(), d_output.get(), volume_size);
    check_if_changed << <(dims3d.z + threads - 1) / threads, threads >> > (d_output.get(), d_changed.get(), image_size, dims3d.z);

    cudaDeviceSynchronize();
    CUDA_CHECK_LAST_ERROR;
    //device_reduce_stable(d_output.get(), d_changed.get(), image_size, dims3d.z);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(m_changed.get(), d_changed.get(), dims3d.z * sizeof(T), cudaMemcpyDeviceToHost));
    auto changed = std::accumulate(m_changed.get(), m_changed.get() + dims3d.z, 0, [](T a, T b) {return std::max<T>(a,b); });
    if (changed) {
      CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(d_input.get(), d_buffer.get(), volume_size * sizeof(T), cudaMemcpyDeviceToDevice));
    } else {
      break;
    }
  }
      std::cout << "count:" << count << std::endl;
      //  CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy(h_output, thrust::raw_pointer_cast(d_input.data()), image_size * sizeof(uint8_t), cudaMemcpyDeviceToHost));
      //copy_thru_managed(h_output, d_input.get(), m_buffer.get(), image_size, cudaMemcpyDeviceToHost);
  copy_thru_managed(h_output_volume, d_buffer.get(), m_buffer.get(), volume_size, cudaMemcpyDeviceToHost);
}

