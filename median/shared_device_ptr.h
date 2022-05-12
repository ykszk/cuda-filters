#ifndef SHARED_DEVICE_PTR_H
#define SHARED_DEVICE_PTR_H
#include <cuda_runtime_api.h>
#include <cstddef>
#include <memory>
#include <thrust/device_ptr.h>

template <typename T>
class shared_device_ptr
{
public:
  shared_device_ptr(size_t size)
  {
    T *p;
    cudaMalloc(&p, size * sizeof(T));
    device_ptr_ = PointerType(p, deleter);
  }
  shared_device_ptr(size_t size, const T* host_ptr)
  {
    T *p;
    cudaMalloc(&p, size * sizeof(T));
    device_ptr_ = PointerType(p, deleter);
    cudaMemcpyAsync(p, host_ptr, size * sizeof(T), cudaMemcpyHostToDevice);
  }
  shared_device_ptr(T& device_ptr)
  {
    device_ptr_ = PointerType(device_ptr, deleter);
  }
  T* get()
  {
    return device_ptr_.get();
  }
  thrust::device_ptr<T> get_thrust_device_ptr()
  {
    return thrust::device_ptr<T>(device_ptr_.get());
  }
  operator T*()
  {
    return device_ptr_.get();
  }
  operator thrust::device_ptr<T>()
  {
    return thrust::device_ptr<T>(device_ptr_.get());
  }
private:
  static void deleter(T* device_ptr)
  {
    cudaFree(device_ptr);
  }
  typedef std::shared_ptr <T> PointerType;
  PointerType device_ptr_;

};

#endif /* SHARED_DEVICE_PTR_H */
