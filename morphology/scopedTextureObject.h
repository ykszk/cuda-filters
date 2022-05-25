#include <cuda_runtime.h>
#include "cuda_safe_call.h"
#include <cstring> //for memset
#include <memory>

template <typename T>
class scopedTextureObject
{
public:
  scopedTextureObject(const T *data, const int3 &dims, cudaMemcpyKind kind, unsigned int array_flags, cudaTextureAddressMode addressMode, cudaTextureFilterMode filterMode)
    : dims_(dims)
  {
    cuArray = createCudaArray(dims, array_flags);
    updateArray(data, kind);
    tex = create3DTexture(cuArray, addressMode, filterMode);
  }
  ~scopedTextureObject()
  {
    cudaDestroyTextureObject(tex);
    cudaFreeArray(cuArray);
  }
  cudaTextureObject_t get()
  {
    return tex;
  }
  void updateArray(const T *data, cudaMemcpyKind kind)
  {
    cudaExtent extent = make_cudaExtent(dims_.x, dims_.y, dims_.z);
    cudaMemcpy3DParms parms = { 0 };
    parms.srcPtr = make_cudaPitchedPtr(const_cast<T*>(data), dims_.x * sizeof(T), dims_.x, dims_.y);
    parms.dstArray = cuArray;
    parms.extent = extent;
    parms.kind = kind;
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy3DAsync(&parms));
   }
  static cudaArray* createCudaArray(const int3 &dims, unsigned int flags)
  {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    channelDesc.x = 8 * sizeof(T);
    channelDesc.y = channelDesc.z = channelDesc.w = 0;
    cudaExtent extent = make_cudaExtent(dims.x, dims.y, dims.z);
    cudaArray *tmp_ptr;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc3DArray(&tmp_ptr, &channelDesc, extent, flags));
    return tmp_ptr;
  }
private:
  cudaArray* cuArray;
  cudaTextureObject_t tex;
  int3 dims_;

  cudaTextureObject_t create3DTexture(cudaArray *cuArray, cudaTextureAddressMode addressMode, cudaTextureFilterMode filterMode)
  {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = texDesc.addressMode[1] = texDesc.addressMode[2] = addressMode;
    texDesc.filterMode = filterMode;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj = 0;
    CUDA_SAFE_CALL_NO_SYNC(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
  }
};

template <typename T>
class scopedTextureObject2D
{
public:
  scopedTextureObject2D(const T *data, const int2 &dims, cudaMemcpyKind kind, unsigned int array_flags, cudaTextureAddressMode addressMode, cudaTextureFilterMode filterMode)
    : dims_(dims)
  {
    cuArray = createCudaArray(dims, array_flags);
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpyToArray(cuArray, 0, 0, data, sizeof(T)*dims.x*dims.y, kind));
    tex = create2DTexture(cuArray, addressMode, filterMode);
  }
  ~scopedTextureObject2D()
  {
    cudaDestroyTextureObject(tex);
    cudaFreeArray(cuArray);
  }
  void updateArray(const T *data, cudaMemcpyKind kind)
  {
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpyToArray(cuArray, 0, 0, data, sizeof(T)*dims_.x*dims_.y, kind));
  }
  cudaTextureObject_t get()
  {
    return tex;
  }
  static cudaArray* createCudaArray(const int2 &dims, unsigned int flags)
  {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    channelDesc.x = 8 * sizeof(T);
    channelDesc.y = channelDesc.z = channelDesc.w = 0;
    cudaArray *tmp_ptr;
    CUDA_SAFE_CALL_NO_SYNC(cudaMallocArray(&tmp_ptr, &channelDesc, dims.x, dims.y));
    return tmp_ptr;
  }
private:
  cudaArray* cuArray;
  cudaTextureObject_t tex;
  int2 dims_;
  scopedTextureObject2D(const scopedTextureObject2D&) = delete;
  scopedTextureObject2D& operator=(const scopedTextureObject2D&) = delete;

  cudaTextureObject_t create2DTexture(cudaArray *cuArray, cudaTextureAddressMode addressMode, cudaTextureFilterMode filterMode)
  {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = texDesc.addressMode[1] = addressMode;
    texDesc.filterMode = filterMode;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;
    cudaTextureObject_t texObj = 0;
    CUDA_SAFE_CALL_NO_SYNC(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
  }
};

