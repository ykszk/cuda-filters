#include "tempTextureObject.h"
#include "cuda_safe_call.h"
#include <cstring> //for memset
namespace
{
  template <typename T>
  cudaArray* createCudaArray(const T *data, const int3 &dims, cudaMemcpyKind kind, unsigned int flags)
  {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
    cudaExtent extent = make_cudaExtent(dims.x, dims.y, dims.z);
    cudaArray *cuArray;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc3DArray(&cuArray, &channelDesc, extent, flags));
    cudaMemcpy3DParms parms = { 0 };
    parms.srcPtr = make_cudaPitchedPtr(const_cast<T*>(data), dims.x * sizeof(T), dims.x, dims.y);
    parms.dstArray = cuArray;
    parms.extent = extent;
    parms.kind = kind;
    CUDA_SAFE_CALL_NO_SYNC(cudaMemcpy3DAsync(&parms));
    return cuArray;
  }

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
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    return texObj;
  }
}

template <typename T>
tempTextureObject<T>::tempTextureObject(const T *data, const int3 &dims, cudaMemcpyKind kind, unsigned int array_flags, cudaTextureAddressMode addressMode, cudaTextureFilterMode filterMode)
{
  cuArray = createCudaArray(data, dims, kind, array_flags);
  tex = create3DTexture(cuArray, addressMode, filterMode);
}

template <typename T>
tempTextureObject<T>::~tempTextureObject()
{
  cudaDestroyTextureObject(tex);
  cudaFreeArray(cuArray);
}
template <typename T>
cudaTextureObject_t& tempTextureObject<T>::get()
{
  return tex;
}

template class tempTextureObject<float>;
template class tempTextureObject<unsigned  char>;
template class tempTextureObject<short>;
