#include "tempTextureObject.h"
#include "cuda_safe_call.h"
#include <cstring> //for memset
namespace
{
  cudaArray* createCudaArray(const float *data, const int3 &dims, cudaMemcpyKind kind, unsigned int flags)
  {
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaExtent extent = make_cudaExtent(dims.x, dims.y, dims.z);
    cudaArray *cuArray;
    CUDA_SAFE_CALL_NO_SYNC(cudaMalloc3DArray(&cuArray, &channelDesc, extent, flags));
    cudaMemcpy3DParms parms = { 0 };
    parms.srcPtr = make_cudaPitchedPtr(const_cast<float*>(data), dims.x * sizeof(float), dims.x, dims.y);
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

tempTextureObject::tempTextureObject(const float *data, const int3 &dims, cudaMemcpyKind kind, unsigned int array_flags, cudaTextureAddressMode addressMode, cudaTextureFilterMode filterMode)
{
  cuArray = createCudaArray(data, dims, kind, array_flags);
  tex = create3DTexture(cuArray, addressMode, filterMode);
}

tempTextureObject::~tempTextureObject()
{
  cudaDestroyTextureObject(tex);
  cudaFreeArray(cuArray);
}
cudaTextureObject_t& tempTextureObject::get()
{
  return tex;
}
