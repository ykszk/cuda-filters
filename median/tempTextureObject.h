#include <cuda_runtime.h>

template <typename T>
class tempTextureObject
{
public:
  tempTextureObject(const T *data, const int3 &dims, cudaMemcpyKind kind, unsigned int array_flags, cudaTextureAddressMode addressMode, cudaTextureFilterMode filterMode);
  ~tempTextureObject();
  cudaTextureObject_t& get();
private:
  cudaArray *cuArray;
  cudaTextureObject_t tex;
};
