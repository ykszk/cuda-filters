#include <cstdio>
#include <cstdlib>
#  define CUDA_SAFE_CALL_NO_SYNC(call) do {                            \
    cudaError err = call;                                               \
    if(cudaSuccess != err) {                                           \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",     \
              __FILE__, __LINE__, cudaGetErrorString( err) );           \
      throw(cudaGetErrorString(err));                                               \
    } } while (0)

#define CUFFT_SAFE_CALL(call) do {                                      \
    cufftResult_t err = call;                                           \
    if (CUFFT_SUCCESS != err) {                                         \
      fprintf(stderr, "cuFFT error %i in file '%s' in line %i : " ,err, __FILE__, __LINE__); \
      throw(cudaGetErrorString(err));                                               \
    } } while (0)
