#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ " : " TOSTRING(__LINE__)

#define CUDA_SAFE_CALL_NO_SYNC(call) do {                            \
    cudaError err = call;                                               \
    if(cudaSuccess != err) {                                           \
      fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",     \
              __FILE__, __LINE__, cudaGetErrorString( err) );           \
      throw(std::runtime_error(std::string(cudaGetErrorString(err))+AT));                                               \
    } } while (0)

#define CUFFT_SAFE_CALL(call) do {                                      \
    cufftResult_t err = call;                                           \
    if (CUFFT_SUCCESS != err) {                                         \
      fprintf(stderr, "cuFFT error %i in file '%s' in line %i : " ,err, __FILE__, __LINE__); \
      throw(std::runtime_error(cudaGetErrorString(err)));                                               \
    } } while (0)
