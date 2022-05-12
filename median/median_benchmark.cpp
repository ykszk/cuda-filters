#include "medianFilter3D.h"
#include "shared_device_ptr.h"
#include "timer.h"
#include <iostream>
#include <string>

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <curand.h>
#include <cuda_profiler_api.h>
#include "RawImageIO.h"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

using namespace std;

void show_help(const char *argv0)
{
  cerr << "Usage:" << argv0 << " [options]\n";
  cerr << "Options:\n";
  cerr << " -s size : Filter size. 3, 5 or 7. Default is 3.\n";
  cerr << " -o filename : Output filename.\n";
  cerr << " -g filename : Save generated file.\n";
  cerr << " -z size : Z size.\n";
}

template <typename T>
int benchmark(std::string &filename, std::string &g_filename, int size, int z_size)
{
  int3 dims = make_int3(512, 512, z_size);
  int n = dims.x*dims.y*dims.z;
  curandGenerator_t gen;

  using PixelType = short;
  /* Allocate n floats on device */
  timer<> tt(false);
  shared_device_ptr<PixelType> devData(n);
  shared_device_ptr<PixelType> d_output(n);
  cout << "memory " << tt.elapsed_time() << endl;

  /* Create pseudo-random number generator */
  CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

  /* Set seed */
  CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

  /* Generate n floats on device */
//  CURAND_CALL(curandGenerateUniform(gen, devData, n));
  /* Cleanup */
  CURAND_CALL(curandDestroyGenerator(gen));

  {
    cout << "Filter size = " << size << endl;
    timer<> t(true);
    cudaProfilerStart();
    medianFilter3D(devData.get_thrust_device_ptr(), d_output.get_thrust_device_ptr(), dims, size, cudaAddressModeMirror);
    cudaDeviceSynchronize();
    cudaProfilerStop();
  }

  if (!g_filename.empty()) {
    save_raw<PixelType>(d_output.get_thrust_device_ptr(), dims.x*dims.y*dims.z, g_filename);
  }
  if (!filename.empty()) {
    save_raw<PixelType>(d_output.get_thrust_device_ptr(), dims.x*dims.y*dims.z, filename);
  }
  {
    timer<> t(true);
    std::vector<T> v(n);
    cudaMemcpy(v.data(), d_output.get(), n, cudaMemcpyDeviceToHost);
  }
  return 0;

}
#include <sstream>
std::string gpu_info()
{
  std::stringstream ss;
  ss << "CUDA version:   v" << CUDART_VERSION << endl;
  int devCount;
  cudaGetDeviceCount(&devCount);

  for (int i = 0; i < devCount; ++i)
  {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, i);
    ss << "GPU " << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
  }
  return ss.str();
}

int main(int argc, char *argv[])
{
  int size = 3;
  int z_size = 512;
  std::string filename;
  std::string g_filename;
  std::string type("float");
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help")==0) {
      show_help(argv[0]);
      return 1;
    }
    if (strcmp(argv[i], "-s") == 0) {
      ++i;
      if (argc <= i) {
        cerr << "-s option requires integer value.";
        return 1;
      } else {
        size = argv[i][0] - '0'; //convert to int
      }
      continue;
    }
    if (strcmp(argv[i], "-o") == 0) {
      ++i;
      if (argc <= i) {
        cerr << "-o option requires filename.";
        return 1;
      } else {
        filename = argv[i];
      }
      continue;
    }
    if (strcmp(argv[i], "-g") == 0) {
      ++i;
      if (argc <= i) {
        cerr << "-g option requires filename.";
        return 1;
      } else {
        g_filename = argv[i];
      }
      continue;
    }
    if (strcmp(argv[i], "-t") == 0) {
      ++i;
      if (argc <= i) {
        cerr << "-t option requires typename.";
        return 1;
      }
      else {
        type = argv[i];
      }
      continue;
    }
    if (strcmp(argv[i], "-z") == 0) {
      ++i;
      if (argc <= i) {
        cerr << "-z option requires integer value.";
        return 1;
      } else {
        z_size = atoi(argv[i]);
      }
      continue;
    }
    cerr << "Illegal option : " << argv[i] << endl;
    show_help(argv[0]);
    return 1;
  }
  cout << gpu_info() << endl;

  if (type == "float") {
  auto ret = benchmark<float>(filename, g_filename, size, z_size);
  cout << ret << endl;
  return ret;
  }
  else if (type == "short") {
  auto ret = benchmark<short>(filename, g_filename, size, z_size);
  cout << ret << endl;
  return ret;
  }
  else if (type == "uchar") {
  auto ret = benchmark<unsigned char>(filename, g_filename, size, z_size);
  cout << ret << endl;
  return ret;
  }
  else {
    cout << "Unknown type " << type << endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
