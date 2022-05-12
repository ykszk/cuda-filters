#include "fillhole.h"
#include <iostream>
#include <vector>
#include "RawImageIO.h"
#include <cuda_runtime.h>
#include "timer.h"
#include <cuda_profiler_api.h>
using namespace std;
int main(int argc, char *argv[])
{
  if (argc < 6) {
    cerr << "Usage:" << argv[0] << " <input> <output> <dims.x> <dims.y> <dims.z>" << endl;
    return 1;
  }
  int dx = atoi(argv[3]);
  int dy = atoi(argv[4]);
  int dz = atoi(argv[5]);
  cout << dx << "x" << dy << "x" << dz << endl;
  int3 dims = make_int3(dx,dy,dz);
cudaProfilerStart();
  int image_size = dims.x*dims.y*dims.z;
  {
    cout << "initialize device " << flush;
    timer<> t(true);
    cudaSetDevice(0);
  }
  typedef uint8_t T;
  T *input;
  input = load_raw<T>(argv[1], image_size);
  T *output;
  cudaMallocManaged(&output, image_size * sizeof(T));
  cout << "2d fill hole" << endl;
  {
    timer<> t(true);
    try {
      fillhole2D_consecutive(input, output, dims);
    } catch (std::exception &e) {
      cout << e.what() << endl;
    }
  }
cudaProfilerStop();
  //save_raw(output, image_size, argv[2]);
  return 0;
}
