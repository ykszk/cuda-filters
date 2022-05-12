#ifndef REDUCE_H
#define REDUCE_H
#include <cstdint>

template <typename T>
__inline__ __device__
T warpReduceSum(T val) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val |= __shfl_down(val, offset);
  return val;
}

template <typename T>
__inline__ __device__
T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x%warpSize;
  int wid = threadIdx.x / warpSize;
  val = warpReduceSum(val);
  if (val) {
    return val;
  }

  //write reduced value to shared memory
  if (lane == 0) shared[wid] = val;
  __syncthreads();

  //ensure we only grab a value from shared memory if that warp existed
  val = (threadIdx.x<blockDim.x / warpSize) ? shared[lane] : T(0);
  if (wid == 0) val = warpReduceSum(val);

  return val;
}

template <typename InputType, typename OutputType>
__global__ void device_reduce_stable_kernel(InputType *in, OutputType *out, int N) {
  OutputType sum = OutputType(0);
  for (int i = blockIdx.x*blockDim.x + threadIdx.x; i<N; i += blockDim.x*gridDim.x) {
    sum |= in[i];
  }
  sum = blockReduceSum(sum);
  if (threadIdx.x == 0)
    out[blockIdx.x] = sum;
}

template <typename InputType, typename OutputType>
void device_reduce_stable(InputType *in, OutputType *out, size_t N, int count) {
  int threads = 512;
  int blocks = std::min<int>((N + threads - 1) / threads, 1024);

  for (int i = 0; i < count; ++i) {
    device_reduce_stable_kernel << <blocks, threads >> > (in+i*N, out+i, N);
    device_reduce_stable_kernel << <1, 1024 >> > (out+i, out+i, blocks);
  }
}


#endif /* REDUCE_H */
