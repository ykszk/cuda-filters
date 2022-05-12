#ifndef UTIL_H
#define UTIL_H

template <typename T>
constexpr __host__ __device__ bool AlwaysFalse()
{
  return false;
}

#endif /* UTIL_H */
