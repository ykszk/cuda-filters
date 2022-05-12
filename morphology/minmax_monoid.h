#include <npp.h>
#include "util.h"

template <typename T>
 __host__ __device__ constexpr T numeric_limits_min()
{
  static_assert(AlwaysFalse<T>(), "Specialized function is always needed.");
  return 1;
}

template<> inline int8_t numeric_limits_min<int8_t>() { return NPP_MIN_8S; };
template<> inline int16_t numeric_limits_min<int16_t>() { return NPP_MIN_16S; };
template<> inline int32_t numeric_limits_min<int32_t>() { return NPP_MIN_32S; };
template<> inline int64_t numeric_limits_min<int64_t>() { return NPP_MIN_64S; };
template<> inline uint8_t numeric_limits_min<uint8_t>() { return NPP_MIN_8U; };
template<> inline uint16_t numeric_limits_min<uint16_t>() { return NPP_MIN_16U; };
template<> inline uint32_t numeric_limits_min<uint32_t>() { return NPP_MIN_32U; };
template<> inline uint64_t numeric_limits_min<uint64_t>() { return NPP_MIN_64U; };
template<> inline float numeric_limits_min<float>() { return -NPP_MAXABS_32F; };
template<> inline double numeric_limits_min<double>() { return -NPP_MAXABS_64F; };

template <typename T>
__host__ __device__ constexpr T numeric_limits_max()
{
  static_assert(AlwaysFalse<T>(), "Specialized function is always needed.");
  return 0;
}
template<> inline int8_t numeric_limits_max<int8_t>() { return NPP_MAX_8S; };
template<> inline int16_t numeric_limits_max<int16_t>() { return NPP_MAX_16S; };
template<> inline int32_t numeric_limits_max<int32_t>() { return NPP_MAX_32S; };
template<> inline int64_t numeric_limits_max<int64_t>() { return NPP_MAX_64S; };
template<> inline uint8_t numeric_limits_max<uint8_t>() { return NPP_MAX_8U; };
template<> inline uint16_t numeric_limits_max<uint16_t>() { return NPP_MAX_16U; };
template<> inline uint32_t numeric_limits_max<uint32_t>() { return NPP_MAX_32U; };
template<> inline uint64_t numeric_limits_max<uint64_t>() { return NPP_MAX_64U; };
template<> inline float numeric_limits_max<float>() { return NPP_MAXABS_32F; };
template<> inline double numeric_limits_max<double>() { return NPP_MAXABS_64F; };

template <typename T>
struct MaxMonoid
{
  static __host__ __device__ T identity_element()
  {
    return numeric_limits_min<T>();
  }
  static __host__ __device__ T op(T a, T b)
  {
    return max(a, b);
  }
};
template <typename T>
struct MinMonoid
{
  static __host__ __device__ T identity_element()
  {
    return numeric_limits_max<T>();
  }
  static __host__ __device__ T op(T a, T b)
  {
    return min(a, b);
  }
};
