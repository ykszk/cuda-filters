#ifndef TIMER_H
#define TIMER_H
#include <chrono>
#include <iostream>

template <typename T>
inline const char* TimeUnitString() {return "unknown_unit";};

template <>
inline const char* TimeUnitString<std::chrono::nanoseconds>() {return "ns";};
template <>
inline const char* TimeUnitString<std::chrono::microseconds>() {return "us";};
template <>
inline const char* TimeUnitString<std::chrono::milliseconds>() {return "ms";};
template <>
inline const char* TimeUnitString<std::chrono::seconds>() {return "sec";};
template <>
inline const char* TimeUnitString<std::chrono::minutes>() {return "min";};
template <>
inline const char* TimeUnitString<std::chrono::hours>() {return "h";};

template <typename ToDuration = std::chrono::milliseconds>
class timer
{
public:
  timer(bool auto_output = false)
    : auto_output_(auto_output)
  {
    start = std::chrono::system_clock::now();
  }
  ~timer()
  {
    if (auto_output_) {
      std::cout << elapsed_time() << " [" << TimeUnitString<ToDuration>() << "]" << std::endl;
    }
  }
  typename ToDuration::rep elapsed_time()
  {
    auto elapsed = std::chrono::duration_cast<ToDuration>(std::chrono::system_clock::now() - start);
    return elapsed.count();
  }
  const char* unit()
  {
    return TimeUnitString<ToDuration>();
  }
private:
  std::chrono::system_clock::time_point start;
  bool auto_output_;
};

#endif /* TIMER_H */
