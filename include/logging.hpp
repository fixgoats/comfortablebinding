#include <format>
#include <iostream>
#include <print>

template <class... Args>
inline void logDebug(std::format_string<Args...> fstr, Args&&... args) {
#ifndef NDEBUG
  std::print("Debug: ");
  std::print(fstr, args...);
  std::cout << std::endl;
#endif // !NDEBUG
}
