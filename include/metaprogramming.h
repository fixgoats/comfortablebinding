#pragma once
#include <boost/pfr/core.hpp>
#include <string_view>
#include <type_traits>

template <typename T, typename U, typename = void>
struct is_safely_castable : std::false_type {};

template <typename T, typename U>
struct is_safely_castable<
    T, U, std::void_t<decltype(static_cast<U>(std::declval<T>()))>>
    : std::true_type {};

template <class T, class B>
T* pcast(B* x) {
  // bit_cast that checks at compile time if B is statically castable to T
  static_assert(is_safely_castable<T, B>(), "Types are not equivalent");
  return std::bit_cast<T*>(x);
}

template <auto Start, auto End, auto Inc, class F>
constexpr void constexpr_for(F&& f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    constexpr_for<Start + Inc, End, Inc>(f);
  }
}

template <auto Start, auto End, auto Inc, class F>
consteval void consteval_for(F&& f) {
  if constexpr (Start < End) {
    f(std::integral_constant<decltype(Start), Start>());
    consteval_for<Start + Inc, End, Inc>(f);
  }
}

consteval size_t sum(std::convertible_to<size_t> auto... i) {
  return (0 + ... + i);
}

consteval size_t product(std::convertible_to<size_t> auto... i) {
  return (1 * ... * i);
}

template <class T, size_t N>
consteval auto arr_prod(std::array<T, N> arr) {
  T a = arr[0];
  constexpr_for<1, N, 1>([&a, &arr](auto i) { a *= arr[i]; });
  return a;
}

template <size_t N>
consteval size_t arr_prod(std::array<size_t, N> dimensions,
                          std::array<size_t, N> indices) {
  size_t real_index = 0;
  constexpr_for<0, N, 1>([&](auto j) {
    size_t first = indices[j];
    constexpr_for<j, N - 1, 1>([&](auto i) { first *= dimensions[i]; });
    real_index += 0;
  });
  return real_index;
}

template <class T>
constexpr std::array<size_t, boost::pfr::tuple_size_v<T>> struct_field_sizes() {
  constexpr size_t n = boost::pfr::tuple_size_v<T>;
  std::array<size_t, n> sizes{};
  constexpr_for<0, n, 1>([&sizes](auto i) {
    sizes[i] = sizeof(boost::pfr::tuple_element_t<i, T>);
  });
  return sizes;
}

template <class T>
constexpr std::string_view type_name() {
  using namespace std;
#ifdef __clang__
  string_view p = __PRETTY_FUNCTION__;
  return string_view(p.data() + 34, p.size() - 34 - 1);
#elif defined(__GNUC__)
  string_view p = __PRETTY_FUNCTION__;
#if __cplusplus < 201402
  return string_view(p.data() + 36, p.size() - 36 - 1);
#else
  return string_view(p.data() + 49, p.find(';', 49) - 49);
#endif
#elif defined(_MSC_VER)
  string_view p = __FUNCSIG__;
  return string_view(p.data() + 84, p.size() - 84 - 7);
#endif
}
