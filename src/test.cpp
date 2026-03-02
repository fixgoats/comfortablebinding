#include "typedefs.h"
#include <iostream>

// template<typename... Args>
// decltype(auto) last(Args&&... args){
//    return (std::forward<Args>(args), ...);
// }

template <class Arg, class... Args>
void myPrint(std::ostream& out, Arg&& arg, Args&&... args) {
  out << std::forward<Arg>(arg);
  ((out << ' ' << std::forward<Args>(args)), ...);
}

int main(int argc, char* argv[]) {
  myPrint(std::cout, "halló", "kælir");
  return 0;
}
