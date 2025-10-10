#include <iostream>
#include <vector>

template<int... ns>
void uhh() {
  std::cout << sizeof...(ns);
}

template<class... Bs>
void foo(Bs... bs) {
  std::cout << sizeof...(bs);
}

int main(int argc, char* argv[]) {
  uhh<1, 2, 3, 4, 5, 6>();
  foo(4, 3, 2, 1, 5);
  return 0;
}
