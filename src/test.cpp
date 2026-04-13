#include "Eigen/Core"
#include "Eigen/Dense"
#include "geometry.h"
#include "highfive/highfive.hpp"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cxxopts.hpp>
#include <execution>
#include <iostream>
#include <random>
#include <vector>

using Eigen::Vector3d;

template <class T>
struct Node {
  std::vector<std::shared_ptr<Node<T>>> children;
  T data{};

  Node(T obj) : data{obj} { std::cout << "Node(T) constructor.\n"; }
  Node(std::vector<std::shared_ptr<Node<T>>> ch, T obj)
      : children{ch}, data{obj} {}
};

template <class T>
struct Tree {
  std::shared_ptr<Node<T>> root;
  Tree<T>(Node<T>* r) : root{r} {
    std::cout << "Tree<T>(Node<T>) constructor.\n";
  }
  Tree<T>(std::shared_ptr<Node<T>> r) : root{r} {}
  Tree<T>(T r) : root{std::make_shared<Node<T>>(r)} {}
};

Tree<u64> add_trees(Tree<u64>& a, Tree<u64>& b) {
  spdlog::debug("Function: add_trees.");
  spdlog::debug("Making array of pointers to a and b roots.");
  return Tree<u64>{
      new Node<u64>{{a.root, b.root}, a.root->data + b.root->data}};
}

int main(int argc, char* argv[]) {
  cxxopts::Options options("test program", "bleh");
  options.add_options()("v,verbose", "Verbose output", cxxopts::value<bool>());
  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
  if (result["v"].count()) {
    spdlog::set_level(spdlog::level::trace);
  }

  Tree<u64> x{1};
  Tree<u64> y{2};
  Tree<u64> u{1};
  Tree<u64> v{3};

  auto d = add_trees(x, y);
  auto e = add_trees(u, v);
  auto f = add_trees(d, e);
  std::cout << "x num: " << x.root->data << '\n';
  std::cout << "y num: " << y.root->data << '\n';
  std::cout << "u num: " << u.root->data << '\n';
  std::cout << "v num: " << v.root->data << '\n';
  std::cout << "d num: " << d.root->data << '\n';
  std::cout << "e num: " << e.root->data << '\n';
  std::cout << "f num: " << f.root->data << '\n';
  // auto cur_node = pascal.root;
  // for (u64 i = 0; i < 5; ++i) {
  //   cur_node = cur_node->children
  // }
  return 0;
}
