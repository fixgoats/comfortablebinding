#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "colormaps.hpp"
#include "geometry.h"
#include "highfive/eigen.hpp"
#include "highfive/highfive.hpp"
#include "io.h"
#include "kdtree.h"
#include "raylib.h"
#include "spdlog/spdlog.h"
#include <algorithm>
#include <cmath>
#include <cxxopts.hpp>
#include <gsl/gsl_sf.h>
#include <random>

using Eigen::SparseMatrix;
constexpr Color rtwilight[256]{
    {244, 235, 245, 255}, {241, 233, 242, 255}, {238, 231, 240, 255},
    {235, 230, 237, 255}, {232, 228, 235, 255}, {229, 226, 233, 255},
    {226, 224, 231, 255}, {223, 222, 228, 255}, {220, 221, 226, 255},
    {217, 219, 224, 255}, {214, 217, 223, 255}, {211, 215, 221, 255},
    {208, 214, 219, 255}, {205, 212, 217, 255}, {202, 210, 216, 255},
    {199, 209, 214, 255}, {195, 207, 213, 255}, {192, 205, 212, 255},
    {189, 204, 210, 255}, {186, 202, 209, 255}, {183, 200, 208, 255},
    {180, 199, 207, 255}, {176, 197, 206, 255}, {173, 195, 205, 255},
    {170, 194, 204, 255}, {167, 192, 203, 255}, {164, 190, 202, 255},
    {161, 189, 202, 255}, {158, 187, 201, 255}, {155, 185, 200, 255},
    {153, 184, 200, 255}, {150, 182, 199, 255}, {147, 180, 199, 255},
    {145, 178, 198, 255}, {142, 177, 198, 255}, {140, 175, 197, 255},
    {137, 173, 197, 255}, {135, 171, 196, 255}, {133, 169, 196, 255},
    {130, 167, 196, 255}, {128, 166, 195, 255}, {126, 164, 195, 255},
    {124, 162, 195, 255}, {122, 160, 194, 255}, {120, 158, 194, 255},
    {118, 156, 194, 255}, {117, 154, 193, 255}, {115, 152, 193, 255},
    {113, 150, 193, 255}, {112, 148, 192, 255}, {110, 146, 192, 255},
    {109, 144, 192, 255}, {108, 142, 192, 255}, {107, 140, 191, 255},
    {105, 138, 191, 255}, {104, 136, 191, 255}, {103, 134, 190, 255},
    {102, 132, 190, 255}, {102, 130, 189, 255}, {101, 127, 189, 255},
    {100, 125, 188, 255}, {99, 123, 188, 255},  {99, 121, 188, 255},
    {98, 119, 187, 255},  {98, 117, 186, 255},  {97, 114, 186, 255},
    {97, 112, 185, 255},  {97, 110, 185, 255},  {96, 108, 184, 255},
    {96, 105, 183, 255},  {96, 103, 182, 255},  {95, 101, 182, 255},
    {95, 98, 181, 255},   {95, 96, 180, 255},   {95, 94, 179, 255},
    {95, 91, 178, 255},   {95, 89, 177, 255},   {95, 87, 176, 255},
    {95, 84, 175, 255},   {94, 82, 174, 255},   {94, 79, 172, 255},
    {94, 77, 171, 255},   {94, 75, 170, 255},   {94, 72, 168, 255},
    {94, 70, 167, 255},   {94, 67, 165, 255},   {93, 65, 164, 255},
    {93, 62, 162, 255},   {93, 60, 160, 255},   {93, 57, 158, 255},
    {92, 55, 156, 255},   {92, 53, 154, 255},   {91, 50, 152, 255},
    {91, 48, 150, 255},   {90, 46, 147, 255},   {90, 43, 145, 255},
    {89, 41, 142, 255},   {88, 39, 139, 255},   {87, 37, 137, 255},
    {87, 35, 134, 255},   {86, 33, 131, 255},   {84, 31, 127, 255},
    {83, 29, 124, 255},   {82, 28, 121, 255},   {81, 26, 117, 255},
    {79, 25, 114, 255},   {78, 23, 110, 255},   {76, 22, 107, 255},
    {74, 21, 103, 255},   {73, 20, 100, 255},   {71, 19, 96, 255},
    {69, 18, 93, 255},    {67, 18, 89, 255},    {65, 17, 86, 255},
    {63, 16, 82, 255},    {61, 15, 79, 255},    {59, 15, 76, 255},
    {57, 14, 72, 255},    {55, 13, 69, 255},    {52, 12, 66, 255},
    {50, 11, 63, 255},    {48, 11, 60, 255},    {46, 10, 57, 255},
    {44, 9, 54, 255},     {41, 8, 51, 255},     {39, 7, 48, 255},
    {37, 6, 45, 255},     {34, 6, 42, 255},     {35, 6, 41, 255},
    {37, 7, 43, 255},     {40, 7, 45, 255},     {42, 8, 46, 255},
    {45, 9, 48, 255},     {47, 10, 50, 255},    {50, 11, 52, 255},
    {53, 12, 53, 255},    {55, 13, 55, 255},    {58, 14, 56, 255},
    {60, 15, 58, 255},    {63, 15, 59, 255},    {66, 16, 61, 255},
    {68, 17, 62, 255},    {71, 17, 64, 255},    {74, 18, 65, 255},
    {76, 19, 66, 255},    {79, 19, 67, 255},    {81, 20, 69, 255},
    {84, 21, 70, 255},    {87, 21, 71, 255},    {89, 22, 72, 255},
    {92, 23, 73, 255},    {95, 23, 74, 255},    {97, 24, 75, 255},
    {100, 25, 76, 255},   {103, 25, 76, 255},   {105, 26, 77, 255},
    {108, 27, 78, 255},   {111, 28, 78, 255},   {113, 28, 78, 255},
    {116, 29, 79, 255},   {118, 30, 79, 255},   {121, 31, 79, 255},
    {123, 32, 80, 255},   {126, 34, 80, 255},   {128, 35, 80, 255},
    {131, 36, 80, 255},   {133, 37, 80, 255},   {135, 39, 80, 255},
    {137, 40, 80, 255},   {140, 42, 80, 255},   {142, 44, 80, 255},
    {144, 45, 80, 255},   {146, 47, 80, 255},   {148, 49, 80, 255},
    {150, 51, 79, 255},   {152, 52, 79, 255},   {154, 54, 79, 255},
    {156, 56, 79, 255},   {158, 58, 79, 255},   {159, 60, 79, 255},
    {161, 62, 79, 255},   {163, 64, 79, 255},   {165, 66, 79, 255},
    {166, 68, 79, 255},   {168, 70, 80, 255},   {169, 73, 80, 255},
    {171, 75, 80, 255},   {172, 77, 80, 255},   {174, 79, 81, 255},
    {175, 81, 81, 255},   {177, 84, 81, 255},   {178, 86, 82, 255},
    {179, 88, 82, 255},   {181, 90, 83, 255},   {182, 93, 83, 255},
    {183, 95, 84, 255},   {184, 97, 85, 255},   {185, 100, 86, 255},
    {186, 102, 87, 255},  {187, 105, 88, 255},  {188, 107, 89, 255},
    {189, 109, 90, 255},  {190, 112, 91, 255},  {191, 114, 92, 255},
    {192, 117, 94, 255},  {193, 119, 95, 255},  {194, 122, 97, 255},
    {194, 124, 98, 255},  {195, 126, 100, 255}, {196, 129, 102, 255},
    {197, 131, 104, 255}, {197, 134, 106, 255}, {198, 136, 108, 255},
    {198, 139, 110, 255}, {199, 141, 112, 255}, {200, 144, 115, 255},
    {200, 146, 117, 255}, {201, 149, 120, 255}, {201, 151, 122, 255},
    {202, 154, 125, 255}, {202, 156, 128, 255}, {203, 159, 131, 255},
    {204, 161, 134, 255}, {204, 163, 137, 255}, {205, 166, 140, 255},
    {205, 168, 143, 255}, {206, 171, 146, 255}, {207, 173, 150, 255},
    {207, 175, 153, 255}, {208, 178, 156, 255}, {209, 180, 160, 255},
    {210, 182, 163, 255}, {211, 185, 167, 255}, {212, 187, 170, 255},
    {212, 189, 174, 255}, {213, 191, 177, 255}, {214, 194, 181, 255},
    {216, 196, 184, 255}, {217, 198, 188, 255}, {218, 200, 192, 255},
    {219, 202, 195, 255}, {220, 205, 199, 255}, {222, 207, 202, 255},
    {223, 209, 206, 255}, {224, 211, 209, 255}, {226, 213, 213, 255},
    {227, 216, 216, 255}, {229, 218, 219, 255}, {230, 220, 223, 255},
    {232, 222, 226, 255}, {234, 224, 229, 255}, {236, 226, 232, 255},
    {238, 229, 235, 255}, {240, 231, 238, 255}, {242, 233, 241, 255},
    {244, 235, 245, 255},
};

constexpr Color rviridis[256]{
    {68, 1, 84, 255},    {68, 2, 85, 255},    {69, 3, 87, 255},
    {69, 5, 88, 255},    {69, 6, 90, 255},    {70, 8, 91, 255},
    {70, 9, 93, 255},    {70, 11, 94, 255},   {70, 12, 96, 255},
    {71, 14, 97, 255},   {71, 15, 98, 255},   {71, 17, 100, 255},
    {71, 18, 101, 255},  {71, 20, 102, 255},  {72, 21, 104, 255},
    {72, 22, 105, 255},  {72, 24, 106, 255},  {72, 25, 108, 255},
    {72, 26, 109, 255},  {72, 28, 110, 255},  {72, 29, 111, 255},
    {72, 30, 112, 255},  {72, 32, 113, 255},  {72, 33, 115, 255},
    {72, 34, 116, 255},  {72, 36, 117, 255},  {72, 37, 118, 255},
    {72, 38, 119, 255},  {72, 39, 120, 255},  {71, 41, 121, 255},
    {71, 42, 121, 255},  {71, 43, 122, 255},  {71, 44, 123, 255},
    {71, 46, 124, 255},  {70, 47, 125, 255},  {70, 48, 126, 255},
    {70, 49, 126, 255},  {70, 51, 127, 255},  {69, 52, 128, 255},
    {69, 53, 129, 255},  {69, 54, 129, 255},  {68, 56, 130, 255},
    {68, 57, 131, 255},  {68, 58, 131, 255},  {67, 59, 132, 255},
    {67, 60, 132, 255},  {67, 62, 133, 255},  {66, 63, 133, 255},
    {66, 64, 134, 255},  {65, 65, 134, 255},  {65, 66, 135, 255},
    {65, 67, 135, 255},  {64, 69, 136, 255},  {64, 70, 136, 255},
    {63, 71, 136, 255},  {63, 72, 137, 255},  {62, 73, 137, 255},
    {62, 74, 137, 255},  {61, 75, 138, 255},  {61, 77, 138, 255},
    {60, 78, 138, 255},  {60, 79, 138, 255},  {59, 80, 139, 255},
    {59, 81, 139, 255},  {58, 82, 139, 255},  {58, 83, 139, 255},
    {57, 84, 140, 255},  {57, 85, 140, 255},  {56, 86, 140, 255},
    {56, 87, 140, 255},  {55, 88, 140, 255},  {55, 89, 140, 255},
    {54, 91, 141, 255},  {54, 92, 141, 255},  {53, 93, 141, 255},
    {53, 94, 141, 255},  {52, 95, 141, 255},  {52, 96, 141, 255},
    {51, 97, 141, 255},  {51, 98, 141, 255},  {51, 99, 141, 255},
    {50, 100, 142, 255}, {50, 101, 142, 255}, {49, 102, 142, 255},
    {49, 103, 142, 255}, {48, 104, 142, 255}, {48, 105, 142, 255},
    {47, 106, 142, 255}, {47, 107, 142, 255}, {47, 108, 142, 255},
    {46, 109, 142, 255}, {46, 110, 142, 255}, {45, 111, 142, 255},
    {45, 112, 142, 255}, {45, 112, 142, 255}, {44, 113, 142, 255},
    {44, 114, 142, 255}, {43, 115, 142, 255}, {43, 116, 142, 255},
    {43, 117, 142, 255}, {42, 118, 142, 255}, {42, 119, 142, 255},
    {41, 120, 142, 255}, {41, 121, 142, 255}, {41, 122, 142, 255},
    {40, 123, 142, 255}, {40, 124, 142, 255}, {40, 125, 142, 255},
    {39, 126, 142, 255}, {39, 127, 142, 255}, {38, 128, 142, 255},
    {38, 129, 142, 255}, {38, 130, 142, 255}, {37, 131, 142, 255},
    {37, 131, 142, 255}, {37, 132, 142, 255}, {36, 133, 142, 255},
    {36, 134, 142, 255}, {35, 135, 142, 255}, {35, 136, 142, 255},
    {35, 137, 142, 255}, {34, 138, 141, 255}, {34, 139, 141, 255},
    {34, 140, 141, 255}, {33, 141, 141, 255}, {33, 142, 141, 255},
    {33, 143, 141, 255}, {32, 144, 141, 255}, {32, 145, 140, 255},
    {32, 146, 140, 255}, {32, 147, 140, 255}, {31, 147, 140, 255},
    {31, 148, 140, 255}, {31, 149, 139, 255}, {31, 150, 139, 255},
    {31, 151, 139, 255}, {30, 152, 139, 255}, {30, 153, 138, 255},
    {30, 154, 138, 255}, {30, 155, 138, 255}, {30, 156, 137, 255},
    {30, 157, 137, 255}, {30, 158, 137, 255}, {30, 159, 136, 255},
    {30, 160, 136, 255}, {31, 161, 136, 255}, {31, 162, 135, 255},
    {31, 163, 135, 255}, {31, 163, 134, 255}, {32, 164, 134, 255},
    {32, 165, 134, 255}, {33, 166, 133, 255}, {33, 167, 133, 255},
    {34, 168, 132, 255}, {35, 169, 131, 255}, {35, 170, 131, 255},
    {36, 171, 130, 255}, {37, 172, 130, 255}, {38, 173, 129, 255},
    {39, 174, 129, 255}, {40, 175, 128, 255}, {41, 175, 127, 255},
    {42, 176, 127, 255}, {43, 177, 126, 255}, {44, 178, 125, 255},
    {46, 179, 124, 255}, {47, 180, 124, 255}, {48, 181, 123, 255},
    {50, 182, 122, 255}, {51, 183, 121, 255}, {53, 183, 121, 255},
    {54, 184, 120, 255}, {56, 185, 119, 255}, {57, 186, 118, 255},
    {59, 187, 117, 255}, {61, 188, 116, 255}, {62, 189, 115, 255},
    {64, 190, 114, 255}, {66, 190, 113, 255}, {68, 191, 112, 255},
    {70, 192, 111, 255}, {72, 193, 110, 255}, {73, 194, 109, 255},
    {75, 194, 108, 255}, {77, 195, 107, 255}, {79, 196, 106, 255},
    {81, 197, 105, 255}, {83, 198, 104, 255}, {85, 198, 102, 255},
    {88, 199, 101, 255}, {90, 200, 100, 255}, {92, 201, 99, 255},
    {94, 201, 98, 255},  {96, 202, 96, 255},  {98, 203, 95, 255},
    {101, 204, 94, 255}, {103, 204, 92, 255}, {105, 205, 91, 255},
    {108, 206, 90, 255}, {110, 206, 88, 255}, {112, 207, 87, 255},
    {115, 208, 85, 255}, {117, 208, 84, 255}, {119, 209, 82, 255},
    {122, 210, 81, 255}, {124, 210, 79, 255}, {127, 211, 78, 255},
    {129, 212, 76, 255}, {132, 212, 75, 255}, {134, 213, 73, 255},
    {137, 213, 72, 255}, {139, 214, 70, 255}, {142, 215, 68, 255},
    {144, 215, 67, 255}, {147, 216, 65, 255}, {149, 216, 63, 255},
    {152, 217, 62, 255}, {155, 217, 60, 255}, {157, 218, 58, 255},
    {160, 218, 57, 255}, {163, 219, 55, 255}, {165, 219, 53, 255},
    {168, 220, 51, 255}, {171, 220, 50, 255}, {173, 221, 48, 255},
    {176, 221, 46, 255}, {179, 221, 45, 255}, {181, 222, 43, 255},
    {184, 222, 41, 255}, {187, 223, 39, 255}, {189, 223, 38, 255},
    {192, 223, 36, 255}, {195, 224, 35, 255}, {197, 224, 33, 255},
    {200, 225, 32, 255}, {203, 225, 30, 255}, {205, 225, 29, 255},
    {208, 226, 28, 255}, {211, 226, 27, 255}, {213, 226, 26, 255},
    {216, 227, 25, 255}, {219, 227, 24, 255}, {221, 227, 24, 255},
    {224, 228, 24, 255}, {226, 228, 24, 255}, {229, 228, 24, 255},
    {232, 229, 25, 255}, {234, 229, 25, 255}, {237, 229, 26, 255},
    {239, 230, 27, 255}, {242, 230, 28, 255}, {244, 230, 30, 255},
    {247, 230, 31, 255}, {249, 231, 33, 255}, {251, 231, 35, 255},
    {254, 231, 36, 255}};

#define SET_STRUCT_FIELD(key, tbl)                                             \
  if (tbl.contains(#key))                                                      \
  key = *tbl[#key].value<decltype(key)>()

struct DrivenDissConf {
  std::string outfile;
  std::string pointPath;
  std::optional<f64> searchRadius;
  f64 p;
  f64 alpha;
  f64 j;
  f64 rscale;
  RangeConf<f64> t;

  DrivenDissConf(const toml::table& tbl) {
    SET_STRUCT_FIELD(outfile, tbl);
    SET_STRUCT_FIELD(pointPath, tbl);
    SET_STRUCT_FIELD(alpha, tbl);
    SET_STRUCT_FIELD(p, tbl);
    SET_STRUCT_FIELD(j, tbl);
    SET_STRUCT_FIELD(rscale, tbl);
    if (tbl.contains("searchRadius")) {
      searchRadius = tbl["searchRadius"].value<f64>().value();
    }
    t = tblToRange(*tbl["t"].as_table());
  }
};

#undef SET_STRUCT_FIELD

s32 toScreenX(f64 r, f64 min, f64 max, s32 dim) {
  return (s32)(((r - min) / (max - min)) * (f64)dim);
}

s32 toScreenY(f64 r, f64 min, f64 max, s32 dim) {
  return (s32)(((r - max) / (min - max)) * (f64)dim);
}

template <class T>
u8 mapToColor(T x, T min, T max) {
  u32 bleh = 256 * ((x - min) / (max - min));
  return bleh > 256 ? 255 : bleh < 0 ? 0 : bleh;
}

template <class IIt, class T>
std::vector<Color> valuesToColor(IIt it, IIt end, const Color* cmap, T min,
                                 T max) {
  std::vector<Color> out(end - it);
  std::transform(it, end, out.begin(),
                 [&](auto x) { return cmap[mapToColor(x, min, max)]; });
  return out;
}

template <class IIt>
std::vector<Color> valuesToColor(IIt it, IIt end, const Color* cmap) {
  std::vector<Color> out(end - it);
  auto minmax = std::ranges::minmax(it, end);
  std::transform(it, end, out.begin(), [&](auto x) {
    return cmap[mapToColor(x, minmax.min, minmax.max)];
  });
  return out;
}

static void AddCodepointRange(Font* font, const char* fontPath, int start,
                              int stop) {
  int rangeSize = stop - start + 1;
  int currentRangeSize = font->glyphCount;

  // TODO: Load glyphs from provided vector font (if available),
  // add them to existing font, regenerating font image and texture

  int updatedCodepointCount = currentRangeSize + rangeSize;
  int* updatedCodepoints = (int*)RL_CALLOC(updatedCodepointCount, sizeof(int));

  // Get current codepoint list
  for (int i = 0; i < currentRangeSize; i++)
    updatedCodepoints[i] = font->glyphs[i].value;

  // Add new codepoints to list (provided range)
  for (int i = currentRangeSize; i < updatedCodepointCount; i++)
    updatedCodepoints[i] = start + (i - currentRangeSize);

  UnloadFont(*font);
  *font = LoadFontEx(fontPath, 32, updatedCodepoints, updatedCodepointCount);
  RL_FREE(updatedCodepoints);
}

struct Sim {
  f64 p;
  f64 alpha;
  f64 dt;
  SparseMatrix<c64> J;
  VectorXcd psi;
  VectorXcd k1;
  VectorXcd k2;
  VectorXcd k3;
  VectorXcd k4;

  Sim(f64 p, f64 alpha, f64 dt, SparseMatrix<c64> J, u32 n)
      : p{p}, alpha{alpha}, dt{dt}, J{J}, psi(n), k1(n), k2(n), k3(n), k4(n) {}

  VectorXcd f(const VectorXcd& psi) {
    return p * psi - c64{1, alpha} * psi.cwiseAbs2().cwiseProduct(psi) +
           J * psi;
  }

  void rk4step() {
    k1 = f(psi);
    k2 = f(psi + 0.5 * dt * k1);
    k3 = f(psi + 0.5 * dt * k2);
    k4 = f(psi + dt * k3);
    psi += (dt / 6.) * (k1 + 2 * k2 + 2 * k3 + k4);
  }
};

int main(int argc, char* argv[]) {
  cxxopts::Options options("Dynamic Simulations", "bleh");
  options.add_options()("config", "TOML configuration",
                        cxxopts::value<std::string>());
  options.parse_positional({"config"});
  cxxopts::ParseResult result;
  try {
    result = options.parse(argc, argv);
  } catch (const std::exception& exc) {
    std::cerr << "Exception: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
  // for (const auto& color : cm::twilight) {
  //   std::cout << (u32)(256 * color.rgb[0]) << ' ' << (u32)(256 *
  //   color.rgb[1])
  //             << ' ' << (u32)(256 * color.rgb[2]) << '\n';
  // }
  if (result["config"].count()) {
    std::string fname = result["config"].as<std::string>();

    toml::table tbl;
    spdlog::debug("Function: main");
    try {
      tbl = toml::parse_file(fname);
    } catch (const std::exception& err) {
      std::cerr << "Parsing file " << fname
                << " failed with exception: " << err.what() << '\n';
      return {};
    }

    auto pointfile = tbl["pointPath"].value<std::string>().value();
    HighFive::File pc(pointfile, HighFive::File::ReadOnly);
    spdlog::debug("Read pointfile.");
    Eigen::MatrixX2d points = pc.getDataSet("points").read<Eigen::MatrixX2d>();
    spdlog::debug("Read points.");
    Eigen::MatrixX2i couplings =
        pc.getDataSet("couplings").read<Eigen::MatrixX2i>();
    f64 rscale = tbl["rscale"].value<f64>().value();
    f64 p = tbl["p"].value<f64>().value();
    f64 alpha = tbl["alpha"].value<f64>().value();
    f64 j = tbl["j"].value<f64>().value();
    f64 dt = tbl["dt"].value<f64>().value();
    auto J = SparseC(points, couplings, [j, rscale](Vector2d d) {
      return j * c64{gsl_sf_bessel_J0(rscale * d.norm()),
                     gsl_sf_bessel_Y0(rscale * d.norm())};
    });
    const auto sol = Eigen::ComplexEigenSolver<MatrixXcd>(MatrixXcd(J));
    auto min_coeff = std::ranges::min_element(
        sol.eigenvalues(), [](c64 a, c64 b) { return a.imag() < b.imag(); });
    const f64 eff_p = (p - 1) * min_coeff->imag();
    Sim sim(eff_p, alpha, dt, J, points.rows());

    s32 width = 800, height = 800;

    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_WINDOW_RESIZABLE |
                   FLAG_WINDOW_TRANSPARENT);
    InitWindow(width, height, "raylib test");

    SetTargetFPS(60);

    std::random_device dev;
    std::mt19937 gen(dev());
    std::uniform_real_distribution<> dis(0.0, 2 * M_PI);

    for (u32 i = 0; i < points.rows(); ++i) {
      f64 x = dis(gen);
      sim.psi(i) = {1e-4 * cos(x), 1e-4 * sin(x)};
    }
    double xmin = points(Eigen::indexing::all, 0).minCoeff();
    double xmax = points(Eigen::indexing::all, 0).maxCoeff();
    double ymin = points(Eigen::indexing::all, 1).minCoeff();
    double ymax = points(Eigen::indexing::all, 1).maxCoeff();
    double exmin = xmin - 0.05 * (xmax - xmin);
    double eymin = ymin - 0.05 * (ymax - ymin);
    double exmax = xmax + 0.05 * (xmax - xmin);
    double eymax = ymax + 0.05 * (ymax - ymin);
    bool start = false;
    std::array<u32, 6> stepsPerFrame{1, 10, 20, 50, 100, 200};
    s32 stepOrder = 0;
    bool showStepsPerFrame = false;
    std::chrono::time_point<std::chrono::high_resolution_clock>
        timeWhenStepsShown;
    while (!WindowShouldClose()) {
      if (IsKeyPressed(KEY_SPACE)) {
        start = !start;
      }
      if (IsKeyPressed(KEY_R)) {
        for (u32 i = 0; i < points.rows(); ++i) {
          f64 x = dis(gen);
          sim.psi(i) = {1e-4 * cos(x), 1e-4 * sin(x)};
        }
      }
      if (IsKeyPressed(KEY_RIGHT)) {
        ++stepOrder;
        stepOrder = euclid_mod(stepOrder, 6);
        showStepsPerFrame = true;
        timeWhenStepsShown = std::chrono::high_resolution_clock::now();
      }
      if (IsKeyPressed(KEY_LEFT)) {
        --stepOrder;
        stepOrder = euclid_mod(stepOrder, 6);
        showStepsPerFrame = true;
        timeWhenStepsShown = std::chrono::high_resolution_clock::now();
      }
      width = GetScreenWidth();
      height = GetScreenHeight();
      VectorXd angles = (std::conj(sim.psi(0)) * sim.psi).cwiseArg();
      VectorXd norms = sim.psi.cwiseAbs2();
      auto maxnorm = norms.maxCoeff();
      auto colors =
          valuesToColor(angles.cbegin(), angles.cend(), rtwilight, -M_PI, M_PI);
      BeginDrawing();
      ClearBackground(WHITE);
      for (u32 i = 0; i < points.rows(); ++i) {
        DrawCircle(toScreenX(points(i, 0), exmin, exmax, width),
                   toScreenY(points(i, 1), eymin, eymax, height),
                   norms(i) * 12.0 / maxnorm, colors[i]);
      }
      // for (const auto& nb : couplings.rowwise()) {
      //   auto startx = toScreenX(points[nb[0]][0], exmin, exmax, width);
      //   auto starty = toScreenY(points[nb[0]][1], eymin, eymax, height);
      //   auto endx = toScreenX(points[nb[1]][0], exmin, exmax, width);
      //   auto endy = toScreenY(points[nb[1]][1], eymin, eymax, height);
      //   DrawLine(startx, starty, endx, endy, BLUE);
      // }
      if (showStepsPerFrame) {
        DrawText(TextFormat("Steps per frame: %d", stepsPerFrame[stepOrder]),
                 static_cast<f32>(width) - 230, 10, 20, BLACK);
        auto now = std::chrono::high_resolution_clock::now();
        showStepsPerFrame = std::chrono::duration_cast<std::chrono::seconds>(
                                now - timeWhenStepsShown)
                                .count() < 3;
      }
      if (!start) {
        DrawLineEx({10, 5}, {10, 45}, 10, BLACK);
        DrawLineEx({25, 5}, {25, 45}, 10, BLACK);
      }
      EndDrawing();
      if (IsKeyPressed(KEY_P)) {
        TakeScreenshot("graph.png");
      }
      if (start) {
        for (u32 i = 0; i < stepsPerFrame[stepOrder]; ++i) {
          sim.rk4step();
        }
      }
    }
    CloseWindow();
  }

  return 0;
}
