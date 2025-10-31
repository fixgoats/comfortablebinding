StructuredBuffer<float2> uh;
StructuredBuffer<float2> points;
StructuredBuffer<float> delta;
RWStructuredBuffer<float> density;

[SpecializationConstant]
const uint XGROUPSIZE = 1;
[SpecializationConstant]
const uint YGROUPSIZE = 1;
[SpecializationConstant]
const uint DELTASIZE = 1;
[SpecializationConstant]
const double dk = 0.0;

[shader("compute")]
[numthreads(XGROUPSIZE, YGROUPSIZE, 1)]
void section(uint3 threadId: SV_DispatchThreadID, uint3 groupId: SV_GroupID) {
  uint2 idx = threadId.xy;
  float2 kvec[DELTASIZE];
  for (uint j = 0; j < points.length(); ++j) {
    float2 kj = c_exp(float2(0, dot(points[j], float2(dk*idx.x, dk*idx.y))));
    for (uint i = 0; i < DELTASIZE; ++i) {
      kvec[i] += c_mul(uh[i * DELTASIZE + j], kj);
    }
  }
  for (uint i = 0; i < DELTASIZE; ++i) {
    density[idx.y * XGROUPSIZE * ]
  }
}
