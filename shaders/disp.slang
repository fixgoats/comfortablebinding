import complex;

struct Pair {
  float f;
  uint idx;
}

StructuredBuffer<float2> uh;
StructuredBuffer<float2> points;
StructuredBuffer<Pair> delta;
StructuredBuffer<uint> indices;
StructuredBuffer<float2> 
RWStructuredBuffer<float> density;

[SpecializationConstant]
const uint XGROUPSIZE = 1;
[SpecializationConstant]
const uint YGROUPSIZE = 1;
[SpecializationConstant]
const uint DELTASIZE = 1;
[SpecializationConstant]
const float dk = 0.0;
[SpecializationConstant]
const float de = 0.0;
[SpecializationConstant]
const float kstart = 0.0;
[SpecializationConstant]
const float estart = 0.0;

[shader("compute")]
[numthreads(XGROUPSIZE, YGROUPSIZE, 1)]
void section(uint3 threadId: SV_DispatchThreadID) {
  uint2 idx = threadId.xy;
  uint npoints;
  uint stride;
  points.GetDimensions(npoints, stride);
  float e = estart + idx.y * de;
  float k = kstart + idx.x * dk;
  for ()

}
