RWTexture2D<float4> field;

// StructuredBuffer<float4> colors;
// min is at minmax[0], max at minmax[NMINMAXVALUES-1]
// StructuredBuffer<float> values;
// StructuredBuffer<float> minmax;

[[SpecializationConstant]]
const uint NMINMAXVALUES = 1;

[shader("compute")]
[numthreads(8, 4, 1)]
void writecolor(uint2 threadId: SV_DispatchThreadID, uniform float4* colors, uniform float* values, uniform float* minmax, uniform float2 *vminmax) {
  uint width = 0;
  uint height = 0;
  field.GetDimensions(width, height);
  uint vidx = threadId.y * width + threadId.x;
  float val = 0;
  float effmax = minmax[NMINMAXVALUES-1];
  float effmin = minmax[0];
  if (vminmax->x == vminmax->y) {
    val = values[vidx];
  } else {
    effmin = vminmax->x;
    effmax = vminmax->y;
    val = clamp(values[vidx], effmin, effmax);
  }
  uint coloridx = uint(255.01 * (val - effmin) /
                       (effmax - effmin));
  int2 coord = int2(threadId.xy);
  field[coord] = colors[coloridx];
}

