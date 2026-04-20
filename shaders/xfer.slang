StructuredBuffer<float2> source;
RWStructuredBuffer<float> dest;

[shader("compute")]
[numthreads(32, 1, 1)]
void xfer(uint i: SV_DispatchThreadID) {
  dest[i] = dot(source[i], source[i]);
}
