[shader("compute")]
[numthreads(32, 1, 1)]
void xfer(uint i: SV_DispatchThreadID, uniform float2* source, uniform float* dest) {
  dest[i] = dot(source[i], source[i]);
}
