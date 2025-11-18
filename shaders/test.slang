StructuredBuffer<float> a;
RWStructuredBuffer<float> b;

[shader("compute")]
[numthreads(1024)]
void add(uint3 threadId: SV_DispatchThreadID) {
  uint i = threadId.x;
  b[i] += a[i];
}
