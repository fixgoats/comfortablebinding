RWStructuredBuffer<float> a;
StructuredBuffer<float> b;

[shader("compute")]
[numthreads(64)]
void add(uint threadId: SV_DispatchThreadID) {
  a[threadId.x] += b[threadId.x];
}
