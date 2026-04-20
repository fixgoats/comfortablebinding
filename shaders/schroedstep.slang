import complex;

RWStructuredBuffer<float2> psir;
StructuredBuffer<float2> rprop;
StructuredBuffer<float2> kprop;

[shader("compute")]
[numthreads(32, 1, 1)]
void kstep(uint i: SV_DispatchThreadID) {
  psir[i] =  c_mul(float2(1, 0), psir[i]); // c_mul(kprop[i], psir[i]);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void rstep(uint i: SV_DispatchThreadID) {
  psir[i] = c_mul(rprop[i], psir[i]);
}
