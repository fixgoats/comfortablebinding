import complex;

[shader("compute")]
[numthreads(32, 1, 1)]
void kstep(uint i: SV_DispatchThreadID, float2* psir, float2* kprop) {
  psir[i] =  c_mul(kprop[i], psir[i]);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void rstep(uint i: SV_DispatchThreadID, float2* psir, float2* rprop) {
  psir[i] = c_mul(rprop[i], psir[i]);
}
