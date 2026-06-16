import complex;

struct Params {
  float a;
  float b;
  float c;
  float d;
  float e;
  float dt;
};

[shader("compute")]
[numthreads(32, 1, 1)]
void rk4(uint i: SV_DispatchThreadID, float2* psir, float2* old_psir, float* nR, float* p, Params* pars) {
  float2 psi2 = psir[i];
  float2 psi1 = old_psir[i];
  float2 inter_psi = (psi2 + psi1) / 2;
  float k1 = -(pars->Gamma + pars->R * dot(psi1, psi1)) * nR[i] + p[i];
  float k2 = -(pars->Gamma + pars->R * dot(inter_psi, inter_psi)) * (nR[i] + 0.5 * k1 * pars->dt) + p[i];
  float k3 = -(pars->Gamma + pars->R * dot(inter_psi, inter_psi)) * (nR[i] + 0.5 * k2 * pars->dt) + p[i];
  float k4 = -(pars->Gamma + pars->R * dot(psi2, psi2)) * (nR[i] + pars->dt * k3) + p[i];
}

float2 rprop(float2 psi, float nR, float p, Params pars) {
  return c_exp(0.5 * dt * float2(pars.d * nR + pars.f, pars.b * dot(psi, psi) + pars.c * p + pars.e * nR));
}

[shader("compute")]
[numthreads(32, 1, 1)]
void rstep(uint i: SV_DispatchThreadID, float2* psir, float2* old_psir, float* nR, float* p, Params* pars) {
  float2 prop = rprop(old_psir[i], nR[i], p[i], pars);
  psir[i] =  c_mul(rprop[i], old_psir[i]);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void rstep_last(uint i: SV_DispatchThreadID, float2* psir, float* nR, float* p, Params* pars) {
  float2 prop = rprop(psir[i], nR[i], p[i], pars);
  psir[i] =  c_mul(rprop[i], psir[i]);
}

