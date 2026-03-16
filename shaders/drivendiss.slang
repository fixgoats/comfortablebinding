import complex;

RWStructuredBuffer<float2> psi;
RWStructuredBuffer<float2> k1;
RWStructuredBuffer<float2> k2;
RWStructuredBuffer<float2> k3;
RWStructuredBuffer<float2> k4;
StructuredBuffer<float> ps;
StructuredBuffer<float2> J;
StructuredBuffer<uint> row_indices;
StructuredBuffer<uint> col_indices;
RWStructuredBuffer<float2> psisum;

[vk::specialization_constant]
const uint LEN = 0;
[vk::specialization_constant]
const uint WAVE_SIZE = 0;
[vk::specialization_constant]
const float dt = 0;
[vk::specialization_constant]
const float alpha = 0;

void f1(uint i) {
  float2 y = psi[i];
  k1[i] = ps[i] * y - dot(y,y) * c_mul(float2(1, alpha), y);
  uint row_start = row_indices[i];
  uint row_end = row_indices[i+1];
  for (uint j = row_start; j < row_end; ++j) {
    k1[i] += c_mul(J[j], psi[col_indices[j]]);
  }
}

void f2(uint i) {
  float2 y = psi[i] + k1[i] * (0.5 * dt);
  k2[i] = ps[i] * y - dot(y,y) * c_mul(float2(1, alpha), y);
  uint row_start = row_indices[i];
  uint row_end = row_indices[i+1];
  for (uint j = row_start; j < row_end; ++j) {
    k2[i] += c_mul(J[j], (psi[col_indices[j]] + (0.5 * dt) * k1[col_indices[j]]));
  }
}

void f3(uint i) {
  float2 y = psi[i] + k2[i] * (0.5 * dt);
  k3[i] = ps[i] * y - dot(y,y) * c_mul(float2(1, alpha), y);
  uint row_start = row_indices[i];
  uint row_end = row_indices[i+1];
  for (uint j = row_start; j < row_end; ++j) {
    k3[i] += c_mul(J[j], (psi[col_indices[j]] + (0.5 * dt) * k2[col_indices[j]]));
  }
}

void f4(uint i) {
  float2 y = psi[i] + k3[i] * dt;
  k4[i] = ps[i] * y - dot(y,y) * c_mul(float2(1, alpha), y);
  uint row_start = row_indices[i];
  uint row_end = row_indices[i+1];
  for (uint j = row_start; j < row_end; ++j) {
    k4[i] += c_mul(J[j], (psi[col_indices[j]] + dt * k3[col_indices[j]]));
  }
}

void parsumfirst(uint threadId: SV_DispatchThreadID, uint gId: SV_GroupID) {
  psisum[gId] = WaveActiveSum(psi[threadId]);
}
void parsum(uint threadId: SV_DispatchThreadID, uint gId: SV_GroupID) {
  psisum[gId] = WaveActiveSum(psisum[threadId]);
}

void commit(uint i) {
  psi[i] += (dt/6.) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
}

[shader("compute")]
[numthreads(WAVE_SIZE)]
void main(uint threadId: SV_DispatchThreadID, uint gId: SV_GroupID, uniform uint stage) {
  uint i = threadId;
  // psi[i] = J[i];
  if(i < LEN) {
  switch (stage) {
  case 0:
    f1(i);
    break;
  case 1:
    f2(i);
    break;
  case 2:
    f3(i);
    break;
  case 3:
    f4(i);
    break;
  case 4:
    commit(i);
    break;
  case 5:
    psisum[gId] = WaveActiveSum(psi[threadId]);
    // parsumfirst(threadId, gId);
    break;
  case 6:
    parsum(threadId, gId);
    break;
  }
  }
}
