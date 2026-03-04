RWStructuredBuffer<float> u;
StructuredBuffer<float> v;
StructuredBuffer<float> M;
StructuredBuffer<uint> row_indices;
StructuredBuffer<uint> col_indices;

[SpecializationConstant]
const uint LEN = 0;

[shader("compute")]
[numthreads(64)]
void sparse(uint threadId: SV_DispatchThreadID) {
  uint i = threadId.x;
  if (i < LEN) {
    uint row_start = row_indices[i];
    uint row_end = row_indices[i+1];
    for (uint j = row_start; j < row_end; ++j) {
      u[i] += M[j] * v[col_indices[j]];
    }
  }
}
