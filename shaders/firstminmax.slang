StructuredBuffer<float> source;
RWStructuredBuffer<float> minmax_buffer;

[[SpecializationConstant]]
const uint NELEMENTS = 2;

[shader("compute")]
[numthreads(32, 1, 1)]
void firstminmax(uint t_id: SV_DispatchThreadID, uint g_id: SV_GroupID) {
  if (t_id < NELEMENTS) {
    float lesser = source[t_id];
    float greater = source[NELEMENTS - t_id - 1];
    minmax_buffer[g_id] = WaveActiveMin(lesser);
    minmax_buffer[((NELEMENTS + 15) / 16) - g_id - 1] = WaveActiveMax(greater);
  }
}

