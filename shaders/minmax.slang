RWStructuredBuffer<float> minmax_buffer;

[[SpecializationConstant]]
const uint NELEMENTS = 2;

[shader("compute")]
[numthreads(32, 1, 1)]
void minmax(uint t_id: SV_DispatchThreadID, uint g_id: SV_GroupID) {
  if (t_id < NELEMENTS) {
    float lesser = minmax_buffer[t_id];
    float greater = minmax_buffer[NELEMENTS - t_id - 1];
    minmax_buffer[g_id] = WaveActiveMin(lesser);
    minmax_buffer[NELEMENTS - g_id - 1] = WaveActiveMax(greater);
  }
}
