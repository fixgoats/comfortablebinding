/* Copyright (c) 2025-2026, Sascha Willems
 *
 * SPDX-License-Identifier: MIT
 *
 */

struct VSInput {
    float2 Pos;
    float3 color;
  };

struct VSOutput {
     float4 Pos : SV_POSITION;
     float2 uv;
};

Sampler2D texture;

static const float2 positions[6] = {
  {-1.0, -1.0},
  {1.0, -1.0},
  {1.0, 1.0},
  {-1.0, -1.0},
  {-1.0, 1.0},
  {1.0, 1.0}
};

static const float2 uv[6] = {
  {0.0, 0.0},
  {1.0, 0.0},
  {1.0, 1.0},
  {0.0, 0.0},
  {0.0, 1.0},
  {1.0, 1.0},
};

[shader("vertex")]
VSOutput main(VSInput input, uint v_id: SV_VulkanVertexID) {
    VSOutput output;
    output.Pos = float4(positions[v_id], 0., 1.);
    output.uv = uv[v_id];
    return output;
}

[shader("fragment")]
float4 main(VSOutput input) {
  return texture.Sample(input.uv);
}
