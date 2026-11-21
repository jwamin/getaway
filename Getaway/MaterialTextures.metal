#include "ShaderCommon.h"

// Procedural, seamlessly tiling ground materials written into a texture array:
// layer 0 grass, 1 sand, 2 rock, 3 snow.

inline float tileHash(float2 cell, float2 period) {
    return hash12(fmod(cell + period * 64.0, period));
}

inline float tileValueNoise(float2 p, float2 period) {
    float2 i = floor(p);
    float2 f = p - i;
    float2 u = f * f * (3.0 - 2.0 * f);
    float a = tileHash(i, period);
    float b = tileHash(i + float2(1, 0), period);
    float c = tileHash(i + float2(0, 1), period);
    float d = tileHash(i + float2(1, 1), period);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

inline float tileFbm(float2 uv, float2 frequency, int octaves) {
    float sum = 0.0;
    float amplitude = 0.5;
    float norm = 0.0;
    for (int i = 0; i < octaves; ++i) {
        sum += amplitude * tileValueNoise(uv * frequency, frequency);
        norm += amplitude;
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    return sum / norm;
}

// Returns the distances to the nearest and second nearest feature points.
inline float2 tileVoronoi(float2 uv, float2 frequency) {
    float2 p = uv * frequency;
    float2 i = floor(p);
    float2 f = p - i;
    float f1 = 8.0;
    float f2 = 8.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float2 g = float2(x, y);
            float2 cell = i + g;
            float2 o = float2(tileHash(cell, frequency), tileHash(cell + 17.0, frequency));
            float d = length(g + o - f);
            if (d < f1) { f2 = f1; f1 = d; }
            else if (d < f2) { f2 = d; }
        }
    }
    return float2(f1, f2);
}

inline float3 grassMaterial(float2 uv, float speck) {
    float patches = tileFbm(uv, float2(4), 5);
    float fine = tileFbm(uv + 0.5, float2(48), 3);
    float dry = smoothstep(0.55, 0.75, tileFbm(uv + 0.37, float2(3), 4));
    float3 c = mix(float3(0.05, 0.13, 0.02), float3(0.15, 0.25, 0.05), patches);
    c = mix(c, float3(0.22, 0.21, 0.07), dry * 0.6);
    c *= 0.6 + 0.8 * fine;
    c *= 0.85 + 0.3 * speck;
    return c;
}

inline float3 sandMaterial(float2 uv, float speck) {
    float base = tileFbm(uv, float2(6), 5);
    float2 v = tileVoronoi(uv, float2(20));
    float pebbles = smoothstep(0.0, 0.1, v.y - v.x);
    float3 c = mix(float3(0.30, 0.23, 0.14), float3(0.52, 0.43, 0.29), base);
    c *= mix(0.7, 1.0, pebbles);
    c *= 0.8 + 0.4 * speck;
    return c;
}

inline float3 rockMaterial(float2 uv, float speck) {
    float n = tileFbm(uv, float2(4), 6);
    float strata = 0.5 + 0.5 * sin((uv.y * 10.0 + n * 1.5) * 6.2831853);
    float2 v = tileVoronoi(uv, float2(6));
    float cracks = smoothstep(0.0, 0.06, v.y - v.x);
    float grit = tileFbm(uv + 0.21, float2(64), 3);
    float3 c = mix(float3(0.14, 0.13, 0.12), float3(0.38, 0.35, 0.31), n);
    c *= 0.78 + 0.28 * strata;
    c *= mix(0.4, 1.0, cracks);
    c *= 0.75 + 0.4 * grit;
    c *= 0.92 + 0.16 * speck;
    return c;
}

inline float3 snowMaterial(float2 uv, float speck) {
    float n = tileFbm(uv, float2(5), 5);
    float3 c = mix(float3(0.66, 0.72, 0.80), float3(0.92, 0.94, 0.98), n);
    c += step(0.996, speck) * 0.25;
    return c;
}

kernel void generateMaterials(texture2d_array<float, access::write> output [[texture(0)]],
                              uint3 gid [[thread_position_in_grid]])
{
    uint width = output.get_width();
    uint height = output.get_height();
    if (gid.x >= width || gid.y >= height || gid.z >= output.get_array_size()) { return; }

    float2 uv = (float2(gid.xy) + 0.5) / float2(width, height);
    float speck = hash12(float2(gid.xy) + float(gid.z) * 101.0);

    float3 color;
    switch (gid.z) {
        case 0:  color = grassMaterial(uv, speck); break;
        case 1:  color = sandMaterial(uv, speck);  break;
        case 2:  color = rockMaterial(uv, speck);  break;
        default: color = snowMaterial(uv, speck);  break;
    }
    output.write(float4(saturate(color), 1.0), gid.xy, gid.z);
}
