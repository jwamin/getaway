#pragma once

#include <metal_stdlib>
using namespace metal;

// Must match `Uniforms` in Renderer.swift.
struct Uniforms {
    float4x4 viewProjection;
    float4x4 invViewProjection;
    float4   cameraPosition;   // xyz: camera position, w: time in seconds
    float4   lodOrigin;        // xyz: point tessellation distances are measured from
    float4   sunDirection;     // xyz: normalized direction towards the sun
    float4   frustumPlanes[4]; // left, right, bottom, top (inward facing)
    float4   terrainParams;    // x: height scale, y: base frequency, z: water level, w: terrain size
    float4   tessParams;       // x: max tess factor, y: lod scale, z: patch size, w: unused
};

constant int kGeometryOctaves = 10;
constant int kNormalOctaves   = 11;
constant int kShadowOctaves   = 7;

constant float2x2 kOctaveRotation = float2x2(float2(0.8, 0.6), float2(-0.6, 0.8));

inline float hash12(float2 p) {
    float3 p3 = fract(float3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

// Quintic value noise in [-1, 1]; yz holds the analytic derivative.
inline float3 noised(float2 x) {
    float2 i = floor(x);
    float2 f = x - i;
    float2 u  = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    float2 du = 30.0 * f * f * (f * (f - 2.0) + 1.0);

    float a = hash12(i)                  * 2.0 - 1.0;
    float b = hash12(i + float2(1, 0))   * 2.0 - 1.0;
    float c = hash12(i + float2(0, 1))   * 2.0 - 1.0;
    float d = hash12(i + float2(1, 1))   * 2.0 - 1.0;

    float k1 = b - a;
    float k2 = c - a;
    float k4 = a - b - c + d;
    return float3(a + k1 * u.x + k2 * u.y + k4 * u.x * u.y,
                  du * (float2(k1, k2) + k4 * u.yx));
}

// Derivative-damped fBm: accumulated slope suppresses detail on steep areas,
// which gives an eroded look. Output is bounded by +/- 2 * height scale.
inline float terrainHeight(float2 p, float4 terrainParams, int octaves) {
    float2 x = p * terrainParams.y;
    float a = 0.0;
    float b = 1.0;
    float2 d = float2(0.0);
    for (int i = 0; i < octaves; ++i) {
        float3 n = noised(x);
        d += n.yz;
        a += b * n.x / (1.0 + dot(d, d));
        b *= 0.5;
        x = kOctaveRotation * x * 2.0;
    }
    return a * terrainParams.x;
}

inline float3 skyColor(float3 dir, float3 sun) {
    float3 horizon = float3(0.62, 0.72, 0.86);
    float3 zenith  = float3(0.12, 0.30, 0.68);
    float3 col = mix(horizon, zenith, pow(saturate(dir.y), 0.45));
    col = mix(col, horizon * 0.85, saturate(-dir.y * 3.0));
    float s = saturate(dot(dir, sun));
    col += float3(1.0, 0.72, 0.42) * (pow(s, 6.0) * 0.25 + pow(s, 64.0) * 0.5);
    return col;
}

inline float3 applyFog(float3 color, float3 worldPos, float3 camera, float3 sun) {
    float3 v = worldPos - camera;
    float dist = length(v);
    float3 dir = v / max(dist, 1e-4);
    float fog = 1.0 - exp(-dist * 0.00032);
    return mix(color, skyColor(dir, sun), fog);
}
