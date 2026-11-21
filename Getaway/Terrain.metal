#include "ShaderCommon.h"

// MARK: - Tessellation factor kernel

inline float edgeTessFactor(float2 a, float2 b, constant Uniforms& u) {
    // Factors depend only on the shared edge, so neighbouring patches agree and no cracks appear.
    float2 mid = 0.5 * (a + b);
    float3 m = float3(mid.x, terrainHeight(mid, u.terrainParams, 5), mid.y);
    float dist = max(distance(m, u.lodOrigin.xyz), 1.0);
    float f = u.tessParams.y * distance(a, b) / dist;
    return clamp(f, 1.0, u.tessParams.x);
}

inline bool patchVisible(float2 minXZ, float2 maxXZ, constant Uniforms& u) {
    float maxH = 2.0 * u.terrainParams.x;
    float3 bmin = float3(minXZ.x, -maxH, minXZ.y);
    float3 bmax = float3(maxXZ.x,  maxH, maxXZ.y);
    for (int i = 0; i < 4; ++i) {
        float4 plane = u.frustumPlanes[i];
        float3 positive = select(bmin, bmax, plane.xyz >= 0.0);
        if (dot(plane.xyz, positive) + plane.w < 0.0) {
            return false;
        }
    }
    return true;
}

// Low-frequency cloud coverage shared by surface lighting and atmospheric
// in-scattering, keeping shafts and moving shadows visually aligned.
inline float volumeHash13(float3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

inline float volumeNoise3D(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float x00 = mix(volumeHash13(i), volumeHash13(i + float3(1, 0, 0)), f.x);
    float x10 = mix(volumeHash13(i + float3(0, 1, 0)), volumeHash13(i + float3(1, 1, 0)), f.x);
    float x01 = mix(volumeHash13(i + float3(0, 0, 1)), volumeHash13(i + float3(1, 0, 1)), f.x);
    float x11 = mix(volumeHash13(i + float3(0, 1, 1)), volumeHash13(i + float3(1, 1, 1)), f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

inline float cloudSunVisibility(float3 p, float3 sun, float time) {
    constexpr float cloudMiddle = 850.0;
    if (p.y >= 1280.0 || sun.y <= 0.001) {
        return 1.0;
    }

    float distanceToCloud = max((cloudMiddle - p.y) / sun.y, 0.0);
    float3 cloudPoint = p + sun * distanceToCloud;
    float3 q = cloudPoint * 0.00115;
    q.xz += float2(time * 0.012, time * 0.004);

    float shape = volumeNoise3D(q) * 0.68
                + volumeNoise3D(q * 2.03 + float3(7.1, 3.4, 5.8)) * 0.32;
    float coverage = volumeNoise3D(q * 0.24 + float3(12.0, 0.0, -8.0));
    float extinction = saturate((shape + coverage * 0.38 - 0.66) * 3.2);
    return exp(-extinction * 2.8);
}

// Control points per patch: 0 = (x0, z0), 1 = (x1, z0), 2 = (x1, z1), 3 = (x0, z1).
// Metal quad edge order: [0] u == 0, [1] v == 0, [2] u == 1, [3] v == 1.
kernel void terrainTessFactors(constant Uniforms& u [[buffer(0)]],
                               device const float2* controlPoints [[buffer(1)]],
                               device MTLQuadTessellationFactorsHalf* factors [[buffer(2)]],
                               constant uint& patchCount [[buffer(3)]],
                               uint pid [[thread_position_in_grid]])
{
    if (pid >= patchCount) { return; }

    float2 c0 = controlPoints[pid * 4 + 0];
    float2 c1 = controlPoints[pid * 4 + 1];
    float2 c2 = controlPoints[pid * 4 + 2];
    float2 c3 = controlPoints[pid * 4 + 3];

    if (!patchVisible(min(c0, c2), max(c0, c2), u)) {
        for (int i = 0; i < 4; ++i) { factors[pid].edgeTessellationFactor[i] = 0; }
        factors[pid].insideTessellationFactor[0] = 0;
        factors[pid].insideTessellationFactor[1] = 0;
        return;
    }

    float e0 = edgeTessFactor(c0, c3, u);
    float e1 = edgeTessFactor(c0, c1, u);
    float e2 = edgeTessFactor(c1, c2, u);
    float e3 = edgeTessFactor(c3, c2, u);

    factors[pid].edgeTessellationFactor[0] = half(e0);
    factors[pid].edgeTessellationFactor[1] = half(e1);
    factors[pid].edgeTessellationFactor[2] = half(e2);
    factors[pid].edgeTessellationFactor[3] = half(e3);
    factors[pid].insideTessellationFactor[0] = half(max(e1, e3));
    factors[pid].insideTessellationFactor[1] = half(max(e0, e2));
}

// MARK: - Terrain

struct ControlPoint {
    float2 position [[attribute(0)]];
};

struct PatchIn {
    patch_control_point<ControlPoint> controlPoints;
};

struct TerrainVertexOut {
    float4 position [[position]];
    float3 worldPos;
    float3 coarseNormal;
};

[[patch(quad, 4)]]
vertex TerrainVertexOut terrainVertex(PatchIn patch [[stage_in]],
                                      float2 uv [[position_in_patch]],
                                      constant Uniforms& u [[buffer(1)]])
{
    float2 c0 = patch.controlPoints[0].position;
    float2 c1 = patch.controlPoints[1].position;
    float2 c2 = patch.controlPoints[2].position;
    float2 c3 = patch.controlPoints[3].position;
    float2 p = mix(mix(c0, c1, uv.x), mix(c3, c2, uv.x), uv.y);

    float h = terrainHeight(p, u.terrainParams, kGeometryOctaves);

    const float e = 4.0;
    float hc = terrainHeight(p, u.terrainParams, 6);
    float hx = terrainHeight(p + float2(e, 0), u.terrainParams, 6);
    float hz = terrainHeight(p + float2(0, e), u.terrainParams, 6);

    TerrainVertexOut out;
    out.worldPos = float3(p.x, h, p.y);
    out.position = u.viewProjection * float4(out.worldPos, 1.0);
    out.coarseNormal = normalize(float3(hc - hx, e, hc - hz));
    return out;
}

inline float terrainShadow(float3 p, float3 sun, float4 terrainParams) {
    float res = 1.0;
    float t = 6.0;
    float ceiling = 2.0 * terrainParams.x;
    for (int i = 0; i < 24; ++i) {
        float3 q = p + sun * t;
        if (q.y > ceiling) { break; }
        float h = q.y - terrainHeight(q.xz, terrainParams, kShadowOctaves);
        res = min(res, 12.0 * h / t);
        if (res < 0.0) { break; }
        t += clamp(h * 0.6, 3.0, 120.0);
    }
    return smoothstep(0.0, 1.0, saturate(res));
}

inline float3 sampleTwoScales(texture2d_array<float> tex, sampler s, float2 uv, uint layer) {
    float3 nearSample = tex.sample(s, uv, layer).rgb;
    float3 farSample  = tex.sample(s, uv * 0.137 + float2(0.31, 0.71), layer).rgb;
    return mix(nearSample, farSample, 0.4);
}

inline float3 sampleTriplanar(texture2d_array<float> tex, sampler s, float3 p, float3 n, float scale, uint layer) {
    float3 w = pow(abs(n), 4.0);
    w /= (w.x + w.y + w.z);
    float3 x = tex.sample(s, p.zy * scale, layer).rgb;
    float3 y = tex.sample(s, p.xz * scale, layer).rgb;
    float3 z = tex.sample(s, p.xy * scale, layer).rgb;
    return x * w.x + y * w.y + z * w.z;
}

fragment float4 terrainFragment(TerrainVertexOut in [[stage_in]],
                                constant Uniforms& u [[buffer(0)]],
                                texture2d_array<float> materials [[texture(0)]],
                                sampler samp [[sampler(0)]])
{
    float3 camera = u.cameraPosition.xyz;
    float3 sun = u.sunDirection.xyz;
    float3 wp = in.worldPos;
    float dist = distance(wp, camera);

    // Per-pixel normal from the full-detail heightfield; the step widens with
    // distance so far-away normals are naturally low-passed.
    float eps = max(0.05, dist * 0.0015);
    float h0 = terrainHeight(wp.xz, u.terrainParams, kNormalOctaves);
    float hx = terrainHeight(wp.xz + float2(eps, 0), u.terrainParams, kNormalOctaves);
    float hz = terrainHeight(wp.xz + float2(0, eps), u.terrainParams, kNormalOctaves);
    float3 n = normalize(float3(h0 - hx, eps, h0 - hz));
    float3 cn = normalize(in.coarseNormal);

    float hs = u.terrainParams.x;
    float water = u.terrainParams.z;
    float variation = noised(wp.xz * 0.008).x;
    float slope = 1.0 - cn.y;

    float2 uv = wp.xz / 7.0;
    float3 grass = sampleTwoScales(materials, samp, uv, 0);
    float3 sand  = sampleTwoScales(materials, samp, uv, 1);
    float3 rock  = sampleTriplanar(materials, samp, wp, cn, 1.0 / 18.0, 2);
    float3 snow  = sampleTwoScales(materials, samp, uv * 0.5, 3);

    float wSand = 1.0 - smoothstep(water + 3.0, water + 12.0, wp.y + variation * 5.0);
    float wDry  = smoothstep(0.35 * hs, 0.65 * hs, wp.y + variation * 30.0) * 0.6;
    float wRock = smoothstep(0.16, 0.32, slope + variation * 0.06);
    float wSnow = smoothstep(0.72 * hs, 0.84 * hs, wp.y + variation * 40.0 + n.y * 20.0)
                * (1.0 - smoothstep(0.35, 0.55, slope));

    float3 albedo = grass;
    albedo = mix(albedo, sand * float3(0.75, 0.7, 0.65), wDry);
    albedo = mix(albedo, sand, wSand);
    albedo = mix(albedo, rock, wRock);
    albedo = mix(albedo, snow, wSnow);
    albedo *= 0.85 + 0.3 * (noised(wp.xz * 0.03).x * 0.5 + 0.5);

    float shadow = terrainShadow(wp + float3(0, 2.0, 0), sun, u.terrainParams);
    float sunVisibility = cloudSunVisibility(wp, sun, u.cameraPosition.w);
    float directLight = shadow * sunVisibility;
    float diffuse = saturate(dot(n, sun)) * directLight;
    float3 sunColor = float3(1.0, 0.9, 0.76) * 2.6;
    float3 skyAmbient = mix(float3(0.10, 0.09, 0.07), float3(0.30, 0.40, 0.55), n.y * 0.5 + 0.5);
    float bounce = saturate(dot(n, normalize(float3(-sun.x, 0.0, -sun.z)))) * 0.15;

    float3 color = albedo * (diffuse * sunColor + skyAmbient + bounce * float3(0.4, 0.35, 0.25));

    float3 view = normalize(camera - wp);
    float3 halfVector = normalize(view + sun);
    color += wSnow * pow(saturate(dot(n, halfVector)), 40.0) * 0.4 * directLight * sunColor;

    color = applyFog(color, wp, camera, sun);
    return float4(color, 1.0);
}

fragment float4 terrainWireFragment(TerrainVertexOut in [[stage_in]],
                                    constant Uniforms& u [[buffer(0)]])
{
    float dist = distance(in.worldPos, u.cameraPosition.xyz);
    float fade = exp(-dist * 0.0004);
    return float4(mix(float3(0.6, 0.75, 0.9), float3(0.1, 1.0, 0.35), fade), 1.0);
}

// MARK: - Sky

struct SkyVertexOut {
    float4 position [[position]];
    float2 ndc;
};

vertex SkyVertexOut skyVertex(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    SkyVertexOut out;
    out.ndc = p * 2.0 - 1.0;
    out.position = float4(out.ndc, 0.0, 1.0);
    return out;
}

fragment float4 skyFragment(SkyVertexOut in [[stage_in]],
                            constant Uniforms& u [[buffer(0)]])
{
    float4 world = u.invViewProjection * float4(in.ndc, 0.5, 1.0);
    float3 dir = normalize(world.xyz / world.w - u.cameraPosition.xyz);
    float3 sun = u.sunDirection.xyz;
    float3 color = skyColor(dir, sun);
    color += smoothstep(0.9992, 0.9997, dot(dir, sun)) * float3(4.0, 3.6, 3.0);
    return float4(color, 1.0);
}

// MARK: - Volumetric clouds

inline float hash13(float3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

inline float noise3D(float3 p) {
    float3 i = floor(p);
    float3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float x00 = mix(hash13(i), hash13(i + float3(1, 0, 0)), f.x);
    float x10 = mix(hash13(i + float3(0, 1, 0)), hash13(i + float3(1, 1, 0)), f.x);
    float x01 = mix(hash13(i + float3(0, 0, 1)), hash13(i + float3(1, 0, 1)), f.x);
    float x11 = mix(hash13(i + float3(0, 1, 1)), hash13(i + float3(1, 1, 1)), f.x);
    return mix(mix(x00, x10, f.y), mix(x01, x11, f.y), f.z);
}

inline float cloudFBM(float3 p) {
    float value = 0.0;
    float amplitude = 0.55;
    for (int i = 0; i < 3; ++i) {
        value += noise3D(p) * amplitude;
        p = p * 2.03 + float3(7.1, 3.4, 5.8);
        amplitude *= 0.5;
    }
    return value;
}

inline float cloudDensity(float3 p, float time) {
    constexpr float cloudBottom = 520.0;
    constexpr float cloudTop = 1280.0;
    float height = saturate((p.y - cloudBottom) / (cloudTop - cloudBottom));
    float heightShape = smoothstep(0.0, 0.18, height) * (1.0 - smoothstep(0.62, 1.0, height));

    float3 q = p * 0.00115;
    q.xz += float2(time * 0.012, time * 0.004);
    float base = cloudFBM(q);
    float coverage = noise3D(q * 0.24 + float3(12.0, 0.0, -8.0));
    return saturate((base + coverage * 0.38 - 0.66) * 3.2) * heightShape;
}

inline float cloudPhase(float cosTheta) {
    // A forward lobe around the sun plus a soft isotropic component.
    constexpr float g = 0.62;
    float denominator = pow(max(1.0 + g * g - 2.0 * g * cosTheta, 0.01), 1.5);
    return 0.35 + 0.65 * (1.0 - g * g) / denominator;
}

fragment float4 cloudFragment(SkyVertexOut in [[stage_in]],
                              constant Uniforms& u [[buffer(0)]],
                              texture2d<float, access::read> sceneColor [[texture(0)]],
                              depth2d<float, access::read> sceneDepth [[texture(1)]])
{
    uint2 pixel = uint2(in.position.xy);
    float3 background = sceneColor.read(pixel).rgb;

    float4 farWorld = u.invViewProjection * float4(in.ndc, 0.0001, 1.0);
    float3 rayDirection = normalize(farWorld.xyz / farWorld.w - u.cameraPosition.xyz);
    float rayLimit = 30000.0;

    float depth = sceneDepth.read(pixel);
    if (depth > 0.000001) {
        float4 hit = u.invViewProjection * float4(in.ndc, depth, 1.0);
        rayLimit = distance(hit.xyz / hit.w, u.cameraPosition.xyz);
    }

    constexpr float cloudBottom = 520.0;
    constexpr float cloudTop = 1280.0;
    if (abs(rayDirection.y) < 0.00001) {
        return float4(background, 1.0);
    }

    float tBottom = (cloudBottom - u.cameraPosition.y) / rayDirection.y;
    float tTop = (cloudTop - u.cameraPosition.y) / rayDirection.y;
    float rayStart = max(min(tBottom, tTop), 0.0);
    float rayEnd = min(max(tBottom, tTop), rayLimit);
    if (rayEnd <= rayStart) {
        return float4(background, 1.0);
    }

    constexpr int stepCount = 36;
    float stepLength = (rayEnd - rayStart) / float(stepCount);
    float jitter = hash13(float3(float2(pixel), floor(u.cameraPosition.w * 24.0)));
    float distanceAlongRay = rayStart + jitter * stepLength;
    float transmittance = 1.0;
    float3 cloudLight = float3(0.0);
    float3 sun = u.sunDirection.xyz;
    float phase = cloudPhase(saturate(dot(rayDirection, sun)));

    for (int i = 0; i < stepCount; ++i) {
        float3 samplePosition = u.cameraPosition.xyz + rayDirection * distanceAlongRay;
        float density = cloudDensity(samplePosition, u.cameraPosition.w);
        if (density > 0.002) {
            float lightDensity = cloudDensity(samplePosition + sun * 130.0, u.cameraPosition.w)
                               + cloudDensity(samplePosition + sun * 300.0, u.cameraPosition.w) * 0.7;
            float sunlight = exp(-lightDensity * 1.7);
            float3 litColor = mix(float3(0.42, 0.48, 0.58),
                                  float3(1.0, 0.91, 0.76) * (0.75 + phase),
                                  sunlight);
            float sampleAlpha = 1.0 - exp(-density * stepLength * 0.012);
            cloudLight += transmittance * sampleAlpha * litColor;
            transmittance *= 1.0 - sampleAlpha;
            if (transmittance < 0.015) {
                break;
            }
        }
        distanceAlongRay += stepLength;
    }

    constexpr int shaftStepCount = 12;
    float shaftEnd = min(rayLimit, 12000.0);
    float shaftStepLength = shaftEnd / float(shaftStepCount);
    float shaftDistance = (0.35 + jitter * 0.65) * shaftStepLength;
    float shaftScattering = 0.0;
    for (int i = 0; i < shaftStepCount; ++i) {
        float3 samplePosition = u.cameraPosition.xyz + rayDirection * shaftDistance;
        float airDensity = exp(-max(samplePosition.y, 0.0) / 8000.0);
        shaftScattering += cloudSunVisibility(samplePosition, sun, u.cameraPosition.w)
                         * airDensity * shaftStepLength;
        shaftDistance += shaftStepLength;
    }

    float sunAlignment = saturate(dot(rayDirection, sun));
    float shaftPhase = 0.12 + 2.2 * pow(sunAlignment, 8.0);
    float3 shaftLight = float3(1.0, 0.82, 0.58)
                      * shaftScattering * 0.000012 * shaftPhase
                      * sqrt(transmittance);

    return float4(cloudLight + shaftLight + background * transmittance, 1.0);
}

// MARK: - Water

struct WaterVertexOut {
    float4 position [[position]];
    float3 worldPos;
};

vertex WaterVertexOut waterVertex(uint vid [[vertex_id]],
                                  constant Uniforms& u [[buffer(0)]])
{
    const float2 corners[4] = { float2(-1, -1), float2(1, -1), float2(-1, 1), float2(1, 1) };
    float extent = u.terrainParams.w * 2.0;
    WaterVertexOut out;
    out.worldPos = float3(corners[vid].x * extent, u.terrainParams.z, corners[vid].y * extent);
    out.position = u.viewProjection * float4(out.worldPos, 1.0);
    return out;
}

fragment float4 waterFragment(WaterVertexOut in [[stage_in]],
                              constant Uniforms& u [[buffer(0)]])
{
    float t = u.cameraPosition.w;
    float2 p = in.worldPos.xz;
    float3 camera = u.cameraPosition.xyz;
    float3 sun = u.sunDirection.xyz;

    float3 w1 = noised(p * 0.04 + float2(t * 0.25, t * 0.18));
    float3 w2 = noised(kOctaveRotation * p * 0.11 - float2(t * 0.4, -t * 0.3));
    float3 w3 = noised(p * 0.35 + float2(-t * 0.8, t * 0.6));
    float2 grad = w1.yz * 0.04 * 1.2 + w2.yz * 0.11 * 0.5 + w3.yz * 0.35 * 0.12;

    float dist = distance(in.worldPos, camera);
    grad *= exp(-dist * 0.0006);
    float3 n = normalize(float3(-grad.x, 1.0, -grad.y));

    float3 v = normalize(in.worldPos - camera);
    float3 r = reflect(v, n);
    r.y = abs(r.y);
    float fresnel = 0.02 + 0.98 * pow(1.0 - saturate(dot(-v, n)), 5.0);

    float3 deep = float3(0.01, 0.06, 0.09);
    float3 color = mix(deep, skyColor(r, sun), fresnel);
    float sunVisibility = cloudSunVisibility(in.worldPos, sun, t);
    color += pow(saturate(dot(r, sun)), 400.0) * float3(1.0, 0.9, 0.7) * 6.0 * sunVisibility;
    color += saturate(dot(n, sun)) * float3(1.0, 0.88, 0.68)
           * (1.0 - fresnel) * 0.12 * sunVisibility;
    color = applyFog(color, in.worldPos, camera, sun);

    return float4(color, mix(0.72, 1.0, fresnel));
}
