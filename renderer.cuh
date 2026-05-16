#pragma once
#include "math_utils.cuh"
#include "world.cuh"
#include "camera.cuh"
#include "light.cuh"

// ---- RNG (xorshift32, seeded per pixel per frame) ----

__device__ inline uint32_t xorshift(uint32_t& s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}
__device__ inline float randf(uint32_t& s) {
    return (float)(xorshift(s) >> 8) * (1.0f / 16777216.0f);
}
__device__ inline uint32_t rngSeed(uint32_t px, uint32_t py, uint32_t frame) {
    uint32_t s = px * 1973u ^ py * 9277u ^ frame * 26699u;
    s ^= s >> 16; s *= 0x45d9f3bu; s ^= s >> 16;
    return s ? s : 1u;
}

// ---- sampling helpers ----

// cosine-weighted hemisphere sample around normal n
__device__ inline float3 cosHemi(float3 n, uint32_t& rng) {
    float u   = randf(rng), v = randf(rng);
    float r   = sqrtf(u);
    float phi = 6.28318530f * v;
    float lx  = r * cosf(phi), lz = r * sinf(phi), ly = sqrtf(fmaxf(0.0f, 1.0f - u));
    float3 up = fabsf(n.y) < 0.9f ? float3{0.0f,1.0f,0.0f} : float3{1.0f,0.0f,0.0f};
    float3 t  = normalize(cross(up, n));
    float3 b  = cross(n, t);
    return {lx*t.x + ly*n.x + lz*b.x,
            lx*t.y + ly*n.y + lz*b.y,
            lx*t.z + ly*n.z + lz*b.z};
}

// jitter direction within a cone (spread in radians)
__device__ inline float3 jitterDir(float3 d, float spread, uint32_t& rng) {
    float3 up = fabsf(d.y) < 0.9f ? float3{0.0f,1.0f,0.0f} : float3{1.0f,0.0f,0.0f};
    float3 t  = normalize(cross(up, d));
    float3 b  = cross(d, t);
    float  u  = (randf(rng) * 2.0f - 1.0f) * spread;
    float  v  = (randf(rng) * 2.0f - 1.0f) * spread;
    return normalize(d + t * u + b * v);
}

// ---- DDA traversal ----

struct HitInfo {
    bool hit;
    BlockType block;
    float3 normal;
    int ix, iy, iz;
};

__device__ HitInfo dda(const Ray& ray, const VoxelGrid* grid, int maxSteps = 1500) {
    float3 pos = ray.origin;
    float3 dir = ray.dir;

    int ix = (int)floorf(pos.x), iy = (int)floorf(pos.y), iz = (int)floorf(pos.z);
    int sx = dir.x > 0.0f ? 1 : -1;
    int sy = dir.y > 0.0f ? 1 : -1;
    int sz = dir.z > 0.0f ? 1 : -1;

    float tDX = fabsf(dir.x) > 1e-6f ? fabsf(1.0f / dir.x) : 1e30f;
    float tDY = fabsf(dir.y) > 1e-6f ? fabsf(1.0f / dir.y) : 1e30f;
    float tDZ = fabsf(dir.z) > 1e-6f ? fabsf(1.0f / dir.z) : 1e30f;

    float tMX = fabsf(dir.x) > 1e-6f ? (dir.x > 0.0f ? (ix+1-pos.x)/dir.x : (ix-pos.x)/dir.x) : 1e30f;
    float tMY = fabsf(dir.y) > 1e-6f ? (dir.y > 0.0f ? (iy+1-pos.y)/dir.y : (iy-pos.y)/dir.y) : 1e30f;
    float tMZ = fabsf(dir.z) > 1e-6f ? (dir.z > 0.0f ? (iz+1-pos.z)/dir.z : (iz-pos.z)/dir.z) : 1e30f;

    float3 normal = {0.0f, 0.0f, 0.0f};

    for (int step = 0; step < maxSteps; step++) {
        if (ix < 0 || ix >= WORLD_WIDTH || iy < 0 || iy >= WORLD_HEIGHT || iz < 0 || iz >= WORLD_DEPTH) break;
        BlockType b = grid->a[I(ix, iy, iz)];
        if (b != BlockType::Air) return {true, b, normal, ix, iy, iz};

        if (tMX < tMY && tMX < tMZ) { ix += sx; tMX += tDX; normal = {-(float)sx, 0.0f,      0.0f     }; }
        else if (tMY < tMZ)          { iy += sy; tMY += tDY; normal = {0.0f,      -(float)sy, 0.0f     }; }
        else                         { iz += sz; tMZ += tDZ; normal = {0.0f,       0.0f,      -(float)sz}; }
    }
    return {false, BlockType::Air, {0.0f,0.0f,0.0f}, 0, 0, 0};
}

// ---- shadow helpers ----

__device__ inline bool traceShadow(float3 origin, float3 dir, const VoxelGrid* grid) {
    return dda({origin, dir}, grid).hit;
}

__device__ inline bool shadowRayToPoint(float3 origin, float3 dir, float maxDist, const VoxelGrid* grid) {
    HitInfo h = dda({origin, dir}, grid);
    if (!h.hit) return false;
    float3 d = float3{(float)h.ix + 0.5f, (float)h.iy + 0.5f, (float)h.iz + 0.5f} - origin;
    return len(d) < maxDist - 1.0f;
}

// ---- sky ----

__device__ inline float3 skyColor(float3 dir, float3 horizon, float3 zenith) {
    return lerp3(horizon, zenith, fmaxf(0.0f, dir.y));
}

// ---- kernel ----

__global__ void renderKernel(
    unsigned char*   output,
    float4*          accum,
    int              frameIdx,
    Camera           cam,
    Light            sun,
    Light            fill,
    float3           skyHorizon,
    float3           skyZenith,
    const VoxelGrid* grid,
    const float3*    campfirePos,
    int              numCampfires
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= cam.width || py >= cam.height) return;

    uint32_t rng = rngSeed((uint32_t)px, (uint32_t)py, (uint32_t)frameIdx);

    Ray     ray = generateRay(cam, px, py);
    HitInfo hit = dda(ray, grid);

    float3 color = {0.0f, 0.0f, 0.0f};

    if (!hit.hit) {
        color = skyColor(ray.dir, skyHorizon, skyZenith);
    } else if (isEmissive(hit.block)) {
        color = blockColor(hit.block) * 2.0f;
    } else {
        float3 base    = blockColor(hit.block);
        float3 surfPos = {
            hit.ix + 0.5f + hit.normal.x * 0.5001f,
            hit.iy + 0.5f + hit.normal.y * 0.5001f,
            hit.iz + 0.5f + hit.normal.z * 0.5001f
        };

        // --- soft shadows: 3 jittered rays toward sun ---
        float sunVis = 0.0f;
        if (sun.dir.y > 0.0f) {
            for (int i = 0; i < 3; i++) {
                float3 jdir = jitterDir(sun.dir, 0.04f, rng);
                float  diff = fmaxf(0.0f, dot(hit.normal, jdir));
                if (diff > 0.0f && !traceShadow(surfPos, jdir, grid))
                    sunVis += diff;
            }
            sunVis /= 3.0f;
        }

        // --- fill light (sky bounce) ---
        float diffFill = fmaxf(0.0f, dot(hit.normal, fill.dir));

        // --- hemisphere AO: 4 short-range cosine rays ---
        float ao = 0.0f;
        for (int i = 0; i < 4; i++) {
            float3 aoDir = cosHemi(hit.normal, rng);
            if (!dda({surfPos, aoDir}, grid, 7).hit) ao += 1.0f;
        }
        ao /= 4.0f;

        // --- one diffuse bounce (indirect GI) ---
        float3 indirect = {0.0f, 0.0f, 0.0f};
        {
            float3 bounceDir = cosHemi(hit.normal, rng);
            HitInfo bh       = dda({surfPos, bounceDir}, grid);
            if (bh.hit) {
                float3 bPos  = {bh.ix + 0.5f + bh.normal.x * 0.5001f,
                                bh.iy + 0.5f + bh.normal.y * 0.5001f,
                                bh.iz + 0.5f + bh.normal.z * 0.5001f};
                float  bDiff = fmaxf(0.0f, dot(bh.normal, sun.dir));
                bool   bShad = sun.dir.y > 0.0f && traceShadow(bPos, sun.dir, grid);
                float  bLit  = (bShad ? 0.0f : bDiff * 0.85f) + 0.08f;
                indirect     = blockColor(bh.block) * bLit * 0.22f;
            } else {
                indirect = skyColor(bounceDir, skyHorizon, skyZenith) * 0.18f;
            }
        }

        // --- campfire point lights ---
        float3 pointLight = {0.0f, 0.0f, 0.0f};
        for (int i = 0; i < numCampfires; i++) {
            float3 toFire = campfirePos[i] - surfPos;
            float  dist   = len(toFire);
            if (dist > 14.0f) continue;
            float3 dir    = toFire / dist;
            float  diff   = fmaxf(0.0f, dot(hit.normal, dir));
            if (diff <= 0.0f) continue;
            if (shadowRayToPoint(surfPos, dir, dist, grid)) continue;
            float atten   = 12.0f / ((dist + 1.0f) * (dist + 1.0f));
            pointLight   += float3{1.0f, 0.45f, 0.05f} * (diff * atten);
        }

        // --- combine ---
        float lighting = (sunVis * 0.85f + diffFill * 0.25f + 0.05f) * ao;
        lighting = fmaxf(lighting, 0.01f);

        color = base * lighting + base * indirect + base * pointLight;
    }

    // --- temporal accumulation (progressive running mean) ---
    int    idx   = py * cam.width + px;
    float  blend = 1.0f / (float)(frameIdx + 1);
    float4 prev  = accum[idx];
    float4 acc   = (frameIdx == 0)
        ? float4{color.x, color.y, color.z, 1.0f}
        : float4{prev.x + blend * (color.x - prev.x),
                 prev.y + blend * (color.y - prev.y),
                 prev.z + blend * (color.z - prev.z),
                 1.0f};
    accum[idx] = acc;

    // --- Reinhard tonemap + gamma 2.2 ---
    float r = powf(acc.x / (acc.x + 1.0f), 1.0f / 2.2f);
    float g = powf(acc.y / (acc.y + 1.0f), 1.0f / 2.2f);
    float b = powf(acc.z / (acc.z + 1.0f), 1.0f / 2.2f);

    int bidx = idx * 4;
    output[bidx + 0] = (unsigned char)(fminf(r, 1.0f) * 255.0f);
    output[bidx + 1] = (unsigned char)(fminf(g, 1.0f) * 255.0f);
    output[bidx + 2] = (unsigned char)(fminf(b, 1.0f) * 255.0f);
    output[bidx + 3] = 255;
}
