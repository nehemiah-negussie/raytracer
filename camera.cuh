#pragma once
#include <cuda_runtime.h>

struct Ray {
    float3 origin;
    float3 dir;
};

struct Camera {
    float3 pos;
    float3 forward;
    float3 up;
    float vfov;   // vertical FOV in degrees
    int width;
    int height;
};

__device__ Ray generateRay(const Camera& cam, int px, int py) {
    // derive right and up axes from forward + up
    float3 right = {
        cam.forward.y * cam.up.z - cam.forward.z * cam.up.y,
        cam.forward.z * cam.up.x - cam.forward.x * cam.up.z,
        cam.forward.x * cam.up.y - cam.forward.y * cam.up.x
    };
    float rlen = sqrtf(right.x*right.x + right.y*right.y + right.z*right.z);
    right = {right.x/rlen, right.y/rlen, right.z/rlen};

    // camera-space up: perpendicular to both right and forward
    float3 up = {
        right.y * cam.forward.z - right.z * cam.forward.y,
        right.z * cam.forward.x - right.x * cam.forward.z,
        right.x * cam.forward.y - right.y * cam.forward.x
    };

    float aspect = (float)cam.width / (float)cam.height;
    float half_h = tanf(cam.vfov * 3.14159265f / 360.0f);
    float half_w = aspect * half_h;

    float sx = (2.0f * (px + 0.5f) / cam.width  - 1.0f) * half_w;
    float sy = (1.0f - 2.0f * (py + 0.5f) / cam.height) * half_h;

    float3 dir = {
        cam.forward.x + sx * right.x + sy * up.x,
        cam.forward.y + sx * right.y + sy * up.y,
        cam.forward.z + sx * right.z + sy * up.z
    };

    float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir = {dir.x/len, dir.y/len, dir.z/len};

    return {cam.pos, dir};
}
