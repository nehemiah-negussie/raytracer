#pragma once
#include <cuda_runtime.h>

struct Light {
    float3 dir;    // direction light is coming FROM (normalized)
    float3 color;  // RGB intensity, each component 0-1
};
