#pragma once
#include <cuda_runtime.h>

__host__ __device__ inline float3 operator+(float3 a, float3 b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }
__host__ __device__ inline float3 operator-(float3 a, float3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
__host__ __device__ inline float3 operator-(float3 a)           { return {-a.x, -a.y, -a.z}; }
__host__ __device__ inline float3 operator*(float3 a, float  s) { return {a.x*s, a.y*s, a.z*s}; }
__host__ __device__ inline float3 operator*(float  s, float3 a) { return {a.x*s, a.y*s, a.z*s}; }
__host__ __device__ inline float3 operator*(float3 a, float3 b) { return {a.x*b.x, a.y*b.y, a.z*b.z}; }
__host__ __device__ inline float3 operator/(float3 a, float  s) { return {a.x/s, a.y/s, a.z/s}; }
__host__ __device__ inline float3& operator+=(float3& a, float3 b) { a.x+=b.x; a.y+=b.y; a.z+=b.z; return a; }
__host__ __device__ inline float3& operator*=(float3& a, float  s) { a.x*=s;   a.y*=s;   a.z*=s;   return a; }

__host__ __device__ inline float  dot(float3 a, float3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
__host__ __device__ inline float  len(float3 a)             { return sqrtf(dot(a,a)); }
__host__ __device__ inline float3 normalize(float3 a)       { return a * (1.0f / len(a)); }
__host__ __device__ inline float3 cross(float3 a, float3 b) {
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}
__host__ __device__ inline float3 lerp3(float3 a, float3 b, float t) { return a + (b - a) * t; }
