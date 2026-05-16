# Voxel Ray Tracer

A real-time GPU path tracer built from scratch in CUDA. No game engine. No graphics framework. Just raw ray marching through 67 million voxels at 1920×1080.

---

## What it does

Every frame, the GPU fires ~18 million rays through a 512×512×256 procedurally generated voxel world. Each pixel computes:

- **Primary ray** — DDA traversal (Amanatides-Woo algorithm) to find the first solid voxel
- **Soft shadows** — 3 jittered rays toward the sun to produce a penumbra
- **Hemisphere AO** — 4 cosine-weighted short-range rays to darken occluded corners
- **One-bounce GI** — a secondary ray that lets surfaces exchange color with their surroundings
- **Point lights** — campfires that cast warm, attenuated light with shadow occlusion

Results accumulate across frames. Stop moving and the image converges to noise-free global illumination in under 2 seconds.

---

## Features

**Rendering**
- DDA voxel ray marching — up to 1500 steps per primary ray
- Soft shadows via jittered directional sampling
- Cosine-weighted hemisphere ambient occlusion
- Single-bounce diffuse global illumination
- Campfire point lights with inverse-square attenuation and shadow occlusion
- Emissive block rendering (campfires glow at 2× intensity)
- Reinhard tonemapping + gamma 2.2
- Temporal accumulation — progressive refinement when camera is still

**World**
- 512 × 256 × 512 voxel grid (~67M voxels, 64MB)
- 6-octave value noise terrain with biome layering (sand, grass, dirt, stone, snow)
- 4 procedural tree archetypes: oak (jittered sphere canopy), pine (layered cone), birch (slim oval crown), twisted (irregular multi-stem)
- Campfires with stone rings placed across the world
- Day/night cycle — animated sun orbit, dynamic sky gradient, moonlight at night

**Display**
- Real-time Vulkan window via a custom zero-render-pass pipeline
- Staging buffer copied directly into swapchain images — no render pass overhead
- Left-click to place blocks, right-click to remove them
- WASD movement, mouse look

**Benchmarking**
- CUDA event timing for kernel and full frame separately
- Mrays/s throughput reported every 60 frames
- Session summary on exit

---

## Architecture

```
main.cu          — game loop, camera, day/night, block editing, benchmarks
renderer.cuh     — renderKernel, DDA, soft shadows, AO, GI, temporal accum
world.cuh        — VoxelGrid, terrain gen, tree placement, campfire placement
camera.cuh       — Ray/Camera structs, generateRay
light.cuh        — Light struct (directional)
math_utils.cuh   — float3 operators, dot, cross, normalize, lerp
vk_display.h/cpp — Vulkan window, swapchain, input (GLFW)
output.h/cpp     — PNG export via stb_image_write (kept separate to avoid nvcc crash)
```

The renderer is a single CUDA kernel — one thread per pixel, no inter-thread communication. The temporal accumulation buffer (`float4`, HDR) lives on the GPU and is blended in-kernel each frame using a running mean. Camera movement resets the accumulator instantly.

---

## Build

**Requirements**
- CUDA 12.6 (CUDA 13.x has a known nvcc Windows bug)
- VS 2022 Build Tools with MSVC C++ workload
- GLFW3
- Vulkan SDK

**Command** (run from Developer PowerShell for VS 2022)

```powershell
nvcc -arch=sm_89 -std=c++17 -O2 -Xcompiler "/MD" -o voxel_rt main.cu vk_display.cpp `
  -I"C:/path/to/glfw/include" `
  -I"C:/VulkanSDK/x.x.x.x/Include" `
  "C:/path/to/glfw/lib/glfw3.lib" `
  "C:/VulkanSDK/x.x.x.x/Lib/vulkan-1.lib" `
  gdi32.lib user32.lib shell32.lib
```

`-arch=sm_89` targets RTX 40-series. For RTX 30-series use `sm_86`, RTX 20-series use `sm_75`.

---

## Controls

| Input | Action |
|---|---|
| W / S | Move forward / back |
| A / D | Strafe left / right |
| Mouse | Look around |
| Left click | Place block |
| Right click | Remove block |

---

## Performance (RTX 4070 Super, 1920×1080)

~9 rays per pixel (primary + 3 shadow + 4 AO + 1 bounce) = ~18M rays per frame.

| State | Behavior |
|---|---|
| Moving | Single noisy sample, real-time response |
| Still | Accumulates — converges in ~1–2 seconds |

Console prints kernel time, frame time, fps, and Mrays/s every 60 frames.

---

## Why

Built to learn how GPU ray tracing actually works — not through a framework, but by writing every piece by hand: the traversal algorithm, the lighting math, the Vulkan pipeline, the noise functions, the tree shapes. Everything visible on screen is computed from first principles each frame.
