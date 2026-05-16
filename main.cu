#include "math_utils.cuh"
#include "world.cuh"
#include "camera.cuh"
#include "light.cuh"
#include "renderer.cuh"
#include "vk_display.h"
#include <cmath>
#include <cstdio>
#include <cstring>

#define WIDTH       1920
#define HEIGHT      1080
#define MOVE_SPEED  0.3f
#define SENSITIVITY 0.002f
#define REACH       10
#define DAY_SPEED   0.001f

// ---- benchmark ----

struct Bench {
    cudaEvent_t start, stop;
    float       kernelMs, frameMs;
    double      totalKernelMs, totalFrameMs;
    long long   frames, printEvery, nextPrint;
    float       peakMrays;
    int         numCampfires;

    void init() {
        cudaEventCreate(&start); cudaEventCreate(&stop);
        kernelMs = frameMs = 0.0f;
        totalKernelMs = totalFrameMs = 0.0;
        frames = 0; printEvery = 60; nextPrint = printEvery;
        peakMrays = 0.0f; numCampfires = 0;
    }
    void destroy() { cudaEventDestroy(start); cudaEventDestroy(stop); }

    void beginKernel() { cudaEventRecord(start); }
    void endKernel()   { cudaEventRecord(stop); cudaEventSynchronize(stop); cudaEventElapsedTime(&kernelMs, start, stop); }

    void endFrame(float wallMs, int accumFrames) {
        frameMs = wallMs;
        totalKernelMs += kernelMs;
        totalFrameMs  += frameMs;
        frames++;
        float mrays = (WIDTH * HEIGHT * 9.0f) / (kernelMs * 0.001f) / 1e6f;  // ~9 rays/pixel
        if (mrays > peakMrays) peakMrays = mrays;

        if (frames >= nextPrint) {
            double avgK   = totalKernelMs / frames;
            double avgF   = totalFrameMs  / frames;
            double avgMR  = (WIDTH * HEIGHT * 9.0) / (avgK * 0.001) / 1e6;
            printf("\n=== Benchmark [frame %lld | accum %d spp] ===\n", frames, accumFrames);
            printf("  kernel:   %6.2f ms   avg %6.2f ms\n", kernelMs, avgK);
            printf("  frame:    %6.2f ms   avg %6.2f ms\n", frameMs, avgF);
            printf("  fps:      %6.1f      avg %6.1f\n",    1000.0f/frameMs, 1000.0/avgF);
            printf("  Mrays/s:  %6.1f      peak %.1f\n",    avgMR, peakMrays);
            printf("  res:      %dx%d   %d campfires\n",    WIDTH, HEIGHT, numCampfires);
            fflush(stdout);
            nextPrint += printEvery;
        }
    }
};

// ---- CPU DDA for block picking ----

struct CPUHit { bool hit; int ix,iy,iz, nx,ny,nz; };

CPUHit cpuDDA(float px, float py, float pz,
              float dx, float dy, float dz,
              const VoxelGrid* grid) {
    int ix=(int)floorf(px), iy=(int)floorf(py), iz=(int)floorf(pz);
    int sx=dx>0?1:-1, sy=dy>0?1:-1, sz=dz>0?1:-1;
    float tDX=fabsf(dx)>1e-6f?fabsf(1.0f/dx):1e30f;
    float tDY=fabsf(dy)>1e-6f?fabsf(1.0f/dy):1e30f;
    float tDZ=fabsf(dz)>1e-6f?fabsf(1.0f/dz):1e30f;
    float tMX=fabsf(dx)>1e-6f?(dx>0?(ix+1-px)/dx:(ix-px)/dx):1e30f;
    float tMY=fabsf(dy)>1e-6f?(dy>0?(iy+1-py)/dy:(iy-py)/dy):1e30f;
    float tMZ=fabsf(dz)>1e-6f?(dz>0?(iz+1-pz)/dz:(iz-pz)/dz):1e30f;
    int nx=0,ny=0,nz=0;
    for (int s=0; s<REACH; s++) {
        if (ix<0||ix>=WORLD_WIDTH||iy<0||iy>=WORLD_HEIGHT||iz<0||iz>=WORLD_DEPTH) break;
        if (grid->a[I(ix,iy,iz)] != BlockType::Air) return {true,ix,iy,iz,nx,ny,nz};
        if (tMX<tMY && tMX<tMZ) { ix+=sx; tMX+=tDX; nx=-sx; ny=0;  nz=0;  }
        else if (tMY<tMZ)        { iy+=sy; tMY+=tDY; nx=0;  ny=-sy; nz=0;  }
        else                     { iz+=sz; tMZ+=tDZ; nx=0;  ny=0;   nz=-sz; }
    }
    return {false,0,0,0,0,0,0};
}

int main() {
    // build scene
    float3 campfirePos[MAX_CAMPFIRES];
    int    numCampfires = 0;
    printf("Building scene...\n"); fflush(stdout);
    VoxelGrid* host_grid = buildScene(campfirePos, &numCampfires);
    printf("Scene built: %d campfires\n", numCampfires); fflush(stdout);

    VoxelGrid* dev_grid;
    cudaMalloc(&dev_grid, sizeof(VoxelGrid));
    cudaMemcpy(dev_grid, host_grid, sizeof(VoxelGrid), cudaMemcpyHostToDevice);

    float3* dev_campfires = nullptr;
    if (numCampfires > 0) {
        cudaMalloc(&dev_campfires, sizeof(float3) * numCampfires);
        cudaMemcpy(dev_campfires, campfirePos, sizeof(float3) * numCampfires, cudaMemcpyHostToDevice);
    }

    // output + accumulation buffers
    unsigned char* dev_output;
    cudaMalloc(&dev_output, WIDTH * HEIGHT * 4);
    unsigned char* host_output = new unsigned char[WIDTH * HEIGHT * 4];

    float4* dev_accum;
    cudaMalloc(&dev_accum, WIDTH * HEIGHT * sizeof(float4));
    cudaMemset(dev_accum, 0, WIDTH * HEIGHT * sizeof(float4));

    dim3 block(16, 16);
    dim3 grid_dim((WIDTH + 15) / 16, (HEIGHT + 15) / 16);

    // camera
    float yaw = 0.0f, pitch = -0.3f;
    Camera cam;
    cam.pos     = {256.0f, 160.0f, 256.0f};
    cam.up      = {0.0f, 1.0f, 0.0f};
    cam.vfov    = 60.0f;
    cam.width   = WIDTH;
    cam.height  = HEIGHT;

    // lights
    Light sun;  sun.dir  = {0.57f, 0.77f, 0.29f}; sun.color  = {1.0f, 1.0f, 1.0f};
    Light fill; fill.dir = {-0.5f, 0.5f, -0.5f};
    float flen = len(fill.dir);
    fill.dir   = fill.dir / flen;
    fill.color = {0.4f, 0.6f, 1.0f};

    float timeOfDay = 1.5f;
    BlockType selectedBlock = BlockType::Stone;

    // temporal accumulation state
    int   frameIdx = 0;
    float lastYaw = yaw, lastPitch = pitch;
    float3 lastPos = cam.pos;

    Bench bench; bench.init(); bench.numCampfires = numCampfires;
    cudaEvent_t fStart, fStop;
    cudaEventCreate(&fStart); cudaEventCreate(&fStop);

    printf("Voxel RT | %dx%d | path tracing + temporal accumulation\n", WIDTH, HEIGHT);
    printf("Soft shadows (3spp) | Hemisphere AO (4spp) | 1-bounce GI | Reinhard+gamma\n\n");
    fflush(stdout);

    VkDisplay* display = vkdisplay_create(WIDTH, HEIGHT, "Voxel RT — Path Tracing");

    int   kup,kdown,kleft,kright, lclick,rclick;
    float mdx, mdy;
    bool  gridDirty = false;

    while (vkdisplay_poll(display, &kup,&kdown,&kleft,&kright, &mdx,&mdy, &lclick,&rclick)) {
        cudaEventRecord(fStart);

        // mouse look
        yaw   -= mdx * SENSITIVITY;
        pitch -= mdy * SENSITIVITY;
        pitch  = fmaxf(-1.4f, fminf(1.4f, pitch));

        cam.forward = {
            cosf(pitch) * sinf(yaw),
            sinf(pitch),
            cosf(pitch) * cosf(yaw)
        };

        float rx = cosf(yaw), rz = -sinf(yaw);
        if (kup)    { cam.pos.x += cam.forward.x*MOVE_SPEED; cam.pos.y += cam.forward.y*MOVE_SPEED; cam.pos.z += cam.forward.z*MOVE_SPEED; }
        if (kdown)  { cam.pos.x -= cam.forward.x*MOVE_SPEED; cam.pos.y -= cam.forward.y*MOVE_SPEED; cam.pos.z -= cam.forward.z*MOVE_SPEED; }
        if (kleft)  { cam.pos.x -= rx*MOVE_SPEED; cam.pos.z -= rz*MOVE_SPEED; }
        if (kright) { cam.pos.x += rx*MOVE_SPEED; cam.pos.z += rz*MOVE_SPEED; }

        // detect camera movement — reset accumulation
        bool camMoved = (cam.pos.x != lastPos.x || cam.pos.y != lastPos.y || cam.pos.z != lastPos.z
                      || yaw != lastYaw || pitch != lastPitch);
        if (camMoved || gridDirty) {
            frameIdx  = 0;
            lastPos   = cam.pos;
            lastYaw   = yaw;
            lastPitch = pitch;
        } else {
            frameIdx = (frameIdx < 8192) ? frameIdx + 1 : frameIdx;
        }

        // animate sun
        timeOfDay += DAY_SPEED;
        float sh   = sinf(timeOfDay);
        float slen2 = sqrtf(0.36f + sh*sh + 0.09f);
        sun.dir = {0.6f/slen2, sh/slen2, 0.3f/slen2};

        float3 skyHorizon, skyZenith;
        float  t = fmaxf(0.0f, fminf(sh / 0.4f, 1.0f));
        if (sh > 0.0f) {
            skyHorizon = {1.0f - t*0.25f, 0.4f + t*0.48f, 0.2f + t*0.8f};
            skyZenith  = {0.3f - t*0.2f,  0.2f + t*0.2f,  0.6f + t*0.2f};
            sun.color  = {1.0f, 0.6f + t*0.35f, 0.2f + t*0.65f};
            fill.color = {0.4f*t, 0.6f*t, 1.0f*t};
        } else {
            float n    = fminf(-sh / 0.5f, 1.0f);
            skyHorizon = {0.04f*n, 0.04f*n, 0.12f*n};
            skyZenith  = {0.01f*n, 0.01f*n, 0.06f*n};
            sun.color  = {0.0f, 0.0f, 0.0f};
            fill.dir   = {-sun.dir.x, -sun.dir.y, -sun.dir.z};
            fill.color = {0.05f, 0.05f, 0.15f};
        }

        // block interaction
        if (lclick || rclick) {
            CPUHit h = cpuDDA(cam.pos.x, cam.pos.y, cam.pos.z,
                              cam.forward.x, cam.forward.y, cam.forward.z,
                              host_grid);
            if (h.hit) {
                if (rclick) {
                    host_grid->a[I(h.ix,h.iy,h.iz)] = BlockType::Air;
                    gridDirty = true;
                } else if (lclick) {
                    int bx=h.ix+h.nx, by=h.iy+h.ny, bz=h.iz+h.nz;
                    if (bx>=0&&bx<WORLD_WIDTH&&by>=0&&by<WORLD_HEIGHT&&bz>=0&&bz<WORLD_DEPTH) {
                        host_grid->a[I(bx,by,bz)] = selectedBlock;
                        gridDirty = true;
                    }
                }
            }
        }

        if (gridDirty) {
            cudaMemcpy(dev_grid, host_grid, sizeof(VoxelGrid), cudaMemcpyHostToDevice);
            gridDirty = false;
        }

        bench.beginKernel();
        renderKernel<<<grid_dim, block>>>(
            dev_output, dev_accum, frameIdx,
            cam, sun, fill, skyHorizon, skyZenith,
            dev_grid, dev_campfires, numCampfires
        );
        bench.endKernel();

        cudaMemcpy(host_output, dev_output, WIDTH * HEIGHT * 4, cudaMemcpyDeviceToHost);
        vkdisplay_present(display, host_output, WIDTH, HEIGHT);

        cudaEventRecord(fStop);
        cudaEventSynchronize(fStop);
        float wallMs = 0.0f;
        cudaEventElapsedTime(&wallMs, fStart, fStop);
        bench.endFrame(wallMs, frameIdx);
    }

    // session summary
    if (bench.frames > 0) {
        double avgK  = bench.totalKernelMs / bench.frames;
        double avgF  = bench.totalFrameMs  / bench.frames;
        printf("\n=== Session Summary ===\n");
        printf("  frames:       %lld\n",  bench.frames);
        printf("  avg fps:      %.1f\n",  1000.0 / avgF);
        printf("  avg kernel:   %.2f ms\n", avgK);
        printf("  avg Mrays/s:  %.1f\n",  (WIDTH * HEIGHT * 9.0) / (avgK * 0.001) / 1e6);
        printf("  peak Mrays/s: %.1f\n",  bench.peakMrays);
    }

    bench.destroy();
    cudaEventDestroy(fStart); cudaEventDestroy(fStop);
    vkdisplay_destroy(display);
    delete[] host_output;
    delete   host_grid;
    cudaFree(dev_output);
    cudaFree(dev_accum);
    cudaFree(dev_grid);
    if (dev_campfires) cudaFree(dev_campfires);
    return 0;
}
