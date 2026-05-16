#pragma once
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>

#define WORLD_WIDTH   512
#define WORLD_HEIGHT  256
#define WORLD_DEPTH   512
#define MAX_CAMPFIRES 128
#define I(x,y,z) ((x)*WORLD_HEIGHT*WORLD_DEPTH + (y)*WORLD_DEPTH + (z))

enum class BlockType : uint8_t {
    Air = 0, Grass, Dirt, Stone, Sand, Snow, Wood, Leaves, Campfire
};

__device__ float3 blockColor(BlockType b) {
    switch (b) {
        case BlockType::Grass:    return {0.27f, 0.54f, 0.18f};
        case BlockType::Dirt:     return {0.40f, 0.26f, 0.13f};
        case BlockType::Stone:    return {0.47f, 0.47f, 0.47f};
        case BlockType::Sand:     return {0.76f, 0.70f, 0.50f};
        case BlockType::Snow:     return {0.90f, 0.92f, 0.95f};
        case BlockType::Wood:     return {0.38f, 0.24f, 0.11f};
        case BlockType::Leaves:   return {0.13f, 0.42f, 0.08f};
        case BlockType::Campfire: return {1.00f, 0.45f, 0.05f};
        default:                  return {0.0f,  0.0f,  0.0f};
    }
}

__device__ bool isEmissive(BlockType b) { return b == BlockType::Campfire; }

typedef struct VoxelGrid {
    BlockType a[WORLD_WIDTH * WORLD_HEIGHT * WORLD_DEPTH];
} VoxelGrid;

// --- noise ---

static float hashf(int x, int z) {
    unsigned int h = (unsigned int)(x * 1619 + z * 31337 + 13);
    h = (h ^ (h >> 16)) * 0x45d9f3bU;
    h = (h ^ (h >> 16)) * 0x45d9f3bU;
    h ^= h >> 16;
    return (float)(h & 0xFFFF) / 65535.0f;
}

static float noise2d(float x, float z) {
    int ix = (int)floorf(x), iz = (int)floorf(z);
    float fx = x-ix, fz = z-iz;
    fx = fx*fx*(3-2*fx); fz = fz*fz*(3-2*fz);
    return hashf(ix,iz)*(1-fx)*(1-fz) + hashf(ix+1,iz)*fx*(1-fz)
         + hashf(ix,iz+1)*(1-fx)*fz   + hashf(ix+1,iz+1)*fx*fz;
}

static float octaveNoise(float x, float z, int oct, float scale) {
    float v=0,a=1,f=1,m=0;
    for (int i=0;i<oct;i++) { v+=noise2d(x*f/scale,z*f/scale)*a; m+=a; a*=.5f; f*=2.f; }
    return v/m;
}

// --- block placement helpers ---

static void setBlock(VoxelGrid* g, int x, int y, int z, BlockType b) {
    if (x<0||x>=WORLD_WIDTH||y<0||y>=WORLD_HEIGHT||z<0||z>=WORLD_DEPTH) return;
    if (g->a[I(x,y,z)] == BlockType::Air) g->a[I(x,y,z)] = b;
}

// Oak: round canopy, slight random lean
static void placeOak(VoxelGrid* g, int x, int y, int z) {
    int h = 5 + (int)(hashf(x,z) * 3);
    for (int i=0;i<h;i++) { g->a[I(x,y+i,z)] = BlockType::Wood; }
    int lx = x + (int)(hashf(x+3,z)*3)-1;
    int lz = z + (int)(hashf(x,z+3)*3)-1;
    int ly = y+h, r = 3;
    for (int dx=-r;dx<=r;dx++) for (int dy=-2;dy<=r;dy++) for (int dz=-r;dz<=r;dz++) {
        float jitter = hashf(lx+dx,lz+dz)*0.8f;
        if ((float)(dx*dx+dy*dy+dz*dz) <= (r+jitter)*(r+jitter))
            setBlock(g, lx+dx, ly+dy, lz+dz, BlockType::Leaves);
    }
}

// Pine: tall, layered cone
static void placePine(VoxelGrid* g, int x, int y, int z) {
    int h = 11 + (int)(hashf(x+1,z)*5);
    for (int i=0;i<h;i++) { if(y+i<WORLD_HEIGHT) g->a[I(x,y+i,z)] = BlockType::Wood; }
    for (int layer=0;layer<5;layer++) {
        int r = 5-layer, ly = y+h-2-layer*2;
        for (int dx=-r;dx<=r;dx++) for (int dz=-r;dz<=r;dz++)
            if (dx*dx+dz*dz <= r*r+r)
                setBlock(g, x+dx, ly, z+dz, BlockType::Leaves);
    }
    setBlock(g, x, y+h, z, BlockType::Leaves);
}

// Birch: tall slim, small oval crown
static void placeBirch(VoxelGrid* g, int x, int y, int z) {
    int h = 9 + (int)(hashf(x,z+1)*4);
    for (int i=0;i<h;i++) { if(y+i<WORLD_HEIGHT) g->a[I(x,y+i,z)] = BlockType::Wood; }
    int ly = y+h, r = 2;
    for (int dx=-r;dx<=r;dx++) for (int dy=-1;dy<=r+1;dy++) for (int dz=-r;dz<=r;dz++)
        if (dx*dx + dy*dy/2 + dz*dz <= r*r+1)
            setBlock(g, x+dx, ly+dy, z+dz, BlockType::Leaves);
}

// Twisted: irregular multi-stem
static void placeTwisted(VoxelGrid* g, int x, int y, int z) {
    int h = 7 + (int)(hashf(x+2,z+2)*4);
    for (int i=0;i<h;i++) {
        int ox = (i > h/2) ? (int)(hashf(x+i,z)*3)-1 : 0;
        int oz = (i > h/2) ? (int)(hashf(x,z+i)*3)-1 : 0;
        setBlock(g, x+ox, y+i, z+oz, BlockType::Wood);
    }
    // two small leaf clusters
    for (int c=0;c<2;c++) {
        int cx2 = x+(int)(hashf(x+c*7,z)*5)-2;
        int cz2 = z+(int)(hashf(x,z+c*7)*5)-2;
        int ly = y+h-c, r=2;
        for (int dx=-r;dx<=r;dx++) for (int dy=-1;dy<=r;dy++) for (int dz=-r;dz<=r;dz++)
            if (dx*dx+dy*dy+dz*dz <= r*r+1)
                setBlock(g, cx2+dx, ly+dy, cz2+dz, BlockType::Leaves);
    }
}

VoxelGrid* buildScene(float3 campfirePositions[], int* numCampfires) {
    VoxelGrid* grid = new VoxelGrid();
    memset(grid->a, 0, sizeof(grid->a));
    *numCampfires = 0;

    // height map for reuse
    int* hmap = new int[WORLD_WIDTH * WORLD_DEPTH];

    // --- terrain ---
    for (int x=0; x<WORLD_WIDTH; x++) {
        for (int z=0; z<WORLD_DEPTH; z++) {
            float n = octaveNoise((float)x,(float)z,6,200.f);
            int height = (int)(50 + n*100);
            if (height >= WORLD_HEIGHT) height = WORLD_HEIGHT-1;
            hmap[x*WORLD_DEPTH+z] = height;

            for (int y=0; y<height; y++) {
                BlockType b;
                if      (y < height-5) b = BlockType::Stone;
                else if (y < height-1) b = BlockType::Dirt;
                else if (height < 62)  b = BlockType::Sand;
                else if (height > 125) b = BlockType::Snow;
                else                   b = BlockType::Grass;
                grid->a[I(x,y,z)] = b;
            }
        }
    }

    // --- trees (every 14x14 cell, grass only) ---
    for (int cx=0; cx<WORLD_WIDTH; cx+=14) {
        for (int cz=0; cz<WORLD_DEPTH; cz+=14) {
            int tx = cx + (int)(hashf(cx,cz)*12);
            int tz = cz + (int)(hashf(cx+1,cz)*12);
            if (tx>=WORLD_WIDTH-5 || tz>=WORLD_DEPTH-5 || tx<5 || tz<5) continue;
            if (hashf(cx*3+7,cz*3+7) < 0.35f) continue;

            int ty = hmap[tx*WORLD_DEPTH+tz];
            if (ty <= 0 || ty >= WORLD_HEIGHT-20) continue;
            if (grid->a[I(tx,ty-1,tz)] != BlockType::Grass) continue;

            float t = hashf(cx*7+3,cz*7+3);
            if      (t < 0.30f) placeOak    (grid, tx, ty, tz);
            else if (t < 0.55f) placePine   (grid, tx, ty, tz);
            else if (t < 0.78f) placeBirch  (grid, tx, ty, tz);
            else                placeTwisted (grid, tx, ty, tz);
        }
    }

    // --- campfires (every 55x55 cell, grass only) ---
    for (int cx=25; cx<WORLD_WIDTH-25; cx+=55) {
        for (int cz=25; cz<WORLD_DEPTH-25; cz+=55) {
            if (*numCampfires >= MAX_CAMPFIRES) continue;
            if (hashf(cx+500,cz+500) < 0.5f) continue;

            int fx = cx + (int)(hashf(cx*2,cz*2)*40)-20;
            int fz = cz + (int)(hashf(cx*2+1,cz*2)*40)-20;
            if (fx<3||fx>=WORLD_WIDTH-3||fz<3||fz>=WORLD_DEPTH-3) continue;

            int fy = hmap[fx*WORLD_DEPTH+fz];
            if (fy<=0||fy>=WORLD_HEIGHT-3) continue;
            if (grid->a[I(fx,fy-1,fz)] != BlockType::Grass) continue;

            grid->a[I(fx,fy,fz)] = BlockType::Campfire;
            campfirePositions[*numCampfires] = {fx+0.5f, fy+0.9f, fz+0.5f};
            (*numCampfires)++;

            // stone ring
            int ring[4][2] = {{1,0},{-1,0},{0,1},{0,-1}};
            for (int r=0;r<4;r++) {
                int rx=fx+ring[r][0], rz=fz+ring[r][1];
                if (grid->a[I(rx,fy,rz)] == BlockType::Air)
                    grid->a[I(rx,fy,rz)] = BlockType::Stone;
            }
        }
    }

    delete[] hmap;
    return grid;
}
