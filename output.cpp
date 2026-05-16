#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "output.h"

void writePNG(const char* filename, unsigned char* data, int width, int height) {
    stbi_write_png(filename, width, height, 3, data, width * 3);
}
