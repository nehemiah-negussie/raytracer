#pragma once

struct VkDisplay;

VkDisplay* vkdisplay_create(int width, int height, const char* title);
void       vkdisplay_destroy(VkDisplay* d);
int        vkdisplay_poll(VkDisplay* d, int* up, int* down, int* left, int* right, float* mdx, float* mdy, int* lclick, int* rclick);
void       vkdisplay_present(VkDisplay* d, const unsigned char* rgba, int width, int height);
