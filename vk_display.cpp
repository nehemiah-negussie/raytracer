#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include "vk_display.h"
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#define VK_CHECK(x) do { \
    VkResult _r = (x); \
    if (_r != VK_SUCCESS) { fprintf(stderr, "Vulkan error %d at line %d\n", _r, __LINE__); exit(1); } \
} while(0)

struct VkDisplay {
    GLFWwindow*      window;
    VkInstance       instance;
    VkSurfaceKHR     surface;
    VkPhysicalDevice phys;
    VkDevice         dev;
    VkQueue          queue;
    uint32_t         qfam;

    VkSwapchainKHR         swapchain;
    VkFormat               swapFmt;
    VkExtent2D             swapExtent;
    std::vector<VkImage>   swapImages;

    VkCommandPool                cmdPool;
    std::vector<VkCommandBuffer> cmds;
    std::vector<VkFence>         fences;

    VkSemaphore semAcquire;
    VkSemaphore semPresent;

    VkBuffer       staging;
    VkDeviceMemory stagingMem;
    void*          stagingPtr;
    size_t         stagingSize;

    bool   isBGRA;
    double lastMouseX, lastMouseY;
    bool   firstMouse;
    bool   prevLMB, prevRMB;
};

VkDisplay* vkdisplay_create(int width, int height, const char* title) {
    VkDisplay* d = new VkDisplay();

    // --- GLFW ---
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE,  GLFW_FALSE);
    d->window = glfwCreateWindow(width, height, title, nullptr, nullptr);

    // --- Instance ---
    uint32_t extCount = 0;
    const char** glfwExts = glfwGetRequiredInstanceExtensions(&extCount);
    VkApplicationInfo appInfo = {};
    appInfo.sType      = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo instCI = {};
    instCI.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo        = &appInfo;
    instCI.enabledExtensionCount   = extCount;
    instCI.ppEnabledExtensionNames = glfwExts;
    VK_CHECK(vkCreateInstance(&instCI, nullptr, &d->instance));

    // --- Surface ---
    VK_CHECK(glfwCreateWindowSurface(d->instance, d->window, nullptr, &d->surface));

    // --- Physical device ---
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(d->instance, &devCount, nullptr);
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(d->instance, &devCount, devs.data());
    d->phys = devs[0];

    // --- Queue family (graphics + present) ---
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(d->phys, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qfProps(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(d->phys, &qfCount, qfProps.data());
    d->qfam = 0;
    for (uint32_t i = 0; i < qfCount; i++) {
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(d->phys, i, d->surface, &present);
        if ((qfProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && present) {
            d->qfam = i;
            break;
        }
    }

    // --- Logical device ---
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qCI = {};
    qCI.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qCI.queueFamilyIndex = d->qfam;
    qCI.queueCount       = 1;
    qCI.pQueuePriorities = &prio;
    const char* devExts[] = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };
    VkDeviceCreateInfo devCI = {};
    devCI.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCI.queueCreateInfoCount    = 1;
    devCI.pQueueCreateInfos       = &qCI;
    devCI.enabledExtensionCount   = 1;
    devCI.ppEnabledExtensionNames = devExts;
    VK_CHECK(vkCreateDevice(d->phys, &devCI, nullptr, &d->dev));
    vkGetDeviceQueue(d->dev, d->qfam, 0, &d->queue);

    // --- Swapchain ---
    VkSurfaceCapabilitiesKHR caps = {};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(d->phys, d->surface, &caps);

    uint32_t fmtCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(d->phys, d->surface, &fmtCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fmtCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(d->phys, d->surface, &fmtCount, formats.data());

    VkSurfaceFormatKHR chosen = formats[0];
    for (auto& f : formats)
        if (f.format == VK_FORMAT_R8G8B8A8_UNORM) { chosen = f; break; }

    d->swapFmt    = chosen.format;
    d->swapExtent = caps.currentExtent;
    d->isBGRA     = (chosen.format == VK_FORMAT_B8G8R8A8_UNORM ||
                     chosen.format == VK_FORMAT_B8G8R8A8_SRGB);

    uint32_t imgCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imgCount > caps.maxImageCount)
        imgCount = caps.maxImageCount;

    VkSwapchainCreateInfoKHR scCI = {};
    scCI.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    scCI.surface          = d->surface;
    scCI.minImageCount    = imgCount;
    scCI.imageFormat      = chosen.format;
    scCI.imageColorSpace  = chosen.colorSpace;
    scCI.imageExtent      = d->swapExtent;
    scCI.imageArrayLayers = 1;
    scCI.imageUsage       = VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    scCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    scCI.preTransform     = caps.currentTransform;
    scCI.compositeAlpha   = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    scCI.presentMode      = VK_PRESENT_MODE_FIFO_KHR;
    scCI.clipped          = VK_TRUE;
    VK_CHECK(vkCreateSwapchainKHR(d->dev, &scCI, nullptr, &d->swapchain));

    vkGetSwapchainImagesKHR(d->dev, d->swapchain, &imgCount, nullptr);
    d->swapImages.resize(imgCount);
    vkGetSwapchainImagesKHR(d->dev, d->swapchain, &imgCount, d->swapImages.data());

    // --- Command pool + buffers ---
    VkCommandPoolCreateInfo cpCI = {};
    cpCI.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpCI.queueFamilyIndex = d->qfam;
    cpCI.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(d->dev, &cpCI, nullptr, &d->cmdPool));

    d->cmds.resize(imgCount);
    VkCommandBufferAllocateInfo cbAI = {};
    cbAI.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbAI.commandPool        = d->cmdPool;
    cbAI.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAI.commandBufferCount = imgCount;
    VK_CHECK(vkAllocateCommandBuffers(d->dev, &cbAI, d->cmds.data()));

    // --- Staging buffer (host visible, RGBA) ---
    d->stagingSize = (size_t)width * height * 4;
    VkBufferCreateInfo bufCI = {};
    bufCI.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufCI.size        = d->stagingSize;
    bufCI.usage       = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufCI.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(d->dev, &bufCI, nullptr, &d->staging));

    VkMemoryRequirements memReq = {};
    vkGetBufferMemoryRequirements(d->dev, d->staging, &memReq);
    VkPhysicalDeviceMemoryProperties memProps = {};
    vkGetPhysicalDeviceMemoryProperties(d->phys, &memProps);
    uint32_t memIdx = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
        bool fits = memReq.memoryTypeBits & (1u << i);
        bool host = memProps.memoryTypes[i].propertyFlags &
                    (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (fits && host) { memIdx = i; break; }
    }
    VkMemoryAllocateInfo allocI = {};
    allocI.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocI.allocationSize  = memReq.size;
    allocI.memoryTypeIndex = memIdx;
    VK_CHECK(vkAllocateMemory(d->dev, &allocI, nullptr, &d->stagingMem));
    vkBindBufferMemory(d->dev, d->staging, d->stagingMem, 0);
    vkMapMemory(d->dev, d->stagingMem, 0, d->stagingSize, 0, &d->stagingPtr);

    // --- Semaphores ---
    VkSemaphoreCreateInfo semCI = { VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VK_CHECK(vkCreateSemaphore(d->dev, &semCI, nullptr, &d->semAcquire));
    VK_CHECK(vkCreateSemaphore(d->dev, &semCI, nullptr, &d->semPresent));

    // --- Per-image fences ---
    d->fences.resize(imgCount);
    VkFenceCreateInfo fenceCI = {};
    fenceCI.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCI.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (auto& f : d->fences)
        VK_CHECK(vkCreateFence(d->dev, &fenceCI, nullptr, &f));

    // capture mouse
    glfwSetInputMode(d->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwGetCursorPos(d->window, &d->lastMouseX, &d->lastMouseY);
    d->firstMouse = true;

    return d;
}

int vkdisplay_poll(VkDisplay* d, int* up, int* down, int* left, int* right, float* mdx, float* mdy, int* lclick, int* rclick) {
    glfwPollEvents();
    *up    = glfwGetKey(d->window, GLFW_KEY_W) == GLFW_PRESS;
    *down  = glfwGetKey(d->window, GLFW_KEY_S) == GLFW_PRESS;
    *left  = glfwGetKey(d->window, GLFW_KEY_D) == GLFW_PRESS;
    *right = glfwGetKey(d->window, GLFW_KEY_A) == GLFW_PRESS;

    double mx, my;
    glfwGetCursorPos(d->window, &mx, &my);
    if (d->firstMouse) {
        *mdx = *mdy = 0.0f;
        d->firstMouse = false;
    } else {
        *mdx = (float)(mx - d->lastMouseX);
        *mdy = (float)(my - d->lastMouseY);
    }
    d->lastMouseX = mx;
    d->lastMouseY = my;

    bool curLMB = glfwGetMouseButton(d->window, GLFW_MOUSE_BUTTON_LEFT)  == GLFW_PRESS;
    bool curRMB = glfwGetMouseButton(d->window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    *lclick = (!d->prevLMB && curLMB) ? 1 : 0;
    *rclick = (!d->prevRMB && curRMB) ? 1 : 0;
    d->prevLMB = curLMB;
    d->prevRMB = curRMB;

    if (glfwWindowShouldClose(d->window) ||
        glfwGetKey(d->window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        return 0;
    return 1;
}

void vkdisplay_present(VkDisplay* d, const unsigned char* rgba, int width, int height) {
    // Upload pixels to staging buffer (swap R/B if swapchain is BGRA)
    uint8_t* dst = (uint8_t*)d->stagingPtr;
    if (d->isBGRA) {
        for (int i = 0; i < width * height; i++) {
            dst[i*4+0] = rgba[i*4+2];
            dst[i*4+1] = rgba[i*4+1];
            dst[i*4+2] = rgba[i*4+0];
            dst[i*4+3] = rgba[i*4+3];
        }
    } else {
        memcpy(dst, rgba, (size_t)width * height * 4);
    }

    // Acquire next swapchain image
    uint32_t imgIdx = 0;
    VkResult res = vkAcquireNextImageKHR(d->dev, d->swapchain, UINT64_MAX,
                                          d->semAcquire, VK_NULL_HANDLE, &imgIdx);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) return;

    vkWaitForFences(d->dev, 1, &d->fences[imgIdx], VK_TRUE, UINT64_MAX);
    vkResetFences(d->dev, 1, &d->fences[imgIdx]);

    // Record commands
    VkCommandBuffer cmd = d->cmds[imgIdx];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginI = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &beginI);

    // Transition: UNDEFINED → TRANSFER_DST
    VkImageMemoryBarrier bar = {};
    bar.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    bar.oldLayout                       = VK_IMAGE_LAYOUT_UNDEFINED;
    bar.newLayout                       = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    bar.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
    bar.image                           = d->swapImages[imgIdx];
    bar.subresourceRange                = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    bar.srcAccessMask                   = 0;
    bar.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &bar);

    // Copy staging buffer → swapchain image
    VkBufferImageCopy region = {};
    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    region.imageExtent      = { (uint32_t)width, (uint32_t)height, 1 };
    vkCmdCopyBufferToImage(cmd, d->staging, d->swapImages[imgIdx],
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition: TRANSFER_DST → PRESENT_SRC
    bar.oldLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    bar.newLayout     = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    bar.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    bar.dstAccessMask = 0;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
        0, 0, nullptr, 0, nullptr, 1, &bar);

    vkEndCommandBuffer(cmd);

    // Submit
    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submit = {};
    submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount   = 1;
    submit.pWaitSemaphores      = &d->semAcquire;
    submit.pWaitDstStageMask    = &waitStage;
    submit.commandBufferCount   = 1;
    submit.pCommandBuffers      = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores    = &d->semPresent;
    vkQueueSubmit(d->queue, 1, &submit, d->fences[imgIdx]);

    // Present
    VkPresentInfoKHR presentI = {};
    presentI.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentI.waitSemaphoreCount = 1;
    presentI.pWaitSemaphores    = &d->semPresent;
    presentI.swapchainCount     = 1;
    presentI.pSwapchains        = &d->swapchain;
    presentI.pImageIndices      = &imgIdx;
    vkQueuePresentKHR(d->queue, &presentI);
}

void vkdisplay_destroy(VkDisplay* d) {
    vkDeviceWaitIdle(d->dev);
    for (auto& f : d->fences) vkDestroyFence(d->dev, f, nullptr);
    vkDestroySemaphore(d->dev, d->semAcquire, nullptr);
    vkDestroySemaphore(d->dev, d->semPresent, nullptr);
    vkUnmapMemory(d->dev, d->stagingMem);
    vkDestroyBuffer(d->dev, d->staging, nullptr);
    vkFreeMemory(d->dev, d->stagingMem, nullptr);
    vkDestroyCommandPool(d->dev, d->cmdPool, nullptr);
    vkDestroySwapchainKHR(d->dev, d->swapchain, nullptr);
    vkDestroyDevice(d->dev, nullptr);
    vkDestroySurfaceKHR(d->instance, d->surface, nullptr);
    vkDestroyInstance(d->instance, nullptr);
    glfwDestroyWindow(d->window);
    glfwTerminate();
    delete d;
}
