/*
    1. Create Ray struct
    2. Create Obj struct
    3. For (frames per second * seconds):
        a. Generate image using CUDA
        b. output image from array to file
*/
#include <atomic>

#include "headers/raytracer.h"
#include "headers/vec.cuh"
#include "headers/ray.cuh"
//#include "headers/intercepts.cuh"
#include "headers/cudaHelpers.cuh"

// Other important Stuff
std::atomic<int> deviceId;
cudaDeviceProp deviceProperties;

// setup atomic and mutex
std::atomic<int> frameCounterSync(0);

struct Camera {
    dVec3 origin = dVec3{0.0, 0.0, 0.0};
    float focalLength = 1.0;
    float fov;
    // might need more, idk yet
};

__global__
void raytrace(unsigned char* image, ImageInfo imageInfo) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = (gridDim.x * blockDim.x);

    Camera camera;
    camera.origin = dVec3{sin(0.01f * static_cast<float>(imageInfo.frameNumber)), 0.0f, (cos(0.01f *static_cast<float>(imageInfo.frameNumber)))};

    dVec3 viewportUpperLeft = camera.origin - dVec3{0, 0, camera.focalLength} - (dVec3{imageInfo.viewportU} / 2) - (dVec3{imageInfo.viewportV} / 2);
    dVec3 pixel00Loc = viewportUpperLeft + 0.5 * (dVec3{imageInfo.viewportUDelta} + dVec3{imageInfo.viewportVDelta});

    for (int i = idx; i < (imageInfo.width * imageInfo.height); i += stride) {
        int h = imageInfo.height - (i / imageInfo.width);
        int w = (i % imageInfo.width);

        dVec3 pixelCenter = pixel00Loc + (w * dVec3{imageInfo.viewportUDelta}) + (h * dVec3{imageInfo.viewportVDelta});
        dVec3 rayDir = pixelCenter - camera.origin;

        Ray ray{rayDir, camera.origin};
        dVec3 color = ray.rayColor();

        int colorIndex = i * 3;
        image[colorIndex] = static_cast<unsigned char>(color.data.x * 255.999);
        image[colorIndex + 1] = static_cast<unsigned char>(color.data.y * 255.999);
        image[colorIndex + 2] = static_cast<unsigned char>(color.data.z * 255.999);
    }

    return;
}

__host__
void setup() {
    int devId;
    cudaGetDevice(&devId);
    deviceId.store(devId);
    cudaGetDeviceProperties(&deviceProperties, deviceId);
    checkCuda(cudaSetDevice(deviceId));
    return;
}

__host__
void workerFunction() {
    ImageInfo imageInfo{};
    imageInfo.totalFrames = (GlobalConstants::kFPS * GlobalConstants::kSeconds);
    size_t bytes = imageInfo.width * imageInfo.height * 3 * sizeof(unsigned char);

    unsigned char* imageHost;
    unsigned char* imageDevice;

    checkCuda(cudaMallocHost(&imageHost, bytes));
    checkCuda(cudaMalloc(&imageDevice, bytes));

    int blocks = deviceProperties.multiProcessorCount * deviceProperties.maxBlocksPerMultiProcessor;
    int threads = deviceProperties.maxThreadsPerBlock;

    while (frameCounterSync < (GlobalConstants::kFPS * GlobalConstants::kSeconds)) {
        imageInfo.frameNumber = frameCounterSync.fetch_add(1);

        raytrace<<<blocks, threads>>>(imageDevice, imageInfo);
        checkCuda(cudaGetLastError());

        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemcpy(imageHost, imageDevice, bytes, cudaMemcpyDeviceToHost));

        saveImage(imageHost, imageInfo);
    }
    checkCuda(cudaFree(imageDevice));
    checkCuda(cudaFreeHost(imageHost));
}