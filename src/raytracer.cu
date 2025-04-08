/*
    1. Create Ray struct
    2. Create Obj struct
    3. For (frames per second * seconds):
        a. Generate image using CUDA
        b. output image from array to file
*/
#include <atomic>

#include <cassert>
#include <stdio.h>
#include <cstdlib>

#include "headers/raytracer.h"

// Other important Stuff
std::atomic<int> deviceId;
cudaDeviceProp deviceProperties;

// setup atomic and mutex
std::atomic<int> frameCounterSync(0);

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }

    return result;
}

__device__
float dot(vec3 a, vec3 b) {
    return ((a.x * b.x) + (a.y * b.y) + (a.z * b.z));
}

__device__
vec3 cross(vec3 a, vec3 b) {
    float x = (a.y * b.z) - (a.z * b.y);
    float y = (a.z * b.x) - (a.x * b.z);
    float z = (a.x * b.y) - (a.y * b.x);

    return vec3{x, y, z};
}

__device__
float intercept_sphere(Sphere sphere, Ray ray) {
    return 0.0f;
}

__global__
void raytrace(unsigned char* image, ImageInfo imageInfo) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = (gridDim.x * blockDim.x);

    for (int i = idx; i < (imageInfo.width * imageInfo.height); i += stride) {
        int colorIndex = i * 3;

        if (colorIndex + 2 > (imageInfo.width * imageInfo.height * 3)) {
            break;
        }

        image[colorIndex] = static_cast<unsigned char>(static_cast<float>((i) / imageInfo.width) / (imageInfo.height) * 255.999);
        image[colorIndex + 1] = static_cast<unsigned char>(static_cast<float>((i) % imageInfo.width) / (imageInfo.width) * 255.999);
        image[colorIndex + 2] = static_cast<unsigned char>(static_cast<float>(imageInfo.frameNumber) / (imageInfo.totalFrames) * 255.999);
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