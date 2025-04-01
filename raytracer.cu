/*
    1. Create Ray struct
    2. Create Obj struct
    3. For (frames per second * seconds):
        a. Generate image using CUDA
        b. output image from array to file
*/

#include <stdio.h>
#include <cstdlib>
#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <iostream>

#include <thread>
#include <atomic>
#include <vector>

#if __linux__
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <windows.h>
#endif

// Global Constants
int kWidth = 3840;
int kHeight = 2160;
int kFPS = 30;
int kSeconds = 2;

// Other important Stuff
int deviceId;
cudaDeviceProp deviceProperties;

// setup atomic and mutex
std::atomic<int> frameCounterSync(0);

struct Camera {
    float origin;
    float fov;
    // might need more, idk yet
};

struct ImageInfo {
    int width = kWidth;
    int height = kHeight;
    int frameNumber;
    int totalFrames;
};

struct vec3 {
    float x;
    float y;
    float z;
};

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

__global__
void raytrace(float* image, ImageInfo imageInfo) {
    // for each sphere, plane, shape, etc. in scene
    // keep track of Z of intercept, use closest Z-coord
    // get thread index and stride
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = (gridDim.x * blockDim.x);

    for (int i = idx; i < (imageInfo.width * imageInfo.height); i += stride) {
        int colorIndex = i * 3;

        if (colorIndex + 2 < (imageInfo.width * imageInfo.height * 3)) {
            image[colorIndex] = static_cast<float>((i) / imageInfo.width) / (imageInfo.height);
            image[colorIndex + 1] = static_cast<float>((i) % imageInfo.width) / (imageInfo.width);
            image[colorIndex + 2] = imageInfo.frameNumber / (imageInfo.totalFrames);
        }
    }

    return;
}

__host__
void saveImage(float* image, ImageInfo imageInfo) {
    std::ofstream file;
    std::ostringstream fileName;
    fileName << "images/" << std::setfill('0') << std::setw(4) << imageInfo.frameNumber << ".ppm";
    file.open(fileName.str());
    file << "P3\n" << imageInfo.width << " " << imageInfo.height << "\n255\n";
    for (int i = 0; i < (imageInfo.height * imageInfo.width); i++) {
        int colorIndex = i * 3;
        file << int(image[colorIndex + 0] * 255.999) << " " << int(image[colorIndex + 1] * 255.999) << " " << int(image[colorIndex + 2] * 255.999) << "\n";
    }
    file.close();

    return;
}

__host__
void setUp() {
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&deviceProperties, deviceId);
    checkCuda(cudaSetDevice(deviceId));
    // if image folers exist, remove them
    // create dir
    #if __linux__
        mkdir("images");
    #else
        CreateDirectory((LPCSTR) "images", NULL);
    #endif
    return;
}

__host__
void cleanUp() {
    // run ffmpeg to make video
    std::ostringstream commandStream;
    commandStream << "ffmpeg -framerate " << kFPS << " -i \"images/%04d.ppm\" -c:v libx264 -preset slow -crf 18 -vf \"scale=";
    commandStream << kWidth << ":" << kHeight << "\" output.mp4";
    std::string command = commandStream.str();
    int res = system(command.c_str());

    if (res == 0) {
        printf("It Worked!!!\n");
    }
    // delete all files in folders

    // delete folders
    #if __linux__
        //
    #else
        //
    #endif
    return;
}

__host__
void workerFunction() {
    ImageInfo imageInfo{};
    imageInfo.totalFrames = (kFPS * kSeconds);
    int N = imageInfo.width * imageInfo.height * 3;
    size_t bytes = N * sizeof(float);
    //printf("%d", bytes);

    int blocks = deviceProperties.multiProcessorCount * deviceProperties.maxBlocksPerMultiProcessor;
    int threads = deviceProperties.maxThreadsPerBlock;

    printf("Blocks: %d, Threads: %d\n", blocks, threads);

    float* image;
    checkCuda(cudaMallocManaged(&image, bytes));

    while (frameCounterSync < (kFPS * kSeconds)) {
        imageInfo.frameNumber = frameCounterSync.fetch_add(1);

        //printf("Here Lies the Failure, Prefetch 1\n");
        //checkCuda(cudaMemPrefetchAsync(&image, bytes, deviceId));
        //printf("Here Did Not Lie the Failure\n");
        raytrace<<<blocks, threads>>>(image, imageInfo);
        //printf("Here Lies the Failure, Sync\n");
        checkCuda(cudaDeviceSynchronize());
        //printf("Here Did Not Lie the Failure\n");
        //printf("Here Lies the Failure, Prefetch 2\n");
        //checkCuda(cudaMemPrefetchAsync(&image, bytes, cudaCpuDeviceId));
        //printf("Here Did Not Lie the Failure\n");
        saveImage(image, imageInfo);
    }
    checkCuda(cudaFree(image));
}

int main() {
    setUp();

    std::vector<std::thread> threads(std::thread::hardware_concurrency());
    //std::vector<std::thread> threads(1);

    for (int i = 0; i < threads.size(); i++) {
        threads[i] = std::thread(workerFunction);
    }

    for (int i = 0; i < threads.size(); i++) {
        threads[i].join();
    }

    cleanUp();
}