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
int kSeconds = 1;

// Other important Stuff
std::atomic<int> deviceId;
cudaDeviceProp deviceProperties;

// setup atomic and mutex
std::atomic<int> frameCounterSync(0);

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

struct Camera {
    vec3 origin;
    float fov;
    // might need more, idk yet
};

struct Ray {
    vec3 direction;
    vec3 origin;
};

struct Sphere {
    vec3 origin;
    float radius;
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

__device__
float intercept_sphere(Sphere sphere, Ray ray) {
    return 0.0f;
}

__global__
void raytrace(float* image, ImageInfo imageInfo) {
    int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    int stride = (gridDim.x * blockDim.x);

    for (int i = idx; i < (imageInfo.width * imageInfo.height); i += stride) {
        int colorIndex = i * 3;

        if (colorIndex + 2 > (imageInfo.width * imageInfo.height * 3)) {
            break;
        }

        image[colorIndex] = static_cast<float>((i) / imageInfo.width) / (imageInfo.height);
        image[colorIndex + 1] = static_cast<float>((i) % imageInfo.width) / (imageInfo.width);
        image[colorIndex + 2] = static_cast<float>(imageInfo.frameNumber) / (imageInfo.totalFrames);
    }

    return;
}

__host__
void saveImage(float* image, ImageInfo imageInfo) {
    std::ofstream file;
    std::ostringstream fileName;
    fileName << "../images/" << std::setfill('0') << std::setw(4) << imageInfo.frameNumber << ".ppm";
    file.open(fileName.str());
    file << "P3\n" << imageInfo.width << " " << imageInfo.height << "\n255\n";
    for (int i = 0; i < (imageInfo.height * imageInfo.width); i++) {
        int colorIndex = i * 3;
        if (colorIndex + 2 < (imageInfo.width * imageInfo.height * 3)) {
            file << int(image[colorIndex + 0] * 255.999) << " " << int(image[colorIndex + 1] * 255.999) << " " << int(image[colorIndex + 2] * 255.999) << "\n";
        }
    }

    file.close();

    return;
}

__host__
void setUp() {
    int devId;
    cudaGetDevice(&devId);
    deviceId.store(devId);
    cudaGetDeviceProperties(&deviceProperties, deviceId);
    checkCuda(cudaSetDevice(deviceId));
    // if image folers exist, remove them
    // create dir
    #if __linux__
        mkdir("images");
    #else
        CreateDirectory((LPCSTR) "../images", NULL);
    #endif
    return;
}

__host__
void cleanUp() {
    // run ffmpeg to make video
    std::ostringstream commandStream;
    commandStream << "ffmpeg -framerate " << kFPS << " -i \"../images/%04d.ppm\" -c:v libx264 -preset slow -crf 18 -vf \"scale=";
    commandStream << kWidth << ":" << kHeight << "\" ../output/output.mp4";
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
void workerFunction(int blocks, int threads) {
    ImageInfo imageInfo{};
    imageInfo.totalFrames = (kFPS * kSeconds);
    size_t bytes = imageInfo.width * imageInfo.height * 3 * sizeof(float);

    float* imageHost;
    float* imageDevice;
    checkCuda(cudaMallocHost(&imageHost, bytes));
    checkCuda(cudaMalloc(&imageDevice, bytes));

    while (frameCounterSync < (kFPS * kSeconds)) {
        imageInfo.frameNumber = frameCounterSync.fetch_add(1);

        raytrace<<<blocks, threads>>>(imageDevice, imageInfo);
        checkCuda(cudaDeviceSynchronize());
        checkCuda(cudaMemcpy(imageHost, imageDevice, bytes, cudaMemcpyDeviceToHost));

        saveImage(imageHost, imageInfo);
    }
    checkCuda(cudaFree(imageDevice));
    checkCuda(cudaFreeHost(imageHost));
}

/*
int main() {
    setUp();

    std::vector<std::thread> threadList(std::thread::hardware_concurrency());
    //std::vector<std::thread> threads(1);

    int blocks = deviceProperties.multiProcessorCount * deviceProperties.maxBlocksPerMultiProcessor;
    int threads = deviceProperties.maxThreadsPerBlock;

    for (int i = 0; i < threadList.size(); i++) {
        threadList[i] = std::thread(workerFunction, blocks, threads);
    }

    for (int i = 0; i < threadList.size(); i++) {
        threadList[i].join();
    }

    cleanUp();
}

*/