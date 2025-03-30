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

#if __linux__
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <windows.h>
#endif


int kWidth = 3840;
int kHeight = 2160;
int kFPS = 30;
int kSeconds = 2;

struct Camera {
    float origin;
    float fov;
    // might need more, idk yet
};

struct ImageInfo {
    int width = kWidth;
    int height = kHeight;
};

__device__
float dot() {
    return 0.0f;
}

__global__
void raytrace(float* image, ImageInfo imageInfo) {
    return;
}

__host__
void saveImage(int n, float* image, ImageInfo imageInfo) {
    std::ofstream file;
    std::ostringstream fileName;
    fileName << "images/" << std::setfill('0') << std::setw(4) << n << ".ppm";
    file.open(fileName.str());
    file << "P3\n" << kWidth << " " << kHeight << "\n255\n";
    for (int j = 0; j < kHeight; j++) {
        for (int i = 0; i < kWidth; i++) {
            file << int(double(i) / kWidth * 255.999) << " " << int(double(j) / kHeight * 255.999) << " " << int(double(n) / (kFPS * kSeconds) * 255.999) << "\n";
        }
    }
    file.close();

    return;
}

__host__
void setUp() {
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

int main() {
    int deviceId;
    cudaDeviceProp deviceProperties;
    cudaGetDevice(&deviceId);
    cudaGetDeviceProperties(&deviceProperties, deviceId);

    setUp();

    // use C++ threads to determine max PC threads
    // have each thread work on its own image by using an atomic or something
    // each thread will determine location of camera/lights if they move
    // allows for multiple images to render at once, could speed up
    //      computations since CPU is the bottleneck
    // use an atomic and some jthreads, start at 0 and go to fps * seconds
    // increment number when getting frame number, use number to determine positions

    ImageInfo imageInfo{};
    int N = imageInfo.width * imageInfo.height;
    int bytes = N * sizeof(float) * 3;

    // try and use 2d kernel
    // bitshift to get the greatest power of 2 for blocks and threads
    int blocks = deviceProperties.multiProcessorCount * deviceProperties.maxBlocksPerMultiProcessor;
    int threads = deviceProperties.maxThreadsPerBlock;

    for (int i = 0; i < (kFPS * kSeconds); i++) {
        float* image;
        cudaMallocManaged(&image, bytes);
        cudaMemPrefetchAsync(&image, N, deviceId);
        raytrace<<<blocks, threads>>>(image, imageInfo);
        cudaDeviceSynchronize();
        cudaMemPrefetchAsync(&image, N, cudaCpuDeviceId);
        saveImage(i, image, imageInfo);
        cudaFree(image);
    }

    cleanUp();

    printf("Blocks: %d\n", blocks);
    printf("Threads: %d\n", threads);
}