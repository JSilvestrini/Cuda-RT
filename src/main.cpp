#include <iostream>
#include <fstream>
#include <iomanip>

#include <thread>
#include <vector>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "headers/stb_image_write.h"
#include "headers/raytracer.h"

void saveImage(unsigned char* image, ImageInfo imageInfo) {
    std::ofstream file;
    std::ostringstream fileName;
    fileName << "images/" << std::setfill('0') << std::setw(8) << imageInfo.frameNumber << ".png";
    int step = (GlobalConstants::kWidth * 3 * sizeof(unsigned char));

    stbi_write_png(fileName.str().c_str(), GlobalConstants::kWidth, GlobalConstants::kHeight, 3, (void*)image, step);

    return;
}

void cleanup() {
    // run ffmpeg to make video
    std::ostringstream commandStream;
    commandStream << "ffmpeg -framerate " << GlobalConstants::kFPS << " -i \"images/%08d.png\" -c:v libx264 -preset slow -crf 18 -vf \"scale=";
    commandStream << GlobalConstants::kWidth << ":" << GlobalConstants::kHeight << "\" output/output.mp4";
    std::string command = commandStream.str();
    int res = system(command.c_str());

    if (res == 0) {
        printf("It Worked!!!\n");
    }

    return;
}

int main() {
    setup();

    std::vector<std::thread> threadList(std::thread::hardware_concurrency());
    //std::vector<std::thread> threads(1);

    for (int i = 0; i < threadList.size(); i++) {
        threadList[i] = std::thread(workerFunction);
    }

    for (int i = 0; i < threadList.size(); i++) {
        threadList[i].join();
    }

    cleanup();
}