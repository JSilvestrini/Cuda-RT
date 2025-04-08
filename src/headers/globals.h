#ifndef CUDA_GLOBALS_H
#define CUDA_GLOBALS_H

namespace GlobalConstants {
    inline int kWidth = 3840;
    inline int kHeight = 2160;
    inline int kFPS = 60;
    inline int kSeconds = 2;
}

struct ImageInfo {
    int width = GlobalConstants::kWidth;
    int height = GlobalConstants::kHeight;
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

struct vertex {
    vec3 pos;
    vec3 normal;
}

#endif