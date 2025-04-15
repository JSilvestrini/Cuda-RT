#ifndef CUDA_GLOBALS_H
#define CUDA_GLOBALS_H

// TODO: Eventually remove vec.h and place completely on device
// TODO: Need to move shape interceptions and other structs that
        // depend on vec3
#include "vec.h"

namespace GlobalConstants {
    inline int kWidth = 3840 / 2;
    inline int kHeight = 2160 / 2;
    inline float kViewportHeight = 2.0;
    inline float kViewportWidth = kViewportHeight * (float(kWidth) / kHeight);
    inline int kFPS = 60;
    inline int kSeconds = 1;
    inline float aspectRatio = 16.0 / 9.0;
}

struct ImageInfo {
    int width = GlobalConstants::kWidth;
    int height = GlobalConstants::kHeight;

    vec3 viewportU = vec3{GlobalConstants::kViewportWidth, 0.0, 0.0};
    vec3 viewportV = vec3{0.0, GlobalConstants::kViewportHeight, 0.0};
    vec3 viewportUDelta = viewportU / GlobalConstants::kWidth;
    vec3 viewportVDelta = viewportV / GlobalConstants::kHeight;

    int frameNumber;
    int totalFrames;
};

struct Vertex {
    vec3 pos;
    vec3 normal;
};

// Bound to change as time goes on
struct Materials {
    vec3 baseColor;
    float roughness;
    float metalness;
    vec3 specularColor;
    vec3 emissionColor;
    float opacity;
    float ior;
};

struct Sphere {
    vec3 origin;
    float radius;
};

struct Plane {
    vec3 point;
    vec3 normal;
};

struct Box {
    vec3 min;
    vec3 max;
};

struct Triangle {
    Vertex v1;
    Vertex v2;
    Vertex v3;
};

#endif