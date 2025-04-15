#ifndef CUDA_INTERCEPTS_H
#define CUDA_INTERCEPTS_H

#include "globals.h"
#include "ray.cuh"

__device__ inline float kEpsilon = 0.0001;

namespace CustomIntercepts {
    __device__ vec4 sphereIntercept(Sphere sphere, Ray r);
    __device__ vec4 triangleIntercept(Triangle triangle, Ray r);
    __device__ vec4 boxIntercept(Box box, Ray r);
    __device__ vec4 planeIntercept(Plane plane, Ray r);
}

#endif
