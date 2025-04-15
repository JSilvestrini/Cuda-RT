#ifndef CUSTOM_RAY_STRUCT_H
#define CUSTOM_RAY_STRUCT_H

#include "vec.cuh"

struct Ray {
    dVec3 direction;
    dVec3 origin;

    __device__ dVec3 rayColor();
};

#endif