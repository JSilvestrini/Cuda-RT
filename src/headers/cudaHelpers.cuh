#ifndef CUSTOM_CUDA_HELPERS_H
#define CUSTOM_CUDA_HELPERS_H

#include <cassert>
#include <stdio.h>
#include <cstdlib>

inline cudaError_t checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
        assert(result == cudaSuccess);
    }

    return result;
}

#endif