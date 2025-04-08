#ifndef SHARED_CUDA_FUNCTIONS_H
#define SHARED_CUDA_FUNCTIONS_H

#include "globals.h"

#include <string>
#include <sstream>

// CUDA FILE FUNCTIONS
void setup();
void workerFunction();

// CPP FILE FUNCTIONS
void cleanup();
void saveImage(unsigned char* image, ImageInfo imageInfo);

#endif