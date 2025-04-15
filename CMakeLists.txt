cmake_minimum_required(VERSION 3.18)
project(cuda_offline_raytracing LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17) # Or a later CUDA standard
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories(src/headers)

find_package(CUDA REQUIRED)
if(CUDA_FOUND)
    message(STATUS "CUDA found (version ${CUDA_VERSION})")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
else()
    message(FATAL_ERROR "CUDA not found. Please ensure CUDA is installed and in your PATH.")
endif()

file(GLOB CMAKE_CXX_SOURCES
    "src/*.cpp"
)

file(GLOB CMAKE_CUDA_SOURCES
    "src/*.cu"
)

add_executable(my_cuda_app
    ${CMAKE_CXX_SOURCES}
    ${CMAKE_CUDA_SOURCES}
)

#target_link_options(my_cuda_app PRIVATE
#    "$<$<COMPILE_LANGUAGE:CUDA>:-arch=${CUDA_ARCHITECTURES}>"
#)