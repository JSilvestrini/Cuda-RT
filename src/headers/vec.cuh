#ifndef VEC_DEVICE_STRUCT_H
#define VEC_DEVICE_STRUCT_H

#include "vec.h"

struct dVec3 {
    vec3 data;  

    __device__ dVec3 operator-() const;
    __device__ dVec3& operator+=(const dVec3 other);
    __device__ dVec3& operator-=(const dVec3 other);
    __device__ dVec3& operator*=(const dVec3 other);
    __device__ dVec3& operator/=(const dVec3 other);
    __device__ float length() const;
    __device__ float lengthSquared() const;
};

__device__ inline dVec3 operator+(const dVec3& first, const dVec3& last) {
    return dVec3{first.data.x + last.data.x, first.data.y + last.data.y, first.data.z + last.data.z};
}

__device__ inline dVec3 operator-(const dVec3& first, const dVec3& last) {
    return dVec3{first.data.x - last.data.x, first.data.y - last.data.y, first.data.z - last.data.z};
}

__device__ inline dVec3 operator*(const dVec3& first, const dVec3& last) {
    return dVec3{first.data.x * last.data.x, first.data.y * last.data.y, first.data.z * last.data.z};
}

__device__ inline dVec3 operator*(float t, const dVec3& v) {
    return dVec3{v.data.x * t, v.data.y * t, v.data.z * t};
}

__device__ inline dVec3 operator*(const dVec3& v, float t) {
    return t * v;
}

__device__ inline dVec3 operator/(const dVec3& v, float t) {
    return (1 / t) * v;
}

__device__ float dot(dVec3 a, dVec3 b);
__device__ dVec3 cross(dVec3 a, dVec3 b);
__device__ dVec3 normalize(dVec3 a);

#endif