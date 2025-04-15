#include "headers/vec.cuh"

__device__
dVec3 dVec3::operator-() const {
    return dVec3{vec3{-data.x, -data.y, -data.z}};
}

__device__
dVec3& dVec3::operator+=(const dVec3 other) {
    data.x += other.data.x;
    data.y += other.data.y;
    data.z += other.data.z;
    return *this;
}

__device__
dVec3& dVec3::operator-=(const dVec3 other) {
    data.x -= other.data.x;
    data.y -= other.data.y;
    data.z -= other.data.z;
    return *this;
}

__device__
dVec3& dVec3::operator*=(const dVec3 other) {
    data.x *= other.data.x;
    data.y *= other.data.y;
    data.z *= other.data.z;
    return *this;
}

__device__
dVec3& dVec3::operator/=(const dVec3 other) {
    data.x /= other.data.x;
    data.y /= other.data.y;
    data.z /= other.data.z;
    return *this;
}

__device__
float dVec3::length() const {
    return sqrtf(lengthSquared());
}

__device__
float dVec3::lengthSquared() const {
    return (powf(data.x, 2.0) + powf(data.y, 2.0) + powf(data.z, 2.0));
}

__device__
float dot(dVec3 a, dVec3 b) {
    return ((a.data.x * b.data.x) + (a.data.y * b.data.y) + (a.data.z * b.data.z));
}

__device__
dVec3 cross(dVec3 a, dVec3 b) {
    float x = (a.data.y * b.data.z) - (a.data.z * b.data.y);
    float y = (a.data.z * b.data.x) - (a.data.x * b.data.z);
    float z = (a.data.x * b.data.y) - (a.data.y * b.data.x);

    return dVec3{vec3{x, y, z}};
}

__device__
dVec3 normalize(dVec3 a) {
    return a / a.length();
}