#ifndef VEC_BASE_STRUCT_H
#define VEC_BASE_STRUCT_H

struct vec3 {
    float x;
    float y;
    float z;
};

inline vec3 operator*(float t, const vec3& v) {
    return vec3{v.x * t, v.y * t, v.z * t};
}

inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

inline vec3 operator/(const vec3& v, float t) {
    return (1 / t) * v;
}

struct vec4 {
    float x;
    float y;
    float z;
    float t;
};

#endif