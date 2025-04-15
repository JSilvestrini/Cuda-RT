#include "headers/intercepts.cuh"

__device__
vec4 CustomIntercepts::sphereIntercept(Sphere sphere, Ray r) {
    return vec4{};
}
__device__
vec4 CustomIntercepts::triangleIntercept(Triangle triangle, Ray r) {
    dVec3 e1 = dVec3{triangle.v2.pos} - dVec3{triangle.v1.pos};
    dVec3 e2 = dVec3{triangle.v3.pos} - dVec3{triangle.v1.pos};
    dVec3 nDir = normalize(r.direction);
    dVec3 rayCrossE2 = cross(nDir, e2);
    float det = dot(e1, rayCrossE2);

    if (det > -kEpsilon && det < kEpsilon) {
        return vec4{0.0f, 0.0f, 0.0f, -1.0f};
    }

    float invDet = 1.0 / det;
    dVec3 s = r.origin - dVec3{triangle.v1.pos};
    float u = invDet * dot(s, rayCrossE2);

    if ((u < 0 && abs(u) > kEpsilon) || (u > 1 && abs(u - 1) > kEpsilon)) {
        return vec4{0.0f, 0.0f, 0.0f, -1.0f};
    }

    dVec3 sCrossE1 = cross(s, e1);
    float v = invDet * dot(nDir, sCrossE1);

    if ((v < 0 && abs(v) > kEpsilon) || (u + v > 1 && abs(u + v - 1) > kEpsilon)) {
        return vec4{0.0f, 0.0f, 0.0f, -1.0f};
    }

    float t = invDet * dot(e2, sCrossE1);

    if (t < kEpsilon) {
        return vec4{0.0f, 0.0f, 0.0f, -1.0f};
    }

    vec3 intercept = (r.origin + (nDir * t)).data;
    return vec4{intercept.x, intercept.y, intercept.z, t};
}

__device__
vec4 CustomIntercepts::boxIntercept(Box box, Ray r) {
    return vec4{};
}

__device__
vec4 CustomIntercepts::planeIntercept(Plane plane, Ray r) {
    return vec4{};
}