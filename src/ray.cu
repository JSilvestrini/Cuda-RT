#include "headers/ray.cuh"
#include "headers/intercepts.cuh"

__device__
dVec3 Ray::rayColor() {
    Triangle tri1{{{-0.5, -0.5, -0.5}, {0.0f, 0.0f, 1.0f}},
                    {{0.0f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}},
                    {{0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}}};

    vec4 intercept = CustomIntercepts::triangleIntercept(tri1, *this);

    if (intercept.t < 0.0f) {
        dVec3 nDir = normalize(direction);
        float a = 0.5 * (nDir.data.y + 1.0);
        return (1.0 - a) * dVec3{1.0, 1.0, 1.0} + a * dVec3{0.5, 0.7, 1.0};
    }

    return normalize({intercept.x, intercept.y, intercept.z});
}