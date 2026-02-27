#pragma once

namespace jsa::core {

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

Vec3 normalizedToWorld(float normalizedX,
                       float normalizedY,
                       float depthMeters,
                       float horizontalFovDeg,
                       float verticalFovDeg);

} // namespace jsa::core
