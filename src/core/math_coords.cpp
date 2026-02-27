#include <jsa/core/math_coords.hpp>

#include <algorithm>
#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979323846f;

} // namespace

namespace jsa::core {

Vec3 normalizedToWorld(float normalizedX,
                       float normalizedY,
                       float depthMeters,
                       float horizontalFovDeg,
                       float verticalFovDeg) {
    const float clampedX = std::clamp(normalizedX, 0.0f, 1.0f);
    const float clampedY = std::clamp(normalizedY, 0.0f, 1.0f);
    const float safeDepth = std::max(depthMeters, 0.001f);

    const float centeredX = clampedX - 0.5f;
    const float centeredY = 0.5f - clampedY;

    const float fovHRad = horizontalFovDeg * kPi / 180.0f;
    const float fovVRad = verticalFovDeg * kPi / 180.0f;

    Vec3 position{};
    position.x = safeDepth * std::tan(centeredX * fovHRad);
    position.y = safeDepth * std::tan(centeredY * fovVRad);
    position.z = -safeDepth;
    return position;
}

} // namespace jsa::core
