#include <jsa/core/math_coords.hpp>

#include <algorithm>
#include <cmath>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDegToRad = kPi / 180.0f;
constexpr float kMinDirectionLengthSq = 1.0e-12f;

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

    const float fovHRad = horizontalFovDeg * kDegToRad;
    const float fovVRad = verticalFovDeg * kDegToRad;

    Vec3 position{};
    position.x = safeDepth * std::tan(centeredX * fovHRad);
    position.y = safeDepth * std::tan(centeredY * fovVRad);
    position.z = -safeDepth;
    return position;
}

Vec3 widenAudioAzimuthDirection(const Vec3& position,
                                float azimuthScale,
                                float maxAzimuthDeg) {
    const float lengthSq =
        position.x * position.x + position.y * position.y + position.z * position.z;
    if (!std::isfinite(lengthSq) || lengthSq <= kMinDirectionLengthSq) {
        return {0.0f, 0.0f, -1.0f};
    }

    const float scale =
        (std::isfinite(azimuthScale) && azimuthScale > 0.0f) ? azimuthScale : 1.0f;
    const float maxDeg =
        (std::isfinite(maxAzimuthDeg) && maxAzimuthDeg > 0.0f)
            ? std::min(maxAzimuthDeg, 180.0f)
            : 90.0f;

    const float invLength = 1.0f / std::sqrt(lengthSq);
    const float y = std::clamp(position.y * invLength, -1.0f, 1.0f);
    const float azimuth = std::atan2(position.x, -position.z);
    const float warpedAzimuth =
        std::clamp(azimuth * scale, -maxDeg * kDegToRad, maxDeg * kDegToRad);
    const float horizontalLength = std::sqrt(std::max(0.0f, 1.0f - (y * y)));

    return {
        std::sin(warpedAzimuth) * horizontalLength,
        y,
        -std::cos(warpedAzimuth) * horizontalLength,
    };
}

} // namespace jsa::core
