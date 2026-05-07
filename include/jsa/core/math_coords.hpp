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

Vec3 socket3DToHeadSpace(const Vec3& socketPosition);

Vec3 widenAudioAzimuthDirection(const Vec3& position,
                                float azimuthScale,
                                float maxAzimuthDeg);

} // namespace jsa::core
