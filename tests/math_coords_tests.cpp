#include <jsa/core/math_coords.hpp>

#include <cmath>
#include <iostream>
#include <string>

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kDefaultAudioAzimuthScale = 180.0f / 73.0f;

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

bool nearlyEqual(float a, float b, float tolerance = 1.0e-4f) {
    return std::abs(a - b) <= tolerance;
}

jsa::core::Vec3 positionForAzimuth(float azimuthDeg, float distance) {
    const float azimuthRad = azimuthDeg * kPi / 180.0f;
    return {
        std::sin(azimuthRad) * distance,
        0.0f,
        -std::cos(azimuthRad) * distance,
    };
}

float azimuthDegForDirection(const jsa::core::Vec3& direction) {
    return std::atan2(direction.x, -direction.z) * 180.0f / kPi;
}

float length(const jsa::core::Vec3& value) {
    return std::sqrt(value.x * value.x + value.y * value.y + value.z * value.z);
}

bool testCenterStaysCentered() {
    const jsa::core::Vec3 direction =
        jsa::core::widenAudioAzimuthDirection({0.0f, 0.0f, -4.0f},
                                              kDefaultAudioAzimuthScale,
                                              90.0f);
    return expect(nearlyEqual(direction.x, 0.0f), "center x stays centered") &&
           expect(nearlyEqual(direction.y, 0.0f), "center y stays centered") &&
           expect(nearlyEqual(direction.z, -1.0f), "center z stays forward");
}

bool testRightEdgeWidensToNinetyDegrees() {
    const jsa::core::Vec3 position = positionForAzimuth(36.5f, 3.0f);
    const jsa::core::Vec3 direction =
        jsa::core::widenAudioAzimuthDirection(position, kDefaultAudioAzimuthScale, 90.0f);
    return expect(nearlyEqual(azimuthDegForDirection(direction), 90.0f, 0.2f),
                  "right camera edge widens near +90 degrees");
}

bool testLeftEdgeWidensSymmetrically() {
    const jsa::core::Vec3 position = positionForAzimuth(-36.5f, 3.0f);
    const jsa::core::Vec3 direction =
        jsa::core::widenAudioAzimuthDirection(position, kDefaultAudioAzimuthScale, 90.0f);
    return expect(nearlyEqual(azimuthDegForDirection(direction), -90.0f, 0.2f),
                  "left camera edge widens near -90 degrees");
}

bool testDistanceDoesNotMatter() {
    const jsa::core::Vec3 nearDirection =
        jsa::core::widenAudioAzimuthDirection(positionForAzimuth(20.0f, 1.0f),
                                              kDefaultAudioAzimuthScale,
                                              90.0f);
    const jsa::core::Vec3 farDirection =
        jsa::core::widenAudioAzimuthDirection(positionForAzimuth(20.0f, 10.0f),
                                              kDefaultAudioAzimuthScale,
                                              90.0f);
    return expect(nearlyEqual(nearDirection.x, farDirection.x), "distance independent x") &&
           expect(nearlyEqual(nearDirection.y, farDirection.y), "distance independent y") &&
           expect(nearlyEqual(nearDirection.z, farDirection.z), "distance independent z") &&
           expect(nearlyEqual(length(farDirection), 1.0f), "widened direction is normalized");
}

bool testNearZeroPositionReturnsForward() {
    const jsa::core::Vec3 direction =
        jsa::core::widenAudioAzimuthDirection({0.0f, 0.0f, 0.0f},
                                              kDefaultAudioAzimuthScale,
                                              90.0f);
    return expect(nearlyEqual(direction.x, 0.0f), "zero x returns forward") &&
           expect(nearlyEqual(direction.y, 0.0f), "zero y returns forward") &&
           expect(nearlyEqual(direction.z, -1.0f), "zero z returns forward");
}

bool testSocket3DRightStaysRight() {
    const jsa::core::Vec3 position = jsa::core::socket3DToHeadSpace({1.0f, 0.0f, 0.0f});
    return expect(nearlyEqual(position.x, 1.0f), "socket +x stays head +x") &&
           expect(nearlyEqual(position.y, 0.0f), "socket +x keeps y level") &&
           expect(nearlyEqual(position.z, 0.0f), "socket +x keeps z centered");
}

bool testSocket3DDownBecomesBelow() {
    const jsa::core::Vec3 position = jsa::core::socket3DToHeadSpace({0.0f, 1.0f, 0.0f});
    return expect(nearlyEqual(position.x, 0.0f), "socket +y keeps x centered") &&
           expect(nearlyEqual(position.y, -1.0f), "socket +y becomes head -y") &&
           expect(nearlyEqual(position.z, 0.0f), "socket +y keeps z centered");
}

bool testSocket3DForwardBecomesNegativeZ() {
    const jsa::core::Vec3 position = jsa::core::socket3DToHeadSpace({0.0f, 0.0f, 1.0f});
    return expect(nearlyEqual(position.x, 0.0f), "socket +z keeps x centered") &&
           expect(nearlyEqual(position.y, 0.0f), "socket +z keeps y level") &&
           expect(nearlyEqual(position.z, -1.0f), "socket +z becomes head -z");
}

} // namespace

int main() {
    bool ok = true;
    ok = testCenterStaysCentered() && ok;
    ok = testRightEdgeWidensToNinetyDegrees() && ok;
    ok = testLeftEdgeWidensSymmetrically() && ok;
    ok = testDistanceDoesNotMatter() && ok;
    ok = testNearZeroPositionReturnsForward() && ok;
    ok = testSocket3DRightStaysRight() && ok;
    ok = testSocket3DDownBecomesBelow() && ok;
    ok = testSocket3DForwardBecomesNegativeZ() && ok;
    return ok ? 0 : 1;
}
