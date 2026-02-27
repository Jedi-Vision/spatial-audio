#pragma once

#include <phonon.h>

#include <jsa/protocol/frame_parser.hpp>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace jsa::tracking {

class Tracker2D {
public:
    Tracker2D(int sampleRate,
              float cameraFovHorizontalDeg,
              float cameraFovVerticalDeg,
              float fadeInMs,
              float fadeOutMs,
              size_t maxSimultaneousObjects,
              uint64_t releaseGracePeriodUs = 2000000ULL);

    void updateFromFrame(const jsa::protocol::SocketFrameData& frame,
                         uint64_t currentTimeUs,
                         float updateSamples);

    std::vector<std::pair<int, IPLVector3>> getInterpolatedPositions(float interpolationFactor) const;
    float getFadeVolume(int objectId) const;

private:
    struct TrackedObject {
        int id = -1;
        IPLVector3 currentPosition{0.0f, 0.0f, -1.0f};
        IPLVector3 previousPosition{0.0f, 0.0f, -1.0f};
        float currentDistance = 1.0f;
        float previousDistance = 1.0f;
        float fadeVolume = 0.0f;
        bool active = false;
        uint64_t lastSeenTimestampUs = 0;
    };

    std::unordered_map<int, TrackedObject> trackedObjects_;
    float cameraFovHorizontalDeg_ = 60.0f;
    float cameraFovVerticalDeg_ = 45.0f;
    float fadeInSamples_ = 1.0f;
    float fadeOutSamples_ = 1.0f;
    size_t maxSimultaneousObjects_ = 4;
    uint64_t releaseGracePeriodUs_ = 2000000ULL;
};

} // namespace jsa::tracking
