#pragma once

#include <phonon.h>

#include <jsa/protocol/frame_parser.hpp>

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace jsa::tracking {

struct ActiveObjectSnapshot {
    int id = -1;
    int label = -1;
    IPLVector3 position{0.0f, 0.0f, -1.0f};
    float fade = 0.0f;
};

class Tracker3D {
public:
    Tracker3D(int sampleRate,
              float noFrameFadeMs,
              float holdLastPositionMs,
              float fadeInMs = 100.0f,
              uint64_t releaseGracePeriodUs = 2000000ULL);

    void updateFromFrame(const jsa::protocol::SocketFrame3D& frame,
                         uint64_t currentTimeUs,
                         float updateSamples);
    void updateWithoutFrame(uint64_t currentTimeUs, float updateSamples);

    std::vector<ActiveObjectSnapshot> getInterpolatedActiveObjects(float interpolationFactor) const;
    std::vector<int> collectReleasableObjects(uint64_t currentTimeUs);
    bool isActiveInLatestFrame(int objectId) const;

private:
    struct TrackedObjectState {
        int id = -1;
        int label = -1;
        IPLVector3 previousPosition{0.0f, 0.0f, -1.0f};
        IPLVector3 currentPosition{0.0f, 0.0f, -1.0f};
        float fade = 0.0f;
        bool activeThisFrame = false;
        uint64_t lastSeenTimestampUs = 0;
    };

    std::unordered_map<int, TrackedObjectState> trackedObjects_;
    float fadeInSamples_ = 1.0f;
    float fadeOutSamples_ = 1.0f;
    uint64_t holdLastPositionUs_ = 0;
    uint64_t releaseGracePeriodUs_ = 2000000ULL;
};

} // namespace jsa::tracking
