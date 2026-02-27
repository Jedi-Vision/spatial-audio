#include <jsa/tracking/tracker_2d.hpp>

#include <jsa/core/math_coords.hpp>

#include <algorithm>
#include <cmath>

namespace {

float clampNormalized(double value) {
    if (!std::isfinite(value)) {
        return 0.5f;
    }
    const float asFloat = static_cast<float>(value);
    return std::clamp(asFloat, 0.0f, 1.0f);
}

} // namespace

namespace jsa::tracking {

Tracker2D::Tracker2D(int sampleRate,
                     float cameraFovHorizontalDeg,
                     float cameraFovVerticalDeg,
                     float fadeInMs,
                     float fadeOutMs,
                     size_t maxSimultaneousObjects,
                     uint64_t releaseGracePeriodUs)
    : cameraFovHorizontalDeg_(cameraFovHorizontalDeg),
      cameraFovVerticalDeg_(cameraFovVerticalDeg),
      fadeInSamples_(std::max(1.0f, (fadeInMs / 1000.0f) * static_cast<float>(sampleRate))),
      fadeOutSamples_(std::max(1.0f, (fadeOutMs / 1000.0f) * static_cast<float>(sampleRate))),
      maxSimultaneousObjects_(std::max<size_t>(1, maxSimultaneousObjects)),
      releaseGracePeriodUs_(releaseGracePeriodUs) {}

void Tracker2D::updateFromFrame(const jsa::protocol::SocketFrameData& frame,
                                uint64_t currentTimeUs,
                                float updateSamples) {
    for (auto& pair : trackedObjects_) {
        pair.second.active = false;
    }

    const float fadeInStep = std::max(0.0f, updateSamples / fadeInSamples_);
    const float fadeOutStep = std::max(0.0f, updateSamples / fadeOutSamples_);

    for (const auto& obj : frame.objects) {
        const float xNorm = clampNormalized(obj.x_2d);
        const float yNorm = clampNormalized(obj.y_2d);

        float depth = static_cast<float>(obj.depth);
        if (!std::isfinite(depth) || depth <= 0.0f) {
            depth = 0.1f;
        }

        const jsa::core::Vec3 worldPos = jsa::core::normalizedToWorld(
            xNorm, yNorm, depth, cameraFovHorizontalDeg_, cameraFovVerticalDeg_);
        const IPLVector3 position{worldPos.x, worldPos.y, worldPos.z};

        auto it = trackedObjects_.find(obj.id);
        if (it != trackedObjects_.end()) {
            TrackedObject& tracked = it->second;
            tracked.previousPosition = tracked.currentPosition;
            tracked.previousDistance = tracked.currentDistance;
            tracked.currentPosition = position;
            tracked.currentDistance = depth;
            tracked.active = true;
            tracked.lastSeenTimestampUs = currentTimeUs;
            tracked.fadeVolume = std::min(1.0f, tracked.fadeVolume + fadeInStep);
        } else {
            TrackedObject tracked;
            tracked.id = obj.id;
            tracked.currentPosition = position;
            tracked.previousPosition = position;
            tracked.currentDistance = depth;
            tracked.previousDistance = depth;
            tracked.active = true;
            tracked.fadeVolume = std::min(1.0f, fadeInStep);
            tracked.lastSeenTimestampUs = currentTimeUs;
            trackedObjects_[obj.id] = tracked;
        }
    }

    for (auto& pair : trackedObjects_) {
        TrackedObject& tracked = pair.second;
        if (!tracked.active) {
            tracked.fadeVolume = std::max(0.0f, tracked.fadeVolume - fadeOutStep);
        }
    }

    for (auto it = trackedObjects_.begin(); it != trackedObjects_.end();) {
        const TrackedObject& tracked = it->second;
        const bool stale = !tracked.active &&
                           tracked.fadeVolume <= 0.0f &&
                           currentTimeUs > tracked.lastSeenTimestampUs &&
                           (currentTimeUs - tracked.lastSeenTimestampUs) > releaseGracePeriodUs_;
        if (stale) {
            it = trackedObjects_.erase(it);
        } else {
            ++it;
        }
    }
}

std::vector<std::pair<int, IPLVector3>> Tracker2D::getInterpolatedPositions(
    float interpolationFactor) const {
    float clampedFactor = std::clamp(interpolationFactor, 0.0f, 1.0f);
    std::vector<std::pair<int, IPLVector3>> positions;

    for (const auto& pair : trackedObjects_) {
        const TrackedObject& tracked = pair.second;
        if (tracked.fadeVolume <= 0.0f) {
            continue;
        }

        IPLVector3 interpolated{};
        interpolated.x = tracked.previousPosition.x +
                         (tracked.currentPosition.x - tracked.previousPosition.x) * clampedFactor;
        interpolated.y = tracked.previousPosition.y +
                         (tracked.currentPosition.y - tracked.previousPosition.y) * clampedFactor;
        interpolated.z = tracked.previousPosition.z +
                         (tracked.currentPosition.z - tracked.previousPosition.z) * clampedFactor;
        positions.push_back({tracked.id, interpolated});
    }

    if (positions.size() > maxSimultaneousObjects_) {
        std::sort(
            positions.begin(),
            positions.end(),
            [this](const auto& a, const auto& b) { return getFadeVolume(a.first) > getFadeVolume(b.first); });
        positions.resize(maxSimultaneousObjects_);
    }

    return positions;
}

float Tracker2D::getFadeVolume(int objectId) const {
    const auto it = trackedObjects_.find(objectId);
    if (it == trackedObjects_.end()) {
        return 0.0f;
    }
    return it->second.fadeVolume;
}

} // namespace jsa::tracking
