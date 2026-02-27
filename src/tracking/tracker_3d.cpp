#include <jsa/tracking/tracker_3d.hpp>

#include <algorithm>
#include <cmath>

namespace jsa::tracking {

Tracker3D::Tracker3D(int sampleRate,
                     float noFrameFadeMs,
                     float holdLastPositionMs,
                     float fadeInMs,
                     uint64_t releaseGracePeriodUs)
    : fadeInSamples_(std::max(1.0f, (fadeInMs / 1000.0f) * static_cast<float>(sampleRate))),
      fadeOutSamples_(std::max(
          1.0f,
          (std::max(1.0f, noFrameFadeMs) / 1000.0f) * static_cast<float>(sampleRate))),
      holdLastPositionUs_(static_cast<uint64_t>(
          std::llround(std::max(0.0f, holdLastPositionMs) * 1000.0f))),
      releaseGracePeriodUs_(releaseGracePeriodUs) {}

void Tracker3D::updateFromFrame(const jsa::protocol::SocketFrame3D& frame,
                                uint64_t currentTimeUs,
                                float updateSamples) {
    for (auto& entry : trackedObjects_) {
        entry.second.activeThisFrame = false;
    }

    const float safeFadeInSamples = std::max(1.0f, fadeInSamples_);
    const float safeFadeOutSamples = std::max(1.0f, fadeOutSamples_);
    const float fadeInStep = std::max(0.0f, updateSamples / safeFadeInSamples);
    const float fadeOutStep = std::max(0.0f, updateSamples / safeFadeOutSamples);

    for (const auto& object : frame.objects) {
        if (!std::isfinite(object.x) || !std::isfinite(object.y) || !std::isfinite(object.z)) {
            continue;
        }

        IPLVector3 position{
            static_cast<float>(object.x),
            static_cast<float>(object.y),
            static_cast<float>(object.z)};

        auto it = trackedObjects_.find(object.id);
        if (it == trackedObjects_.end()) {
            TrackedObjectState state;
            state.id = object.id;
            state.label = object.label;
            state.previousPosition = position;
            state.currentPosition = position;
            state.fade = std::min(1.0f, fadeInStep);
            state.activeThisFrame = true;
            state.lastSeenTimestampUs = currentTimeUs;
            trackedObjects_.emplace(object.id, state);
        } else {
            TrackedObjectState& state = it->second;
            state.label = object.label;
            state.previousPosition = state.currentPosition;
            state.currentPosition = position;
            state.fade = std::min(1.0f, state.fade + fadeInStep);
            state.activeThisFrame = true;
            state.lastSeenTimestampUs = currentTimeUs;
        }
    }

    for (auto& entry : trackedObjects_) {
        TrackedObjectState& state = entry.second;
        if (!state.activeThisFrame) {
            state.fade = std::max(0.0f, state.fade - fadeOutStep);
        }
    }
}

void Tracker3D::updateWithoutFrame(uint64_t currentTimeUs, float updateSamples) {
    for (auto& entry : trackedObjects_) {
        entry.second.activeThisFrame = false;
    }

    const float safeFadeOutSamples = std::max(1.0f, fadeOutSamples_);
    const float fadeOutStep = std::max(0.0f, updateSamples / safeFadeOutSamples);
    for (auto& entry : trackedObjects_) {
        TrackedObjectState& state = entry.second;
        const bool withinHoldWindow =
            holdLastPositionUs_ > 0 &&
            currentTimeUs >= state.lastSeenTimestampUs &&
            (currentTimeUs - state.lastSeenTimestampUs) <= holdLastPositionUs_;
        if (!withinHoldWindow) {
            state.fade = std::max(0.0f, state.fade - fadeOutStep);
        }
    }
}

std::vector<ActiveObjectSnapshot> Tracker3D::getInterpolatedActiveObjects(
    float interpolationFactor) const {
    const float clampedFactor = std::clamp(interpolationFactor, 0.0f, 1.0f);

    std::vector<ActiveObjectSnapshot> result;
    result.reserve(trackedObjects_.size());
    for (const auto& entry : trackedObjects_) {
        const TrackedObjectState& state = entry.second;
        if (state.fade <= 0.0f) {
            continue;
        }

        ActiveObjectSnapshot snapshot;
        snapshot.id = state.id;
        snapshot.label = state.label;
        snapshot.fade = state.fade;
        snapshot.position.x = state.previousPosition.x +
                              (state.currentPosition.x - state.previousPosition.x) * clampedFactor;
        snapshot.position.y = state.previousPosition.y +
                              (state.currentPosition.y - state.previousPosition.y) * clampedFactor;
        snapshot.position.z = state.previousPosition.z +
                              (state.currentPosition.z - state.previousPosition.z) * clampedFactor;
        result.push_back(snapshot);
    }

    return result;
}

std::vector<int> Tracker3D::collectReleasableObjects(uint64_t currentTimeUs) {
    std::vector<int> releasableIds;

    for (auto it = trackedObjects_.begin(); it != trackedObjects_.end();) {
        const TrackedObjectState& state = it->second;
        const bool stale = !state.activeThisFrame &&
                           state.fade <= 0.0f &&
                           currentTimeUs > state.lastSeenTimestampUs &&
                           (currentTimeUs - state.lastSeenTimestampUs) > releaseGracePeriodUs_;

        if (stale) {
            releasableIds.push_back(it->first);
            it = trackedObjects_.erase(it);
        } else {
            ++it;
        }
    }

    return releasableIds;
}

bool Tracker3D::isActiveInLatestFrame(int objectId) const {
    const auto it = trackedObjects_.find(objectId);
    if (it == trackedObjects_.end()) {
        return false;
    }
    return it->second.activeThisFrame;
}

} // namespace jsa::tracking
