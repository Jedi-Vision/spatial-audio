#include <jsa/protocol/frame_serializer.hpp>

#include <cstdint>
#include <cstring>
#include <limits>

namespace {

template <typename T>
void appendPod(std::vector<uint8_t>& out, const T& value) {
    const size_t oldSize = out.size();
    out.resize(oldSize + sizeof(T));
    std::memcpy(out.data() + oldSize, &value, sizeof(T));
}

} // namespace

namespace jsa::protocol {

bool serializeFrame2DV1(const Frame2DV1& frame, std::vector<uint8_t>& out, std::string& err) {
    err.clear();
    out.clear();

    if (frame.objects.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        err = "Object count exceeds int32 payload limit.";
        return false;
    }

    const size_t perObjectBytes =
        1 + sizeof(int32_t) + sizeof(int32_t) + 3 * sizeof(double);
    out.reserve(sizeof(int32_t) + sizeof(double) + 1 + sizeof(int32_t) +
                frame.objects.size() * perObjectBytes + 1);

    appendPod<int32_t>(out, static_cast<int32_t>(frame.frame_number));
    appendPod<double>(out, frame.timestamp_ms);
    out.push_back(static_cast<uint8_t>('^'));
    appendPod<int32_t>(out, static_cast<int32_t>(frame.objects.size()));

    for (const Object2DV1& object : frame.objects) {
        out.push_back(static_cast<uint8_t>('|'));
        appendPod<int32_t>(out, static_cast<int32_t>(object.id));
        appendPod<int32_t>(out, static_cast<int32_t>(object.label));
        appendPod<double>(out, object.x_2d);
        appendPod<double>(out, object.y_2d);
        appendPod<double>(out, object.depth);
    }

    out.push_back(static_cast<uint8_t>('^'));
    return true;
}

bool serializeFrame3DV1(const Frame3DV1& frame, std::vector<uint8_t>& out, std::string& err) {
    err.clear();
    out.clear();

    if (frame.objects.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        err = "Object count exceeds int32 payload limit.";
        return false;
    }

    const size_t perObjectBytes =
        1 + sizeof(int32_t) + sizeof(int32_t) + 3 * sizeof(double);
    out.reserve(sizeof(int32_t) + sizeof(double) + 1 + sizeof(int32_t) +
                frame.objects.size() * perObjectBytes + 1);

    appendPod<int32_t>(out, static_cast<int32_t>(frame.frame_number));
    appendPod<double>(out, frame.timestamp_ms);
    out.push_back(static_cast<uint8_t>('^'));
    appendPod<int32_t>(out, static_cast<int32_t>(frame.objects.size()));

    for (const Object3DV1& object : frame.objects) {
        out.push_back(static_cast<uint8_t>('|'));
        appendPod<int32_t>(out, static_cast<int32_t>(object.id));
        appendPod<int32_t>(out, static_cast<int32_t>(object.label));
        appendPod<double>(out, object.x);
        appendPod<double>(out, object.y);
        appendPod<double>(out, object.z);
    }

    out.push_back(static_cast<uint8_t>('^'));
    return true;
}

} // namespace jsa::protocol
