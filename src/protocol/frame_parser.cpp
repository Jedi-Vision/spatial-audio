#include <jsa/protocol/frame_parser.hpp>

#include <cstring>

namespace {

bool readInt32(const uint8_t* data, size_t len, size_t& offset, int32_t& outValue) {
    if (offset + sizeof(int32_t) > len) {
        return false;
    }

    std::memcpy(&outValue, data + offset, sizeof(int32_t));
    offset += sizeof(int32_t);
    return true;
}

bool readDouble(const uint8_t* data, size_t len, size_t& offset, double& outValue) {
    if (offset + sizeof(double) > len) {
        return false;
    }

    std::memcpy(&outValue, data + offset, sizeof(double));
    offset += sizeof(double);
    return true;
}

bool readChar(const uint8_t* data, size_t len, size_t& offset, char expected) {
    if (offset >= len) {
        return false;
    }

    if (static_cast<char>(data[offset]) != expected) {
        return false;
    }

    ++offset;
    return true;
}

} // namespace

namespace jsa::protocol {

bool parseFrame2DV1(const uint8_t* data, size_t len, Frame2DV1& out, std::string& err) {
    out = {};
    err.clear();

    if (data == nullptr) {
        err = "Input data pointer is null.";
        return false;
    }

    if (len < sizeof(int32_t) + sizeof(double) + 1 + sizeof(int32_t)) {
        err = "Payload too short.";
        return false;
    }

    size_t offset = 0;
    int32_t frameNumber = 0;
    double timestampMs = 0.0;

    if (!readInt32(data, len, offset, frameNumber)) {
        err = "Failed to parse frame_number.";
        return false;
    }

    if (!readDouble(data, len, offset, timestampMs)) {
        err = "Failed to parse timestamp_ms.";
        return false;
    }

    if (!readChar(data, len, offset, '^')) {
        err = "Missing list start marker '^'.";
        return false;
    }

    int32_t objectCount = 0;
    if (!readInt32(data, len, offset, objectCount)) {
        err = "Failed to parse object count.";
        return false;
    }

    if (objectCount < 0) {
        err = "Object count is negative.";
        return false;
    }

    constexpr int32_t kMaxObjects = 10000;
    if (objectCount > kMaxObjects) {
        err = "Object count exceeds safety limit.";
        return false;
    }

    out.frame_number = frameNumber;
    out.timestamp_ms = timestampMs;
    out.objects.clear();
    out.objects.reserve(static_cast<size_t>(objectCount));

    for (int32_t i = 0; i < objectCount; ++i) {
        if (!readChar(data, len, offset, '|')) {
            err = "Missing object delimiter '|'.";
            return false;
        }

        Object2DV1 object;
        int32_t id = -1;
        int32_t label = -1;
        double x = 0.5;
        double y = 0.5;
        double depth = 1.0;

        if (!readInt32(data, len, offset, id)) {
            err = "Failed to parse object id.";
            return false;
        }
        if (!readInt32(data, len, offset, label)) {
            err = "Failed to parse object label.";
            return false;
        }
        if (!readDouble(data, len, offset, x)) {
            err = "Failed to parse object x_2d.";
            return false;
        }
        if (!readDouble(data, len, offset, y)) {
            err = "Failed to parse object y_2d.";
            return false;
        }
        if (!readDouble(data, len, offset, depth)) {
            err = "Failed to parse object depth.";
            return false;
        }

        object.id = id;
        object.label = label;
        object.x_2d = x;
        object.y_2d = y;
        object.depth = depth;
        out.objects.push_back(object);
    }

    if (offset < len && static_cast<char>(data[offset]) == '^') {
        ++offset;
    }

    if (offset != len) {
        err = "Trailing bytes after payload parse.";
        return false;
    }

    return true;
}

bool parseFrame3DV1(const uint8_t* data, size_t len, Frame3DV1& out, std::string& err) {
    out = {};
    err.clear();

    if (data == nullptr) {
        err = "Input data pointer is null.";
        return false;
    }

    if (len < sizeof(int32_t) + sizeof(double) + 1 + sizeof(int32_t)) {
        err = "Payload too short.";
        return false;
    }

    size_t offset = 0;
    int32_t frameNumber = 0;
    double timestampMs = 0.0;

    if (!readInt32(data, len, offset, frameNumber)) {
        err = "Failed to parse frame_number.";
        return false;
    }

    if (!readDouble(data, len, offset, timestampMs)) {
        err = "Failed to parse timestamp_ms.";
        return false;
    }

    if (!readChar(data, len, offset, '^')) {
        err = "Missing list start marker '^'.";
        return false;
    }

    int32_t objectCount = 0;
    if (!readInt32(data, len, offset, objectCount)) {
        err = "Failed to parse object count.";
        return false;
    }

    if (objectCount < 0) {
        err = "Object count is negative.";
        return false;
    }

    constexpr int32_t kMaxObjectsSafetyLimit = 100000;
    if (objectCount > kMaxObjectsSafetyLimit) {
        err = "Object count exceeds parser safety limit.";
        return false;
    }

    out.frame_number = frameNumber;
    out.timestamp_ms = timestampMs;
    out.objects.clear();
    out.objects.reserve(static_cast<size_t>(objectCount));

    for (int32_t i = 0; i < objectCount; ++i) {
        if (!readChar(data, len, offset, '|')) {
            err = "Missing object delimiter '|'.";
            return false;
        }

        Object3DV1 object;
        int32_t id = -1;
        int32_t label = -1;
        double x = 0.0;
        double y = 0.0;
        double z = -1.0;

        if (!readInt32(data, len, offset, id)) {
            err = "Failed to parse object id.";
            return false;
        }
        if (!readInt32(data, len, offset, label)) {
            err = "Failed to parse object label.";
            return false;
        }
        if (!readDouble(data, len, offset, x)) {
            err = "Failed to parse object x.";
            return false;
        }
        if (!readDouble(data, len, offset, y)) {
            err = "Failed to parse object y.";
            return false;
        }
        if (!readDouble(data, len, offset, z)) {
            err = "Failed to parse object z.";
            return false;
        }

        object.id = id;
        object.label = label;
        object.x = x;
        object.y = y;
        object.z = z;
        out.objects.push_back(object);
    }

    if (offset < len && static_cast<char>(data[offset]) == '^') {
        ++offset;
    }

    if (offset != len) {
        err = "Trailing bytes after payload parse.";
        return false;
    }

    return true;
}

} // namespace jsa::protocol
