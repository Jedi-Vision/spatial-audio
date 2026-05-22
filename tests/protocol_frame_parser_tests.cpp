#include <jsa/protocol/frame_parser.hpp>
#include <jsa/protocol/frame_json.hpp>
#include <jsa/protocol/frame_serializer.hpp>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

bool testRoundTrip3D() {
    jsa::protocol::Frame3DV1 input{};
    input.frame_number = 42;
    input.timestamp_ms = 1337.5;
    jsa::protocol::Object3DV1 first{};
    first.id = 7;
    first.label = 3;
    first.x = 1.0;
    first.y = -2.0;
    first.z = -4.0;
    input.objects.push_back(first);

    jsa::protocol::Object3DV1 second{};
    second.id = 9;
    second.label = 8;
    second.x = -0.5;
    second.y = 0.25;
    second.z = -1.5;
    input.objects.push_back(second);

    std::vector<uint8_t> payload;
    std::string err;
    if (!expect(jsa::protocol::serializeFrame3DV1(input, payload, err), "serialize 3D frame")) {
        return false;
    }

    jsa::protocol::Frame3DV1 parsed{};
    if (!expect(jsa::protocol::parseFrame3DV1(payload.data(), payload.size(), parsed, err),
                "parse serialized 3D frame")) {
        return false;
    }

    if (!expect(parsed.frame_number == input.frame_number, "frame number matches")) {
        return false;
    }
    if (!expect(parsed.objects.size() == input.objects.size(), "object count matches")) {
        return false;
    }
    if (!expect(parsed.objects[0].id == 7 && parsed.objects[1].label == 8,
                "object fields round-trip")) {
        return false;
    }

    return true;
}

bool testZeroObject3DFrame() {
    jsa::protocol::Frame3DV1 input{};
    input.frame_number = 77;
    input.timestamp_ms = 2500.0;

    std::vector<uint8_t> payload;
    std::string err;
    if (!expect(jsa::protocol::serializeFrame3DV1(input, payload, err),
                "serialize zero-object 3D frame")) {
        return false;
    }

    jsa::protocol::Frame3DV1 parsed{};
    if (!expect(jsa::protocol::parseFrame3DV1(payload.data(), payload.size(), parsed, err),
                "parse zero-object 3D frame")) {
        return false;
    }

    return expect(parsed.frame_number == input.frame_number, "zero-object frame number matches") &&
           expect(parsed.timestamp_ms == input.timestamp_ms, "zero-object timestamp matches") &&
           expect(parsed.objects.empty(), "zero-object frame has no objects");
}

bool testRoundTrip2D() {
    jsa::protocol::Frame2DV1 input{};
    input.frame_number = 11;
    input.timestamp_ms = 99.0;
    jsa::protocol::Object2DV1 object{};
    object.id = 2;
    object.label = 4;
    object.x_2d = 0.25;
    object.y_2d = 0.75;
    object.depth = 2.0;
    input.objects.push_back(object);

    std::vector<uint8_t> payload;
    std::string err;
    if (!expect(jsa::protocol::serializeFrame2DV1(input, payload, err), "serialize 2D frame")) {
        return false;
    }

    jsa::protocol::Frame2DV1 parsed{};
    if (!expect(jsa::protocol::parseFrame2DV1(payload.data(), payload.size(), parsed, err),
                "parse serialized 2D frame")) {
        return false;
    }

    if (!expect(parsed.frame_number == input.frame_number, "2D frame number matches")) {
        return false;
    }
    if (!expect(parsed.objects.size() == 1, "2D object count matches")) {
        return false;
    }
    if (!expect(parsed.objects[0].depth == 2.0, "2D depth preserved")) {
        return false;
    }

    return true;
}

bool testMalformedMarker3D() {
    jsa::protocol::Frame3DV1 input{};
    input.objects.push_back(jsa::protocol::Object3DV1{});

    std::vector<uint8_t> payload;
    std::string err;
    if (!jsa::protocol::serializeFrame3DV1(input, payload, err)) {
        return false;
    }

    payload[sizeof(int32_t) + sizeof(double)] = static_cast<uint8_t>('!');
    jsa::protocol::Frame3DV1 parsed{};
    const bool ok = jsa::protocol::parseFrame3DV1(payload.data(), payload.size(), parsed, err);
    return expect(!ok, "3D parser rejects malformed list marker");
}

bool testInvalidObjectCount2D() {
    std::vector<uint8_t> payload;
    payload.resize(sizeof(int32_t) + sizeof(double) + 1 + sizeof(int32_t));
    size_t offset = 0;

    const int32_t frameNumber = 1;
    const double timestamp = 1.0;
    const char marker = '^';
    const int32_t objectCount = -1;

    std::memcpy(payload.data() + offset, &frameNumber, sizeof(frameNumber));
    offset += sizeof(frameNumber);
    std::memcpy(payload.data() + offset, &timestamp, sizeof(timestamp));
    offset += sizeof(timestamp);
    std::memcpy(payload.data() + offset, &marker, sizeof(marker));
    offset += sizeof(marker);
    std::memcpy(payload.data() + offset, &objectCount, sizeof(objectCount));

    std::string err;
    jsa::protocol::Frame2DV1 parsed{};
    const bool ok = jsa::protocol::parseFrame2DV1(payload.data(), payload.size(), parsed, err);
    return expect(!ok, "2D parser rejects negative object count");
}

bool testTrailingBytes3D() {
    jsa::protocol::Frame3DV1 input{};
    std::vector<uint8_t> payload;
    std::string err;
    if (!jsa::protocol::serializeFrame3DV1(input, payload, err)) {
        return false;
    }

    payload.push_back(static_cast<uint8_t>('x'));
    jsa::protocol::Frame3DV1 parsed{};
    const bool ok = jsa::protocol::parseFrame3DV1(payload.data(), payload.size(), parsed, err);
    return expect(!ok, "3D parser rejects trailing bytes");
}

bool testShortPayload2D() {
    const std::vector<uint8_t> payload{1, 2, 3};
    std::string err;
    jsa::protocol::Frame2DV1 parsed{};
    const bool ok = jsa::protocol::parseFrame2DV1(payload.data(), payload.size(), parsed, err);
    return expect(!ok, "2D parser rejects short payload");
}

bool testValidStrict3DJsonLine() {
    const std::string line =
        R"({"frame_number":0,"timestamp_ms":0.0,"objects":[{"id":1,"label":0,"x":0.0,"y":0.5,"z":-1.0}]})";
    std::string err;
    jsa::protocol::Frame3DV1 parsed{};
    if (!expect(jsa::protocol::parseFrame3DJsonLine(line, parsed, err),
                "strict 3D JSON line parses")) {
        return false;
    }

    return expect(parsed.frame_number == 0, "JSON frame number parsed") &&
           expect(parsed.timestamp_ms == 0.0, "JSON timestamp parsed") &&
           expect(parsed.objects.size() == 1, "JSON object count parsed") &&
           expect(parsed.objects[0].id == 1, "JSON object id parsed") &&
           expect(parsed.objects[0].label == 0, "JSON object label parsed") &&
           expect(parsed.objects[0].x == 0.0, "JSON object x parsed") &&
           expect(parsed.objects[0].y == 0.5, "JSON object y parsed") &&
           expect(parsed.objects[0].z == -1.0, "JSON object z parsed");
}

bool testJsonLineMissingTimestampFails() {
    const std::string line = R"({"frame_number":0,"objects":[]})";
    std::string err;
    jsa::protocol::Frame3DV1 parsed{};
    const bool ok = jsa::protocol::parseFrame3DJsonLine(line, parsed, err);
    return expect(!ok, "JSON parser rejects missing timestamp_ms");
}

bool testJsonLineMissingObjectLabelFails() {
    const std::string line =
        R"({"frame_number":0,"timestamp_ms":0.0,"objects":[{"id":1,"x":0.0,"y":0.0,"z":-1.0}]})";
    std::string err;
    jsa::protocol::Frame3DV1 parsed{};
    const bool ok = jsa::protocol::parseFrame3DJsonLine(line, parsed, err);
    return expect(!ok, "JSON parser rejects missing object label");
}

bool testJsonLineMissingObjectZFails() {
    const std::string line =
        R"({"frame_number":0,"timestamp_ms":0.0,"objects":[{"id":1,"label":0,"x":0.0,"y":0.0}]})";
    std::string err;
    jsa::protocol::Frame3DV1 parsed{};
    const bool ok = jsa::protocol::parseFrame3DJsonLine(line, parsed, err);
    return expect(!ok, "JSON parser rejects missing object z");
}

bool testMalformedJsonLineFails() {
    const std::string line =
        R"({"frame_number":0,"timestamp_ms":0.0,"objects":[{"id":1,"label":0,"x":0.0,"y":0.0,"z":-1.0})";
    std::string err;
    jsa::protocol::Frame3DV1 parsed{};
    const bool ok = jsa::protocol::parseFrame3DJsonLine(line, parsed, err);
    return expect(!ok, "JSON parser rejects malformed JSON");
}

bool testJsonLineNonArrayObjectsFails() {
    const std::string line = R"({"frame_number":0,"timestamp_ms":0.0,"objects":{}})";
    std::string err;
    jsa::protocol::Frame3DV1 parsed{};
    const bool ok = jsa::protocol::parseFrame3DJsonLine(line, parsed, err);
    return expect(!ok, "JSON parser rejects non-array objects");
}

} // namespace

int main() {
    bool ok = true;
    ok = testRoundTrip3D() && ok;
    ok = testZeroObject3DFrame() && ok;
    ok = testRoundTrip2D() && ok;
    ok = testMalformedMarker3D() && ok;
    ok = testInvalidObjectCount2D() && ok;
    ok = testTrailingBytes3D() && ok;
    ok = testShortPayload2D() && ok;
    ok = testValidStrict3DJsonLine() && ok;
    ok = testJsonLineMissingTimestampFails() && ok;
    ok = testJsonLineMissingObjectLabelFails() && ok;
    ok = testJsonLineMissingObjectZFails() && ok;
    ok = testMalformedJsonLineFails() && ok;
    ok = testJsonLineNonArrayObjectsFails() && ok;
    return ok ? 0 : 1;
}
