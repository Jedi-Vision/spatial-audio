#include <jsa/protocol/frame_json.hpp>

#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <utility>

#include <nlohmann/json.hpp>

namespace {

using Json = nlohmann::json;

bool requireObject(const Json& value, std::string& err) {
    if (!value.is_object()) {
        err = "Frame JSON must be an object.";
        return false;
    }
    return true;
}

bool requireInt(const Json& value, const char* key, int& out, std::string& err) {
    const auto it = value.find(key);
    if (it == value.end()) {
        err = std::string("Missing required field: ") + key + ".";
        return false;
    }
    if (!it->is_number_integer() && !it->is_number_unsigned()) {
        err = std::string("Field must be an integer: ") + key + ".";
        return false;
    }
    if (it->is_number_unsigned()) {
        const auto valueUnsigned = it->get<uint64_t>();
        if (valueUnsigned > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            err = std::string("Integer field is out of range: ") + key + ".";
            return false;
        }
        out = static_cast<int>(valueUnsigned);
        return true;
    }

    const auto valueSigned = it->get<int64_t>();
    if (valueSigned < static_cast<int64_t>(std::numeric_limits<int>::min()) ||
        valueSigned > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        err = std::string("Integer field is out of range: ") + key + ".";
        return false;
    }
    out = static_cast<int>(valueSigned);
    return true;
}

bool requireFiniteDouble(const Json& value, const char* key, double& out, std::string& err) {
    const auto it = value.find(key);
    if (it == value.end()) {
        err = std::string("Missing required field: ") + key + ".";
        return false;
    }
    if (!it->is_number()) {
        err = std::string("Field must be numeric: ") + key + ".";
        return false;
    }
    out = it->get<double>();
    if (!std::isfinite(out)) {
        err = std::string("Field must be finite: ") + key + ".";
        return false;
    }
    return true;
}

} // namespace

namespace jsa::protocol {

bool parseFrame3DJsonLine(std::string_view line, Frame3DV1& out, std::string& err) {
    out = {};
    err.clear();

    Json frameJson;
    try {
        frameJson = Json::parse(line.begin(), line.end());
    } catch (const std::exception& ex) {
        err = std::string("Malformed JSON: ") + ex.what();
        return false;
    }

    if (!requireObject(frameJson, err)) {
        return false;
    }

    Frame3DV1 parsed{};
    if (!requireInt(frameJson, "frame_number", parsed.frame_number, err)) {
        return false;
    }
    if (!requireFiniteDouble(frameJson, "timestamp_ms", parsed.timestamp_ms, err)) {
        return false;
    }

    const auto objectsIt = frameJson.find("objects");
    if (objectsIt == frameJson.end()) {
        err = "Missing required field: objects.";
        return false;
    }
    if (!objectsIt->is_array()) {
        err = "Field must be an array: objects.";
        return false;
    }

    parsed.objects.reserve(objectsIt->size());
    for (size_t index = 0; index < objectsIt->size(); ++index) {
        const Json& objectJson = (*objectsIt)[index];
        if (!objectJson.is_object()) {
            err = "Each object entry must be an object.";
            return false;
        }

        Object3DV1 object{};
        if (!requireInt(objectJson, "id", object.id, err) ||
            !requireInt(objectJson, "label", object.label, err) ||
            !requireFiniteDouble(objectJson, "x", object.x, err) ||
            !requireFiniteDouble(objectJson, "y", object.y, err) ||
            !requireFiniteDouble(objectJson, "z", object.z, err)) {
            return false;
        }

        parsed.objects.push_back(object);
    }

    out = std::move(parsed);
    return true;
}

} // namespace jsa::protocol
