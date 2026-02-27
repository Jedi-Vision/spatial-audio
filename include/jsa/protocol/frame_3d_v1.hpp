#pragma once

#include <vector>

namespace jsa::protocol {

struct Object3DV1 {
    int id = -1;
    int label = -1;
    double x = 0.0;
    double y = 0.0;
    double z = -1.0;
};

struct Frame3DV1 {
    int frame_number = 0;
    double timestamp_ms = 0.0;
    std::vector<Object3DV1> objects;
};

} // namespace jsa::protocol
