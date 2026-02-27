#pragma once

#include <vector>

namespace jsa::protocol {

struct Object2DV1 {
    int id = -1;
    int label = -1;
    double x_2d = 0.5;
    double y_2d = 0.5;
    double depth = 1.0;
};

struct Frame2DV1 {
    int frame_number = 0;
    double timestamp_ms = 0.0;
    std::vector<Object2DV1> objects;
};

} // namespace jsa::protocol
