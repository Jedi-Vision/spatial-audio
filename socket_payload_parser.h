#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct SocketObjectData {
    int id = -1;
    int label = -1;
    double x_2d = 0.5;
    double y_2d = 0.5;
    double depth = 1.0;
};

struct SocketFrameData {
    int frame_number = 0;
    double timestamp_ms = 0.0;
    std::vector<SocketObjectData> objects;
};

bool parseSocketObjectRep(const uint8_t* data,
                          size_t len,
                          SocketFrameData& out,
                          std::string& err);
