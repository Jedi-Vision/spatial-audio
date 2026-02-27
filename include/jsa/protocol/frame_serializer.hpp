#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <jsa/protocol/frame_2d_v1.hpp>
#include <jsa/protocol/frame_3d_v1.hpp>

namespace jsa::protocol {

bool serializeFrame2DV1(const Frame2DV1& frame, std::vector<uint8_t>& out, std::string& err);
bool serializeFrame3DV1(const Frame3DV1& frame, std::vector<uint8_t>& out, std::string& err);

} // namespace jsa::protocol
