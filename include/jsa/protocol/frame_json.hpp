#pragma once

#include <jsa/protocol/frame_3d_v1.hpp>

#include <string>
#include <string_view>

namespace jsa::protocol {

bool parseFrame3DJsonLine(std::string_view line, Frame3DV1& out, std::string& err);

} // namespace jsa::protocol
