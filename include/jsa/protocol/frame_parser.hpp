#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <jsa/protocol/frame_v1.hpp>

namespace jsa::protocol {

bool parseSocketObjectRep(const uint8_t* data,
                          size_t len,
                          SocketFrameData& out,
                          std::string& err);

} // namespace jsa::protocol

using SocketObjectData = jsa::protocol::SocketObjectData;
using SocketFrameData = jsa::protocol::SocketFrameData;
using jsa::protocol::parseSocketObjectRep;
