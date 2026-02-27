#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <jsa/protocol/frame_2d_v1.hpp>
#include <jsa/protocol/frame_3d_v1.hpp>

namespace jsa::protocol {

bool parseFrame2DV1(const uint8_t* data, size_t len, Frame2DV1& out, std::string& err);
bool parseFrame3DV1(const uint8_t* data, size_t len, Frame3DV1& out, std::string& err);

// Compatibility aliases for the pre-refactor parser API.
using SocketObjectData = Object2DV1;
using SocketFrameData = Frame2DV1;
using SocketObject3D = Object3DV1;
using SocketFrame3D = Frame3DV1;

inline bool parseSocketObjectRep(const uint8_t* data,
                                 size_t len,
                                 SocketFrameData& out,
                                 std::string& err) {
    return parseFrame2DV1(data, len, out, err);
}

inline bool parseSocketObjectRep3D(const uint8_t* data,
                                   size_t len,
                                   SocketFrame3D& out,
                                   std::string& err) {
    return parseFrame3DV1(data, len, out, err);
}

} // namespace jsa::protocol

using SocketObjectData = jsa::protocol::SocketObjectData;
using SocketFrameData = jsa::protocol::SocketFrameData;
using SocketObject3D = jsa::protocol::SocketObject3D;
using SocketFrame3D = jsa::protocol::SocketFrame3D;
using jsa::protocol::parseSocketObjectRep;
using jsa::protocol::parseSocketObjectRep3D;
