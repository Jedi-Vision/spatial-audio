cmake_minimum_required(VERSION 4.1)

project(spatial-audio-demo)
include(CTest)

# Export compile commands for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(fmt CONFIG REQUIRED)
find_package(flatbuffers CONFIG REQUIRED)
find_package(pffft CONFIG REQUIRED)
find_package(mysofa CONFIG REQUIRED)
find_package(steam-audio CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(raylib CONFIG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED IMPORTED_TARGET libzmq)
pkg_check_modules(GLFW3 REQUIRED IMPORTED_TARGET glfw3)
pkg_check_modules(PORTAUDIO REQUIRED IMPORTED_TARGET portaudio-2.0)

add_executable(spatial-audio-demo spatial-audio-demo.cpp)

target_link_libraries(spatial-audio-demo PRIVATE fmt::fmt unofficial::steam-audio::phonon)

add_executable(object-spatial-audio object-spatial-audio.cpp)

target_link_libraries(object-spatial-audio PRIVATE 
    fmt::fmt 
    unofficial::steam-audio::phonon
    nlohmann_json::nlohmann_json
)

add_executable(
    socket-spatial-audio-live
    socket-spatial-audio-live.cpp
    socket_payload_parser.cpp
)

target_link_libraries(
    socket-spatial-audio-live
    PRIVATE
    fmt::fmt
    unofficial::steam-audio::phonon
    PkgConfig::ZMQ
    PkgConfig::PORTAUDIO
)

add_executable(
    spatial_audio_live_new
    spatial_audio_live_new.cpp
)

target_link_libraries(
    spatial_audio_live_new
    PRIVATE
    fmt::fmt
    unofficial::steam-audio::phonon
    PkgConfig::ZMQ
    PkgConfig::PORTAUDIO
)

add_executable(
    socket-orbit-stream-3d
    socket_orbit_stream_3d.cpp
)

target_link_libraries(
    socket-orbit-stream-3d
    PRIVATE
    PkgConfig::ZMQ
)

add_executable(
    render-visual-stream
    render_visual_stream.cpp
)

target_link_libraries(
    render-visual-stream
    PRIVATE
    raylib
    PkgConfig::GLFW3
    PkgConfig::ZMQ
)

if(BUILD_TESTING)
    function(add_help_smoke_test target_name)
        add_test(
            NAME help-${target_name}
            COMMAND $<TARGET_FILE:${target_name}> --help
        )
    endfunction()

    add_help_smoke_test(spatial-audio-demo)
    add_help_smoke_test(object-spatial-audio)
    add_help_smoke_test(socket-spatial-audio-live)
    add_help_smoke_test(spatial_audio_live_new)
    add_help_smoke_test(socket-orbit-stream-3d)
    add_help_smoke_test(render-visual-stream)
endif()
