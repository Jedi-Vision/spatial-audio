cmake_minimum_required(VERSION 4.1)

project(spatial-audio-demo)
include(CTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectOptions)
include(Dependencies)
include(AddJsaExecutable)

jsa_find_dependencies()

add_library(
    jsa_protocol
    STATIC
    src/protocol/frame_parser.cpp
    src/protocol/frame_serializer.cpp
)
target_include_directories(
    jsa_protocol
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
jsa_apply_project_options(jsa_protocol)

add_library(
    jsa_core
    STATIC
    src/core/resource_locator.cpp
    src/core/wav_io.cpp
    src/core/math_coords.cpp
)
target_include_directories(
    jsa_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
jsa_apply_project_options(jsa_core)

add_library(
    jsa_tracking
    STATIC
    src/tracking/tracker_2d.cpp
    src/tracking/tracker_3d.cpp
)
target_link_libraries(jsa_tracking PUBLIC unofficial::steam-audio::phonon)
target_include_directories(
    jsa_tracking
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_BINARY_DIR}/vcpkg_installed/${VCPKG_TARGET_TRIPLET}/include"
)
jsa_apply_project_options(jsa_tracking)

jsa_add_executable(
    spatial-audio-demo
    SOURCES src/apps/demo/main.cpp
    LINK_LIBS fmt::fmt unofficial::steam-audio::phonon jsa_core
)

jsa_add_executable(
    object-spatial-audio
    SOURCES src/apps/offline_renderer/main.cpp
    LINK_LIBS
        fmt::fmt
        unofficial::steam-audio::phonon
        nlohmann_json::nlohmann_json
        jsa_core
        jsa_tracking
)

jsa_add_executable(
    socket-spatial-audio-live
    SOURCES src/apps/live_receiver_2d/main.cpp
    LINK_LIBS
        fmt::fmt
        unofficial::steam-audio::phonon
        PkgConfig::ZMQ
        PkgConfig::PORTAUDIO
        jsa_protocol
        jsa_core
        jsa_tracking
)

jsa_add_executable(
    spatial_audio_live_new
    SOURCES src/apps/live_receiver_3d/main.cpp
    LINK_LIBS
        fmt::fmt
        unofficial::steam-audio::phonon
        PkgConfig::ZMQ
        PkgConfig::PORTAUDIO
        jsa_protocol
        jsa_core
        jsa_tracking
)

jsa_add_executable(
    socket-orbit-stream-3d
    SOURCES src/apps/orbit_streamer/main.cpp
    LINK_LIBS PkgConfig::ZMQ jsa_protocol
)

jsa_add_executable(
    render-visual-stream
    SOURCES src/apps/visual_monitor/main.cpp
    LINK_LIBS raylib PkgConfig::GLFW3 PkgConfig::ZMQ jsa_protocol
)

if(BUILD_TESTING)
    add_executable(protocol_frame_parser_tests tests/protocol_frame_parser_tests.cpp)
    target_link_libraries(protocol_frame_parser_tests PRIVATE jsa_protocol)
    jsa_apply_project_options(protocol_frame_parser_tests)

    add_test(
        NAME protocol-frame-parser
        COMMAND $<TARGET_FILE:protocol_frame_parser_tests>
    )

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
