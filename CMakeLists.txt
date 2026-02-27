cmake_minimum_required(VERSION 4.1)

project(spatial-audio-demo)
include(CTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ProjectOptions)
include(Dependencies)
include(AddJsaExecutable)

jsa_find_dependencies()

add_library(jsa_protocol STATIC socket_payload_parser.cpp)
target_include_directories(jsa_protocol PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
jsa_apply_project_options(jsa_protocol)

add_library(jsa_core INTERFACE)
add_library(jsa_tracking INTERFACE)

jsa_add_executable(
    spatial-audio-demo
    SOURCES spatial-audio-demo.cpp
    LINK_LIBS fmt::fmt unofficial::steam-audio::phonon jsa_core
)

jsa_add_executable(
    object-spatial-audio
    SOURCES object-spatial-audio.cpp
    LINK_LIBS
        fmt::fmt
        unofficial::steam-audio::phonon
        nlohmann_json::nlohmann_json
        jsa_core
        jsa_tracking
)

jsa_add_executable(
    socket-spatial-audio-live
    SOURCES socket-spatial-audio-live.cpp
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
    SOURCES spatial_audio_live_new.cpp
    LINK_LIBS
        fmt::fmt
        unofficial::steam-audio::phonon
        PkgConfig::ZMQ
        PkgConfig::PORTAUDIO
        jsa_tracking
)

jsa_add_executable(
    socket-orbit-stream-3d
    SOURCES socket_orbit_stream_3d.cpp
    LINK_LIBS PkgConfig::ZMQ
)

jsa_add_executable(
    render-visual-stream
    SOURCES render_visual_stream.cpp
    LINK_LIBS raylib PkgConfig::GLFW3 PkgConfig::ZMQ
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
