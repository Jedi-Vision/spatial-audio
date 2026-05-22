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
    src/protocol/frame_json.cpp
    src/protocol/frame_parser.cpp
    src/protocol/frame_serializer.cpp
)
target_include_directories(
    jsa_protocol
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(jsa_protocol PUBLIC nlohmann_json::nlohmann_json)
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
target_link_libraries(jsa_tracking PRIVATE jsa_core)
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
    LINK_LIBS raylib PkgConfig::GLFW3 PkgConfig::ZMQ jsa_protocol jsa_core
)

jsa_add_executable(
    object-simulator
    SOURCES src/apps/object_simulator/main.cpp
    LINK_LIBS raylib PkgConfig::GLFW3 PkgConfig::ZMQ jsa_protocol
)

function(configure_binary_names target_name new_output legacy_output)
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME "${new_output}")
    add_custom_command(
        TARGET ${target_name}
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${target_name}>
            $<TARGET_FILE_DIR:${target_name}>/${legacy_output}
    )
endfunction()

function(add_binary_alias target_name alias_name)
    add_custom_command(
        TARGET ${target_name}
        POST_BUILD
        COMMAND
            ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:${target_name}>
            $<TARGET_FILE_DIR:${target_name}>/${alias_name}
    )
endfunction()

configure_binary_names(spatial-audio-demo jsa-demo spatial-audio-demo)
configure_binary_names(object-spatial-audio jsa-offline-render object-spatial-audio)
configure_binary_names(socket-spatial-audio-live jsa-live-2d socket-spatial-audio-live)
configure_binary_names(spatial_audio_live_new jsa-live-3d spatial_audio_live_new)
configure_binary_names(socket-orbit-stream-3d jsa-orbit-stream socket-orbit-stream-3d)
configure_binary_names(render-visual-stream jsa-visual-monitor render-visual-stream)
configure_binary_names(object-simulator jsa-object-sim object-simulator)
add_binary_alias(object-simulator jsa-obj-sim)

if(BUILD_TESTING)
    add_executable(protocol_frame_parser_tests tests/protocol_frame_parser_tests.cpp)
    target_link_libraries(protocol_frame_parser_tests PRIVATE jsa_protocol)
    jsa_apply_project_options(protocol_frame_parser_tests)

    add_executable(resource_locator_tests tests/resource_locator_tests.cpp)
    target_link_libraries(resource_locator_tests PRIVATE jsa_core)
    jsa_apply_project_options(resource_locator_tests)

    add_executable(math_coords_tests tests/math_coords_tests.cpp)
    target_link_libraries(math_coords_tests PRIVATE jsa_core)
    jsa_apply_project_options(math_coords_tests)

    add_executable(wav_io_tests tests/wav_io_tests.cpp)
    target_link_libraries(wav_io_tests PRIVATE jsa_core)
    jsa_apply_project_options(wav_io_tests)

    add_test(
        NAME protocol-frame-parser
        COMMAND $<TARGET_FILE:protocol_frame_parser_tests>
    )
    add_test(
        NAME resource-locator
        COMMAND $<TARGET_FILE:resource_locator_tests>
    )
    add_test(
        NAME math-coords
        COMMAND $<TARGET_FILE:math_coords_tests>
    )
    add_test(
        NAME wav-io
        COMMAND $<TARGET_FILE:wav_io_tests>
    )

    function(add_help_smoke_test target_name)
        add_test(
            NAME help-${target_name}
            COMMAND $<TARGET_FILE:${target_name}> --help
        )
    endfunction()

    function(add_help_smoke_alias_test target_name alias_name)
        add_test(
            NAME help-legacy-${alias_name}
            COMMAND $<TARGET_FILE_DIR:${target_name}>/${alias_name} --help
        )
    endfunction()

    add_help_smoke_test(spatial-audio-demo)
    add_help_smoke_test(object-spatial-audio)
    add_help_smoke_test(socket-spatial-audio-live)
    add_help_smoke_test(spatial_audio_live_new)
    set_tests_properties(
        help-spatial_audio_live_new
        PROPERTIES PASS_REGULAR_EXPRESSION "--input-file <jsonl>"
    )
    add_help_smoke_test(socket-orbit-stream-3d)
    add_help_smoke_test(render-visual-stream)
    add_help_smoke_test(object-simulator)
    add_help_smoke_alias_test(spatial-audio-demo spatial-audio-demo)
    add_help_smoke_alias_test(object-spatial-audio object-spatial-audio)
    add_help_smoke_alias_test(socket-spatial-audio-live socket-spatial-audio-live)
    add_help_smoke_alias_test(spatial_audio_live_new spatial_audio_live_new)
    add_help_smoke_alias_test(socket-orbit-stream-3d socket-orbit-stream-3d)
    add_help_smoke_alias_test(render-visual-stream render-visual-stream)
    add_help_smoke_alias_test(object-simulator object-simulator)
    add_help_smoke_alias_test(object-simulator jsa-obj-sim)
endif()
