include_guard(GLOBAL)

function(jsa_add_executable target_name)
    cmake_parse_arguments(JSA "" "" "SOURCES;LINK_LIBS" ${ARGN})

    if(NOT JSA_SOURCES)
        message(FATAL_ERROR "jsa_add_executable(${target_name}) requires SOURCES")
    endif()

    add_executable(${target_name} ${JSA_SOURCES})

    if(JSA_LINK_LIBS)
        target_link_libraries(${target_name} PRIVATE ${JSA_LINK_LIBS})
    endif()

    jsa_apply_project_options(${target_name})
endfunction()
