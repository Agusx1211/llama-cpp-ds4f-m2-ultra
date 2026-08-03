set(TARGET llama-dsv4-direct-amx-probe)

find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

add_executable(${TARGET} dsv4-direct-amx-probe.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${TARGET} PRIVATE
    ACCELERATE_NEW_LAPACK
    ACCELERATE_LAPACK_ILP64)
target_link_libraries(${TARGET} PRIVATE
    ggml
    ${CMAKE_THREAD_LIBS_INIT}
    ${ACCELERATE_FRAMEWORK})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
