set(TARGET llama-dsv4-amx-probe)

find_library(ACCELERATE_FRAMEWORK Accelerate REQUIRED)

add_executable(${TARGET} dsv4-amx-probe.cpp)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE
    llama
    llama-common
    ${CMAKE_THREAD_LIBS_INIT}
    ${ACCELERATE_FRAMEWORK})

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
