set(TARGET llama-dsv4-amx-full-model-gate)

add_executable(${TARGET} dsv4-amx-full-model-gate.cpp)
target_link_libraries(${TARGET} PRIVATE llama-common llama ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
