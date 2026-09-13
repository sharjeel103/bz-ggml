cmake_minimum_required(VERSION 3.19)
project(bz_ggml LANGUAGES C CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(BREEZE_CUDA "Build ggml with CUDA backend" ON)
option(BREEZE_BUILD_SHARED "Build shared C library (libbreeze.so)" ON)
option(BREEZE_BUILD_CLI "Build standalone CLI test executable" ON)

# GGML Submodule or FetchContent fallback
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ggml/CMakeLists.txt")
    message(STATUS "Using local third_party/ggml submodule")
    set(GGML_CUDA ${BREEZE_CUDA} CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/ggml)
else()
    message(STATUS "third_party/ggml submodule missing; fetching via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        ggml
        GIT_REPOSITORY https://github.com/ggml-org/ggml.git
        GIT_TAG master
    )
    set(GGML_CUDA ${BREEZE_CUDA} CACHE BOOL "" FORCE)
    set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(ggml)
endif()

set(BREEZE_SOURCES
    src/common.cpp
    src/audio.cpp
    src/config.cpp
    src/gguf_loader.cpp
    src/tokenizer.cpp
    src/text_encoder.cpp
    src/backbone.cpp
    src/depth_decoder.cpp
    src/codec.cpp
    src/codec_conv.cpp
    src/codec_decoder.cpp
    src/codec_transformer.cpp
    src/sampling.cpp
    src/generation.cpp
    src/text_split.cpp
    src/voice.cpp
    src/model.cpp
)

add_library(breeze_core STATIC ${BREEZE_SOURCES})
target_include_directories(breeze_core PUBLIC include)
target_link_libraries(breeze_core PUBLIC ggml)

if(MINGW)
    add_link_options(-static-libgcc -static-libstdc++ -static)
endif()

if(BREEZE_BUILD_SHARED)
    add_library(breeze SHARED src/c_api.cpp)
    target_link_libraries(breeze PRIVATE breeze_core)
    target_compile_definitions(breeze PRIVATE BREEZE_BUILD_SHARED)
    set_target_properties(breeze PROPERTIES OUTPUT_NAME breeze)
endif()

if(BREEZE_BUILD_CLI AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/apps/cli/main.cpp")
    add_executable(breeze-cli apps/cli/main.cpp)
    target_link_libraries(breeze-cli PRIVATE breeze_core)
endif()
