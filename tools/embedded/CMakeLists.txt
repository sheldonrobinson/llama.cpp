# llama-embedded - libopenai-based embedded inference server/client
#
# Builds a static library that integrates llama.cpp backend with libopenai
# transports for single-process (in-memory) and multi-process (named pipe)
# embedded inference.

cmake_minimum_required(VERSION 3.14)
project(llama-embedded LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Locate libopenai
# ---------------------------------------------------------------------------
set(LIBOPENAI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../libopenai" CACHE PATH
    "Path to libopenai directory")

# Optional: path to a prebuilt llama.cpp work tree (e.g. work/build) that
# provides server-context.lib and its dependencies. When llama-embedded is
# built as part of the main work tree, the server-context target is available
# directly (tools/server is added before tools/embedded). When built
# standalone, set this to link the prebuilt libraries so single-process
# (in-memory) embedded serving can use real llama.cpp inference without
# subprocesses. Named-pipe (multi-process) embedded serving may use
# subprocesses and does not require this.
set(LLAMA_EMBEDDED_SERVER_BUILD_DIR "" CACHE PATH
    "Path to a prebuilt llama.cpp work build providing server-context.lib")

if(NOT EXISTS "${LIBOPENAI_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "libopenai not found at ${LIBOPENAI_DIR}")
endif()

# ---------------------------------------------------------------------------
# Enable named pipe transport in libopenai (needed for multi-process mode)
# ---------------------------------------------------------------------------
set(LIBOPENAI_USE_NAMED_PIPE ON CACHE BOOL "Enable named pipe transport" FORCE)

# ---------------------------------------------------------------------------
# Include libopenai as a subdirectory
# ---------------------------------------------------------------------------
add_subdirectory(${LIBOPENAI_DIR} ${CMAKE_CURRENT_BINARY_DIR}/libopenai-build
    EXCLUDE_FROM_ALL)

# ---------------------------------------------------------------------------
# Source files for llama-embedded library
# ---------------------------------------------------------------------------
# embedded_stubs.cpp is only needed in standalone mode where prebuilt
# libraries don't export ggml_backend_* and llama_build_info symbols.
# In-tree builds link against the real llama/server-context targets.
set(TARGET_SRCS
    embedded_server.hpp
    embedded_server.cpp
    embedded_client.hpp
    embedded_client.cpp
)

# ---------------------------------------------------------------------------
# Build llama-embedded static library
# ---------------------------------------------------------------------------
add_library(llama-embedded STATIC ${TARGET_SRCS})

target_include_directories(llama-embedded PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBOPENAI_DIR}/include
    ${LIBOPENAI_DIR}/include/libopenai
    ${PROJECT_SOURCE_DIR}/../../tools/mtmd
)

target_link_libraries(llama-embedded PUBLIC
    libopenai
    httplib::httplib
)

# ---------------------------------------------------------------------------
# Link llama, ggml, and server-context for single-process (in-memory) embedded
# serving so it can use real llama.cpp inference in-process without subprocesses.
# server-context provides server-task, server-queue, and speculative decoding
# via common_speculative. Named-pipe (multi-process) embedded serving may use
# subprocesses and does not require these libraries.
# ---------------------------------------------------------------------------
if(TARGET llama)
    # Built as part of the main llama.cpp work tree: llama and server-context
    # targets are available (tools/server is added before tools/embedded).
    target_link_libraries(llama-embedded PUBLIC llama server-context)
    target_include_directories(llama-embedded PUBLIC
        ${PROJECT_SOURCE_DIR}/../../include
        ${PROJECT_SOURCE_DIR}/../../ggml/include
        ${PROJECT_SOURCE_DIR}/../../common
        ${PROJECT_SOURCE_DIR}/../../tools/server
    )
elseif(LLAMA_EMBEDDED_SERVER_BUILD_DIR AND EXISTS
           "${LLAMA_EMBEDDED_SERVER_BUILD_DIR}/src")
    # Standalone build: compile server-context from source (to avoid __imp_
    # symbol mismatches from prebuilt server-context.lib compiled without
    # LLAMA_BUILD/GGML_BUILD) and link prebuilt llama/ggml libraries.
    set(_embed_server_build "${LLAMA_EMBEDDED_SERVER_BUILD_DIR}")
    set(_embed_server_src "${PROJECT_SOURCE_DIR}/../../tools/server")

    # Compile all server-context source files (matching upstream
    # tools/server/CMakeLists.txt server-context target). This avoids
    # unresolved externals from cross-file dependencies.
    set(SERVER_CONTEXT_SRCS
        "${_embed_server_src}/server-chat.cpp"
        "${_embed_server_src}/server-task.cpp"
        "${_embed_server_src}/server-queue.cpp"
        "${_embed_server_src}/server-common.cpp"
        "${_embed_server_src}/server-decision.cpp"
        "${_embed_server_src}/server-context.cpp"
        "${_embed_server_src}/server-realtime.cpp"
        "${_embed_server_src}/server-stream.cpp"
        "${_embed_server_src}/server-tools.cpp"
        "${_embed_server_src}/server-mcp.cpp"
        "${_embed_server_src}/server-schema.cpp"
    )
    add_library(server-context-embedded STATIC ${SERVER_CONTEXT_SRCS})
    target_include_directories(server-context-embedded PUBLIC
        "${PROJECT_SOURCE_DIR}/../../include"
        "${PROJECT_SOURCE_DIR}/../../ggml/include"
        "${PROJECT_SOURCE_DIR}/../../common"
        "${PROJECT_SOURCE_DIR}/../../src"
        "${PROJECT_SOURCE_DIR}/../../tools/mtmd"
        "${PROJECT_SOURCE_DIR}/../../tools/server"
        "${LIBOPENAI_DIR}/include"
        "${LIBOPENAI_DIR}/include/libopenai"
        "${_embed_server_build}"
        "${_embed_server_build}/ggml/include"
        "${_embed_server_build}/src"
        "${_embed_server_build}/common"
        "${_embed_server_build}/tools/server"
        "${CMAKE_CURRENT_BINARY_DIR}/_deps"
        "${PROJECT_SOURCE_DIR}/../../vendor/sheredom"
        "${PROJECT_SOURCE_DIR}/../../vendor"
    )
    target_compile_definitions(server-context-embedded PUBLIC GGML_BUILD LLAMA_BUILD)
    target_link_libraries(server-context-embedded PUBLIC
        "${_embed_server_build}/src/$<CONFIG>/llama.lib"
        "${_embed_server_build}/common/$<CONFIG>/llama-common.lib"
        "${_embed_server_build}/ggml/src/$<CONFIG>/ggml-base.lib"
        "${_embed_server_build}/ggml/src/$<CONFIG>/ggml-cpu.lib"
    )

    # Compile embedded_stubs.cpp (llama_build_info from upstream/common,
    # needed by server-context but not exported by prebuilt DLLs compiled
    # without LLAMA_BUILD/GGML_BUILD). ggml_backend_* functions come from
    # the real ggml library (prebuilt ggml-base.lib / ggml-cpu.lib).
    add_library(ggml-embedded-STATIC STATIC embedded_stubs.cpp)
    target_include_directories(ggml-embedded-STATIC PUBLIC
        "${PROJECT_SOURCE_DIR}/../../common"
    )
    target_compile_definitions(ggml-embedded-STATIC PUBLIC GGML_BUILD LLAMA_BUILD)

    # Compile mtmd from source (matching upstream tools/mtmd/CMakeLists.txt)
    set(MTMD_SRC "${PROJECT_SOURCE_DIR}/../../tools/mtmd")
    set(MTMD_SRCS
        "${MTMD_SRC}/mtmd.cpp"
        "${MTMD_SRC}/mtmd-audio.cpp"
        "${MTMD_SRC}/mtmd-image.cpp"
        "${MTMD_SRC}/mtmd-helper.cpp"
        "${MTMD_SRC}/mtmd-helper-gen.cpp"
        "${MTMD_SRC}/clip.cpp"
        # hash library source files (dependency of mtmd)
        "${PROJECT_SOURCE_DIR}/../../vendor/hash/hash.cpp"
        "${PROJECT_SOURCE_DIR}/../../vendor/hash/sha256/sha256.c"
        "${PROJECT_SOURCE_DIR}/../../vendor/hash/sha1/sha1.c"
        "${PROJECT_SOURCE_DIR}/../../vendor/hash/xxhash/xxhash.c"
        "${MTMD_SRC}/models/models.h"
        "${MTMD_SRC}/models/cogvlm.cpp"
        "${MTMD_SRC}/models/conformer.cpp"
        "${MTMD_SRC}/models/deepseek4v.cpp"
        "${MTMD_SRC}/models/dots3note.cpp"
        "${MTMD_SRC}/models/dotsocr.cpp"
        "${MTMD_SRC}/models/exaone4_5.cpp"
        "${MTMD_SRC}/models/gemma4a.cpp"
        "${MTMD_SRC}/models/gemma4v.cpp"
        "${MTMD_SRC}/models/gemma4ua.cpp"
        "${MTMD_SRC}/models/gemma4uv.cpp"
        "${MTMD_SRC}/models/glm4v.cpp"
        "${MTMD_SRC}/models/granite-speech.cpp"
        "${MTMD_SRC}/models/granite4-vision.cpp"
        "${MTMD_SRC}/models/hunyuanvl.cpp"
        "${MTMD_SRC}/models/internvl.cpp"
        "${MTMD_SRC}/models/kimivl.cpp"
        "${MTMD_SRC}/models/kimik25.cpp"
        "${MTMD_SRC}/models/nemotron-v2-vl.cpp"
        "${MTMD_SRC}/models/nemotron-omni.cpp"
        "${MTMD_SRC}/models/muse-glimmer.cpp"
        "${MTMD_SRC}/models/muse-spark-audio.cpp"
        "${MTMD_SRC}/models/llama4.cpp"
        "${MTMD_SRC}/models/llava.cpp"
        "${MTMD_SRC}/models/minicpmv.cpp"
        "${MTMD_SRC}/models/paddleocr.cpp"
        "${MTMD_SRC}/models/pixtral.cpp"
        "${MTMD_SRC}/models/qwen2vl.cpp"
        "${MTMD_SRC}/models/sensenova-u1.cpp"
        "${MTMD_SRC}/models/minimax-m3.cpp"
        "${MTMD_SRC}/models/qwen3vl.cpp"
        "${MTMD_SRC}/models/lfm3vl.cpp"
        "${MTMD_SRC}/models/lfm3a.cpp"
        "${MTMD_SRC}/models/lfm3a-gen.cpp"
        "${MTMD_SRC}/models/ling3vl.cpp"
        "${MTMD_SRC}/models/mimovl.cpp"
        "${MTMD_SRC}/models/qwen3a.cpp"
        "${MTMD_SRC}/models/mimo-audio.cpp"
        "${MTMD_SRC}/models/qwen3tts-spkenc.cpp"
        "${MTMD_SRC}/models/qwen3tts-gen.cpp"
        "${MTMD_SRC}/models/pockettts-seanet.cpp"
        "${MTMD_SRC}/models/pockettts-spkenc.cpp"
        "${MTMD_SRC}/models/pockettts-gen.cpp"
        "${MTMD_SRC}/models/soprano.cpp"
        "${MTMD_SRC}/models/step3vl.cpp"
        "${MTMD_SRC}/models/siglip.cpp"
        "${MTMD_SRC}/models/whisper-enc.cpp"
        "${MTMD_SRC}/models/voxtral-rt.cpp"
        "${MTMD_SRC}/models/deepseekocr.cpp"
        "${MTMD_SRC}/models/deepseekocr2.cpp"
        "${MTMD_SRC}/models/mobilenetv5.cpp"
        "${MTMD_SRC}/models/youtuvl.cpp"
        "${MTMD_SRC}/models/yasa2.cpp"
        "${MTMD_SRC}/models/parakeet.cpp"
    )
    add_library(mtmd-embedded STATIC ${MTMD_SRCS})
    target_include_directories(mtmd-embedded PUBLIC
        "${MTMD_SRC}"
        "${PROJECT_SOURCE_DIR}/../../include"
        "${PROJECT_SOURCE_DIR}/../../ggml/include"
        "${PROJECT_SOURCE_DIR}/../../common"
        "${PROJECT_SOURCE_DIR}/../../vendor/sheredom"
        "${PROJECT_SOURCE_DIR}/../../vendor"
        "${PROJECT_SOURCE_DIR}/../../vendor/hash/sha1"
        "${PROJECT_SOURCE_DIR}/../../vendor/hash/sha256"
        "${PROJECT_SOURCE_DIR}/../../vendor/hash/xxhash"
        "${PROJECT_SOURCE_DIR}/../../vendor/hash/rotate-bits"
        "${PROJECT_SOURCE_DIR}/../../vendor/miniaudio"
        "${PROJECT_SOURCE_DIR}/../../vendor/stb"
    )
    target_compile_definitions(mtmd-embedded PUBLIC GGML_BUILD LLAMA_BUILD)
    target_link_libraries(mtmd-embedded PUBLIC
        "${_embed_server_build}/src/$<CONFIG>/llama.lib"
        "${_embed_server_build}/ggml/src/$<CONFIG>/ggml-base.lib"
        "${_embed_server_build}/ggml/src/$<CONFIG>/ggml-cpu.lib"
    )

    # Link llama, ggml, server-context-embedded, and mtmd-embedded
    target_link_libraries(llama-embedded PUBLIC
        "${_embed_server_build}/src/$<CONFIG>/llama.lib"
        "${_embed_server_build}/common/$<CONFIG>/llama-common.lib"
        "${_embed_server_build}/ggml/src/$<CONFIG>/ggml-base.lib"
        "${_embed_server_build}/ggml/src/$<CONFIG>/ggml-cpu.lib"
        server-context-embedded
        mtmd-embedded
        ggml-embedded-STATIC
    )
    target_include_directories(llama-embedded PUBLIC
        "${PROJECT_SOURCE_DIR}/../../include"
        "${PROJECT_SOURCE_DIR}/../../ggml/include"
        "${PROJECT_SOURCE_DIR}/../../common"
        "${PROJECT_SOURCE_DIR}/../../src"
        "${PROJECT_SOURCE_DIR}/../../tools/server"
        "${_embed_server_build}"
        "${_embed_server_build}/ggml/include"
        "${_embed_server_build}/src"
        "${_embed_server_build}/common"
        "${_embed_server_build}/tools/server"
    )
    message(STATUS "llama-embedded: compiling server-context and mtmd from source, linking prebuilt llama/ggml from ${LLAMA_EMBEDDED_SERVER_BUILD_DIR}")
else()
    message(STATUS "llama-embedded: llama not linked (single-process will use placeholder backend; set LLAMA_EMBEDDED_SERVER_BUILD_DIR to link llama)")
endif()

# Link llama-server-impl source files for multi-process (subprocess) support.
# These match upstream tools/server/CMakeLists.txt llama-server-impl target:
#   server.cpp, server-connect.cpp, server-http.cpp, server-models.cpp,
#   server-models-sched.cpp
set(_embed_server_src "${PROJECT_SOURCE_DIR}/../../tools/server")
set(SERVER_IMPL_SRCS
    "${_embed_server_src}/server.cpp"
    "${_embed_server_src}/server-connect.cpp"
    "${_embed_server_src}/server-http.cpp"
    "${_embed_server_src}/server-models.cpp"
    "${_embed_server_src}/server-models-sched.cpp"
)
if(EXISTS "${_embed_server_src}/server.cpp")
    add_library(llama-server-impl-embedded STATIC ${SERVER_IMPL_SRCS})
    target_include_directories(llama-server-impl-embedded PUBLIC
        "${PROJECT_SOURCE_DIR}/../../include"
        "${PROJECT_SOURCE_DIR}/../../ggml/include"
        "${PROJECT_SOURCE_DIR}/../../common"
        "${PROJECT_SOURCE_DIR}/../../tools/server"
        "${PROJECT_SOURCE_DIR}/../../tools/mtmd"
        "${CMAKE_CURRENT_BINARY_DIR}/_deps"
        "${CMAKE_CURRENT_BINARY_DIR}/_deps/httplib-src"
    )
    target_compile_definitions(llama-server-impl-embedded PUBLIC GGML_BUILD LLAMA_BUILD)
    if(TARGET llama)
        target_link_libraries(llama-server-impl-embedded PUBLIC llama server-context)
    elseif(LLAMA_EMBEDDED_SERVER_BUILD_DIR)
        target_link_libraries(llama-server-impl-embedded PUBLIC
            "${_embed_server_build}/src/$<CONFIG>/llama.lib"
            "${_embed_server_build}/common/$<CONFIG>/llama-common.lib"
            server-context-embedded
        )
    endif()
    message(STATUS "llama-embedded: compiling llama-server-impl source files for multi-process support")
endif()

target_compile_definitions(llama-embedded PRIVATE GGML_BUILD LLAMA_BUILD LIBOPENAI_USE_NAMED_PIPE)
target_compile_features(llama-embedded PRIVATE cxx_std_17)

# ---------------------------------------------------------------------------
# Example: embedded single-process demo
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/embedded_example.cpp")
    add_executable(embedded_example embedded_example.cpp)
    target_link_libraries(embedded_example PRIVATE llama-embedded nlohmann_json::nlohmann_json)
    target_include_directories(embedded_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBOPENAI_DIR}/include
        ${LIBOPENAI_DIR}/include/libopenai
        ${PROJECT_SOURCE_DIR}/../../tools/mtmd
    )
    target_compile_definitions(embedded_example PRIVATE GGML_BUILD LLAMA_BUILD)
    target_compile_features(embedded_example PRIVATE cxx_std_17)
endif()

# ---------------------------------------------------------------------------
# Example: embedded multi-process demo
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/embedded_mp_example.cpp")
    add_executable(embedded_mp_example embedded_mp_example.cpp)
    target_link_libraries(embedded_mp_example PRIVATE llama-embedded nlohmann_json::nlohmann_json)
    target_include_directories(embedded_mp_example PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${LIBOPENAI_DIR}/include
        ${LIBOPENAI_DIR}/include/libopenai
        ${PROJECT_SOURCE_DIR}/../../tools/mtmd
    )
    target_compile_definitions(embedded_mp_example PRIVATE GGML_BUILD LLAMA_BUILD)
    target_compile_features(embedded_mp_example PRIVATE cxx_std_17)
endif()

install(TARGETS llama-embedded RUNTIME)
