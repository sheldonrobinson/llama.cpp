include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET server-context)

include(FetchContent)
######################## libuv
FetchContent_Declare(
  libuv
  GIT_REPOSITORY 		 https://github.com/libuv/libuv.git
  GIT_TAG        		 origin/v1.x
  GIT_SHALLOW 			 TRUE
  GIT_SUBMODULES_RECURSE TRUE
  
  EXCLUDE_FROM_ALL
			 
  OVERRIDE_FIND_PACKAGE
)

set(LIBUV_BUILD_SHARED ON CACHE BOOL "Build shared lib" FORCE)

FetchContent_MakeAvailable(libuv)

file(REAL_PATH ${libuv_SOURCE_DIR} LIBUV_SOURCE_DIR)
file(REAL_PATH ${libuv_BINARY_DIR} LIBUV_BINARY_DIR)

set(LIBUV_LIBRARY ${LIBUV_SOURCE_DIR}/$<CONFIG>/uv.lib CACHE FILEPATH "libuv library" FORCE)

if(NOT TARGET libuv::libuv)
	add_library(libuv::libuv INTERFACE IMPORTED)
    set_target_properties(libuv::libuv PROPERTIES
                          INTERFACE_LINK_LIBRARIES 
							  uv
						  INTERFACE_POSITION_INDEPENDENT_CODE ON
						  )
endif()

add_library(${TARGET} STATIC
    server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
)

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC common mtmd ${CMAKE_THREAD_LIBS_INIT})


# llama-server executable

set(TARGET llama-embedded)

if (NOT LLAMA_HTTPLIB)
    message(FATAL_ERROR "LLAMA_HTTPLIB is OFF, cannot build llama-embedded. Hint: to skip building server, set -DLLAMA_BUILD_SERVER=OFF")
endif()

set(TARGET_SRCS
    server-core.cpp
    server-core.h
    server-embedded.cpp
    server-embedded.h
)

if(BUILD_SHARED_LIBS)
	add_library(${TARGET} SHARED ${TARGET_SRCS})
else()
	add_library(${TARGET} STATIC ${TARGET_SRCS})
endif()
install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context PUBLIC uv common cpp-httplib concurrentqueue ${CMAKE_THREAD_LIBS_INIT})


target_compile_features(${TARGET} PRIVATE cxx_std_17)
