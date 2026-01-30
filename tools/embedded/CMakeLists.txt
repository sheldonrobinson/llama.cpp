include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR})

# server-context containing the core server logic, used by llama-server and CLI

set(TARGET llama-embedded)

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

######################## pthreadpool
FetchContent_Declare(
  pthreadpool
  GIT_REPOSITORY 		 https://github.com/google/pthreadpool.git
  GIT_TAG        		 9003ee6c137cea3b94161bd5c614fb43be523ee1
  GIT_SHALLOW 			 TRUE
  GIT_SUBMODULES_RECURSE TRUE
  
  EXCLUDE_FROM_ALL
			 
  OVERRIDE_FIND_PACKAGE
)

SET(PTHREADPOOL_LIBRARY_TYPE "static" CACHE STRING "Type of library (shared, static, or default) to build" FORCE)
set(PTHREADPOOL_ALLOW_DEPRECATED_API OFF CACHE BOOL "Enable deprecated API functions" FORCE)
set(PTHREADPOOL_BUILD_TESTS OFF CACHE BOOL "Build pthreadpool unit tests" FORCE)
set(PTHREADPOOL_BUILD_BENCHMARKS OFF CACHE BOOL "Build pthreadpool micro-benchmarks" FORCE)

FetchContent_MakeAvailable(pthreadpool)


######################## concurrentqueue
FetchContent_Declare(
  concurrentqueue
  GIT_REPOSITORY 		 https://github.com/cameron314/concurrentqueue.git
  GIT_TAG        		 c68072129c8a5b4025122ca5a0c82ab14b30cb03
  GIT_SHALLOW 			 TRUE
  GIT_SUBMODULES_RECURSE TRUE
  
  EXCLUDE_FROM_ALL
			 
  OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(concurrentqueue)


if (NOT LLAMA_HTTPLIB)
    message(FATAL_ERROR "LLAMA_HTTPLIB is OFF, cannot build llama-embedded. Hint: to skip building server, set -DLLAMA_BUILD_SERVER=OFF")
endif()

set(TARGET_SRCS
	server-task.cpp
    server-task.h
    server-queue.cpp
    server-queue.h
    server-common.cpp
    server-common.h
    server-context.cpp
    server-context.h
    server-core.cpp
    server-core.h
    server-embedded.cpp
    server-embedded.h
	uv-memory-server.hpp
	server-model-manager.hpp
)

if(BUILD_SHARED_LIBS)
	add_library(${TARGET} SHARED ${TARGET_SRCS})
else()
	add_library(${TARGET} STATIC ${TARGET_SRCS})
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

install(TARGETS ${TARGET} RUNTIME)

target_include_directories(${TARGET} PRIVATE ../mtmd)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(${TARGET} PUBLIC common mtmd uv pthreadpool cpp-httplib concurrentqueue ${CMAKE_THREAD_LIBS_INIT})


target_compile_features(${TARGET} PRIVATE cxx_std_17)
