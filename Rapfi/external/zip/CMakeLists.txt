cmake_minimum_required(VERSION 3.14)

project(zip
  LANGUAGES C
  VERSION "0.2.3")

option(ZIP_STATIC_PIC "Build static zip with PIC" ON)

if(ZIP_ENABLE_SHARABLE_FILE_OPEN)
	add_definitions(-DZIP_ENABLE_SHARABLE_FILE_OPEN)
endif()

if(CMAKE_SIZEOF_VOID_P EQUAL 4)
	# large file support
	add_definitions(-D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64)
endif()

# zip
set(SRC src/miniz.h src/zip.h src/zip.c)

add_library(${PROJECT_NAME} ${SRC})
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

if(ZIP_STATIC_PIC)
  set_property(TARGET ${PROJECT_NAME} PROPERTY POSITION_INDEPENDENT_CODE 1)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY C_VISIBILITY_PRESET hidden)
if(BUILD_SHARED_LIBS)
  target_compile_definitions(${PROJECT_NAME}
    PUBLIC ZIP_SHARED
    PRIVATE ZIP_BUILD_SHARED
  )
endif()

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

set(CMAKE_C_STANDARD 90)
if (MSVC)
  if(CMAKE_BUILD_TYPE STREQUAL Release)
    set_property(TARGET ${PROJECT_NAME} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
  endif()

  # Use secure functions by default and suppress warnings about "deprecated" functions
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES=1")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D _CRT_SECURE_CPP_OVERLOAD_STANDARD_NAMES_COUNT=1")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /D _CRT_NONSTDC_NO_WARNINGS=1 /D _CRT_SECURE_NO_WARNINGS=1")

elseif ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR
        "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" OR
        "${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wno-error -pedantic")
endif (MSVC)
