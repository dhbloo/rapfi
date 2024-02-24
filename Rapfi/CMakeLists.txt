cmake_minimum_required(VERSION 3.18.2 FATAL_ERROR)
project(rapfi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#==========================================================
# CMake Options

option(NO_MULTI_THREADING "Disable multi-threading" OFF)
option(NO_COMMAND_MODULES "Disable command modules" OFF)
option(NO_PREFETCH "Disable prefetch in search" OFF)

option(USE_ROTR "Enable ROTR instruction" ON)
option(USE_SSE  "Enable SSE-SSE4.2 instruction" ON)
option(USE_AVX  "Enable AVX instruction" ON)
option(USE_AVX2 "Enable AVX2/FMA instruction" ON)
option(USE_AVX512 "Enable AVX512 instruction" OFF)
option(USE_BMI2 "Enable BMI2 instruction" OFF)

option(ENABLE_LTO "Enable link time optimization" OFF)
option(FORCE_ASSERTION "Enable force assertion for all build type" OFF)
option(USE_ASAN "Enable address sanitizer" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "No build type selected, default to Release" FORCE)
endif()

#==========================================================
# Rapfi Compiling

set(CORE_SOURCES
    command/argutils.cpp
    command/benchmark.cpp
    command/command.cpp
    command/gomocup.cpp

    core/hash.cpp
    core/iohelper.cpp
    core/utils.cpp
    core/platform.cpp

    database/dbclient.cpp
    database/dbutils.cpp
    database/dbtypes.cpp
    database/yxdbstorage.cpp

    eval/eval.cpp
    eval/evaluator.cpp
    eval/mix6nnue.cpp
    eval/mix7nnue.cpp
    eval/mix8nnue.cpp
    eval/nnuev2.cpp

    game/board.cpp
    game/movegen.cpp
    game/pattern.cpp

    search/hashtable.cpp
    search/movepick.cpp
    search/opening.cpp
    search/searchcommon.cpp
    search/searchoutput.cpp
    search/searchthread.cpp
    search/timecontrol.cpp
    search/ab/history.cpp
    search/ab/search.cpp
    
    config.cpp
    internalConfig.cpp
    main.cpp
)

set(MODULE_SOURCES
    command/database.cpp
    command/dataprep.cpp
    command/opengen.cpp
    command/selfplay.cpp
    command/tuning.cpp
    tuning/dataset.cpp
    tuning/datawriter.cpp
    tuning/optimizer.cpp
    tuning/tuner.cpp
)

set(HEADERS
    command/command.h
    command/argutils.h

    core/hash.h
    core/iohelper.h
    core/platform.h
    core/pos.h
    core/types.h
    core/utils.h
    core/version.h

    database/cache.h
    database/dbclient.h
    database/dbstorage.h
    database/dbtypes.h
	database/dbutils.h
    database/yxdbstorage.h

    eval/eval.h
    eval/evaluator.h
    eval/mix6nnue.h
    eval/mix7nnue.h
    eval/mix8nnue.h
    eval/nnuev2.h
    eval/simdops.h
    eval/weightloader.h

    game/board.h
    game/movegen.h
    game/pattern.h
    game/wincheck.h

    tuning/dataentry.h
    tuning/dataset.h
    tuning/datawriter.h
    tuning/optimizer.h
    tuning/tunemap.h
    tuning/tuner.h

    search/hashtable.h
    search/history.h
    search/movepick.h
    search/opening.h
    search/searchcommon.h
    search/searcher.h
    search/searchoutput.h
    search/searchthread.h
    search/skill.h
    search/timecontrol.h
    search/ab/history.h
    search/ab/parameter.h
    search/ab/searcher.h
    search/ab/searchstack.h

    config.h
)

add_executable(rapfi
    ${CORE_SOURCES}
    $<$<NOT:$<BOOL:${NO_COMMAND_MODULES}>>:${MODULE_SOURCES}>
    ${HEADERS})
set_target_properties(rapfi PROPERTIES OUTPUT_NAME "pbrain-rapfi")

# external libraries
add_subdirectory(external/cpptoml)
add_subdirectory(external/cxxopts)
add_subdirectory(external/lz4)
add_subdirectory(external/simde)
target_link_libraries(rapfi PRIVATE cpptoml cxxopts lz4 simde)
if(NOT NO_COMMAND_MODULES)
    add_subdirectory(external/flat.hpp)
    add_subdirectory(external/zip)
    add_subdirectory(external/libnpy)
    target_link_libraries(rapfi PRIVATE flat.hpp zip libnpy)
endif()

#==========================================================
# Web Assembly compiling

if(EMSCRIPTEN)
    message(STATUS "Setting up build for Emscripten.")
    set(WEB_DIR "${CMAKE_SOURCE_DIR}/../Gomocalc")

    if(NOT NO_COMMAND_MODULES)
        message(FATAL_ERROR
            "Command modules is not supported for wasm build. "
            "Must set option NO_COMMAND_MODULES to ON.")
    endif()
    set(USE_ROTR OFF)
    set(USE_AVX OFF)
    set(USE_AVX2 OFF)
    set(USE_AVX512 OFF)
    set(USE_BMI2 OFF)

    target_compile_options(rapfi
        PRIVATE "-fexceptions"
        PRIVATE "-flto"
    #   PRIVATE "-msimd128"
    #   PRIVATE "-msse"
    )
    string(CONCAT EMCC_FLAGS
        "-fexceptions "
        "-flto "
        "--pre-js ${CMAKE_SOURCE_DIR}/emscripten/preamble.js "
        "--preload-file ${WEB_DIR}/configs@/ "
        "--closure=1 "
        "-s MODULARIZE=1 "
        "-s EXPORT_NAME=Rapfi "
        "-s ENVIRONMENT='web,worker' "
        "-s EXPORTED_FUNCTIONS=\"['_gomocupLoopOnce']\" "
        "-s EXPORTED_RUNTIME_METHODS=\"['cwrap']\" "
        "-s DEMANGLE_SUPPORT=1 "
        "-s ALLOW_MEMORY_GROWTH=1 "
        "-s INITIAL_MEMORY=134217728 "
        "-s MAXIMUM_MEMORY=1073741824 "
    )

    if(NO_MULTI_THREADING)
        set_target_properties(rapfi PROPERTIES OUTPUT_NAME "rapfi-single")
    else()
        set_target_properties(rapfi PROPERTIES OUTPUT_NAME "rapfi-multi")
        target_compile_options(rapfi PRIVATE "-pthread")
        string(CONCAT EMCC_FLAGS ${EMCC_FLAGS}
            "-s USE_PTHREADS=1 "
            "-s PTHREAD_POOL_SIZE=1+navigator.hardwareConcurrency "
        )
    endif()

    if(FORCE_ASSERTION)
        string(CONCAT EMCC_FLAGS ${EMCC_FLAGS} "-s ASSERTIONS=1 ")
    endif()

    set_target_properties(rapfi PROPERTIES LINK_FLAGS ${EMCC_FLAGS})

endif()

#==========================================================
# Compile flags

if(NOT NO_MULTI_THREADING)
    target_compile_definitions(rapfi PRIVATE MULTI_THREADING)
endif()
if(NOT NO_COMMAND_MODULES)
    target_compile_definitions(rapfi PRIVATE COMMAND_MODULES)
endif()
if(NO_PREFETCH)
    target_compile_definitions(rapfi PRIVATE NO_PREFETCH)
endif()
if(USE_ROTR)
    target_compile_definitions(rapfi PRIVATE USE_ROTR)
endif()
if(USE_SSE)
    target_compile_definitions(rapfi PRIVATE USE_SSE)
endif()
if(USE_AVX)
    target_compile_definitions(rapfi PRIVATE USE_AVX)
endif()
if(USE_AVX2)
    target_compile_definitions(rapfi PRIVATE USE_AVX2)
endif()
if(USE_AVX512)
    target_compile_definitions(rapfi PRIVATE USE_AVX512)
endif()
if(USE_BMI2 OR USE_AVX512)
    target_compile_definitions(rapfi PRIVATE USE_BMI2)
endif()

if(ENABLE_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPOSupported OUTPUT error)
    if(IPOSupported)
        set_property(TARGET rapfi PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "LTO not supported: <${error}>")
    endif()
endif()

if(MSVC)
    message(STATUS "Setting up build for MSVC.")

    string(REGEX REPLACE "/O2" "/Ox"
        CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")

	if(USE_AVX512)
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
    elseif(USE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    elseif(USE_AVX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
    elseif(USE_SSE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /d2archSSE42")
    endif()

    if(FORCE_ASSERTION)
        string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
        string(REPLACE "/DNDEBUG" "" CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}})
    endif()

    if(USE_ASAN)
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fsanitize=address")
	endif()
	
    if(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
        set_target_properties(rapfi PROPERTIES LINK_FLAGS "/PROFILE")
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL Release)
        set_property(TARGET rapfi PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded")
    endif()

elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
    message(STATUS "Setting up build for GNU or Clang.")

	if(USE_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512dq -mavx512bw")
    elseif(USE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    elseif(USE_AVX)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    elseif(USE_SSE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -mssse3 -msse4 -msse4a -msse4.1 -msse4.2")
    endif()

    if(USE_BMI2 OR USE_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mbmi2")
    endif()

    if(FORCE_ASSERTION)
        string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
        string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}})
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-variable -Wno-unused-function -Wno-missing-braces -Wno-logical-op-parentheses -Wno-parentheses -Wno-sign-compare -Wno-strict-prototypes")

	if(USE_ASAN)
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
	endif()

    if(WIN32)
        if(NO_COMMAND_MODULES)
            target_link_libraries(rapfi PRIVATE -static)
		else()
            target_link_libraries(rapfi PRIVATE tbb12)
        endif()
    elseif(NOT EMSCRIPTEN)
        if(NO_MULTI_THREADING AND NO_COMMAND_MODULES)
            target_link_libraries(rapfi PRIVATE -static)
        else()
		    target_link_libraries(rapfi PRIVATE pthread)
            if(NOT NO_COMMAND_MODULES)
                target_link_libraries(rapfi PRIVATE tbb)
            endif()
        endif()
    endif()
endif()

#==========================================================
# IDE Helper

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Sources" FILES ${CORE_SOURCES} ${MODULE_SOURCES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Headers" FILES ${HEADERS})
