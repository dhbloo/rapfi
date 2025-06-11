cmake_minimum_required(VERSION 3.18.2 FATAL_ERROR)
project(rapfi)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#==========================================================
# Auto-detect SIMD/ISA capability of the host
include(CheckCXXSourceCompiles)

function(bool_to_onoff in_var out_var)
    if(${in_var})
        set(${out_var} ON  PARENT_SCOPE)
    else()
        set(${out_var} OFF PARENT_SCOPE)
    endif()
endfunction()

set(DEFAULT_USE_SSE               OFF)
set(DEFAULT_USE_AVX2              OFF)
set(DEFAULT_USE_AVX512            OFF)
set(DEFAULT_USE_BMI2              OFF)
set(DEFAULT_USE_VNNI              OFF)
set(DEFAULT_USE_NEON              OFF)
set(DEFAULT_USE_NEON_DOTPROD      OFF)
set(DEFAULT_USE_WASM_SIMD         OFF)
set(DEFAULT_USE_WASM_SIMD_RELAXED OFF)

if(NOT CMAKE_CROSSCOMPILING)
    set(CMAKE_REQUIRED_FLAGS "-march=native")
    set(CMAKE_REQUIRED_QUIET TRUE)

    # ---- x86/AMD64 ----
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){ __m128i a = _mm_setzero_si128(); (void)a; return 0; }"
        HAS_SSE)
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){ __m256i a = _mm256_setzero_si256(); (void)a; return 0; }"
        HAS_AVX2)
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){ __m512i a = _mm512_setzero_si512(); (void)a; return 0; }"
        HAS_AVX512)
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){ unsigned long long r = _pext_u64(0xFFFFULL, 0xAAULL); (void)r; return 0; }"
        HAS_BMI2)
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){
            __m512i z = _mm512_setzero_si512();
            __m512i r = _mm512_dpbusd_epi32(z, z, z);
            (void)r; return 0;
        }"
        HAS_AVX512_VNNI)
    check_cxx_source_compiles("
        #include <immintrin.h>
        int main(){
            __m256i z = _mm256_setzero_si256();
            __m256i r = _mm256_dpbusd_avx_epi32(z, z, z);
            (void)r; return 0;
        }"
        HAS_AVX_VNNI)
    set(HAS_VNNI FALSE)
    if(HAS_AVX512_VNNI OR HAS_AVX_VNNI)
        set(HAS_VNNI TRUE)
    endif()

    # ---- ARM/ARM64 ----
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main(){ int32x4_t a = vdupq_n_s32(0); (void)a; return 0; }"
        HAS_NEON)
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main(){
            int32x4_t acc = vdupq_n_s32(0);
            int8x16_t  a  = vdupq_n_s8(0);
            int8x16_t  b  = vdupq_n_s8(0);
            int32x4_t  r  = vdotq_s32(acc, a, b);
            (void)r; return 0;
        }"
        HAS_NEON_DOTPROD)

    # ---- WebAssembly ----
    check_cxx_source_compiles("
        #include <wasm_simd128.h>
        int main(){ v128_t x = wasm_i32x4_splat(0); (void)x; return 0; }"
        HAS_WASM_SIMD)
    check_cxx_source_compiles("
        #include <wasm_simd128.h>
        int main(){
            v128_t acc = wasm_i32x4_splat(0);
            v128_t  a  = wasm_i8x16_splat(0);
            v128_t  b  = wasm_i8x16_splat(0);
            v128_t  r  = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(acc, a, b);
            (void)r; return 0;
        }"
        HAS_WASM_SIMD_RELAXED)

    bool_to_onoff(HAS_SSE               DEFAULT_USE_SSE)
    bool_to_onoff(HAS_AVX2              DEFAULT_USE_AVX2)
    bool_to_onoff(HAS_AVX512            DEFAULT_USE_AVX512)
    bool_to_onoff(HAS_BMI2              DEFAULT_USE_BMI2)
    bool_to_onoff(HAS_VNNI              DEFAULT_USE_VNNI)
    bool_to_onoff(HAS_NEON              DEFAULT_USE_NEON)
    bool_to_onoff(HAS_NEON_DOTPROD      DEFAULT_USE_NEON_DOTPROD)
    bool_to_onoff(HAS_WASM_SIMD         DEFAULT_USE_WASM_SIMD)
    bool_to_onoff(HAS_WASM_SIMD_RELAXED DEFAULT_USE_WASM_SIMD_RELAXED)

    message(STATUS "Detected SIMD/ISA capability of the host:")
    message(STATUS "  SSE:               ${DEFAULT_USE_SSE}")
    message(STATUS "  AVX2:              ${DEFAULT_USE_AVX2}")
    message(STATUS "  AVX512:            ${DEFAULT_USE_AVX512}")
    message(STATUS "  BMI2:              ${DEFAULT_USE_BMI2}")
    message(STATUS "  VNNI:              ${DEFAULT_USE_VNNI}")
    message(STATUS "  NEON:              ${DEFAULT_USE_NEON}")
    message(STATUS "  NEON_DOTPROD:      ${DEFAULT_USE_NEON_DOTPROD}")
    message(STATUS "  WASM_SIMD:         ${DEFAULT_USE_WASM_SIMD}")
    message(STATUS "  WASM_SIMD_RELAXED: ${DEFAULT_USE_WASM_SIMD_RELAXED}")
else()
    message(STATUS
        "Cross-compiling detected. Please specify the SIMD/ISA capability manually like -DUSE_AVX2=ON.")
endif()

#==========================================================
# CMake Options

option(NO_MULTI_THREADING "Disable multi-threading" OFF)
option(NO_COMMAND_MODULES "Disable command modules" OFF)
option(NO_PREFETCH "Disable prefetch in search" OFF)

option(USE_SSE  "Enable SSE2/SSSE3/SSE4.1 instruction" ${DEFAULT_USE_SSE})
option(USE_AVX2 "Enable AVX2/FMA instruction" ${DEFAULT_USE_AVX2})
option(USE_AVX512 "Enable AVX512 instruction" ${DEFAULT_USE_AVX512})
option(USE_BMI2 "Enable BMI2 instruction" ${DEFAULT_USE_BMI2})
option(USE_VNNI "Enable AVX512_VNNI/AVX_VNNI instruction" ${DEFAULT_USE_VNNI})
option(USE_NEON "Enable NEON instruction" ${DEFAULT_USE_NEON})
option(USE_NEON_DOTPROD "Enable NEON DotProd instruction" ${DEFAULT_USE_NEON_DOTPROD})
option(USE_WASM_SIMD "Enable WebAssembly SIMD instruction" ${DEFAULT_USE_WASM_SIMD})
option(USE_WASM_SIMD_RELAXED "Enable WebAssembly Relaxed SIMD instruction" ${DEFAULT_USE_WASM_SIMD_RELAXED})

option(USE_ORT_EVALUATOR "Enable ONNX runtime evaluator" OFF)
option(USE_ORT_GPU_EP "Enable ONNX GPU execution providers" OFF)
option(ORT_ROOTDIR "Root directory of onnxruntime" "")

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
    core/version.cpp

    database/dbclient.cpp
    database/dbutils.cpp
    database/dbtypes.cpp
    database/yxdbstorage.cpp

    eval/eval.cpp
    eval/evaluator.cpp
    eval/mix9svqnnue.cpp
    eval/mix10nnue.cpp

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
    search/mcts/node.cpp
    search/mcts/search.cpp

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

    database/cache.h
    database/dbclient.h
    database/dbstorage.h
    database/dbtypes.h
	database/dbutils.h
    database/yxdbstorage.h

    eval/crosscheck.h
    eval/eval.h
    eval/evaluator.h
    eval/mix9svqnnue.h
    eval/mix10nnue.h
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
    search/mcts/node.h
    search/mcts/nodetable.h
    search/mcts/searcher.h
    search/mcts/parameter.h

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
    add_subdirectory(external/thread-pool)
    target_link_libraries(rapfi PRIVATE flat.hpp zip libnpy BS_thread_pool)
endif()

#==========================================================
# Web Assembly compiling

if(EMSCRIPTEN)
    message(STATUS "Setting up build for Emscripten.")
    set(NETWORKS_DIR "${CMAKE_SOURCE_DIR}/../Networks")

    if(NOT NO_COMMAND_MODULES)
        message(FATAL_ERROR
            "Command modules is not supported for wasm build. "
            "Must set option NO_COMMAND_MODULES to ON.")
    endif()
    if(USE_SSE OR USE_AVX2 OR USE_AVX512 OR USE_BMI2 OR USE_VNNI)
        message(FATAL_ERROR
			"SSE, AVX2, AVX512, BMI2, VNNI instruction set is not supported for wasm build.")
    endif()

    # common flags for emscripten
    target_compile_options(rapfi PRIVATE "-fexceptions" PRIVATE "-flto")
    string(CONCAT EMCC_FLAGS
        "-fexceptions "
        "-flto "
        "--pre-js ${CMAKE_SOURCE_DIR}/emscripten/preamble.js "
        "--closure=1 "
        "-s MODULARIZE=1 "
        "-s EXPORT_NAME=Rapfi "
        "-s EXPORTED_FUNCTIONS=\"['_main','_gomocupLoopOnce','_emscripten_force_exit']\" "
        "-s EXPORTED_RUNTIME_METHODS=\"['cwrap']\" "
        "-s EXIT_RUNTIME=1 "
        "-s ALLOW_MEMORY_GROWTH=1 "
        "-s IMPORTED_MEMORY=1 "
        "-s STACK_SIZE=2MB "
    )

    # get the list of files to preload
    file(READ ${NETWORKS_DIR}/wasm_preloads.txt PRELOADS)
    string(REGEX REPLACE "\n" ";" PRELOADS "${PRELOADS}")
    foreach(PRELOAD ${PRELOADS})
        message(STATUS "Preloading: ${NETWORKS_DIR}/${PRELOAD}")
		string(CONCAT EMCC_FLAGS ${EMCC_FLAGS} "--preload-file \"${NETWORKS_DIR}/${PRELOAD}\" ")
    endforeach()

    set(WASM_OUTPUT_NAME "rapfi")
    if(NO_MULTI_THREADING)
        string(CONCAT WASM_OUTPUT_NAME ${WASM_OUTPUT_NAME} "-single")
    else()
        string(CONCAT WASM_OUTPUT_NAME ${WASM_OUTPUT_NAME} "-multi")
        target_compile_options(rapfi PRIVATE "-pthread")
        string(CONCAT EMCC_FLAGS ${EMCC_FLAGS}
            "-s USE_PTHREADS=1 "
            "-s PROXY_TO_PTHREAD=1 "
            "-s TEXTDECODER=0 "
        )
    endif()

    if(USE_WASM_SIMD OR USE_WASM_SIMD_RELAXED)
        string(CONCAT WASM_OUTPUT_NAME ${WASM_OUTPUT_NAME} "-simd128")
        target_compile_options(rapfi PRIVATE "-msimd128")
	    target_compile_definitions(rapfi PRIVATE USE_WASM_SIMD)
        if(USE_WASM_SIMD_RELAXED)
            string(CONCAT WASM_OUTPUT_NAME ${WASM_OUTPUT_NAME} "-relaxed")
            target_compile_options(rapfi PRIVATE "-mrelaxed-simd")
            target_compile_definitions(rapfi PRIVATE USE_WASM_SIMD_RELAXED)
        endif()
    endif()

    if(FORCE_ASSERTION)
        string(CONCAT EMCC_FLAGS ${EMCC_FLAGS} "-s ASSERTIONS=1 ")
    endif()
    if(CMAKE_BUILD_TYPE EQUAL "DEBUG")
        string(CONCAT EMCC_FLAGS ${EMCC_FLAGS}
            "--emit-symbol-map "
        )
    endif()

    set_target_properties(rapfi PROPERTIES
        OUTPUT_NAME ${WASM_OUTPUT_NAME}
        LINK_FLAGS ${EMCC_FLAGS}
    )

endif()

#==========================================================
# Onnx runtime linking

if(USE_ORT_EVALUATOR)
    if(NOT (EXISTS "${ORT_ROOTDIR}"))
	    message(FATAL_ERROR "Must provide a valid directory path to Onnx Runtime SDK.")
    endif()

    target_compile_definitions(rapfi PRIVATE USE_ORT_EVALUATOR)
    target_sources(rapfi PRIVATE
		eval/onnxevaluator.cpp
		eval/onnxevaluator.h
	)
	target_include_directories(rapfi PRIVATE "${ORT_ROOTDIR}/include")
	target_link_directories(rapfi PRIVATE "${ORT_ROOTDIR}/lib")
    target_link_libraries(rapfi PRIVATE "onnxruntime")

    if(USE_ORT_GPU_EP)
        target_compile_definitions(rapfi PRIVATE USE_ORT_GPU_EP)
		target_link_libraries(rapfi PRIVATE "onnxruntime_providers_shared")
		target_link_libraries(rapfi PRIVATE "onnxruntime_providers_cuda")
		target_link_libraries(rapfi PRIVATE "onnxruntime_providers_tensorrt")
    endif()
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
if(USE_SSE)
    target_compile_definitions(rapfi PRIVATE USE_SSE)
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
if(USE_VNNI)
	target_compile_definitions(rapfi PRIVATE USE_VNNI)
endif()
if(USE_NEON OR USE_NEON_DOTPROD)
    target_compile_definitions(rapfi PRIVATE USE_NEON)
endif()
if(USE_NEON_DOTPROD)
	target_compile_definitions(rapfi PRIVATE USE_NEON_DOTPROD)
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
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(STATUS "Setting up build for MSVC Clang-CL.")
        string(REGEX REPLACE "/O2" "/clang:-O3" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    else()
        message(STATUS "Setting up build for MSVC.")
        string(REGEX REPLACE "/O2" "/Ox" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    endif()

	if(USE_AVX512)
        # Fix SSE3, SSSE3, SSE4.1, AVX, AVX2, FMA, AVX512F, AVX512DQ, AVX512BW flags on MSVC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE3__ /D__SSSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /D__FMA__ /D__AVX512F__ /D__AVX512DQ__ /D__AVX512BW__")
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavx512f /clang:-mavx512dq /clang:-mavx512bw")
            if(USE_VNNI)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__AVX512VNNI__")
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavx512vnni")
            endif()
        endif()
    elseif(USE_AVX2)
        # Fix SSE3, SSSE3, SSE4.1, AVX, AVX2, FMA flags on MSVC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE3__ /D__SSSE3__ /D__SSE4_1__ /D__AVX__ /D__AVX2__ /D__FMA__")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
			set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavx2 /clang:-mfma")
            if(USE_VNNI)
                set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-mavxvnni")
            endif()
        endif()
    elseif(USE_SSE)
        # Fix SSE3, SSSE3, SSE4.1 flags on MSVC
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__SSE3__ /D__SSSE3__ /D__SSE4_1__")
        if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-msse /clang:-msse2 /clang:-msse3 /clang:-mssse3 /clang:-msse4 /clang:-msse4.1")
        endif()
    endif()

    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        if(USE_NEON_DOTPROD)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-march=armv8.2-a+dotprod")
	    elseif(USE_NEON)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /clang:-march=armv8-a+simd")
        endif()
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
        if(USE_VNNI)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vnni -mavx512vl")
        endif()
    elseif(USE_AVX2)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
        if(USE_VNNI)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavxvnni")
        endif()
    elseif(USE_SSE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -mssse3 -msse4 -msse4.1")
    endif()

    if(USE_BMI2 OR USE_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mbmi2")
    endif()

    if(USE_NEON_DOTPROD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod")
	elseif(USE_NEON)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    endif()

    if(FORCE_ASSERTION)
        string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
        string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}})
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-variable -Wno-unused-function -Wno-missing-braces -Wno-logical-op-parentheses -Wno-parentheses -Wno-sign-compare")

	if(USE_ASAN)
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
	endif()

    if(WIN32)
        if(NO_COMMAND_MODULES)
            target_link_libraries(rapfi PRIVATE -static)
        endif()
    elseif(NOT EMSCRIPTEN)
        if(NO_MULTI_THREADING AND NO_COMMAND_MODULES)
            target_link_libraries(rapfi PRIVATE -static)
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
            target_link_libraries(rapfi PRIVATE pthread)
        else()
		    target_link_libraries(rapfi PRIVATE pthread atomic)
        endif()
    endif()
endif()

#==========================================================
# IDE Helper

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Sources" FILES ${CORE_SOURCES} ${MODULE_SOURCES})
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "Headers" FILES ${HEADERS})
