/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "version.h"

#include <iomanip>
#include <sstream>

#define RAPFI_MAJOR_VER    0
#define RAPFI_MINOR_VER    43
#define RAPFI_REVISION_VER 2

// Two-step stringify so macro arguments are themselves expanded before becoming text. Used to
// format compiler version macros like __clang_major__.
#define STRINGIFY(s)         #s
#define VERSION_STR(a, b, c) STRINGIFY(a) "." STRINGIFY(b) "." STRINGIFY(c)
#define VERSION_STR2(a, b)   STRINGIFY(a) "." STRINGIFY(b)

std::tuple<int, int, int> getVersionNumbers()
{
    return {RAPFI_MAJOR_VER, RAPFI_MINOR_VER, RAPFI_REVISION_VER};
}

/// One-liner describing the toolchain, target OS, and enabled instruction-set features that
/// the binary was built with. Appended in parentheses to the version string.
static std::string getBuildInfo()
{
    std::stringstream ss;

#if defined(__INTEL_LLVM_COMPILER)
    ss << "ICX " << STRINGIFY(__INTEL_LLVM_COMPILER);
#elif defined(__clang__)
    ss << "clang++ " << VERSION_STR(__clang_major__, __clang_minor__, __clang_patchlevel__);
#elif _MSC_VER
    ss << "MSVC " << VERSION_STR2(_MSC_FULL_VER, _MSC_BUILD);
#elif __GNUC__
    ss << "g++ " << VERSION_STR(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    ss << "Unknown compiler";
#endif

#if defined(__APPLE__)
    ss << " on Apple";
#elif defined(__MINGW64__)
    ss << " on MinGW64";
#elif defined(__ANDROID__)
    ss << " on Android";
#elif defined(__linux__)
    ss << " on Linux";
#elif defined(_WIN64) || defined(_WIN32)
    ss << " on Windows";
#elif defined(__EMSCRIPTEN__)
    ss << " on Wasm";
#else
    ss << " on unknown system";
#endif

#if defined(USE_SSE)
    ss << " SSE41";
#endif
#if defined(USE_AVX2)
    ss << " AVX2";
#endif
#if defined(USE_AVX512)
    ss << " AVX512";
#endif
#if defined(USE_BMI2)
    ss << " BMI2";
#endif
#if defined(USE_VNNI)
    ss << " VNNI";
#endif
#if defined(USE_NEON_DOTPROD)
    ss << " NEON_DOTPROD";
#elif defined(USE_NEON)
    ss << " NEON";
#endif
#if defined(USE_WASM_SIMD_RELAXED)
    ss << " WASM_SIMD_RELAXED";
#elif defined(USE_WASM_SIMD)
    ss << " WASM_SIMD";
#endif

#if !defined(MULTI_THREADING)
    ss << " SINGLE_THREAD";
#endif
#if defined(NO_PREFETCH)
    ss << " NO_PREFETCH";
#endif
#if !defined(NDEBUG)
    ss << " DEBUG";
#endif

    return ss.str();
}

std::string getVersionInfo()
{
    // Revision is zero-padded to two digits (e.g. "0.43.01") at runtime, so the numeric
    // RAPFI_REVISION_VER macro stays a plain decimal and can safely exceed 7.
    std::stringstream ss;
    ss << RAPFI_MAJOR_VER << '.' << RAPFI_MINOR_VER << '.' << std::setfill('0') << std::setw(2)
       << RAPFI_REVISION_VER << " (" << getBuildInfo() << ')';
    return ss.str();
}

std::string getEngineInfo()
{
    std::stringstream ss;
    ss << "name=\"Rapfi\", ";
    ss << "version=\"" << getVersionInfo() << "\", ";
    ss << "author=\"Rapfi developers (see AUTHORS file)\", ";
    ss << "country=\"China\"";
    return ss.str();
}
