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

#pragma once

#if defined(USE_BMI2) || defined(USE_AVX512)
    #include <immintrin.h>  // BMI2, AVX512
#endif

#if !defined(NO_PREFETCH) && (defined(__INTEL_COMPILER) || defined(_MSC_VER))
    #include <xmmintrin.h>  // SSE
#endif

#ifdef USE_ROTR
    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
        #include <x86intrin.h>
    #else
        #include <cstdlib>
    #endif
#endif

#include <cstdint>

// Define some macros for platform specific optimization hint
#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
    #define FORCE_INLINE inline __attribute__((always_inline))
    #define NO_INLINE    __attribute__((noinline))
    #define RESTRICT     __restrict__
    #define LIKELY(x)    __builtin_expect(!!(x), 1)
    #define UNLIKELY(x)  __builtin_expect(!!(x), 0)
#elif defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
    #define NO_INLINE    __declspec(noinline)
    #define RESTRICT     __restrict
    #define LIKELY(x)    (x)
    #define UNLIKELY(x)  (x)
#else
    #define FORCE_INLINE inline
    #define NO_INLINE
    #define RESTRICT
    #define LIKELY(x)   (x)
    #define UNLIKELY(x) (x)
#endif

// -------------------------------------------------
// Platform related functions

/// A right logical shift function that supports negetive shamt.
/// It might be implemented as rotr64 to avoid conditional branch.
inline uint64_t rightShift(uint64_t x, int shamt)
{
#ifdef USE_ROTR
    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
    return __rorq(x, shamt);
    #else
    return _rotr64(x, shamt);
    #endif
#else
    return shamt < 0 ? x << (-shamt) : x >> shamt;
#endif
}

/// mulhi64() returns the higher 64 bits from two 64 bits multiply.
/// @see stackoverflow "getting-the-high-part-of-64-bit-integer-multiplication".

#ifdef __SIZEOF_INT128__  // GNU C

inline uint64_t mulhi64(uint64_t a, uint64_t b)
{
    return ((unsigned __int128)a * (unsigned __int128)b) >> 64;
}

#elif defined(_M_X64) || defined(_M_ARM64)  // MSVC

    // MSVC for x86-64 or AArch64
    // possibly also  || defined(_M_IA64) || defined(_WIN64)
    // but the docs only guarantee x86-64!  Don't use *just* _WIN64;
    // it doesn't include AArch64 Android / Linux
    // https://docs.microsoft.com/en-gb/cpp/intrinsics/umulh
    #include <intrin.h>
    #define mulhi64 __umulh

#elif defined(_M_IA64)  // || defined(_M_ARM)       // MSVC again

    // https://docs.microsoft.com/en-gb/cpp/intrinsics/umul128
    // incorrectly say that _umul128 is available for ARM
    // which would be weird because there's no single insn on AArch32
    #include <intrin.h>
inline uint64_t mulhi64(uint64_t a, uint64_t b)
{
    unsigned __int64 HighProduct;
    (void)_umul128(a, b, &HighProduct);
    return HighProduct;
}

#else

inline uint64_t mulhi64(uint64_t a, uint64_t b)
{
    uint64_t aL = (uint32_t)a, aH = a >> 32;
    uint64_t bL = (uint32_t)b, bH = b >> 32;
    uint64_t c1 = (aL * bL) >> 32;
    uint64_t c2 = aH * bL + c1;
    uint64_t c3 = aL * bH + (uint32_t)c2;
    return aH * bH + (c2 >> 32) + (c3 >> 32);
}

#endif

/// Preloads the given address in L1/L2 cache. This is a non-blocking
/// function that doesn't stall the CPU waiting for data to be loaded
/// from memory, which can be quite slow.
inline void prefetch(const void *addr)
{
#ifndef NO_PREFETCH

    #if defined(__INTEL_COMPILER)
    // This hack prevents prefetches from being optimized away by
    // Intel compiler. Both MSVC and gcc seem not be affected by this.
    __asm__("");
    #endif

    #if defined(__INTEL_COMPILER) || defined(_MSC_VER)
    _mm_prefetch((char *)addr, _MM_HINT_T0);
    #else
    __builtin_prefetch(addr);
    #endif

#endif
}

namespace _PrefetchImpl {

template <int N>
struct PrefetchImpl
{};

template <>
struct PrefetchImpl<1>
{
    inline static void call(const char *addr) { ::prefetch(addr); }
};

template <>
struct PrefetchImpl<2>
{
    inline static void call(const char *addr)
    {
        ::prefetch(addr);
        ::prefetch(addr + 64);
    }
};

template <>
struct PrefetchImpl<3>
{
    inline static void call(const char *addr)
    {
        ::prefetch(addr);
        ::prefetch(addr + 64);
        ::prefetch(addr + 128);
    }
};

template <>
struct PrefetchImpl<4>
{
    inline static void call(const char *addr)
    {
        ::prefetch(addr);
        ::prefetch(addr + 64);
        ::prefetch(addr + 128);
        ::prefetch(addr + 192);
    }
};

template <>
struct PrefetchImpl<5>
{
    inline static void call(const char *addr)
    {
        ::prefetch(addr);
        ::prefetch(addr + 64);
        ::prefetch(addr + 128);
        ::prefetch(addr + 192);
        ::prefetch(addr + 256);
    }
};

template <>
struct PrefetchImpl<6>
{
    inline static void call(const char *addr)
    {
        ::prefetch(addr);
        ::prefetch(addr + 64);
        ::prefetch(addr + 128);
        ::prefetch(addr + 192);
        ::prefetch(addr + 256);
        ::prefetch(addr + 320);
    }
};

template <>
struct PrefetchImpl<7>
{
    inline static void call(const char *addr)
    {
        ::prefetch(addr);
        ::prefetch(addr + 64);
        ::prefetch(addr + 128);
        ::prefetch(addr + 192);
        ::prefetch(addr + 256);
        ::prefetch(addr + 320);
        ::prefetch(addr + 384);
    }
};

template <>
struct PrefetchImpl<8>
{
    inline static void call(const char *addr)
    {
        ::prefetch(addr);
        ::prefetch(addr + 64);
        ::prefetch(addr + 128);
        ::prefetch(addr + 192);
        ::prefetch(addr + 256);
        ::prefetch(addr + 320);
        ::prefetch(addr + 384);
        ::prefetch(addr + 448);
    }
};

}  // namespace _PrefetchImpl

template <int NumBytes>
inline void multiPrefetch(const void *addr)
{
    constexpr int CacheLineSize = 64;
    constexpr int NumCacheLines = (NumBytes + CacheLineSize - 1) / CacheLineSize;
    _PrefetchImpl::PrefetchImpl<NumCacheLines>::call(reinterpret_cast<const char *>(addr));
}

// -------------------------------------------------
// NUMA-aware helper

namespace WinProcGroup {

/// Under Windows it is not possible for a process to run on more than one
/// logical processor group. This usually means to be limited to use max 64
/// cores. To overcome this, some special platform specific API should be
/// called to set group affinity for each thread. Original code from Texel by
/// Peter Österlund.
void bindThisThread(size_t idx);

}  // namespace WinProcGroup

// -------------------------------------------------
// Large-Page memory allocator

namespace MemAlloc {

/// A warpper around std::aligned_alloc, which uses system calls if available.
/// Memory allocated using this function should be freed with alignedFree().
void *alignedAlloc(size_t alignment, size_t size);
/// Free memory allocated by alignedAlloc().
void alignedFree(void *ptr);
/// A helper to alloc an aligned array of type T. Ptr should be freed with alignedFree().
template <typename T, size_t Alignment = alignof(T)>
T *alignedArrayAlloc(size_t arraySize)
{
    return reinterpret_cast<T *>(alignedAlloc(Alignment, sizeof(T) * arraySize));
}

/// Allocate large page memory, with min alignment 4KiB. Memory allocated
/// using this function should be freed with alignedLargePageFree().
void *alignedLargePageAlloc(size_t size);

/// Free memory allocated by alignedLargePageAlloc().
void alignedLargePageFree(void *ptr);

}  // namespace MemAlloc
