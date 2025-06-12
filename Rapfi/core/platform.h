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

#if defined(USE_NEON)
    #include <arm_neon.h>  // NEON
#endif

#if defined(USE_WASM_SIMD)
    #include <wasm_simd128.h>  // WASM SIMD128
#endif

#if defined(_MSC_VER)
    #include <cstdlib>   // for _rotr64
    #include <intrin.h>  // for __umulh, _mm_prefetch, __prefetch
#endif

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>

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

/// popcount(x) – portable population-count (Hamming weight) for 64-bit words.
/// Tries to use the fastest compiler/CPU intrinsic that is available at
/// compile time and falls back to a branch-free SWAR algorithm otherwise.
/// • GCC / Clang  →  __builtin_popcountll
/// • MSVC x64     →  __popcnt64
/// • C++20        →  std::popcount
/// • Fallback     →  SWAR (Parallel bit-count) algorithm
inline int popcount(uint64_t x)
{
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    return std::popcount(x);
#elif defined(__clang__) || defined(__GNUC__)
    return __builtin_popcountll(x);
#elif defined(_MSC_VER) && defined(_M_X64)
    return static_cast<int>(__popcnt64(x));
#else
    // 64-bit variant of “Hacker’s Delight” 5-step popcount
    x -= (x >> 1) & 0x5555'5555'5555'5555ULL;
    x = (x & 0x3333'3333'3333'3333ULL) + ((x >> 2) & 0x3333'3333'3333'3333ULL);
    x = (x + (x >> 4)) & 0x0F0F'0F0F'0F0F'0F0FULL;
    return static_cast<int>((x * 0x0101'0101'0101'0101ULL) >> 56);
#endif
}

/// A right logical shift function that supports negetive shamt.
/// It might be implemented as rotr64 to avoid conditional branch.
inline uint64_t rotr(uint64_t x, int shamt)
{
#if defined(__clang__)
    return __builtin_rotateright64(x, shamt);
#elif defined(_MSC_VER)
    return _rotr64(x, shamt);
#else
    shamt &= 63;
    return (x << (64 - shamt)) | (x >> shamt);
#endif
}

/// mulhi64() returns the higher 64 bits from two 64 bits multiply.
/// @see stackoverflow "getting-the-high-part-of-64-bit-integer-multiplication".

#ifdef __SIZEOF_INT128__  // GNU C

inline uint64_t mulhi64(uint64_t a, uint64_t b)
{
    return ((unsigned __int128)a * (unsigned __int128)b) >> 64;
}

#elif defined(_M_X64) || defined(_M_ARM64)  // MSVC for x86-64 or AArch64

    #define mulhi64 __umulh

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
    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
    __builtin_prefetch(addr);
    #elif defined(_M_ARM) || defined(_M_ARM64)
    __prefetch(addr);
    #else
    _mm_prefetch((char *)addr, _MM_HINT_T0);
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

namespace Numa {

/// NumaNodeId is a type representing a NUMA node ID.
typedef int32_t NumaNodeId;

/// Default NUMA node ID, used when NUMA is not supported or not available.
constexpr NumaNodeId DefaultNumaNodeId = 0;

/// The threshold for the number of bind groups. If the number of threads is
/// greater than this threshold, we will bind threads to NUMA groups to
/// improve performance. This is a heuristic value to determine when to
/// use NUMA-aware logic.
constexpr int BindGroupThreshold = 8;

/// Under Windows it is not possible for a process to run on more than one
/// logical processor group. This usually means to be limited to use max 64
/// cores. To overcome this, some special platform specific API should be
/// called to set group affinity for each thread. Original code from Texel by
/// Peter Österlund. We also let this function return the numa node ID for
/// the thread, to allow NUMA-aware logics in the thread.
NumaNodeId bindThisThread(size_t idx);

}  // namespace Numa

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

template <typename T>
struct LargePageDeleter
{
    void operator()(T *ptr) const
    {
        if (!ptr)
            return;

        // Explicitly needed to call the destructor
        if constexpr (!std::is_trivially_destructible_v<T>)
            ptr->~T();

        MemAlloc::alignedLargePageFree(ptr);
    }
};

template <typename T>
using LargePagePtr = std::unique_ptr<T, LargePageDeleter<T>>;

// make_unique_large_page for single objects
template <typename T, typename... Args>
LargePagePtr<T> make_unique_large_page(Args &&...args)
{
    static_assert(alignof(T) <= 4096,
                  "alignedLargePageAlloc() may fail for such a big alignment requirement of T");
    void *raw_memory = MemAlloc::alignedLargePageAlloc(sizeof(T));
    T    *obj        = new (raw_memory) T(std::forward<Args>(args)...);
    return LargePagePtr<T>(obj);
}
