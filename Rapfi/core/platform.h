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
    #include <cstdlib>   // _rotr64
    #include <intrin.h>  // __umulh, _mm_prefetch, __prefetch
#endif

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

// -------------------------------------------------
// Compiler hint macros

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
// Bit / integer intrinsics

/// 64-bit population count (Hamming weight). Picks the fastest available implementation:
/// `std::popcount` (C++20) -> `__builtin_popcountll` (GCC/Clang) -> `__popcnt64` (MSVC x64)
/// -> Hacker's Delight SWAR fallback.
inline int popcount(uint64_t x)
{
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    return std::popcount(x);
#elif defined(__clang__) || defined(__GNUC__)
    return __builtin_popcountll(x);
#elif defined(_MSC_VER) && defined(_M_X64)
    return static_cast<int>(__popcnt64(x));
#else
    x -= (x >> 1) & 0x5555'5555'5555'5555ULL;
    x = (x & 0x3333'3333'3333'3333ULL) + ((x >> 2) & 0x3333'3333'3333'3333ULL);
    x = (x + (x >> 4)) & 0x0F0F'0F0F'0F0F'0F0FULL;
    return static_cast<int>((x * 0x0101'0101'0101'0101ULL) >> 56);
#endif
}

/// Index of the least-significant set bit (count of trailing zeros). Undefined for x == 0.
/// Picks the fastest available: `std::countr_zero` (C++20) -> `__builtin_ctzll` (GCC/Clang)
/// -> `_BitScanForward64` (MSVC x64) -> a de Bruijn fallback.
inline int lsb(uint64_t x)
{
    assert(x != 0);
#if defined(__cpp_lib_bitops) && __cpp_lib_bitops >= 201907L
    return std::countr_zero(x);
#elif defined(__clang__) || defined(__GNUC__)
    return __builtin_ctzll(x);
#elif defined(_MSC_VER) && defined(_M_X64)
    unsigned long index;
    _BitScanForward64(&index, x);
    return static_cast<int>(index);
#else
    // De Bruijn sequence multiply-and-lookup: isolate the lowest bit, map its product's top 6
    // bits to a position via a perfect-hash table. See Knuth TAOCP 4A / Leiserson et al. 1998.
    static constexpr int Index64[64] = {
        0,  1,  48, 2,  57, 49, 28, 3,  61, 58, 50, 42, 38, 29, 17, 4,  62, 55, 59, 36, 53,
        51, 43, 22, 45, 39, 33, 30, 24, 18, 12, 5,  63, 47, 56, 27, 60, 41, 37, 16, 54, 35,
        52, 21, 44, 32, 23, 11, 46, 26, 40, 15, 34, 20, 31, 10, 25, 14, 19, 9,  13, 8,  7,  6};
    return Index64[((x & (~x + 1)) * 0x03f79d71b4cb0a89ULL) >> 58];
#endif
}

/// Return the index of the least-significant set bit of `b` and clear it from `b`. Undefined for
/// b == 0. Lets callers iterate set bits without repeating the clear-lowest-bit step at each site.
inline int pop_lsb(uint64_t &b)
{
    assert(b);
    const int s = lsb(b);
    b &= b - 1;
    return s;
}

/// 64-bit rotate right that also accepts negative shift amounts (which rotate left). Lowered
/// to a single instruction on x86 and ARM via the compiler's rotate intrinsic when available.
inline uint64_t rotr(uint64_t x, int shamt)
{
#if defined(__clang__)
    return __builtin_rotateright64(x, shamt);
#elif defined(_MSC_VER)
    return _rotr64(x, shamt);
#else
    shamt &= 63;
    if (shamt == 0)
        return x;  // a 64-bit shift is undefined behaviour; rotating by 0 is identity
    return (x << (64 - shamt)) | (x >> shamt);
#endif
}

/// High 64 bits of a 64*64 -> 128 unsigned multiply. Used in TT bucket addressing.
/// @see https://stackoverflow.com/q/28868367
#ifdef __SIZEOF_INT128__

inline uint64_t mulhi64(uint64_t a, uint64_t b)
{
    return ((unsigned __int128)a * (unsigned __int128)b) >> 64;
}

#elif defined(_M_X64) || defined(_M_ARM64)

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

// -------------------------------------------------
// Memory prefetch

/// Bring `addr` into L1/L2 ahead of an upcoming load. Non-blocking; the CPU does not stall
/// waiting for the line to arrive. Compiled to a no-op when built with `NO_PREFETCH`.
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

namespace detail {

template <size_t... I>
inline void multiPrefetchImpl(const char *addr, std::index_sequence<I...>)
{
    constexpr size_t CacheLineSize = 64;
    (prefetch(addr + I * CacheLineSize), ...);
}

}  // namespace detail

/// Prefetch a contiguous `NumBytes`-byte region, one prefetch per 64-byte cache line. The
/// per-line calls are emitted via a parameter-pack expansion and inline to the same sequence
/// the hand-unrolled version produced.
template <int NumBytes>
inline void multiPrefetch(const void *addr)
{
    constexpr int CacheLineSize = 64;
    constexpr int NumCacheLines = (NumBytes + CacheLineSize - 1) / CacheLineSize;
    detail::multiPrefetchImpl(reinterpret_cast<const char *>(addr),
                              std::make_index_sequence<NumCacheLines> {});
}

// -------------------------------------------------
// NUMA awareness

namespace Numa {

using NumaNodeId = int32_t;

/// Returned by `bindThisThread` when NUMA is unsupported or binding fails.
constexpr NumaNodeId DefaultNumaNodeId = 0;

/// If the thread pool grows above this many threads, threads are spread across NUMA nodes
/// (and on Windows, across processor groups - one process is otherwise capped at 64 cores).
constexpr int BindGroupThreshold = 8;

/// Pin the calling thread to a NUMA node chosen by `idx` (round-robin across detected nodes)
/// and return that node's id. On platforms without NUMA support, returns DefaultNumaNodeId
/// without changing affinity. Original Windows implementation adapted from Texel by Peter
/// Österlund.
NumaNodeId bindThisThread(size_t idx);

}  // namespace Numa

// -------------------------------------------------
// Aligned and large-page allocation

namespace MemAlloc {

/// Aligned allocation backed by the platform's best aligned-allocator (posix_memalign /
/// _aligned_malloc / std::aligned_alloc). Free with `alignedFree`.
void *alignedAlloc(size_t alignment, size_t size);

/// Free memory returned by `alignedAlloc`.
void alignedFree(void *ptr);

/// Aligned allocation for an array of `arraySize` `T`s. Free with `alignedFree`.
template <typename T, size_t Alignment = alignof(T)>
T *alignedArrayAlloc(size_t arraySize)
{
    return reinterpret_cast<T *>(alignedAlloc(Alignment, sizeof(T) * arraySize));
}

/// Allocate `size` bytes backed by large pages where the OS permits it (Windows large pages,
/// Linux transparent huge pages via madvise), falling back to a 4KiB / page-aligned
/// allocation otherwise. Always 4KiB-aligned at minimum. Free with `alignedLargePageFree`.
void *alignedLargePageAlloc(size_t size);

/// Free memory returned by `alignedLargePageAlloc`.
void alignedLargePageFree(void *ptr);

}  // namespace MemAlloc

/// Deleter for `unique_ptr` of objects allocated through `MemAlloc::alignedLargePageAlloc`.
/// Calls the destructor (unless `T` is trivially destructible) and then releases the pages.
template <typename T>
struct LargePageDeleter
{
    void operator()(T *ptr) const
    {
        if (!ptr)
            return;

        if constexpr (!std::is_trivially_destructible_v<T>)
            ptr->~T();

        MemAlloc::alignedLargePageFree(ptr);
    }
};

template <typename T>
using LargePagePtr = std::unique_ptr<T, LargePageDeleter<T>>;

/// Construct a single `T` in large-page memory and return owning `LargePagePtr<T>`.
template <typename T, typename... Args>
LargePagePtr<T> make_unique_large_page(Args &&...args)
{
    static_assert(alignof(T) <= 4096,
                  "alignedLargePageAlloc() may fail for such a big alignment requirement of T");
    void *raw_memory = MemAlloc::alignedLargePageAlloc(sizeof(T));
    if (!raw_memory)
        throw std::bad_alloc {};  // never placement-new into a null pointer
    T *obj = new (raw_memory) T(std::forward<Args>(args)...);
    return LargePagePtr<T>(obj);
}
