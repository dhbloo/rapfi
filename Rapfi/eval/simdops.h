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
#include "../core/platform.h"
#include "../core/utils.h"

#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#include <tuple>
#include <type_traits>

namespace Evaluation::simd {

enum InstructionType {
    SCALAR,
    SSE,
    AVX2,
    AVX512,
};

/// Get simd register width of the given instruction type.
constexpr size_t simdBitsOfInstType(InstructionType instType)
{
    switch (instType) {
    default: return 128;
    case SSE: return 128;
    case AVX2: return 256;
    case AVX512: return 512;
    }
}

#if defined(USE_AVX512)
constexpr size_t          NativeAlignment = 64;
constexpr InstructionType NativeInstType  = AVX512;
#elif defined(USE_AVX2)
constexpr size_t          NativeAlignment = 32;
constexpr InstructionType NativeInstType  = AVX2;
#elif defined(USE_SSE)
constexpr size_t          NativeAlignment = 16;
constexpr InstructionType NativeInstType  = SSE;
#else  // Delegate to SSE with simde's implementation
constexpr size_t          NativeAlignment = 16;
constexpr InstructionType NativeInstType  = SCALAR;
#endif

constexpr bool isAlignSizeOK(size_t alignSize)
{
    return alignSize > 0 && alignSize <= 64 && isPowerOfTwo(alignSize);
}

template <size_t AlignSize, typename T>
constexpr bool isPtrAligned(const T *pointer)
{
    static_assert(isAlignSizeOK(AlignSize), "AlignSize is not valid");
    return (reinterpret_cast<uintptr_t>(pointer) & (AlignSize - 1)) == 0;
}

template <size_t AlignSize, typename T>
constexpr size_t alignDimSize(size_t dimSize)
{
    size_t alignBytes = std::max<size_t>(AlignSize / sizeof(T), 1);
    return alignBytes * ((dimSize + alignBytes - 1) / alignBytes);
}

namespace detail {

    template <typename...>
    inline constexpr bool always_false_v = false;

    // ------------------------------------------------------------------------
    // Vec regwidth & batch num definition

    template <size_t Size, typename T, InstructionType I, bool AllowExtra = false>
    struct VecBatch
    {
        static constexpr size_t SimdBits    = simdBitsOfInstType(I);
        static constexpr size_t RegWidth    = (SimdBits / 8) / sizeof(T);
        static constexpr size_t NumBatch    = Size / RegWidth;
        static constexpr size_t NumExtra    = Size % RegWidth;
        static constexpr size_t BatchedSize = NumBatch * RegWidth;

        static_assert(AllowExtra || NumExtra == 0, "data does not fill a register");
    };

    // ------------------------------------------------------------------------
    // Vec store & load template

    template <typename T, int Alignment, InstructionType I, typename Enabled = void>
    struct VecLoadStore
    {};

    template <typename T, int Alignment>
    struct VecLoadStore<T, Alignment, SSE, std::enable_if_t<std::is_integral_v<T>>>
    {
        static FORCE_INLINE auto load(const void *addr)
        {
            if constexpr (Alignment >= 16)
                return simde_mm_load_si128(reinterpret_cast<const simde__m128i *>(addr));
            else
                return simde_mm_loadu_si128(addr);
        }

        static FORCE_INLINE void store(void *addr, simde__m128i data)
        {
            if constexpr (Alignment >= 16)
                simde_mm_store_si128(reinterpret_cast<simde__m128i *>(addr), data);
            else
                simde_mm_storeu_si128(addr, data);
        }
    };

    template <typename T, int Alignment>
    struct VecLoadStore<T, Alignment, AVX2, std::enable_if_t<std::is_integral_v<T>>>
    {
        static FORCE_INLINE auto load(const void *addr)
        {
            if constexpr (Alignment >= 32)
                return simde_mm256_load_si256(reinterpret_cast<const simde__m256i *>(addr));
            else
                return simde_mm256_loadu_si256(addr);
        }

        static FORCE_INLINE void store(void *addr, simde__m256i data)
        {
            if constexpr (Alignment >= 32)
                simde_mm256_store_si256(reinterpret_cast<simde__m256i *>(addr), data);
            else
                simde_mm256_storeu_si256(addr, data);
        }
    };

#ifdef USE_AVX512
    template <typename T, int Alignment>
    struct VecLoadStore<T, Alignment, AVX512, std::enable_if_t<std::is_integral_v<T>>>
    {
        static FORCE_INLINE auto load(const void *addr)
        {
            if constexpr (Alignment >= 64)
                return _mm512_load_si512(reinterpret_cast<const __m512i *>(addr));
            else
                return _mm512_loadu_si512(addr);
        }

        static FORCE_INLINE void store(void *addr, __m512i data)
        {
            if constexpr (Alignment >= 64)
                _mm512_store_si512(reinterpret_cast<__m512i *>(addr), data);
            else
                _mm512_storeu_si512(addr, data);
        }
    };
#endif

    template <int Alignment>
    struct VecLoadStore<float, Alignment, SSE>
    {
        static FORCE_INLINE auto load(const float *addr)
        {
            if constexpr (Alignment >= 16)
                return simde_mm_load_ps(addr);
            else
                return simde_mm_loadu_ps(addr);
        }

        static FORCE_INLINE void store(float *addr, simde__m128 data)
        {
            if constexpr (Alignment >= 16)
                simde_mm_store_ps(reinterpret_cast<simde_float32 *>(addr), data);
            else
                simde_mm_storeu_ps(addr, data);
        }
    };

    template <int Alignment>
    struct VecLoadStore<float, Alignment, AVX2>
    {
        static FORCE_INLINE auto load(const float *addr)
        {
            if constexpr (Alignment >= 32)
                return simde_mm256_load_ps(addr);
            else
                return simde_mm256_loadu_ps(addr);
        }

        static FORCE_INLINE void store(float *addr, simde__m256 data)
        {
            if constexpr (Alignment >= 32)
                simde_mm256_store_ps(reinterpret_cast<simde_float32 *>(addr), data);
            else
                simde_mm256_storeu_ps(addr, data);
        }
    };

#ifdef USE_AVX512
    template <int Alignment>
    struct VecLoadStore<float, Alignment, AVX512>
    {
        static FORCE_INLINE auto load(const float *addr)
        {
            if constexpr (Alignment >= 64)
                return _mm512_load_ps(addr);
            else
                return _mm512_loadu_ps(addr);
        }

        static FORCE_INLINE void store(float *addr, __m512 data)
        {
            if constexpr (Alignment >= 64)
                _mm512_store_ps(addr, data);
            else
                _mm512_storeu_ps(addr, data);
        }
    };
#endif

    template <typename T, int Alignment>
    struct VecLoadStore<T, Alignment, SCALAR> : VecLoadStore<T, Alignment, SSE>
    {};

    // ------------------------------------------------------------------------
    // Vec type conversion template

    /// Convert vector register from FT type to TT type.
    template <typename FT, typename TT, InstructionType I, typename Enabled = void>
    struct VecCvt
    {};

    template <typename FT, typename TT>
    struct VecCvt<FT, TT, SSE, std::enable_if_t<std::is_integral_v<TT>>>
    {
        typedef simde__m128i FR;
        typedef simde__m128i TR;

        static FORCE_INLINE TR convert1(FR a)
        {
            if constexpr (std::is_same_v<FT, int8_t>) {
                if constexpr (std::is_same_v<TT, int16_t>)
                    return simde_mm_cvtepi8_epi16(a);
                else if constexpr (std::is_same_v<TT, int32_t>)
                    return simde_mm_cvtepi8_epi32(a);
                else if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm_cvtepi8_epi64(a);
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int8_t");
            }
            else if constexpr (std::is_same_v<FT, int16_t>) {
                if constexpr (std::is_same_v<TT, int32_t>)
                    return simde_mm_cvtepi16_epi32(a);
                else if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm_cvtepi16_epi64(a);
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int16_t");
            }
            else if constexpr (std::is_same_v<FT, int32_t>) {
                if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm_cvtepi32_epi64(a);
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int32_t");
            }
            else
                static_assert(always_false_v<FT>, "unsupported convert1 from type");
        }

        static FORCE_INLINE auto convert(TR a)
        {
            if constexpr (sizeof(TT) / sizeof(FT) == 2)
                return std::tuple(convert1(a), convert1(simde_mm_srli_si128(a, 8)));
            else if constexpr (sizeof(TT) / sizeof(FT) == 4)
                return std::tuple(convert1(a),
                                  convert1(simde_mm_srli_si128(a, 4)),
                                  convert1(simde_mm_srli_si128(a, 8)),
                                  convert1(simde_mm_srli_si128(a, 12)));
            else if constexpr (sizeof(TT) / sizeof(FT) == 8)
                return std::tuple(convert1(a),
                                  convert1(simde_mm_srli_si128(a, 2)),
                                  convert1(simde_mm_srli_si128(a, 4)),
                                  convert1(simde_mm_srli_si128(a, 6)),
                                  convert1(simde_mm_srli_si128(a, 8)),
                                  convert1(simde_mm_srli_si128(a, 10)),
                                  convert1(simde_mm_srli_si128(a, 12)),
                                  convert1(simde_mm_srli_si128(a, 14)));
            else
                static_assert(always_false_v<FT, TT>, "unsupported convert type");
        }
    };

    template <typename FT, typename TT>
    struct VecCvt<FT, TT, AVX2, std::enable_if_t<std::is_integral_v<TT>>>
    {
        typedef simde__m128i FR;
        typedef simde__m256i TR;

        static FORCE_INLINE TR convert1(FR a)
        {
            if constexpr (std::is_same_v<FT, int8_t>) {
                if constexpr (std::is_same_v<TT, int16_t>)
                    return simde_mm256_cvtepi8_epi16(a);
                else if constexpr (std::is_same_v<TT, int32_t>)
                    return simde_mm256_cvtepi8_epi32(a);
                else if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm256_cvtepi8_epi64(a);
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int8_t");
            }
            else if constexpr (std::is_same_v<FT, int16_t>) {
                if constexpr (std::is_same_v<TT, int32_t>)
                    return simde_mm256_cvtepi16_epi32(a);
                else if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm256_cvtepi16_epi64(a);
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int16_t");
            }
            else if constexpr (std::is_same_v<FT, int32_t>) {
                if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm256_cvtepi32_epi64(a);
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int32_t");
            }
            else
                static_assert(always_false_v<FT>, "unsupported convert1 from type");
        }

        static FORCE_INLINE auto convert(TR a)
        {
            if constexpr (sizeof(TT) / sizeof(FT) == 2)
                return std::tuple(convert1(simde_mm256_castsi256_si128(a)),
                                  convert1(simde_mm256_extracti128_si256(a, 1)));
            else if constexpr (sizeof(TT) / sizeof(FT) == 4) {
                auto l128 = simde_mm256_castsi256_si128(a);
                auto h128 = simde_mm256_extracti128_si256(a, 1);
                return std::tuple(convert1(l128),
                                  convert1(simde_mm_srli_si128(l128, 8)),
                                  convert1(h128),
                                  convert1(simde_mm_srli_si128(h128, 8)));
            }
            else if constexpr (sizeof(TT) / sizeof(FT) == 8) {
                auto l128 = simde_mm256_castsi256_si128(a);
                auto h128 = simde_mm256_extracti128_si256(a, 1);
                return std::tuple(convert1(l128),
                                  convert1(simde_mm_srli_si128(l128, 4)),
                                  convert1(simde_mm_srli_si128(l128, 8)),
                                  convert1(simde_mm_srli_si128(l128, 12)),
                                  convert1(h128),
                                  convert1(simde_mm_srli_si128(h128, 4)),
                                  convert1(simde_mm_srli_si128(h128, 8)),
                                  convert1(simde_mm_srli_si128(h128, 12)));
            }
            else
                static_assert(always_false_v<FT, TT>, "unsupported convert type");
        }
    };

#ifdef USE_AVX512
    template <typename FT, typename TT>
    struct VecCvt<FT, TT, AVX512, std::enable_if_t<std::is_integral_v<TT>>>
    {
        typedef std::conditional_t<sizeof(TT) / sizeof(FT) >= 4, __m128i, __m256i> FR;
        typedef __m512i                                                            TR;

        static FORCE_INLINE TR convert1(FR a)
        {
            if constexpr (std::is_same_v<FT, int8_t>) {
                if constexpr (std::is_same_v<TT, int16_t>)
                    return _mm512_cvtepi8_epi16(a);
                else if constexpr (std::is_same_v<TT, int32_t>)
                    return _mm512_cvtepi8_epi32(a);
                else if constexpr (std::is_same_v<TT, int64_t>)
                    return _mm512_cvtepi8_epi64(a);
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int8_t");
            }
            else if constexpr (std::is_same_v<FT, int16_t>) {
                if constexpr (std::is_same_v<TT, int32_t>)
                    return _mm512_cvtepi16_epi32(a);
                else if constexpr (std::is_same_v<TT, int64_t>)
                    return _mm512_cvtepi16_epi64(a);
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int16_t");
            }
            else if constexpr (std::is_same_v<FT, int32_t>) {
                if constexpr (std::is_same_v<TT, int64_t>)
                    return _mm512_cvtepi32_epi64(a);
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int32_t");
            }
        }

        static FORCE_INLINE auto convert(TR a)
        {
            if constexpr (sizeof(TT) / sizeof(FT) == 2)
                return std::tuple(convert1(_mm512_castsi512_si256(a)),
                                  convert1(_mm512_extracti64x4_epi64(a, 1)));
            else if constexpr (sizeof(TT) / sizeof(FT) == 4)
                return std::tuple(convert1(_mm512_castsi512_si128(a)),
                                  convert1(_mm512_extracti32x4_epi32(a, 1)),
                                  convert1(_mm512_extracti32x4_epi32(a, 2)),
                                  convert1(_mm512_extracti32x4_epi32(a, 3)));
            else if constexpr (sizeof(TT) / sizeof(FT) == 8) {
                auto a128 = _mm512_castsi512_si128(a);
                auto b128 = _mm512_extracti32x4_epi32(a, 1);
                auto c128 = _mm512_extracti32x4_epi32(a, 2);
                auto d128 = _mm512_extracti32x4_epi32(a, 3);
                return std::tuple(convert1(a128),
                                  convert1(_mm_srli_si128(a128, 8)),
                                  convert1(b128),
                                  convert1(_mm_srli_si128(b128, 8)),
                                  convert1(c128),
                                  convert1(_mm_srli_si128(c128, 8)),
                                  convert1(d128),
                                  convert1(_mm_srli_si128(d128, 8)));
            }
            else
                static_assert(always_false_v<FT, TT>, "unsupported convert type");
        }
    };
#endif

    template <>
    struct VecCvt<int32_t, float, SSE>
    {
        typedef simde__m128i FR;
        typedef simde__m128  TR;

        static FORCE_INLINE TR convert1(FR a) { return simde_mm_cvtepi32_ps(a); }
    };

    template <>
    struct VecCvt<int32_t, float, AVX2>
    {
        typedef simde__m256i FR;
        typedef simde__m256  TR;

        static FORCE_INLINE TR convert1(FR a) { return simde_mm256_cvtepi32_ps(a); }
    };

#ifdef USE_AVX512
    template <>
    struct VecCvt<int32_t, float, AVX512>
    {
        typedef __m512i FR;
        typedef __m512  TR;

        static FORCE_INLINE TR convert1(FR a) { return _mm512_cvtepi32_ps(a); }
    };
#endif

    template <typename FT, typename TT>
    struct VecCvt<FT, TT, SCALAR> : VecCvt<FT, TT, SSE>
    {};

    // ------------------------------------------------------------------------
    // Vec type packing (narrowing-conversion) template

    /// Convert vector register from FT type to TT type.
    template <typename FT, typename TT, InstructionType I, typename Enabled = void>
    struct VecPack
    {};

    template <typename FT, typename TT>
    struct VecPack<FT, TT, SSE, std::enable_if_t<std::is_integral_v<TT>>>
    {
        typedef simde__m128i R;

        static FORCE_INLINE R packs(R a, R b)
        {
            if constexpr (std::is_same_v<FT, int16_t> && std::is_same_v<TT, int8_t>)
                return simde_mm_packs_epi16(a, b);
            else if constexpr (std::is_same_v<FT, uint16_t> && std::is_same_v<TT, uint8_t>)
                return simde_mm_packus_epi16(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, int16_t>)
                return simde_mm_packs_epi32(a, b);
            else if constexpr (std::is_same_v<FT, uint32_t> && std::is_same_v<TT, uint16_t>)
                return simde_mm_packus_epi32(a, b);
            else
                static_assert(always_false_v<FT, TT>, "unsupported packs type");
        }

        static FORCE_INLINE R packs_permuted(R a, R b) { return packs(a, b); }
    };

    template <typename FT, typename TT>
    struct VecPack<FT, TT, AVX2, std::enable_if_t<std::is_integral_v<TT>>>
    {
        typedef simde__m256i R;

        static FORCE_INLINE R packs(R a, R b)
        {
            if constexpr (std::is_same_v<FT, int16_t> && std::is_same_v<TT, int8_t>)
                return simde_mm256_packs_epi16(a, b);
            else if constexpr (std::is_same_v<FT, uint16_t> && std::is_same_v<TT, uint8_t>)
                return simde_mm256_packus_epi16(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, int16_t>)
                return simde_mm256_packs_epi32(a, b);
            else if constexpr (std::is_same_v<FT, uint32_t> && std::is_same_v<TT, uint16_t>)
                return simde_mm256_packus_epi32(a, b);
            else
                static_assert(always_false_v<FT, TT>, "unsupported packs type");
        }

        static FORCE_INLINE R packs_permuted(R a, R b)
        {
            R r = packs(a, b);
            r   = simde_mm256_permute4x64_epi64(r, SIMDE_MM_SHUFFLE(3, 1, 2, 0));
            return r;
        }
    };

#ifdef USE_AVX512
    template <typename FT, typename TT>
    struct VecPack<FT, TT, AVX512, std::enable_if_t<std::is_integral_v<TT>>>
    {
        typedef __m512i R;

        static FORCE_INLINE R packs(R a, R b)
        {
            if constexpr (std::is_same_v<FT, int16_t> && std::is_same_v<TT, int8_t>)
                return _mm512_packs_epi16(a, b);
            else if constexpr (std::is_same_v<FT, uint16_t> && std::is_same_v<TT, uint8_t>)
                return _mm512_packus_epi16(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, int16_t>)
                return _mm512_packs_epi32(a, b);
            else if constexpr (std::is_same_v<FT, uint32_t> && std::is_same_v<TT, uint16_t>)
                return _mm512_packus_epi32(a, b);
            else
                static_assert(always_false_v<FT, TT>, "unsupported packs type");
        }

        static FORCE_INLINE R packs_permuted(R a, R b)
        {
            R r = packs(a, b);
            r   = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7), r);
            return r;
        }
    };
#endif

    template <typename FT, typename TT>
    struct VecPack<FT, TT, SCALAR, std::enable_if_t<std::is_integral_v<TT>>> : VecPack<FT, TT, SSE>
    {};

    // ------------------------------------------------------------------------
    // Vec operation set template

    template <typename T, InstructionType I>
    struct VecOp
    {};

    struct VecOpSISSE
    {
        typedef simde__m128i  R;
        static FORCE_INLINE R setzero() { return simde_mm_setzero_si128(); }
        static FORCE_INLINE R bitwiseor(R a, R b) { return simde_mm_or_si128(a, b); }
        static FORCE_INLINE R bitwiseand(R a, R b) { return simde_mm_and_si128(a, b); }
        static FORCE_INLINE R bitwisexor(R a, R b) { return simde_mm_xor_si128(a, b); }
    };

    struct VecOpSIAVX2
    {
        typedef simde__m256i  R;
        static FORCE_INLINE R setzero() { return simde_mm256_setzero_si256(); }
        static FORCE_INLINE R bitwiseor(R a, R b) { return simde_mm256_or_si256(a, b); }
        static FORCE_INLINE R bitwiseand(R a, R b) { return simde_mm256_and_si256(a, b); }
        static FORCE_INLINE R bitwisexor(R a, R b) { return simde_mm256_xor_si256(a, b); }
    };

#ifdef USE_AVX512
    struct VecOpSIAVX512
    {
        typedef __m512i       R;
        static FORCE_INLINE R setzero() { return _mm512_setzero_si512(); }
        static FORCE_INLINE R bitwiseor(R a, R b) { return _mm512_or_si512(a, b); }
        static FORCE_INLINE R bitwiseand(R a, R b) { return _mm512_and_si512(a, b); }
        static FORCE_INLINE R bitwisexor(R a, R b) { return _mm512_xor_si512(a, b); }
    };
#endif

    template <>
    struct VecOp<int8_t, SSE> : VecOpSISSE
    {
        typedef int8_t        T;
        static FORCE_INLINE R set1(T a) { return simde_mm_set1_epi8(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm_add_epi8(a, b); }
        static FORCE_INLINE R adds(R a, R b) { return simde_mm_adds_epi8(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm_sub_epi8(a, b); }
        static FORCE_INLINE R subs(R a, R b) { return simde_mm_subs_epi8(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm_min_epi8(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm_max_epi8(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return simde_mm_avg_epu8(a, b); }
        static FORCE_INLINE R dot2_u7i8(R a, R b) { return simde_mm_maddubs_epi16(a, b); }

        /// Compute 4-element dot product of [u7x16] and [i8x16] then accumulate into [i32x4].
        static FORCE_INLINE void dot4_u7i8_accum(R &acc, R a, R b)
        {
#if defined(USE_VNNI)
            acc = _mm_dpbusd_avx_epi32(acc, a, b);
#else
            R product0 = simde_mm_maddubs_epi16(a, b);
            product0   = simde_mm_madd_epi16(product0, simde_mm_set1_epi16(1));
            acc        = simde_mm_add_epi32(acc, product0);
#endif
        }

        /// Compute 4-element dot product of [i8x16] and [i8x16] then accumulate into [i32x4].
        static FORCE_INLINE void dot4_i8i8_accum(R &acc, R a, R b)
        {
            const R highest_bit = simde_mm_set1_epi8(0x80);

            R msb  = simde_mm_and_si128(a, highest_bit);
            R low7 = simde_mm_andnot_si128(highest_bit, a);

#if defined(USE_VNNI)
            msb  = _mm_dpbusd_avx_epi32(_mm_setzero_si128(), msb, b);  // 0 or 128
            low7 = _mm_dpbusd_avx_epi32(_mm_setzero_si128(), low7, b);
#else
            // Multiply a * b in two parts and accumulate neighbouring outputs into int16 values
            msb  = simde_mm_maddubs_epi16(msb, b);  // 0 or 128
            low7 = simde_mm_maddubs_epi16(low7, b);

            // Horizontally sum i16 pairs to i32
            const R one = simde_mm_set1_epi16(1);
            low7        = simde_mm_madd_epi16(low7, one);
            msb         = simde_mm_madd_epi16(msb, one);
#endif

            // Place value of the MSB was negative
            R product0 = simde_mm_sub_epi32(low7, msb);
            acc        = simde_mm_add_epi32(acc, product0);
        }
    };

    template <>
    struct VecOp<int8_t, AVX2> : VecOpSIAVX2
    {
        typedef int8_t        T;
        static FORCE_INLINE R set1(T a) { return simde_mm256_set1_epi8(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm256_add_epi8(a, b); }
        static FORCE_INLINE R adds(R a, R b) { return simde_mm256_adds_epi8(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm256_sub_epi8(a, b); }
        static FORCE_INLINE R subs(R a, R b) { return simde_mm256_subs_epi8(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm256_min_epi8(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm256_max_epi8(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return simde_mm256_avg_epu8(a, b); }
        static FORCE_INLINE R dot2_u7i8(R a, R b) { return simde_mm256_maddubs_epi16(a, b); }

        /// Compute 4-element dot product of [u7x32] and [i8x32] then accumulate into [i32x8].
        static FORCE_INLINE void dot4_u7i8_accum(R &acc, R a, R b)
        {
#if defined(USE_VNNI)
            acc = _mm256_dpbusd_avx_epi32(acc, a, b);
#else
            R product0  = simde_mm256_maddubs_epi16(a, b);
            product0    = simde_mm256_madd_epi16(product0, simde_mm256_set1_epi16(1));
            acc         = simde_mm256_add_epi32(acc, product0);
#endif
        }

        /// Compute 4-element dot product of [i8x32] and [i8x32] then accumulate into [i32x8].
        static FORCE_INLINE void dot4_i8i8_accum(R &acc, R a, R b)
        {
            const R highest_bit = simde_mm256_set1_epi8(0x80);

            R msb  = simde_mm256_and_si256(a, highest_bit);
            R low7 = simde_mm256_andnot_si256(highest_bit, a);

#if defined(USE_VNNI)
            msb  = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), msb, b);  // 0 or 128
            low7 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), low7, b);
#else
            // Multiply a * b in two parts and accumulate neighbouring outputs into int16 values
            msb  = simde_mm256_maddubs_epi16(msb, b);  // 0 or 128
            low7 = simde_mm256_maddubs_epi16(low7, b);

            // Horizontally sum i16 pairs to i32
            const R one = simde_mm256_set1_epi16(1);
            low7        = simde_mm256_madd_epi16(low7, one);
            msb         = simde_mm256_madd_epi16(msb, one);
#endif

            // Place value of the MSB was negative
            R product0 = simde_mm256_sub_epi32(low7, msb);
            acc        = simde_mm256_add_epi32(acc, product0);
        }
    };

#ifdef USE_AVX512
    template <>
    struct VecOp<int8_t, AVX512> : VecOpSIAVX512
    {
        typedef int8_t        T;
        static FORCE_INLINE R set1(T a) { return _mm512_set1_epi8(a); }
        static FORCE_INLINE R add(R a, R b) { return _mm512_add_epi8(a, b); }
        static FORCE_INLINE R adds(R a, R b) { return _mm512_adds_epi8(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return _mm512_sub_epi8(a, b); }
        static FORCE_INLINE R subs(R a, R b) { return _mm512_subs_epi8(a, b); }
        static FORCE_INLINE R min(R a, R b) { return _mm512_min_epi8(a, b); }
        static FORCE_INLINE R max(R a, R b) { return _mm512_max_epi8(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return _mm512_avg_epu8(a, b); }
        static FORCE_INLINE R dot2_u7i8(R a, R b) { return _mm512_maddubs_epi16(a, b); }

        /// Compute 4-element dot product of [u7x64] and [i8x64] then accumulate into [i32x16].
        static FORCE_INLINE void dot4_u7i8_accum(R &acc, R a, R b)
        {
    #if defined(USE_VNNI)
            acc = _mm512_dpbusd_epi32(acc, a, b);
    #else
            R product0 = _mm512_maddubs_epi16(a, b);
            product0   = _mm512_madd_epi16(product0, _mm512_set1_epi16(1));
            acc        = _mm512_add_epi32(acc, product0);
    #endif
        }

        /// Compute 4-element dot product of [i8x64] and [i8x64] then accumulate into [i32x16].
        static FORCE_INLINE void dot4_i8i8_accum(R &acc, R a, R b)
        {
            const R highest_bit = _mm512_set1_epi8(0x80);

            R msb  = _mm512_and_si512(a, highest_bit);
            R low7 = _mm512_andnot_si512(highest_bit, a);

    #if defined(USE_VNNI)
            msb  = _mm512_dpbusd_epi32(_mm512_setzero_si512(), msb, b);  // 0 or 128
            low7 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), low7, b);
    #else
            // Multiply a * b in two parts and accumulate neighbouring outputs into int16 values
            msb  = _mm512_maddubs_epi16(msb, b);  // 0 or 128
            low7 = _mm512_maddubs_epi16(low7, b);

            // Horizontally sum i16 pairs to i32
            const R one = _mm512_set1_epi16(1);
            low7        = _mm512_madd_epi16(low7, one);
            msb         = _mm512_madd_epi16(msb, one);
    #endif

            // Place value of the MSB was negative
            R product0 = _mm512_sub_epi32(low7, msb);
            acc        = _mm512_add_epi32(acc, product0);
        }
    };
#endif

    template <>
    struct VecOp<int16_t, SSE> : VecOpSISSE
    {
        typedef int16_t       T;
        static FORCE_INLINE R set1(T a) { return simde_mm_set1_epi16(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm_add_epi16(a, b); }
        static FORCE_INLINE R adds(R a, R b) { return simde_mm_adds_epi16(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm_sub_epi16(a, b); }
        static FORCE_INLINE R subs(R a, R b) { return simde_mm_subs_epi16(a, b); }
        static FORCE_INLINE R mullo(R a, R b) { return simde_mm_mullo_epi16(a, b); }
        static FORCE_INLINE R mulhi(R a, R b) { return simde_mm_mulhi_epi16(a, b); }
        static FORCE_INLINE R mulhrs(R a, R b) { return simde_mm_mulhrs_epi16(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm_min_epi16(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm_max_epi16(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return simde_mm_avg_epu16(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return simde_mm_srai_epi16(a, i);
        }
        template <int i>
        static FORCE_INLINE R srli(R a)
        {
            return simde_mm_srli_epi16(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return simde_mm_slli_epi16(a, i);
        }
        static FORCE_INLINE R dot2(R a, R b) { return simde_mm_madd_epi16(a, b); }
    };

    template <>
    struct VecOp<int16_t, AVX2> : VecOpSIAVX2
    {
        typedef int16_t       T;
        static FORCE_INLINE R set1(T a) { return simde_mm256_set1_epi16(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm256_add_epi16(a, b); }
        static FORCE_INLINE R adds(R a, R b) { return simde_mm256_adds_epi16(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm256_sub_epi16(a, b); }
        static FORCE_INLINE R subs(R a, R b) { return simde_mm256_subs_epi16(a, b); }
        static FORCE_INLINE R mullo(R a, R b) { return simde_mm256_mullo_epi16(a, b); }
        static FORCE_INLINE R mulhi(R a, R b) { return simde_mm256_mulhi_epi16(a, b); }
        static FORCE_INLINE R mulhrs(R a, R b) { return simde_mm256_mulhrs_epi16(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm256_min_epi16(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm256_max_epi16(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return simde_mm256_avg_epu16(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return simde_mm256_srai_epi16(a, i);
        }
        template <int i>
        static FORCE_INLINE R srli(R a)
        {
            return simde_mm256_srli_epi16(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return simde_mm256_slli_epi16(a, i);
        }
        static FORCE_INLINE R       dot2(R a, R b) { return simde_mm256_madd_epi16(a, b); }
        static FORCE_INLINE int32_t reduceadd(R a)
        {
            a          = simde_mm256_madd_epi16(a, set1(1));
            auto lo    = simde_mm256_castsi256_si128(a);
            auto hi    = simde_mm256_extracti128_si256(a, 1);
            lo         = simde_mm_add_epi32(lo, hi);
            auto hi64  = simde_mm_unpackhi_epi64(lo, lo);
            auto sum64 = simde_mm_add_epi32(hi64, lo);
            auto hi32  = simde_mm_shuffle_epi32(sum64, SIMDE_MM_SHUFFLE(2, 3, 0, 1));
            auto sum32 = simde_mm_add_epi32(sum64, hi32);
            return simde_mm_cvtsi128_si32(sum32);  // movd
        }
    };

#ifdef USE_AVX512
    template <>
    struct VecOp<int16_t, AVX512> : VecOpSIAVX512
    {
        typedef int16_t       T;
        static FORCE_INLINE R set1(T a) { return _mm512_set1_epi16(a); }
        static FORCE_INLINE R add(R a, R b) { return _mm512_add_epi16(a, b); }
        static FORCE_INLINE R adds(R a, R b) { return _mm512_adds_epi16(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return _mm512_sub_epi16(a, b); }
        static FORCE_INLINE R subs(R a, R b) { return _mm512_subs_epi16(a, b); }
        static FORCE_INLINE R mullo(R a, R b) { return _mm512_mullo_epi16(a, b); }
        static FORCE_INLINE R mulhi(R a, R b) { return _mm512_mulhi_epi16(a, b); }
        static FORCE_INLINE R mulhrs(R a, R b) { return _mm512_mulhrs_epi16(a, b); }
        static FORCE_INLINE R min(R a, R b) { return _mm512_min_epi16(a, b); }
        static FORCE_INLINE R max(R a, R b) { return _mm512_max_epi16(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return _mm512_avg_epu16(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return _mm512_srai_epi16(a, i);
        }
        template <int i>
        static FORCE_INLINE R srli(R a)
        {
            return _mm512_srli_epi16(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return _mm512_slli_epi16(a, i);
        }
        static FORCE_INLINE R dot2(R a, R b) { return _mm512_madd_epi16(a, b); }
    };
#endif

    template <>
    struct VecOp<int32_t, SSE> : VecOpSISSE
    {
        typedef int32_t       T;
        static FORCE_INLINE R set1(T a) { return simde_mm_set1_epi32(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm_add_epi32(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm_sub_epi32(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm_min_epi32(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm_max_epi32(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return simde_mm_srai_epi32(a, i);
        }
        template <int i>
        static FORCE_INLINE R srli(R a)
        {
            return simde_mm_srli_epi32(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return simde_mm_slli_epi32(a, i);
        }
        static FORCE_INLINE T reduceadd(R a)
        {
            auto hi64  = simde_mm_shuffle_epi32(a, SIMDE_MM_SHUFFLE(1, 0, 3, 2));
            auto sum64 = simde_mm_add_epi32(hi64, a);
            auto hi32  = simde_mm_shuffle_epi32(sum64, SIMDE_MM_SHUFFLE(2, 3, 0, 1));
            auto sum32 = simde_mm_add_epi32(sum64, hi32);
            return simde_mm_cvtsi128_si32(sum32);  // movd
        }

        /// Horizontal sum [i32x4] of 4 groups into one [i32x4].
        static FORCE_INLINE R hsum4(R sum0, R sum1, R sum2, R sum3)
        {
            sum0 = simde_mm_hadd_epi32(sum0, sum1);
            sum2 = simde_mm_hadd_epi32(sum2, sum3);
            sum0 = simde_mm_hadd_epi32(sum0, sum2);
            return sum0;
        }
    };

    template <>
    struct VecOp<int32_t, AVX2> : VecOpSIAVX2
    {
        typedef int32_t       T;
        static FORCE_INLINE R set1(T a) { return simde_mm256_set1_epi32(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm256_add_epi32(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm256_sub_epi32(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm256_min_epi32(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm256_max_epi32(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return simde_mm256_srai_epi32(a, i);
        }
        template <int i>
        static FORCE_INLINE R srli(R a)
        {
            return simde_mm256_srli_epi32(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return simde_mm256_slli_epi32(a, i);
        }
        static FORCE_INLINE T reduceadd(R a)
        {
            auto lo    = simde_mm256_castsi256_si128(a);
            auto hi    = simde_mm256_extracti128_si256(a, 1);
            lo         = simde_mm_add_epi32(lo, hi);
            auto hi64  = simde_mm_unpackhi_epi64(lo, lo);
            auto sum64 = simde_mm_add_epi32(hi64, lo);
            auto hi32  = simde_mm_shuffle_epi32(sum64, SIMDE_MM_SHUFFLE(2, 3, 0, 1));
            auto sum32 = simde_mm_add_epi32(sum64, hi32);
            return simde_mm_cvtsi128_si32(sum32);  // movd
        }

        /// Horizontal sum [i32x8] of 4 groups into one [i32x4].
        static FORCE_INLINE simde__m128i hsum4(R sum0, R sum1, R sum2, R sum3)
        {
            sum0 = simde_mm256_hadd_epi32(sum0, sum1);
            sum2 = simde_mm256_hadd_epi32(sum2, sum3);

            sum0 = simde_mm256_hadd_epi32(sum0, sum2);

            simde__m128i sum128lo = simde_mm256_castsi256_si128(sum0);
            simde__m128i sum128hi = simde_mm256_extracti128_si256(sum0, 1);
            simde__m128i sum128   = simde_mm_add_epi32(sum128lo, sum128hi);

            return sum128;
        }
    };

#ifdef USE_AVX512
    template <>
    struct VecOp<int32_t, AVX512> : VecOpSIAVX512
    {
        typedef int32_t       T;
        static FORCE_INLINE R set1(T a) { return _mm512_set1_epi32(a); }
        static FORCE_INLINE R add(R a, R b) { return _mm512_add_epi32(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return _mm512_sub_epi32(a, b); }
        static FORCE_INLINE R min(R a, R b) { return _mm512_min_epi32(a, b); }
        static FORCE_INLINE R max(R a, R b) { return _mm512_max_epi32(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return _mm512_srai_epi32(a, i);
        }
        template <int i>
        static FORCE_INLINE R srli(R a)
        {
            return _mm512_srli_epi32(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return _mm512_slli_epi32(a, i);
        }
        static FORCE_INLINE T reduceadd(R a) { return _mm512_reduce_add_epi32(a); }

        /// Horizontal sum [i32x16] of 4 groups into one [i32x4].
        static FORCE_INLINE simde__m128i hsum4(R sum0, R sum1, R sum2, R sum3)
        {
            auto sum0lo = _mm512_castsi512_si256(sum0);
            auto sum1lo = _mm512_castsi512_si256(sum1);
            auto sum2lo = _mm512_castsi512_si256(sum2);
            auto sum3lo = _mm512_castsi512_si256(sum3);
            auto sum0hi = _mm512_extracti64x4_epi64(sum0, 1);
            auto sum1hi = _mm512_extracti64x4_epi64(sum1, 1);
            auto sum2hi = _mm512_extracti64x4_epi64(sum2, 1);
            auto sum3hi = _mm512_extracti64x4_epi64(sum3, 1);

            typedef VecOp<int32_t, AVX2> I32OpHalf;
            return I32OpHalf::hsum4(I32OpHalf::add(sum0lo, sum0hi),
                                    I32OpHalf::add(sum1lo, sum1hi),
                                    I32OpHalf::add(sum2lo, sum2hi),
                                    I32OpHalf::add(sum3lo, sum3hi));
        }
    };
#endif

    template <>
    struct VecOp<float, SSE>
    {
        typedef float         T;
        typedef simde__m128   R;
        static FORCE_INLINE R setzero() { return simde_mm_setzero_ps(); }
        static FORCE_INLINE R set1(T a) { return simde_mm_set1_ps(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm_add_ps(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm_sub_ps(a, b); }
        static FORCE_INLINE R mul(R a, R b) { return simde_mm_mul_ps(a, b); }
        static FORCE_INLINE R div(R a, R b) { return simde_mm_div_ps(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm_min_ps(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm_max_ps(a, b); }
        static FORCE_INLINE R fmadd(R a, R b, R c)
        {
#ifdef USE_AVX2
            return simde_mm_fmadd_ps(a, b, c);
#else
            return simde_mm_add_ps(simde_mm_mul_ps(a, b), c);
#endif
        }
        static FORCE_INLINE T reduceadd(R a)
        {
            R shuf = simde_mm_movehdup_ps(a);  // broadcast elements 3,1 to 2,0
            R sums = simde_mm_add_ps(a, shuf);
            shuf   = simde_mm_movehl_ps(shuf, sums);  // high half -> low half
            sums   = simde_mm_add_ss(sums, shuf);
            return simde_mm_cvtss_f32(sums);
        }
    };

    template <>
    struct VecOp<float, AVX2>
    {
        typedef float         T;
        typedef simde__m256   R;
        static FORCE_INLINE R setzero() { return simde_mm256_setzero_ps(); }
        static FORCE_INLINE R set1(T a) { return simde_mm256_set1_ps(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm256_add_ps(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm256_sub_ps(a, b); }
        static FORCE_INLINE R mul(R a, R b) { return simde_mm256_mul_ps(a, b); }
        static FORCE_INLINE R div(R a, R b) { return simde_mm256_div_ps(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm256_min_ps(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm256_max_ps(a, b); }
        static FORCE_INLINE R fmadd(R a, R b, R c) { return simde_mm256_fmadd_ps(a, b, c); }
        static FORCE_INLINE T reduceadd(R a)
        {
            auto lo = simde_mm256_castps256_ps128(a);
            auto hi = simde_mm256_extractf128_ps(a, 1);
            lo      = simde_mm_add_ps(lo, hi);
            return VecOp<float, SSE>::reduceadd(lo);
        }
    };

#ifdef USE_AVX512
    template <>
    struct VecOp<float, AVX512>
    {
        typedef float         T;
        typedef __m512        R;
        static FORCE_INLINE R setzero() { return _mm512_setzero_ps(); }
        static FORCE_INLINE R set1(T a) { return _mm512_set1_ps(a); }
        static FORCE_INLINE R add(R a, R b) { return _mm512_add_ps(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return _mm512_sub_ps(a, b); }
        static FORCE_INLINE R mul(R a, R b) { return _mm512_mul_ps(a, b); }
        static FORCE_INLINE R div(R a, R b) { return _mm512_div_ps(a, b); }
        static FORCE_INLINE R min(R a, R b) { return _mm512_min_ps(a, b); }
        static FORCE_INLINE R max(R a, R b) { return _mm512_max_ps(a, b); }
        static FORCE_INLINE R fmadd(R a, R b, R c) { return _mm512_fmadd_ps(a, b, c); }
        static FORCE_INLINE T reduceadd(R a) { return _mm512_reduce_add_ps(a); }
    };
#endif

    template <typename T>
    struct VecOp<T, SCALAR> : VecOp<T, SSE>
    {};

    // ------------------------------------------------------------------------
    // Affine transform operation (y = Ax + b) template
    template <int OutSize, int InSize, int Alignment, InstructionType I, typename Enabled = void>
    struct Affine
    {};

    template <int OutSize, int InSize, int Alignment, InstructionType I>
    struct Affine<OutSize,
                  InSize,
                  Alignment,
                  I,
                  std::enable_if_t<(OutSize >= 4 && OutSize % 4 == 0)>>
    {
        template <bool SignedInput, bool Bias, bool PreReLU, bool PostReLU>
        static void
        forward(int32_t *output, const int8_t *input, const int8_t *weight, const int32_t *bias)
        {
            constexpr InstructionType I128 = SSE;

            typedef detail::VecLoadStore<int8_t, Alignment, I>     I8LS;
            typedef detail::VecLoadStore<int32_t, Alignment, I128> I32LS128;
            typedef detail::VecOp<int8_t, I>                       I8Op;
            typedef detail::VecOp<int32_t, I>                      I32Op;
            typedef detail::VecOp<int32_t, I128>                   I32Op128;

            constexpr int OutNumBatches = OutSize / 4;
            for (int i = 0; i < OutNumBatches; i++) {
                // Prepare weight offsets. One offset for one row of weights.
                // This is a simple index into a 2d array.
                const int offset0 = (i * 4 + 0) * InSize;
                const int offset1 = (i * 4 + 1) * InSize;
                const int offset2 = (i * 4 + 2) * InSize;
                const int offset3 = (i * 4 + 3) * InSize;

                // Accumulation starts from 0, we add the bias only at the end.
                auto sum0 = I32Op::setzero();
                auto sum1 = I32Op::setzero();
                auto sum2 = I32Op::setzero();
                auto sum3 = I32Op::setzero();

                // Each innermost loop processes a 32x4 chunk of weights, so 128 weights at a time!
                typedef detail::VecBatch<InSize, int8_t, I> B;
                for (int j = 0; j < B::NumBatch; j++) {
                    // We unroll by 4 so that we can reuse this value, reducing the number of
                    // memory operations required.
                    auto in = I8LS::load(input + j * B::RegWidth);
                    if constexpr (PreReLU)
                        in = I8Op::max(in, I32Op::setzero());

                    // Processes a 4Lx1 chunk of int8 and produces a Lx1 chunk of int32.
                    const auto w0 = I8LS::load(weight + offset0 + j * B::RegWidth);
                    const auto w1 = I8LS::load(weight + offset1 + j * B::RegWidth);
                    const auto w2 = I8LS::load(weight + offset2 + j * B::RegWidth);
                    const auto w3 = I8LS::load(weight + offset3 + j * B::RegWidth);
                    if constexpr (SignedInput) {
                        I8Op::dot4_i8i8_accum(sum0, in, w0);
                        I8Op::dot4_i8i8_accum(sum1, in, w1);
                        I8Op::dot4_i8i8_accum(sum2, in, w2);
                        I8Op::dot4_i8i8_accum(sum3, in, w3);
                    }
                    else {
                        I8Op::dot4_u7i8_accum(sum0, in, w0);
                        I8Op::dot4_u7i8_accum(sum1, in, w1);
                        I8Op::dot4_u7i8_accum(sum2, in, w2);
                        I8Op::dot4_u7i8_accum(sum3, in, w3);
                    }
                }

                // Adds horizontally L values from each sum together, producing 4 int32 values.
                auto outval = I32Op::hsum4(sum0, sum1, sum2, sum3);
                if constexpr (Bias)
                    outval = I32Op128::add(outval, I32LS128::load(bias + i * 4));
                if constexpr (PostReLU)
                    outval = I32Op128::max(outval, I32Op128::setzero());
                I32LS128::store(output + i * 4, outval);
            }
        }

        template <bool SignedInput, bool Bias, bool PreReLU, bool PostReLU>
        static void
        forward(int32_t *output, const int16_t *input, const int16_t *weight, const int32_t *bias)
        {
            constexpr InstructionType I128 = SSE;

            typedef detail::VecLoadStore<int16_t, Alignment, I>    I16LS;
            typedef detail::VecLoadStore<int32_t, Alignment, I128> I32LS128;
            typedef detail::VecOp<int16_t, I>                      I16Op;
            typedef detail::VecOp<int32_t, I>                      I32Op;
            typedef detail::VecOp<int32_t, I128>                   I32Op128;

            constexpr int OutNumBatches = OutSize / 4;
            for (int i = 0; i < OutNumBatches; i++) {
                // Prepare weight offsets. One offset for one row of weights.
                // This is a simple index into a 2d array.
                const int offset0 = (i * 4 + 0) * InSize;
                const int offset1 = (i * 4 + 1) * InSize;
                const int offset2 = (i * 4 + 2) * InSize;
                const int offset3 = (i * 4 + 3) * InSize;

                // Accumulation starts from 0, we add the bias only at the end.
                auto sum0 = I32Op::setzero();
                auto sum1 = I32Op::setzero();
                auto sum2 = I32Op::setzero();
                auto sum3 = I32Op::setzero();

                // Each innermost loop processes a 16x4 chunk of weights, so 64 weights at a time!
                typedef detail::VecBatch<InSize, int16_t, I> B;
                for (int j = 0; j < B::NumBatch; j++) {
                    // We unroll by 4 so that we can reuse this value, reducing the number of
                    // memory operations required.
                    auto in = I16LS::load(input + j * B::RegWidth);
                    if constexpr (PreReLU)
                        in = I16Op::max(in, I32Op::setzero());

                    // Processes a 2Lx1 chunk of int16 and produces a Lx1 chunk of int32.
                    const auto w0 = I16LS::load(weight + offset0 + j * B::RegWidth);
                    const auto w1 = I16LS::load(weight + offset1 + j * B::RegWidth);
                    const auto w2 = I16LS::load(weight + offset2 + j * B::RegWidth);
                    const auto w3 = I16LS::load(weight + offset3 + j * B::RegWidth);
                    sum0          = I32Op::add(sum0, I16Op::dot2(in, w0));
                    sum1          = I32Op::add(sum1, I16Op::dot2(in, w1));
                    sum2          = I32Op::add(sum2, I16Op::dot2(in, w2));
                    sum3          = I32Op::add(sum3, I16Op::dot2(in, w3));
                }

                // Adds horizontally L values from each sum together, producing 4 int32 values.
                auto outval = I32Op::hsum4(sum0, sum1, sum2, sum3);
                if constexpr (Bias)
                    outval = I32Op128::add(outval, I32LS128::load(bias + i * 4));
                if constexpr (PostReLU)
                    outval = I32Op128::max(outval, I32Op128::setzero());
                I32LS128::store(output + i * 4, outval);
            }
        }
    };

}  // namespace detail

/// Set an array to zeros. Return the end pointer of the output array.
template <int Size,
          typename T,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
T *zero(T *output)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));

    typedef detail::VecBatch<Size, T, Inst>          B;
    typedef detail::VecLoadStore<T, Alignment, Inst> LS;
    typedef detail::VecOp<T, Inst>                   Op;

    auto zero = Op::setzero();
    for (int i = 0; i < B::NumBatch; i++)
        LS::store(output + i * B::RegWidth, zero);

    return output + B::NumBatch * B::RegWidth;
}

/// Copy an array from input to output. Return the end pointer of the output array.
template <int Size,
          typename T,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
T *copy(T *output, const T *input)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));

    typedef detail::VecBatch<Size, T, Inst>          B;
    typedef detail::VecLoadStore<T, Alignment, Inst> LS;
    for (int i = 0; i < B::NumBatch; i++) {
        auto data = LS::load(input + i * B::RegWidth);
        LS::store(output + i * B::RegWidth, data);
    }

    return output + B::NumBatch * B::RegWidth;
}

template <int Size,
          typename T,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
T *add(T *output, const T *input0, const T *input1)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input0));
    assert(isPtrAligned<Alignment>(input1));

    typedef detail::VecBatch<Size, T, Inst>          B;
    typedef detail::VecLoadStore<T, Alignment, Inst> LS;
    typedef detail::VecOp<T, Inst>                   Op;

    for (int i = 0; i < B::NumBatch; i++) {
        auto data0 = LS::load(input0 + i * B::RegWidth);
        auto data1 = LS::load(input1 + i * B::RegWidth);
        data0      = Op::add(data0, data1);
        LS::store(output + i * B::RegWidth, data0);
    }

    return output + B::NumBatch * B::RegWidth;
}

/// Apply int8/int16 linear layer with int32 accumulation.
template <int             OutSize,
          int             InSize,
          bool            SignedInput = false,
          bool            Bias        = true,
          bool            PreReLU     = false,
          bool            PostReLU    = false,
          int             Alignment   = NativeAlignment,
          InstructionType Inst        = NativeInstType,
          typename AccType            = int32_t,
          typename InputType          = int8_t>
AccType *linear(AccType         *output,
                const InputType *input,
                const InputType  weight[OutSize * InSize],
                const AccType    bias[OutSize])
{
    static_assert(std::is_same_v<AccType, int32_t>, "Only int32_t accumulator is supported");
    static_assert(std::is_same_v<InputType, int8_t> || std::is_same_v<InputType, int16_t>,
                  "Only int8_t or int16_t input is supported");
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));
    assert(isPtrAligned<Alignment>(weight));
    if constexpr (Bias)
        assert(isPtrAligned<Alignment>(bias));

    typedef detail::Affine<OutSize, InSize, Alignment, Inst> Affine;
    Affine::template forward<SignedInput, Bias, PreReLU, PostReLU>(output, input, weight, bias);

    return output + OutSize;
}

/// Divide an int32 array by a 2-exp divisor, then apply clipped relu to the int32
/// array and store the saturated int8 results.
template <int             Size,
          int             Divisor,
          bool            NoReLU    = false,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
int8_t *crelu(int8_t output[Size], const int32_t input[Size])
{
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));
    static_assert(isPowerOfTwo(Divisor), "divisor must be a power of two");
    constexpr int Log2Divisor = floorLog2(Divisor);

    typedef detail::VecBatch<Size, int32_t, Inst>          InB;
    typedef detail::VecBatch<Size, int8_t, Inst>           OutB;
    typedef detail::VecLoadStore<int32_t, Alignment, Inst> I32LS;
    typedef detail::VecLoadStore<int8_t, Alignment, Inst>  I8LS;
    typedef detail::VecPack<int32_t, int16_t, Inst>        I32Pack;
    typedef detail::VecPack<int16_t, int8_t, Inst>         I16Pack;
    typedef detail::VecOp<int16_t, Inst>                   I16Op;
    typedef detail::VecOp<int8_t, Inst>                    I8Op;

    const auto zero = I8Op::setzero();

    for (int i = 0; i < OutB::NumBatch; i++) {
        auto in0  = I32LS::load(input + (i * 4 + 0) * InB::RegWidth);
        auto in1  = I32LS::load(input + (i * 4 + 1) * InB::RegWidth);
        auto in2  = I32LS::load(input + (i * 4 + 2) * InB::RegWidth);
        auto in3  = I32LS::load(input + (i * 4 + 3) * InB::RegWidth);
        auto in01 = I32Pack::packs(in0, in1);
        auto in23 = I32Pack::packs(in2, in3);
        if constexpr (Log2Divisor > 0) {
            in01 = I16Op::template srai<Log2Divisor>(in01);
            in23 = I16Op::template srai<Log2Divisor>(in23);
        }
        auto result = I16Pack::packs(in01, in23);
        if constexpr (!NoReLU)
            result = I8Op::max(result, zero);

        // Permute values in different lanes if required.
        if constexpr (Inst == AVX2) {
            const auto control = simde_mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
            result             = simde_mm256_permutevar8x32_epi32(result, control);
        }
#ifdef USE_AVX512
        else if constexpr (Inst == AVX512) {
            const auto control =
                _mm512_set_epi32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0);
            result = _mm512_permutexvar_epi32(control, result);
        }
#endif

        I8LS::store(output + i * OutB::RegWidth, result);
    }

    return output + Size;
}

/// Divide an int32 array by a 2-exp divisor, then apply clipped relu to the int32
/// array and store the saturated int16 results.
template <int             Size,
          int             Divisor,
          bool            NoReLU    = false,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
int16_t *crelu(int16_t output[Size], const int32_t input[Size])
{
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));
    static_assert(isPowerOfTwo(Divisor), "divisor must be a power of two");
    constexpr int Log2Divisor = floorLog2(Divisor);

    typedef detail::VecBatch<Size, int32_t, Inst>          InB;
    typedef detail::VecBatch<Size, int16_t, Inst>          OutB;
    typedef detail::VecLoadStore<int32_t, Alignment, Inst> I32LS;
    typedef detail::VecLoadStore<int16_t, Alignment, Inst> I16LS;
    typedef detail::VecPack<int32_t, int16_t, Inst>        I32Pack;
    typedef detail::VecOp<int32_t, Inst>                   I32Op;
    typedef detail::VecOp<int16_t, Inst>                   I16Op;

    const auto zero = I16Op::setzero();

    for (int i = 0; i < OutB::NumBatch; i++) {
        auto in0 = I32LS::load(input + (i * 2 + 0) * InB::RegWidth);
        auto in1 = I32LS::load(input + (i * 2 + 1) * InB::RegWidth);
        if constexpr (Log2Divisor > 0) {
            in0 = I32Op::template srai<Log2Divisor>(in0);
            in1 = I32Op::template srai<Log2Divisor>(in1);
        }
        auto result = I32Pack::packs(in0, in1);
        if constexpr (!NoReLU)
            result = I16Op::max(result, zero);

        // Permute values in different lanes if required.
        if constexpr (Inst == AVX2) {
            const auto control = SIMDE_MM_SHUFFLE(3, 1, 2, 0);
            result             = simde_mm256_permute4x64_epi64(result, control);
        }
#ifdef USE_AVX512
        else if constexpr (Inst == AVX512) {
            const auto control = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
            result             = _mm512_permutexvar_epi64(control, result);
        }
#endif

        I16LS::store(output + i * OutB::RegWidth, result);
    }

    return output + Size;
}

enum class Activation { None, Relu };

/// Apply linear layer and relu layer.
template <Activation Activation,
          int        OutDim,
          int        InDim,
          typename T,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType,
          bool            Bias      = true>
void linearLayer(T out[],
                 const T (&in)[InDim],
                 const T (&weight)[InDim][OutDim],
                 const T (&bias)[OutDim])
{
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(out));
    assert(isPtrAligned<Alignment>(in));
    assert(isPtrAligned<Alignment>(weight));
    assert(isPtrAligned<Alignment>(bias));

    constexpr size_t InstAlignSize   = simdBitsOfInstType(Inst) / 8;
    constexpr size_t OutDimAligned   = alignDimSize<InstAlignSize, T>(OutDim);
    constexpr size_t WeightAlignment = OutDimAligned == OutDim ? Alignment : 1;
    typedef detail::VecBatch<OutDimAligned, T, Inst>       B;
    typedef detail::VecLoadStore<T, Alignment, Inst>       LS;
    typedef detail::VecLoadStore<T, WeightAlignment, Inst> LSWeight;
    typedef detail::VecOp<T, Inst>                         Op;

    for (int b = 0; b < B::NumBatch; b++) {
        auto y = Bias ? LS::load(&bias[b * B::RegWidth]) : Op::setzero();
        for (int inC = 0; inC < InDim; inC++) {
            auto x = Op::set1(in[inC]);
            auto w = LSWeight::load(&weight[inC][b * B::RegWidth]);
            y      = Op::fmadd(w, x, y);
        }

        if constexpr (Activation == Activation::Relu) {
            y = Op::max(y, Op::setzero());
        }

        LS::store(&out[b * B::RegWidth], y);
    }
}

template <Activation Activation,
          int        OutDim,
          int        InDim,
          typename T,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
void linearLayer(T out[], const T (&in)[InDim], const T (&weight)[InDim][OutDim])
{
    linearLayer<Activation, OutDim, InDim, T, Alignment, Inst, false>(
        out,
        in,
        weight,
        *reinterpret_cast<const T(*)[OutDim]>(out));
}

template <int Size,
          typename T,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
void preluLayer(T (&out)[Size], const T (&in)[Size], const T (&weight)[Size])
{
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(out));
    assert(isPtrAligned<Alignment>(in));
    assert(isPtrAligned<Alignment>(weight));

    typedef detail::VecBatch<Size, T, Inst>          B;
    typedef detail::VecLoadStore<T, Alignment, Inst> LS;
    typedef detail::VecOp<T, Inst>                   Op;

    for (int i = 0; i < B::NumBatch; i++) {
        auto data0    = LS::load(&in[i * B::RegWidth]);
        auto weight0  = LS::load(&weight[i * B::RegWidth]);
        auto product0 = Op::mul(data0, weight0);
        auto result0  = Op::max(data0, product0);
        LS::store(&out[i * B::RegWidth], result0);
    }
}

}  // namespace Evaluation::simd
