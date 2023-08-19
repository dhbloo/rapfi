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
    default: return 0;
    case SSE: return 128;
    case AVX2: return 256;
    case AVX512: return 512;
    }
}

#if defined(USE_AVX512)
constexpr size_t          NativeAlignment = 64;
constexpr InstructionType NativeInstType  = AVX512;
#elif defined(USE_AVX2) || defined(USE_AVX) || defined(USE_SSE)  // avx2 to sse by simde
constexpr size_t          NativeAlignment = 32;
constexpr InstructionType NativeInstType  = AVX2;
#else
constexpr size_t          NativeAlignment = 8;
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

/// @param SimdBits The width of simd instructions (in bits)
/// @param T The type of elements
/// @param Size The size of element array
/// @param RegWidth The number of elements in one register
/// @param NumBatches Number iterations for one register to loop all elements
#define DEF_BATCH(SimdBits, T, Size, RegWidth, NumBatches)                                  \
    constexpr int RegWidth = SimdBits ? (SimdBits / 8) / sizeof(T) : 1;                     \
    static_assert(Size % RegWidth == 0, "data does not fill a " #SimdBits " bit register"); \
    constexpr int NumBatches = Size / RegWidth;

#define DEF_BATCH128(T, Size, RegWidth, NumBatches) DEF_BATCH(128, T, Size, RegWidth, NumBatches)
#define DEF_BATCH256(T, Size, RegWidth, NumBatches) DEF_BATCH(256, T, Size, RegWidth, NumBatches)
#define DEF_BATCH512(T, Size, RegWidth, NumBatches) DEF_BATCH(512, T, Size, RegWidth, NumBatches)

namespace regop {  // register level operators

    /// Unpack avx2 register [32xI8] to 2x[16xI16].
    /// @return (lower 128bit [16xI16], higher 128bit [16xI16]).
    FORCE_INLINE ::std::tuple<simde__m256i, simde__m256i> unpackI8ToI16(simde__m256i a)
    {
        auto a0i8  = simde_mm256_castsi256_si128(a);
        auto a1i8  = simde_mm256_extracti128_si256(a, 1);
        auto a0i16 = simde_mm256_cvtepi8_epi16(a0i8);
        auto a1i16 = simde_mm256_cvtepi8_epi16(a1i8);
        return {a0i16, a1i16};
    }

    /// Divide 2x[16xI16] by a power of two divisor and pack them into [32xI8].
    template <unsigned Divisor>
    FORCE_INLINE simde__m256i divideAndPackI16ToI8(simde__m256i a, simde__m256i b)
    {
        static_assert(Divisor > 0, "divisor must not be zero");
        static_assert(isPowerOfTwo(Divisor), "divisor must be a power of two");
        constexpr unsigned Log2Divisor = floorLog2(Divisor);

        if constexpr (Log2Divisor > 0) {
            a = simde_mm256_srai_epi16(a, Log2Divisor);
            b = simde_mm256_srai_epi16(b, Log2Divisor);
        }

        return simde_mm256_permute4x64_epi64(simde_mm256_packs_epi16(a, b),
                                             SIMDE_MM_SHUFFLE(3, 1, 2, 0));
    }

    /// Multiply two avx2 register [32xI8] into 2x[16xI16].
    /// @return ([16xI16] from lower [32xI8], [16xI16] from higher [32xI8]).
    FORCE_INLINE ::std::tuple<simde__m256i, simde__m256i> mulI8(simde__m256i a, simde__m256i b)
    {
        simde__m128i a_m128 = simde_mm256_castsi256_si128(a);
        simde__m128i b_m128 = simde_mm256_castsi256_si128(b);

        simde__m256i a_m256 = simde_mm256_cvtepi8_epi16(a_m128);
        simde__m256i b_m256 = simde_mm256_cvtepi8_epi16(b_m128);

        simde__m256i r0 = simde_mm256_mullo_epi16(a_m256, b_m256);

        a_m128 = simde_mm256_extractf128_si256(a, 1);
        b_m128 = simde_mm256_extractf128_si256(b, 1);

        a_m256 = simde_mm256_cvtepi8_epi16(a_m128);
        b_m256 = simde_mm256_cvtepi8_epi16(b_m128);

        simde__m256i r1 = simde_mm256_mullo_epi16(a_m256, b_m256);

        return {r0, r1};
    }

    /// Horizontal sum [8xI32] of 4 groups into one [4xI32].
    FORCE_INLINE simde__m128i hsumI32x4(simde__m256i sum0,
                                        simde__m256i sum1,
                                        simde__m256i sum2,
                                        simde__m256i sum3)
    {
        sum0 = simde_mm256_hadd_epi32(sum0, sum1);
        sum2 = simde_mm256_hadd_epi32(sum2, sum3);

        sum0 = simde_mm256_hadd_epi32(sum0, sum2);

        simde__m128i sum128lo = simde_mm256_castsi256_si128(sum0);
        simde__m128i sum128hi = simde_mm256_extracti128_si256(sum0, 1);
        simde__m128i sum128   = simde_mm_add_epi32(sum128lo, sum128hi);

        return sum128;
    }

    /// Warpper around _mm256_dpbusd_epi32().
    FORCE_INLINE void add_dpbusd_epi32(simde__m256i &acc, simde__m256i a, simde__m256i b)
    {
#if defined(USE_VNNI)
        // This does exactly the same thing as explained below but in one instruction.
        acc = _mm256_dpbusd_epi32(acc, a, b);
#else
        // Multiply a * b and accumulate neighbouring outputs into int16 values
        simde__m256i product0 = simde_mm256_maddubs_epi16(a, b);

        // Multiply product0 by 1 (idempotent) and accumulate neighbouring outputs into int32 values
        product0 = simde_mm256_madd_epi16(product0, simde_mm256_set1_epi16(1));

        // Add to the main int32 accumulator.
        acc = simde_mm256_add_epi32(acc, product0);
#endif
    };

    /// Apply int16 leaky relu to a batch of [16xI16] registers.
    template <int Size, int NegSlopeDivisor>
    simde__m256i *lrelu16(simde__m256i x[])
    {
        static_assert(isPowerOfTwo(NegSlopeDivisor), "divisor must be a power of two");
        constexpr int Log2Divisor = floorLog2(NegSlopeDivisor);
        DEF_BATCH256(int16_t, Size, RegWidth, NumBatches);

        for (int i = 0; i < NumBatches; i++) {
            auto negValue = simde_mm256_srai_epi16(x[i], Log2Divisor);
            x[i]          = simde_mm256_max_epi16(x[i], negValue);
        }

        return x + NumBatches;
    }

    /// Apply int16 prelu to a batch of [16xI16] registers.
    template <int Size, int WeightScale>
    simde__m256i *prelu16(simde__m256i x[], const int16_t weight[Size])
    {
        DEF_BATCH256(int16_t, Size, RegWidth, NumBatches);
        static_assert(isPowerOfTwo(WeightScale), "weight scale must be a power of two");
        constexpr int Log2WeightScale = floorLog2(WeightScale);

        for (int i = 0; i < NumBatches; i++) {
            auto slope    = simde_mm256_loadu_si256(weight + i * RegWidth);
            auto negValue = simde_mm256_mullo_epi16(x[i], slope);
            negValue      = simde_mm256_srai_epi16(negValue, Log2WeightScale);
            x[i]          = simde_mm256_max_epi16(x[i], negValue);
        }

        return x + NumBatches;
    }

}  // namespace regop

namespace detail {

    // ------------------------------------------------------------------------
    // Vec regwidth & batch num definition

    template <size_t Size, typename T, InstructionType I>
    struct VecBatch
    {
        static constexpr size_t SimdBits = simdBitsOfInstType(I);
        static constexpr size_t RegWidth = SimdBits ? (SimdBits / 8) / sizeof(T) : 1;
        static constexpr size_t NumBatch = Size / RegWidth;

        static_assert(Size % RegWidth == 0, "data does not fill a register");
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
                if constexpr (std::is_same_v<TT, int32_t>)
                    return simde_mm_cvtepi8_epi32(a);
                if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm_cvtepi8_epi64(a);
            }
            if constexpr (std::is_same_v<FT, int16_t>) {
                if constexpr (std::is_same_v<TT, int32_t>)
                    return simde_mm_cvtepi16_epi32(a);
                if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm_cvtepi16_epi64(a);
            }
            if constexpr (std::is_same_v<FT, int32_t>) {
                if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm_cvtepi32_epi64(a);
            }
        }

        static FORCE_INLINE auto convert(TR a)
        {
            if constexpr (sizeof(TT) / sizeof(FT) == 2) {
                return std::tuple(convert1(a), convert1(simde_mm_srli_si128(a, 8)));
            }
            if constexpr (sizeof(TT) / sizeof(FT) == 4) {
                return std::tuple(convert1(a),
                                  convert1(simde_mm_srli_si128(a, 4)),
                                  convert1(simde_mm_srli_si128(a, 8)),
                                  convert1(simde_mm_srli_si128(a, 12)));
            }
            if constexpr (sizeof(TT) / sizeof(FT) == 8) {
                return std::tuple(convert1(a),
                                  convert1(simde_mm_srli_si128(a, 2)),
                                  convert1(simde_mm_srli_si128(a, 4)),
                                  convert1(simde_mm_srli_si128(a, 6)),
                                  convert1(simde_mm_srli_si128(a, 8)),
                                  convert1(simde_mm_srli_si128(a, 10)),
                                  convert1(simde_mm_srli_si128(a, 12)),
                                  convert1(simde_mm_srli_si128(a, 14)));
            }
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
                if constexpr (std::is_same_v<TT, int32_t>)
                    return simde_mm256_cvtepi8_epi32(a);
                if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm256_cvtepi8_epi64(a);
            }
            if constexpr (std::is_same_v<FT, int16_t>) {
                if constexpr (std::is_same_v<TT, int32_t>)
                    return simde_mm256_cvtepi16_epi32(a);
                if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm256_cvtepi16_epi64(a);
            }
            if constexpr (std::is_same_v<FT, int32_t>) {
                if constexpr (std::is_same_v<TT, int64_t>)
                    return simde_mm256_cvtepi32_epi64(a);
            }
        }

        static FORCE_INLINE auto convert(TR a)
        {
            if constexpr (sizeof(TT) / sizeof(FT) == 2) {
                return std::tuple(convert1(simde_mm256_castsi256_si128(a)),
                                  convert1(simde_mm256_extracti128_si256(a, 1)));
            }
            if constexpr (sizeof(TT) / sizeof(FT) == 4) {
                auto l128 = simde_mm256_castsi256_si128(a);
                auto h128 = simde_mm256_extracti128_si256(a, 1);
                return std::tuple(convert1(l128),
                                  convert1(simde_mm_srli_si128(l128, 8)),
                                  convert1(h128),
                                  convert1(simde_mm_srli_si128(h128, 8)));
            }
            if constexpr (sizeof(TT) / sizeof(FT) == 8) {
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
                if constexpr (std::is_same_v<TT, int32_t>)
                    return _mm512_cvtepi8_epi32(a);
                if constexpr (std::is_same_v<TT, int64_t>)
                    return _mm512_cvtepi8_epi64(a);
            }
            if constexpr (std::is_same_v<FT, int16_t>) {
                if constexpr (std::is_same_v<TT, int32_t>)
                    return _mm512_cvtepi16_epi32(a);
                if constexpr (std::is_same_v<TT, int64_t>)
                    return _mm512_cvtepi16_epi64(a);
            }
            if constexpr (std::is_same_v<FT, int32_t>) {
                if constexpr (std::is_same_v<TT, int64_t>)
                    return _mm512_cvtepi32_epi64(a);
            }
        }

        static FORCE_INLINE auto convert(TR a)
        {
            if constexpr (sizeof(TT) / sizeof(FT) == 2) {
                return std::tuple(convert1(_mm512_castsi512_si256(a)),
                                  convert1(_mm512_extracti64x4_epi64(a, 1)));
            }
            if constexpr (sizeof(TT) / sizeof(FT) == 4) {
                return std::tuple(convert1(_mm512_castsi512_si128(a)),
                                  convert1(_mm512_extracti32x4_epi32(a, 1)),
                                  convert1(_mm512_extracti32x4_epi32(a, 2)),
                                  convert1(_mm512_extracti32x4_epi32(a, 3)));
            }
            if constexpr (sizeof(TT) / sizeof(FT) == 8) {
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
    };

    template <>
    struct VecOp<int16_t, AVX2> : VecOpSIAVX2
    {
        typedef int16_t             T;
        static FORCE_INLINE R       set1(T a) { return simde_mm256_set1_epi16(a); }
        static FORCE_INLINE R       add(R a, R b) { return simde_mm256_add_epi16(a, b); }
        static FORCE_INLINE R       adds(R a, R b) { return simde_mm256_adds_epi16(a, b); }
        static FORCE_INLINE R       sub(R a, R b) { return simde_mm256_sub_epi16(a, b); }
        static FORCE_INLINE R       subs(R a, R b) { return simde_mm256_subs_epi16(a, b); }
        static FORCE_INLINE R       mullo(R a, R b) { return simde_mm256_mullo_epi16(a, b); }
        static FORCE_INLINE R       mulhi(R a, R b) { return simde_mm256_mulhi_epi16(a, b); }
        static FORCE_INLINE R       mulhrs(R a, R b) { return simde_mm256_mulhrs_epi16(a, b); }
        static FORCE_INLINE R       min(R a, R b) { return simde_mm256_min_epi16(a, b); }
        static FORCE_INLINE R       max(R a, R b) { return simde_mm256_max_epi16(a, b); }
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
        static FORCE_INLINE T reduceadd(R a)
        {
            auto hi64  = simde_mm_shuffle_epi32(a, SIMDE_MM_SHUFFLE(1, 0, 3, 2));
            auto sum64 = simde_mm_add_epi32(hi64, a);
            auto hi32  = simde_mm_shuffle_epi32(sum64, SIMDE_MM_SHUFFLE(2, 3, 0, 1));
            auto sum32 = simde_mm_add_epi32(sum64, hi32);
            return simde_mm_cvtsi128_si32(sum32);  // movd
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
        static FORCE_INLINE T reduceadd(R a) { return _mm512_reduce_add_epi32(a); }
    };
#endif

    template <>
    struct VecOp<int64_t, SSE> : VecOpSISSE
    {
        typedef int64_t       T;
        static FORCE_INLINE R set1(T a) { return simde_mm_set1_epi64x(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm_add_epi64(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm_sub_epi64(a, b); }
    };

    template <>
    struct VecOp<int64_t, AVX2> : VecOpSIAVX2
    {
        typedef int64_t       T;
        static FORCE_INLINE R set1(T a) { return simde_mm256_set1_epi64x(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm256_add_epi64(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm256_sub_epi64(a, b); }
    };

#ifdef USE_AVX512
    template <>
    struct VecOp<int64_t, AVX512> : VecOpSIAVX512
    {
        typedef int64_t       T;
        static FORCE_INLINE R set1(T a) { return _mm512_set1_epi64(a); }
        static FORCE_INLINE R add(R a, R b) { return _mm512_add_epi64(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return _mm512_sub_epi64(a, b); }
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
        static FORCE_INLINE R fmadd(R a, R b, R c) { return simde_mm_fmadd_ps(a, b, c); }
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
            auto lo   = simde_mm256_castps256_ps128(a);
            auto hi   = simde_mm256_extractf128_ps(a, 1);
            lo        = simde_mm_add_ps(lo, hi);
            auto shuf = simde_mm_movehdup_ps(lo);  // broadcast elements 3,1 to 2,0
            auto sums = simde_mm_add_ps(lo, shuf);
            shuf      = simde_mm_movehl_ps(shuf, sums);  // high half -> low half
            sums      = simde_mm_add_ss(sums, shuf);
            return simde_mm_cvtss_f32(sums);
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
    struct VecOp<T, SCALAR>
    {
        typedef T             R;
        static FORCE_INLINE R setzero() { return T(0); }
        static FORCE_INLINE R set1(T a) { return a; }
        static FORCE_INLINE R add(R a, R b) { return a + b; }
        static FORCE_INLINE R sub(R a, R b) { return a - b; }
        static FORCE_INLINE R mul(R a, R b) { return a * b; }
        static FORCE_INLINE R div(R a, R b) { return a / b; }
        static FORCE_INLINE R min(R a, R b) { return std::min(a, b); }
        static FORCE_INLINE R max(R a, R b) { return std::max(a, b); }
        static FORCE_INLINE R fmadd(R a, R b, R c) { return a * b + c; }
        static FORCE_INLINE T reduceadd(R a) { return a; }
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
T *add(T *output, const T *input, const T a)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));

    typedef detail::VecBatch<Size, T, Inst>          B;
    typedef detail::VecLoadStore<T, Alignment, Inst> LS;
    typedef detail::VecOp<T, Inst>                   Op;

    auto A = Op::set1(a);
    for (int i = 0; i < B::NumBatch; i++) {
        auto data = LS::load(input + i * B::RegWidth);
        data      = Op::add(data, A);
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

template <int Size,
          typename T,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
T *min(T *output, const T *input0, const T *input1)
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
        data0      = Op::min(data0, data1);
        LS::store(output + i * B::RegWidth, data0);
    }

    return output + B::NumBatch * B::RegWidth;
}

template <int Size,
          typename T,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
T *max(T *output, const T *input0, const T *input1)
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
        data0      = Op::max(data0, data1);
        LS::store(output + i * B::RegWidth, data0);
    }

    return output + B::NumBatch * B::RegWidth;
}

template <int Size,
          typename T,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
T *relu(T *output, const T *input)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));

    typedef detail::VecBatch<Size, T, Inst>          B;
    typedef detail::VecLoadStore<T, Alignment, Inst> LS;
    typedef detail::VecOp<T, Inst>                   Op;

    auto zero = Op::setzero();
    for (int i = 0; i < B::NumBatch; i++) {
        auto data0 = LS::load(input + i * B::RegWidth);
        data0      = Op::max(data0, zero);
        LS::store(output + i * B::RegWidth, data0);
    }

    return output + B::NumBatch * B::RegWidth;
}

template <int             OutSize,
          int             InSize,
          int             WeightScale,
          int             Alignment = NativeAlignment,
          InstructionType Inst      = NativeInstType>
int32_t *linear(int32_t      *output,
                const int8_t *input,
                const int8_t  weight[OutSize][InSize],
                const int32_t bias[OutSize])
{
    static_assert(Inst == AVX2, "Only avx2 is supported now!");
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));
    assert(isPtrAligned<Alignment>(weight));
    assert(isPtrAligned<Alignment>(bias));

    typedef detail::VecBatch<InSize, int8_t, Inst> B;
    static_assert(OutSize % 4 == 0, "OutSize must be divisble by 4");
    static_assert(isPowerOfTwo(WeightScale), "weight scale must be a power of two");
    constexpr int OutNumBatches   = OutSize / 4;
    constexpr int Log2WeightScale = floorLog2(WeightScale);

    for (int i = 0; i < OutNumBatches; i++) {
        // Prepare weight offsets. One offset for one row of weights.
        // This is a simple index into a 2d array.
        const int offset0 = (i * 4 + 0) * InSize;
        const int offset1 = (i * 4 + 1) * InSize;
        const int offset2 = (i * 4 + 2) * InSize;
        const int offset3 = (i * 4 + 3) * InSize;

        // Accumulation starts from 0, we add the bias only at the end.
        auto sum0 = simde_mm256_setzero_si256();
        auto sum1 = simde_mm256_setzero_si256();
        auto sum2 = simde_mm256_setzero_si256();
        auto sum3 = simde_mm256_setzero_si256();

        // Each innermost loop processes a 32x4 chunk of weights, so 128 weights at a time!
        for (int j = 0; j < B::NumBatch; j++) {
            typedef detail::VecLoadStore<int8_t, Alignment, AVX2> I8LS;

            // We unroll by 4 so that we can reuse this value, reducing the number of
            // memory operations required.
            const auto in = I8LS::load(input + j * B::RegWidth);

            // This function processes a 32x1 chunk of int8 and produces a 8x1 chunk of int32.
            // For definition see below.
            regop::add_dpbusd_epi32(sum0, in, I8LS::load(weight[0] + offset0 + j * B::RegWidth));
            regop::add_dpbusd_epi32(sum1, in, I8LS::load(weight[0] + offset1 + j * B::RegWidth));
            regop::add_dpbusd_epi32(sum2, in, I8LS::load(weight[0] + offset2 + j * B::RegWidth));
            regop::add_dpbusd_epi32(sum3, in, I8LS::load(weight[0] + offset3 + j * B::RegWidth));
        }

        // This function adds horizontally 8 values from each sum together, producing 4 int32
        // values. For the definition see below.
        auto outval = regop::hsumI32x4(sum0, sum1, sum2, sum3);
        outval      = simde_mm_add_epi32(outval, simde_mm_loadu_si128(bias + i * 4));
        // Here we account for the weights scaling.
        outval = simde_mm_srai_epi32(outval, Log2WeightScale);
        simde_mm_storeu_si128(output + i * 4, outval);
    }

    return output + OutSize;
}

template <int Size, int Alignment = NativeAlignment, InstructionType Inst = NativeInstType>
int8_t *crelu32(int8_t output[Size], const int32_t input[Size])
{
    static_assert(Inst == AVX2, "Only avx2 is supported now!");
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));

    typedef detail::VecBatch<Size, int32_t, Inst> InB;
    typedef detail::VecBatch<Size, int8_t, Inst>  OutB;

    const auto zero    = simde_mm256_setzero_si256();
    const auto control = simde_mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    for (int i = 0; i < OutB::NumBatch; i++) {
        typedef detail::VecLoadStore<int32_t, Alignment, Inst> I32LS;
        auto in0  = I32LS::load(input + (i * 4 + 0) * InB::RegWidth);
        auto in1  = I32LS::load(input + (i * 4 + 1) * InB::RegWidth);
        auto in2  = I32LS::load(input + (i * 4 + 2) * InB::RegWidth);
        auto in3  = I32LS::load(input + (i * 4 + 3) * InB::RegWidth);
        auto in01 = simde_mm256_packs_epi32(in0, in1);
        auto in23 = simde_mm256_packs_epi32(in2, in3);

        auto result = simde_mm256_permutevar8x32_epi32(
            simde_mm256_max_epi8(simde_mm256_packs_epi16(in01, in23), zero),
            control);

        detail::VecLoadStore<int8_t, Alignment, Inst>::store(output + i * OutB::RegWidth, result);
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

namespace debug {
    template <typename T, int N>
    void assertInRange(const simde__m256i *x, int min, int max)
    {
#ifndef NDEBUG
        for (int i = 0; i < N; i++) {
            if constexpr (std::is_same_v<T, int8_t>) {
                int8_t v[32] = {
                    (int8_t)simde_mm256_extract_epi8(x[i], 0),
                    (int8_t)simde_mm256_extract_epi8(x[i], 1),
                    (int8_t)simde_mm256_extract_epi8(x[i], 2),
                    (int8_t)simde_mm256_extract_epi8(x[i], 3),
                    (int8_t)simde_mm256_extract_epi8(x[i], 4),
                    (int8_t)simde_mm256_extract_epi8(x[i], 5),
                    (int8_t)simde_mm256_extract_epi8(x[i], 6),
                    (int8_t)simde_mm256_extract_epi8(x[i], 7),
                    (int8_t)simde_mm256_extract_epi8(x[i], 8),
                    (int8_t)simde_mm256_extract_epi8(x[i], 9),
                    (int8_t)simde_mm256_extract_epi8(x[i], 10),
                    (int8_t)simde_mm256_extract_epi8(x[i], 11),
                    (int8_t)simde_mm256_extract_epi8(x[i], 12),
                    (int8_t)simde_mm256_extract_epi8(x[i], 13),
                    (int8_t)simde_mm256_extract_epi8(x[i], 14),
                    (int8_t)simde_mm256_extract_epi8(x[i], 15),
                    (int8_t)simde_mm256_extract_epi8(x[i], 16),
                    (int8_t)simde_mm256_extract_epi8(x[i], 17),
                    (int8_t)simde_mm256_extract_epi8(x[i], 18),
                    (int8_t)simde_mm256_extract_epi8(x[i], 19),
                    (int8_t)simde_mm256_extract_epi8(x[i], 20),
                    (int8_t)simde_mm256_extract_epi8(x[i], 21),
                    (int8_t)simde_mm256_extract_epi8(x[i], 22),
                    (int8_t)simde_mm256_extract_epi8(x[i], 23),
                    (int8_t)simde_mm256_extract_epi8(x[i], 24),
                    (int8_t)simde_mm256_extract_epi8(x[i], 25),
                    (int8_t)simde_mm256_extract_epi8(x[i], 26),
                    (int8_t)simde_mm256_extract_epi8(x[i], 27),
                    (int8_t)simde_mm256_extract_epi8(x[i], 28),
                    (int8_t)simde_mm256_extract_epi8(x[i], 29),
                    (int8_t)simde_mm256_extract_epi8(x[i], 30),
                    (int8_t)simde_mm256_extract_epi8(x[i], 31),
                };

                for (int j = 0; j < 32; j++) {
                    assert(min <= v[j] && v[j] <= max);
                }
            }
            else if constexpr (std::is_same_v<T, uint8_t>) {
                uint8_t v[32] = {
                    (uint8_t)simde_mm256_extract_epi8(x[i], 0),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 1),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 2),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 3),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 4),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 5),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 6),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 7),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 8),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 9),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 10),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 11),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 12),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 13),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 14),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 15),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 16),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 17),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 18),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 19),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 20),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 21),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 22),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 23),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 24),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 25),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 26),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 27),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 28),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 29),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 30),
                    (uint8_t)simde_mm256_extract_epi8(x[i], 31),
                };

                for (int j = 0; j < 32; j++) {
                    assert(min <= v[j] && v[j] <= max);
                }
            }
            else if constexpr (std::is_same_v<T, int16_t>) {
                int16_t v[16] = {
                    (int16_t)simde_mm256_extract_epi16(x[i], 0),
                    (int16_t)simde_mm256_extract_epi16(x[i], 1),
                    (int16_t)simde_mm256_extract_epi16(x[i], 2),
                    (int16_t)simde_mm256_extract_epi16(x[i], 3),
                    (int16_t)simde_mm256_extract_epi16(x[i], 4),
                    (int16_t)simde_mm256_extract_epi16(x[i], 5),
                    (int16_t)simde_mm256_extract_epi16(x[i], 6),
                    (int16_t)simde_mm256_extract_epi16(x[i], 7),
                    (int16_t)simde_mm256_extract_epi16(x[i], 8),
                    (int16_t)simde_mm256_extract_epi16(x[i], 9),
                    (int16_t)simde_mm256_extract_epi16(x[i], 10),
                    (int16_t)simde_mm256_extract_epi16(x[i], 11),
                    (int16_t)simde_mm256_extract_epi16(x[i], 12),
                    (int16_t)simde_mm256_extract_epi16(x[i], 13),
                    (int16_t)simde_mm256_extract_epi16(x[i], 14),
                    (int16_t)simde_mm256_extract_epi16(x[i], 15),
                };

                for (int j = 0; j < 16; j++) {
                    assert(min <= v[j] && v[j] <= max);
                }
            }
            else if constexpr (std::is_same_v<T, uint16_t>) {
                uint16_t v[16] = {
                    (uint16_t)simde_mm256_extract_epi16(x[i], 0),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 1),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 2),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 3),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 4),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 5),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 6),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 7),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 8),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 9),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 10),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 11),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 12),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 13),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 14),
                    (uint16_t)simde_mm256_extract_epi16(x[i], 15),
                };

                for (int j = 0; j < 16; j++) {
                    assert(min <= v[j] && v[j] <= max);
                }
            }
            else {
                assert(false && "unsupported type T");
            }
        }
#endif
    }

    template <typename T, int N>
    void assertInRange(const simde__m256i (&x)[N], int min, int max)
    {
        assertInRange<T, N>(static_cast<const simde__m256i *>(x), min, max);
    }

    template <typename T>
    void assertInRange(simde__m256i x, int min, int max)
    {
        simde__m256i t[1] = {x};
        assertInRange<T>(t, min, max);
    }

}  // namespace debug

}  // namespace Evaluation::simd
