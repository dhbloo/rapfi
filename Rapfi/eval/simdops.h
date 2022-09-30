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

#include <tuple>
#include <type_traits>

namespace Evaluation::simd {

enum InstructionType {
    SCALAR,
    AVX2,
    AVX512,
};

#if defined(USE_AVX512)
constexpr size_t          NativeAlignment = 64;
constexpr InstructionType NativeInstType  = AVX512;
#elif defined(USE_AVX2) || defined(USE_AVX)
constexpr size_t          NativeAlignment = 32;
constexpr InstructionType NativeInstType  = AVX2;
#else
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
    static_assert(AlignSize >= alignof(T), "Incorrect AlignSize");
    return (reinterpret_cast<uintptr_t>(pointer) & (AlignSize - 1)) == 0;
}

/// @param SimdBits The width of simd instructions (in bits)
/// @param T The type of elements
/// @param Size The size of element array
/// @param RegWidth The number of elements in one register
/// @param NumBatches Number iterations for one register to loop all elements
#define DEF_BATCH(SimdBits, T, Size, RegWidth, NumBatches)                                 \
    constexpr int RegWidth = (SimdBits / 8) / sizeof(T);                                   \
    static_assert(Size % RegWidth == 0, "data does not fill a " #SimdBits "bit register"); \
    constexpr int NumBatches = Size / RegWidth;

#define DEF_BATCH128(T, Size, RegWidth, NumBatches) DEF_BATCH(128, T, Size, RegWidth, NumBatches)
#define DEF_BATCH256(T, Size, RegWidth, NumBatches) DEF_BATCH(256, T, Size, RegWidth, NumBatches)
#define DEF_BATCH512(T, Size, RegWidth, NumBatches) DEF_BATCH(512, T, Size, RegWidth, NumBatches)

namespace regop {  // register level operators

    /// Unpack avx2 register [32xI8] to 2x[16xI16].
    /// @return (lower 128bit [16xI16], higher 128bit [16xI16]).
    inline ::std::tuple<simde__m256i, simde__m256i> unpackI8ToI16(simde__m256i a)
    {
        auto a0i8  = simde_mm256_castsi256_si128(a);
        auto a1i8  = simde_mm256_extracti128_si256(a, 1);
        auto a0i16 = simde_mm256_cvtepi8_epi16(a0i8);
        auto a1i16 = simde_mm256_cvtepi8_epi16(a1i8);
        return {a0i16, a1i16};
    }

    /// Divide 2x[16xI16] by a power of two divisor and pack them into [32xI8].
    template <unsigned Divisor>
    inline simde__m256i divideAndPackI16ToI8(simde__m256i a, simde__m256i b)
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
    inline ::std::tuple<simde__m256i, simde__m256i> mulI8(simde__m256i a, simde__m256i b)
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

    /// Horizontal sum [4xI32] into one I32.
    inline int32_t hsumI32(simde__m128i a)
    {
        simde__m128i hi64  = simde_mm_unpackhi_epi64(a, a);
        simde__m128i sum64 = simde_mm_add_epi32(hi64, a);
        simde__m128i hi32  = simde_mm_shuffle_epi32(sum64, SIMDE_MM_SHUFFLE(2, 3, 0, 1));
        simde__m128i sum32 = simde_mm_add_epi32(sum64, hi32);
        return simde_mm_cvtsi128_si32(sum32);  // movd
    }

    /// Horizontal sum [8xI32] into one I32.
    inline int32_t hsumI32(simde__m256i a)
    {
        return hsumI32(simde_mm_add_epi32(simde_mm256_castsi256_si128(a),
                                          simde_mm256_extracti128_si256(a, 1)));
    }

    /// Horizontal sum [16xI16] into one I32.
    inline int32_t hsumI16(simde__m256i a)
    {
#if 0
        a            = simde_mm256_hadds_epi16(a, a);
        a            = simde_mm256_hadds_epi16(a, a);
        a            = simde_mm256_hadds_epi16(a, a);
        auto a0_m128 = simde_mm256_castsi256_si128(a);
        auto a1_m128 = simde_mm256_extractf128_si256(a, 1);
        auto a_m128  = simde_mm_adds_epi16(a0_m128, a1_m128);
        return (int16_t)simde_mm_extract_epi16(a_m128, 0);
#else
        a = simde_mm256_madd_epi16(a, simde_mm256_set1_epi16(1));
        return hsumI32(a);
#endif
    }

    /// Horizontal sum [8xI32] of 4 groups into one [4xI32].
    inline simde__m128i
    hsumI32x4(simde__m256i sum0, simde__m256i sum1, simde__m256i sum2, simde__m256i sum3)
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
    inline void add_dpbusd_epi32(simde__m256i &acc, simde__m256i a, simde__m256i b)
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
    // Vec store & load template

    template <typename T, int Alignment, InstructionType I, typename Enabled = void>
    struct VecLoadStore
    {};

    template <typename T, int Alignment>
    struct VecLoadStore<T, Alignment, AVX2, std::enable_if_t<std::is_integral_v<T>>>
    {
        static inline auto load(const void *addr)
        {
            if constexpr (Alignment >= 32)
                return simde_mm256_load_si256(reinterpret_cast<const simde__m256i *>(addr));
            else
                return simde_mm256_loadu_si256(addr);
        }

        static inline void store(void *addr, simde__m256i data)
        {
            if constexpr (Alignment >= 32)
                simde_mm256_store_si256(reinterpret_cast<simde__m256i *>(addr), data);
            else
                simde_mm256_storeu_si256(addr, data);
        }
    };

    template <int Alignment>
    struct VecLoadStore<float, Alignment, AVX2>
    {
        static inline auto load(const float *addr)
        {
            if constexpr (Alignment >= 32)
                return simde_mm256_load_ps(addr);
            else
                return simde_mm256_loadu_ps(addr);
        }

        static inline void store(float *addr, simde__m256 data)
        {
            if constexpr (Alignment >= 32)
                simde_mm256_store_ps(reinterpret_cast<simde_float32 *>(addr), data);
            else
                simde_mm256_storeu_ps(addr, data);
        }
    };

    // ------------------------------------------------------------------------
    // Vec operation set template

    template <typename T, InstructionType I>
    struct VecOp
    {};

    struct VecOpSIAvx2
    {
        typedef simde__m256i R;
        static inline R      setzero() { return simde_mm256_setzero_si256(); }
    };

    template <>
    struct VecOp<int8_t, AVX2> : VecOpSIAvx2
    {
        typedef int8_t  T;
        static inline R set1(T a) { return simde_mm256_set1_epi8(a); }
        static inline R add(R a, R b) { return simde_mm256_add_epi8(a, b); }
        static inline R sub(R a, R b) { return simde_mm256_sub_epi8(a, b); }
        static inline R min(R a, R b) { return simde_mm256_min_epi8(a, b); }
        static inline R max(R a, R b) { return simde_mm256_max_epi8(a, b); }
    };

    template <>
    struct VecOp<int16_t, AVX2> : VecOpSIAvx2
    {
        typedef int16_t T;
        static inline R set1(T a) { return simde_mm256_set1_epi16(a); }
        static inline R add(R a, R b) { return simde_mm256_add_epi16(a, b); }
        static inline R sub(R a, R b) { return simde_mm256_sub_epi16(a, b); }
        static inline R min(R a, R b) { return simde_mm256_min_epi16(a, b); }
        static inline R max(R a, R b) { return simde_mm256_max_epi16(a, b); }
    };

    template <>
    struct VecOp<int32_t, AVX2> : VecOpSIAvx2
    {
        typedef int32_t T;
        static inline R set1(T a) { return simde_mm256_set1_epi32(a); }
        static inline R add(R a, R b) { return simde_mm256_add_epi32(a, b); }
        static inline R sub(R a, R b) { return simde_mm256_sub_epi32(a, b); }
        static inline R min(R a, R b) { return simde_mm256_min_epi32(a, b); }
        static inline R max(R a, R b) { return simde_mm256_max_epi32(a, b); }
    };

    template <>
    struct VecOp<int64_t, AVX2> : VecOpSIAvx2
    {
        typedef int64_t T;
        static inline R set1(T a) { return simde_mm256_set1_epi64x(a); }
        static inline R add(R a, R b) { return simde_mm256_add_epi64(a, b); }
        static inline R sub(R a, R b) { return simde_mm256_sub_epi64(a, b); }
    };

    template <>
    struct VecOp<float, AVX2>
    {
        typedef float       T;
        typedef simde__m256 R;
        static inline R     setzero() { return simde_mm256_setzero_ps(); }
        static inline R     set1(T a) { return simde_mm256_set1_ps(a); }
        static inline R     add(R a, R b) { return simde_mm256_add_ps(a, b); }
        static inline R     sub(R a, R b) { return simde_mm256_sub_ps(a, b); }
        static inline R     mul(R a, R b) { return simde_mm256_mul_ps(a, b); }
        static inline R     div(R a, R b) { return simde_mm256_div_ps(a, b); }
        static inline R     min(R a, R b) { return simde_mm256_min_ps(a, b); }
        static inline R     max(R a, R b) { return simde_mm256_max_ps(a, b); }
    };

    /*template <typename T, int Size, int Alignment>
    struct VecOpSet
    {
        static_assert(std::is_arithmetic_v<T>);
        static inline void zeros(T *output)
        {
            SIMDE_VECTORIZE
            for (size_t i = 0; i < Size; ++i)
                output[i] = T(0);
        }
        static inline void ones(T *output)
        {
            SIMDE_VECTORIZE
            for (size_t i = 0; i < Size; ++i)
                output[i] = T(1);
        }
        static inline void set(T *output, T num)
        {
            SIMDE_VECTORIZE
            for (size_t i = 0; i < Size; ++i)
                output[i] = num;
        }
        static inline void copy(T *output, T *input)
        {
            SIMDE_VECTORIZE
            for (size_t i = 0; i < Size; ++i)
                output[i] = input[i];
        }
        static inline void add(T *output, T *input0, T *input1)
        {
            SIMDE_VECTORIZE
            for (size_t i = 0; i < Size; ++i)
                output[i] += input0[i] + input1[i];
        }
        static inline void sub(T *output, T *input0, T *input1)
        {
            SIMDE_VECTORIZE
            for (size_t i = 0; i < Size; ++i)
                output[i] += input0[i] - input1[i];
        }
        static inline void mul(T *output, T *input0, T *input1)
        {
            SIMDE_VECTORIZE
            for (size_t i = 0; i < Size; ++i)
                output[i] += input0[i] * input1[i];
        }
        static inline T hsum(T *input)
        {
            T result {};
            SIMDE_VECTORIZE
            for (size_t i = 0; i < Size; ++i)
                result += input[i];
            return result;
        }
    };*/

}  // namespace detail

/// Set an integer array to zeros. Return the end pointer of the array.
template <int Size, typename T, int Alignment = NativeAlignment>
T *zero(T *output)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));

    auto zero = detail::VecOp<T, AVX2>::setzero();

    DEF_BATCH256(T, Size, RegWidth, NumBatches);
    for (int i = 0; i < NumBatches; i++)
        detail::VecLoadStore<T, Alignment, AVX2>::store(output + i * RegWidth, zero);

    return output + NumBatches * RegWidth;
}

template <int Size, typename T, int Alignment = NativeAlignment>
T *copy(T *output, const T *input)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));

    DEF_BATCH256(T, Size, RegWidth, NumBatches);

    for (int i = 0; i < NumBatches; i++) {
        auto data = detail::VecLoadStore<T, Alignment, AVX2>::load(input + i * RegWidth);
        detail::VecLoadStore<T, Alignment, AVX2>::store(output + i * RegWidth, data);
    }

    return output + NumBatches * RegWidth;
}

template <int Size, typename T, int Alignment = NativeAlignment>
T *add(T *output, const T *input, const T a)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));

    auto A = detail::VecOp<T, AVX2>::set1(a);

    DEF_BATCH256(T, Size, RegWidth, NumBatches);
    for (int i = 0; i < NumBatches; i++) {
        auto data = detail::VecLoadStore<T, Alignment, AVX2>::load(input + i * RegWidth);
        data      = detail::VecOp<T, AVX2>::add(data, A);
        detail::VecLoadStore<T, Alignment, AVX2>::store(output + i * RegWidth, data);
    }

    return output + NumBatches * RegWidth;
}

template <int Size, typename T, int Alignment = NativeAlignment>
T *add(T *output, const T *input0, const T *input1)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input0));
    assert(isPtrAligned<Alignment>(input1));

    DEF_BATCH256(T, Size, RegWidth, NumBatches);
    for (int i = 0; i < NumBatches; i++) {
        auto data0 = detail::VecLoadStore<T, Alignment, AVX2>::load(input0 + i * RegWidth);
        auto data1 = detail::VecLoadStore<T, Alignment, AVX2>::load(input1 + i * RegWidth);
        data0      = detail::VecOp<T, AVX2>::add(data0, data1);
        detail::VecLoadStore<T, Alignment, AVX2>::store(output + i * RegWidth, data0);
    }

    return output + NumBatches * RegWidth;
}

template <int Size, typename T, int Alignment = NativeAlignment>
T *min(T *output, const T *input0, const T *input1)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input0));
    assert(isPtrAligned<Alignment>(input1));

    DEF_BATCH256(T, Size, RegWidth, NumBatches);
    for (int i = 0; i < NumBatches; i++) {
        auto data0 = detail::VecLoadStore<T, Alignment, AVX2>::load(input0 + i * RegWidth);
        auto data1 = detail::VecLoadStore<T, Alignment, AVX2>::load(input1 + i * RegWidth);
        data0      = detail::VecOp<T, AVX2>::min(data0, data1);
        detail::VecLoadStore<T, Alignment, AVX2>::store(output + i * RegWidth, data0);
    }

    return output + NumBatches * RegWidth;
}

template <int Size, typename T, int Alignment = NativeAlignment>
T *max(T *output, const T *input0, const T *input1)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input0));
    assert(isPtrAligned<Alignment>(input1));

    DEF_BATCH256(T, Size, RegWidth, NumBatches);
    for (int i = 0; i < NumBatches; i++) {
        auto data0 = detail::VecLoadStore<T, Alignment, AVX2>::load(input0 + i * RegWidth);
        auto data1 = detail::VecLoadStore<T, Alignment, AVX2>::load(input1 + i * RegWidth);
        data0      = detail::VecOp<T, AVX2>::max(data0, data1);
        detail::VecLoadStore<T, Alignment, AVX2>::store(output + i * RegWidth, data0);
    }

    return output + NumBatches * RegWidth;
}

template <int Size, typename T, int Alignment = NativeAlignment>
T *relu(T *output, const T *input)
{
    static_assert(std::is_integral_v<T> || std::is_same_v<T, float>);
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));

    auto zero = detail::VecOp<T, AVX2>::setzero();

    DEF_BATCH256(T, Size, RegWidth, NumBatches);
    for (int i = 0; i < NumBatches; i++) {
        auto data0 = detail::VecLoadStore<T, Alignment, AVX2>::load(input + i * RegWidth);
        data0      = detail::VecOp<T, AVX2>::max(data0, zero);
        detail::VecLoadStore<T, Alignment, AVX2>::store(output + i * RegWidth, data0);
    }

    return output + NumBatches * RegWidth;
}

template <int OutSize, int InSize, int WeightScale, int Alignment = NativeAlignment>
int32_t *linear(int32_t      *output,
                const int8_t *input,
                const int8_t  weight[OutSize][InSize],
                const int32_t bias[OutSize])
{
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));
    assert(isPtrAligned<Alignment>(weight));
    assert(isPtrAligned<Alignment>(bias));

    static_assert(OutSize % 4 == 0, "OutSize must be divisble by 4");
    constexpr int OutNumBatches = OutSize / 4;
    DEF_BATCH256(int8_t, InSize, InRegWidth, InNumBatches);
    static_assert(isPowerOfTwo(WeightScale), "weight scale must be a power of two");
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
        for (int j = 0; j < InNumBatches; j++) {
            typedef detail::VecLoadStore<int8_t, Alignment, AVX2> I8LS;

            // We unroll by 4 so that we can reuse this value, reducing the number of
            // memory operations required.
            const auto in = I8LS::load(input + j * InRegWidth);

            // This function processes a 32x1 chunk of int8 and produces a 8x1 chunk of int32.
            // For definition see below.
            regop::add_dpbusd_epi32(sum0, in, I8LS::load(weight + offset0 + j * InRegWidth));
            regop::add_dpbusd_epi32(sum1, in, I8LS::load(weight + offset1 + j * InRegWidth));
            regop::add_dpbusd_epi32(sum2, in, I8LS::load(weight + offset2 + j * InRegWidth));
            regop::add_dpbusd_epi32(sum3, in, I8LS::load(weight + offset3 + j * InRegWidth));
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

template <int Size, int Alignment = NativeAlignment>
int8_t *crelu32(int8_t output[Size], const int32_t input[Size])
{
    static_assert(isAlignSizeOK(Alignment));
    assert(isPtrAligned<Alignment>(output));
    assert(isPtrAligned<Alignment>(input));

    DEF_BATCH256(int32_t, Size, InRegWidth, InNumBatches);
    DEF_BATCH256(int8_t, Size, OutRegWidth, OutNumBatches);

    const auto zero    = simde_mm256_setzero_si256();
    const auto control = simde_mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

    for (int i = 0; i < OutNumBatches; i++) {
        typedef detail::VecLoadStore<int32_t, Alignment, AVX2> I32LS;
        auto in0  = I32LS::load(input + (i * 4 + 0) * InRegWidth);
        auto in1  = I32LS::load(input + (i * 4 + 1) * InRegWidth);
        auto in2  = I32LS::load(input + (i * 4 + 2) * InRegWidth);
        auto in3  = I32LS::load(input + (i * 4 + 3) * InRegWidth);
        auto in01 = simde_mm256_packs_epi32(in0, in1);
        auto in23 = simde_mm256_packs_epi32(in2, in3);

        auto result = simde_mm256_permutevar8x32_epi32(
            simde_mm256_max_epi8(simde_mm256_packs_epi16(in01, in23), zero),
            control);

        detail::VecLoadStore<int8_t, Alignment, AVX2>::store(output + i * OutRegWidth, result);
    }

    return output + Size;
}

enum class Activation { None, Relu };

/// Apply linear layer and relu layer.
template <Activation activation, int OutDim, int InDim, int OutDimAligned = 8 * ((OutDim + 7) / 8)>
void linearLayer(float (&out)[OutDimAligned],
                 const float (&in)[InDim],
                 const float (&weight)[InDim][OutDim],
                 const float (&bias)[OutDim])
{
    DEF_BATCH256(float, OutDimAligned, RegWidth, OutBatches);

    for (int b = 0; b < OutBatches; b++) {
        auto y = simde_mm256_loadu_ps(&bias[b * RegWidth]);
        for (int inC = 0; inC < InDim; inC++) {
            auto x = simde_mm256_set1_ps(in[inC]);
            auto W = simde_mm256_loadu_ps(&weight[inC][b * RegWidth]);
            y      = simde_mm256_fmadd_ps(W, x, y);  // linear
        }
        if constexpr (activation == Activation::Relu) {
            y = simde_mm256_max_ps(simde_mm256_setzero_ps(), y);  // relu
        }
        simde_mm256_storeu_ps(out + b * RegWidth, y);
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
