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
#include "../../core/platform.h"
#include "isa.h"

#include <simde/x86/avx2.h>
#include <simde/x86/fma.h>
#include <tuple>
#include <type_traits>

/// Register-level SIMD primitives, specialized per instruction set (SSE/AVX2/AVX512/
/// NEON/WASM_SIMD, with SCALAR delegating to SSE through simde's portable fallbacks):
/// VecBatch (size/register bookkeeping), VecLoadStore, VecCvt (widening), VecPack
/// (narrowing) and VecOp (arithmetic op set). Although nested in detail, these are the
/// intended building blocks for evaluator hot loops; the kernel headers (linear.h,
/// activation.h) compose them into full array operations.
namespace Evaluation::simd {

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

        static constexpr InstructionType Inst = I;

        static_assert(AllowExtra || NumExtra == 0, "data does not fill a register");
    };

    // ------------------------------------------------------------------------
    // Neon type helper
#ifdef USE_NEON
    template <typename T>
    struct NeonReg
    {};

    template <>
    struct NeonReg<int8_t>
    {
        typedef int8x16_t type;
    };

    template <>
    struct NeonReg<int16_t>
    {
        typedef int16x8_t type;
    };

    template <>
    struct NeonReg<int32_t>
    {
        typedef int32x4_t type;
    };

    template <>
    struct NeonReg<uint8_t>
    {
        using type = uint8x16_t;
    };

    template <>
    struct NeonReg<uint16_t>
    {
        using type = uint16x8_t;
    };

    template <>
    struct NeonReg<uint32_t>
    {
        using type = uint32x4_t;
    };

    template <>
    struct NeonReg<float>
    {
        typedef float32x4_t type;
    };

    template <typename T>
    using NeonReg_t = typename NeonReg<T>::type;
#endif

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

#ifdef USE_NEON
    template <typename T, int Alignment>
    struct VecLoadStore<T, Alignment, NEON, std::enable_if_t<std::is_integral_v<T>>>
    {
        static FORCE_INLINE auto load(const void *addr)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                return vld1q_s8(reinterpret_cast<const int8_t *>(addr));
            else if constexpr (std::is_same_v<T, uint8_t>)
                return vld1q_u8(reinterpret_cast<const uint8_t *>(addr));
            else if constexpr (std::is_same_v<T, int16_t>)
                return vld1q_s16(reinterpret_cast<const int16_t *>(addr));
            else if constexpr (std::is_same_v<T, uint16_t>)
                return vld1q_u16(reinterpret_cast<const uint16_t *>(addr));
            else if constexpr (std::is_same_v<T, int32_t>)
                return vld1q_s32(reinterpret_cast<const int32_t *>(addr));
            else if constexpr (std::is_same_v<T, uint32_t>)
                return vld1q_u32(reinterpret_cast<const uint32_t *>(addr));
            else
                static_assert(always_false_v<T>, "unsupported load type");
        }

        static FORCE_INLINE void store(void *addr, NeonReg_t<T> data)
        {
            if constexpr (std::is_same_v<T, int8_t>)
                vst1q_s8(reinterpret_cast<int8_t *>(addr), data);
            else if constexpr (std::is_same_v<T, uint8_t>)
                vst1q_u8(reinterpret_cast<uint8_t *>(addr), data);
            else if constexpr (std::is_same_v<T, int16_t>)
                vst1q_s16(reinterpret_cast<int16_t *>(addr), data);
            else if constexpr (std::is_same_v<T, uint16_t>)
                vst1q_u16(reinterpret_cast<uint16_t *>(addr), data);
            else if constexpr (std::is_same_v<T, int32_t>)
                vst1q_s32(reinterpret_cast<int32_t *>(addr), data);
            else if constexpr (std::is_same_v<T, uint32_t>)
                vst1q_u32(reinterpret_cast<uint32_t *>(addr), data);
            else
                static_assert(always_false_v<T>, "unsupported store type");
        }
    };
#endif

#ifdef USE_WASM_SIMD
    template <typename T, int Alignment>
    struct VecLoadStore<T, Alignment, WASM_SIMD, std::enable_if_t<std::is_integral_v<T>>>
    {
        static FORCE_INLINE auto load(const void *addr) { return wasm_v128_load(addr); }

        static FORCE_INLINE void store(void *addr, v128_t data)
        {
            return wasm_v128_store(addr, data);
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

#ifdef USE_NEON
    template <int Alignment>
    struct VecLoadStore<float, Alignment, NEON>
    {
        static FORCE_INLINE auto load(const void *addr)
        {
            return vld1q_f32(reinterpret_cast<const float *>(addr));
        }

        static FORCE_INLINE void store(void *addr, float32x4_t data)
        {
            vst1q_f32(reinterpret_cast<float *>(addr), data);
        }
    };
#endif

#ifdef USE_WASM_SIMD
    template <int Alignment>
    struct VecLoadStore<float, Alignment, WASM_SIMD>
    {
        static FORCE_INLINE auto load(const void *addr) { return wasm_v128_load(addr); }

        static FORCE_INLINE void store(void *addr, v128_t data)
        {
            return wasm_v128_store(addr, data);
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

#ifdef USE_NEON
    template <typename FT, typename TT>
    struct VecCvt<FT, TT, NEON, std::enable_if_t<std::is_integral_v<TT>>>
    {
        typedef NeonReg_t<FT> FR;
        typedef NeonReg_t<TT> TR;

        static FORCE_INLINE TR convert1(FR a)
        {
            if constexpr (std::is_same_v<FT, int8_t>) {
                if constexpr (std::is_same_v<TT, int16_t>)
                    return vmovl_s8(vget_low_s8(a));
                else if constexpr (std::is_same_v<TT, int32_t>)
                    return vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(a))));
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int8_t");
            }
            else if constexpr (std::is_same_v<FT, int16_t>) {
                if constexpr (std::is_same_v<TT, int32_t>)
                    return vmovl_s16(vget_low_s16(a));
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int16_t");
            }
            else
                static_assert(always_false_v<FT>, "unsupported convert1 from type");
        }

        static FORCE_INLINE auto convert(FR a)
        {
            if constexpr (std::is_same_v<FT, int8_t>) {
                if constexpr (std::is_same_v<TT, int16_t>)
                    return std::tuple(vmovl_s8(vget_low_s8(a)), vmovl_s8(vget_high_s8(a)));
                else if constexpr (std::is_same_v<TT, int32_t>)
                    return std::tuple(vmovl_s16(vget_low_s16(vmovl_s8(vget_low_s8(a)))),
                                      vmovl_s16(vget_high_s16(vmovl_s8(vget_low_s8(a)))),
                                      vmovl_s16(vget_low_s16(vmovl_s8(vget_high_s8(a)))),
                                      vmovl_s16(vget_high_s16(vmovl_s8(vget_high_s8(a)))));
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int8_t");
            }
            else if constexpr (std::is_same_v<FT, int16_t>) {
                if constexpr (std::is_same_v<TT, int32_t>)
                    return std::tuple(vmovl_s16(vget_low_s16(a)), vmovl_s16(vget_high_s16(a)));
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int16_t");
            }
            else
                static_assert(always_false_v<FT>, "unsupported convert1 from type");
        }
    };
#endif

#ifdef USE_WASM_SIMD
    template <typename FT, typename TT>
    struct VecCvt<FT, TT, WASM_SIMD, std::enable_if_t<std::is_integral_v<TT>>>
    {
        typedef v128_t FR;
        typedef v128_t TR;

        static FORCE_INLINE TR convert1(FR a)
        {
            if constexpr (std::is_same_v<FT, int8_t>) {
                if constexpr (std::is_same_v<TT, int16_t>)
                    return wasm_i16x8_extend_low_i8x16(a);
                else if constexpr (std::is_same_v<TT, int32_t>)
                    return wasm_i32x4_extend_low_i16x8(wasm_i16x8_extend_low_i8x16(a));
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int8_t");
            }
            else if constexpr (std::is_same_v<FT, int16_t>) {
                if constexpr (std::is_same_v<TT, int32_t>)
                    return wasm_i32x4_extend_low_i16x8(a);
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int16_t");
            }
            else
                static_assert(always_false_v<FT>, "unsupported convert1 from type");
        }

        static FORCE_INLINE auto convert(FR a)
        {
            if constexpr (std::is_same_v<FT, int8_t>) {
                if constexpr (std::is_same_v<TT, int16_t>)
                    return std::tuple(wasm_i16x8_extend_low_i8x16(a),
                                      wasm_i16x8_extend_high_i8x16(a));
                else if constexpr (std::is_same_v<TT, int32_t>)
                    return std::tuple(
                        wasm_i32x4_extend_low_i16x8(wasm_i16x8_extend_low_i8x16(a)),
                        wasm_i32x4_extend_high_i16x8(wasm_i16x8_extend_low_i8x16(a)),
                        wasm_i32x4_extend_low_i16x8(wasm_i16x8_extend_high_i8x16(a)),
                        wasm_i32x4_extend_high_i16x8(wasm_i16x8_extend_high_i8x16(a)));
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int8_t");
            }
            else if constexpr (std::is_same_v<FT, int16_t>) {
                if constexpr (std::is_same_v<TT, int32_t>)
                    return std::tuple(wasm_i32x4_extend_low_i16x8(a),
                                      wasm_i32x4_extend_high_i16x8(a));
                else
                    static_assert(always_false_v<TT>, "unsupported convert1 to type from int16_t");
            }
            else
                static_assert(always_false_v<FT>, "unsupported convert1 from type");
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

#ifdef USE_NEON
    template <>
    struct VecCvt<int32_t, float, NEON>
    {
        typedef int32x4_t   FR;
        typedef float32x4_t TR;

        static FORCE_INLINE TR convert1(FR a) { return vcvtq_f32_s32(a); }
    };
#endif

#ifdef USE_WASM_SIMD
    template <>
    struct VecCvt<int32_t, float, WASM_SIMD>
    {
        typedef v128_t FR;
        typedef v128_t TR;

        static FORCE_INLINE TR convert1(FR a) { return wasm_f32x4_convert_i32x4(a); }
    };
#endif

    template <typename FT, typename TT>
    struct VecCvt<FT, TT, SCALAR> : VecCvt<FT, TT, SSE>
    {};

    // ------------------------------------------------------------------------
    // Vec type packing (narrowing-conversion) template

    /// Narrow two registers of FT into one register of TT with saturation.
    /// Note the x86 semantics: on AVX2/AVX512, packs() interleaves the 128-bit lanes
    /// of the two inputs instead of concatenating them, so packed elements are NOT in
    /// order. Use packs_permuted() when element order must be preserved, or apply the
    /// matching lane fixup at the consumer (as crelu() in activation.h does).
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
            else if constexpr (std::is_same_v<FT, int16_t> && std::is_same_v<TT, uint8_t>)
                return simde_mm_packus_epi16(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, int16_t>)
                return simde_mm_packs_epi32(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, uint16_t>)
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
            else if constexpr (std::is_same_v<FT, int16_t> && std::is_same_v<TT, uint8_t>)
                return simde_mm256_packus_epi16(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, int16_t>)
                return simde_mm256_packs_epi32(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, uint16_t>)
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
            else if constexpr (std::is_same_v<FT, int16_t> && std::is_same_v<TT, uint8_t>)
                return _mm512_packus_epi16(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, int16_t>)
                return _mm512_packs_epi32(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, uint16_t>)
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

#ifdef USE_NEON
    template <typename FT, typename TT>
    struct VecPack<FT, TT, NEON, std::enable_if_t<std::is_integral_v<TT>>>
    {
        typedef NeonReg_t<FT> FR;
        typedef NeonReg_t<TT> TR;

        static FORCE_INLINE TR packs(FR a, FR b)
        {
            if constexpr (std::is_same_v<FT, int16_t> && std::is_same_v<TT, int8_t>)
                return vqmovn_high_s16(vqmovn_s16(a), b);
            else if constexpr (std::is_same_v<FT, int16_t> && std::is_same_v<TT, uint8_t>)
                return vqmovun_high_s16(vqmovun_s16(a), b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, int16_t>)
                return vqmovn_high_s32(vqmovn_s32(a), b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, uint16_t>)
                return vqmovun_high_s32(vqmovun_s32(a), b);
            else
                static_assert(always_false_v<FT, TT>, "unsupported packs type");
        }

        static FORCE_INLINE TR packs_permuted(FR a, FR b) { return packs(a, b); }
    };
#endif

#ifdef USE_WASM_SIMD
    template <typename FT, typename TT>
    struct VecPack<FT, TT, WASM_SIMD, std::enable_if_t<std::is_integral_v<TT>>>
    {
        typedef v128_t FR;
        typedef v128_t TR;

        static FORCE_INLINE TR packs(FR a, FR b)
        {
            if constexpr (std::is_same_v<FT, int16_t> && std::is_same_v<TT, int8_t>)
                return wasm_i8x16_narrow_i16x8(a, b);
            else if constexpr (std::is_same_v<FT, int16_t> && std::is_same_v<TT, uint8_t>)
                return wasm_u8x16_narrow_i16x8(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, int16_t>)
                return wasm_i16x8_narrow_i32x4(a, b);
            else if constexpr (std::is_same_v<FT, int32_t> && std::is_same_v<TT, uint16_t>)
                return wasm_u16x8_narrow_i32x4(a, b);
            else
                static_assert(always_false_v<FT, TT>, "unsupported packs type");
        }

        static FORCE_INLINE TR packs_permuted(FR a, FR b) { return packs(a, b); }
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
    };

    struct VecOpSIAVX2
    {
        typedef simde__m256i  R;
        static FORCE_INLINE R setzero() { return simde_mm256_setzero_si256(); }
    };

#ifdef USE_AVX512
    struct VecOpSIAVX512
    {
        typedef __m512i       R;
        static FORCE_INLINE R setzero() { return _mm512_setzero_si512(); }
    };
#endif

#ifdef USE_WASM_SIMD
    struct VecOpSIWASM_SIMD
    {
        typedef v128_t        R;
        static FORCE_INLINE R setzero() { return wasm_i32x4_splat(0); }
    };
#endif

    template <>
    struct VecOp<int8_t, SSE> : VecOpSISSE
    {
        typedef int8_t        T;
        static FORCE_INLINE R set1(T a) { return simde_mm_set1_epi8(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm_add_epi8(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm_sub_epi8(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm_min_epi8(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm_max_epi8(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return simde_mm_avg_epu8(a, b); }
        static FORCE_INLINE R dot2_u7i8(R a, R b) { return simde_mm_maddubs_epi16(a, b); }

        /// Compute 4-element dot product of [u7x16] and [i8x16] then accumulate into [i32x4].
        static FORCE_INLINE void dot4_u7i8_accum(R &acc, R a, R b)
        {
#if defined(USE_VNNI)
    #if !defined(USE_AVX512)
            acc = _mm_dpbusd_avx_epi32(acc, a, b);
    #else
            acc = _mm_dpbusd_epi32(acc, a, b);
    #endif
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
    #if !defined(USE_AVX512)
            msb  = _mm_dpbusd_avx_epi32(_mm_setzero_si128(), msb, b);  // 0 or 128
            low7 = _mm_dpbusd_avx_epi32(_mm_setzero_si128(), low7, b);
    #else
            msb  = _mm_dpbusd_epi32(_mm_setzero_si128(), msb, b);  // 0 or 128
            low7 = _mm_dpbusd_epi32(_mm_setzero_si128(), low7, b);
    #endif
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
        static FORCE_INLINE R sub(R a, R b) { return simde_mm256_sub_epi8(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm256_min_epi8(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm256_max_epi8(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return simde_mm256_avg_epu8(a, b); }
        static FORCE_INLINE R dot2_u7i8(R a, R b) { return simde_mm256_maddubs_epi16(a, b); }

        /// Compute 4-element dot product of [u7x32] and [i8x32] then accumulate into [i32x8].
        static FORCE_INLINE void dot4_u7i8_accum(R &acc, R a, R b)
        {
#if defined(USE_VNNI)
    #if !defined(USE_AVX512)
            acc = _mm256_dpbusd_avx_epi32(acc, a, b);
    #else
            acc = _mm256_dpbusd_epi32(acc, a, b);
    #endif
#else
            R product0 = simde_mm256_maddubs_epi16(a, b);
            product0   = simde_mm256_madd_epi16(product0, simde_mm256_set1_epi16(1));
            acc        = simde_mm256_add_epi32(acc, product0);
#endif
        }

        /// Compute 4-element dot product of [i8x32] and [i8x32] then accumulate into [i32x8].
        static FORCE_INLINE void dot4_i8i8_accum(R &acc, R a, R b)
        {
            const R highest_bit = simde_mm256_set1_epi8(0x80);

            R msb  = simde_mm256_and_si256(a, highest_bit);
            R low7 = simde_mm256_andnot_si256(highest_bit, a);

#if defined(USE_VNNI)
    #if !defined(USE_AVX512)
            msb  = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), msb, b);  // 0 or 128
            low7 = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), low7, b);
    #else
            msb  = _mm256_dpbusd_epi32(_mm256_setzero_si256(), msb, b);  // 0 or 128
            low7 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), low7, b);
    #endif
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
        static FORCE_INLINE R sub(R a, R b) { return _mm512_sub_epi8(a, b); }
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

#ifdef USE_NEON
    template <>
    struct VecOp<int8_t, NEON>
    {
        typedef int8_t        T;
        typedef int8x16_t     R;
        static FORCE_INLINE R setzero() { return vdupq_n_s8(0); }
        static FORCE_INLINE R set1(T a) { return vdupq_n_s8(a); }
        static FORCE_INLINE R add(R a, R b) { return vaddq_s8(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return vsubq_s8(a, b); }
        static FORCE_INLINE R min(R a, R b) { return vminq_s8(a, b); }
        static FORCE_INLINE R max(R a, R b) { return vmaxq_s8(a, b); }
        static FORCE_INLINE R avg(R a, R b)
        {
            return vreinterpretq_s8_u8(vrhaddq_u8(vreinterpretq_u8_s8(a), vreinterpretq_u8_s8(b)));
        }

        static FORCE_INLINE int16x8_t dot2_u7i8(R a, R b)
        {
            uint8x16_t a_u8 = vreinterpretq_u8_s8(a);
            int16x8_t  tl   = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(a_u8))),
                                     vmovl_s8(vget_low_s8(b)));
            int16x8_t  th   = vmulq_s16(vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(a_u8))),
                                     vmovl_s8(vget_high_s8(b)));
            return vqaddq_s16(vuzp1q_s16(tl, th), vuzp2q_s16(tl, th));
        }

        /// Compute 4-element dot product of [u7x16] and [i8x16] then accumulate into [i32x4].
        static FORCE_INLINE void dot4_u7i8_accum(int32x4_t &acc, R a, R b)
        {
    #if defined(USE_NEON_DOTPROD)
            acc = vdotq_s32(acc, a, b);
    #else
            int16x8_t product0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
            int16x8_t product1 = vmull_high_s8(a, b);
            int16x8_t sum      = vpaddq_s16(product0, product1);
            acc                = vpadalq_s16(acc, sum);
    #endif
        }

        /// Compute 4-element dot product of [i8x16] and [i8x16] then accumulate into [i32x4].
        static FORCE_INLINE void dot4_i8i8_accum(int32x4_t &acc, R a, R b)
        {
    #if defined(USE_NEON_DOTPROD)
            acc = vdotq_s32(acc, a, b);
    #else
            // Widen before adding: two (-128)*(-128) products overflow an i16 sum.
            int16x8_t product0 = vmull_s8(vget_low_s8(a), vget_low_s8(b));
            int16x8_t product1 = vmull_high_s8(a, b);
            int32x4_t sum01    = vpaddq_s32(vpaddlq_s16(product0), vpaddlq_s16(product1));
            acc                = vaddq_s32(acc, sum01);
    #endif
        }
    };
#endif

#ifdef USE_WASM_SIMD
    template <>
    struct VecOp<int8_t, WASM_SIMD> : VecOpSIWASM_SIMD
    {
        typedef int8_t        T;
        static FORCE_INLINE R set1(T a) { return wasm_i8x16_splat(a); }
        static FORCE_INLINE R add(R a, R b) { return wasm_i8x16_add(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return wasm_i8x16_sub(a, b); }
        static FORCE_INLINE R min(R a, R b) { return wasm_i8x16_min(a, b); }
        static FORCE_INLINE R max(R a, R b) { return wasm_i8x16_max(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return wasm_u8x16_avgr(a, b); }

        static FORCE_INLINE R dot2_u7i8(R a, R b)
        {
    #ifdef USE_WASM_SIMD_RELAXED
            return wasm_i16x8_relaxed_dot_i8x16_i7x16(b, a);
    #else
            const R prod_low  = wasm_i16x8_extmul_low_i8x16(a, b);
            const R prod_high = wasm_i16x8_extmul_high_i8x16(a, b);
            const R prod_even = wasm_i16x8_shuffle(prod_low, prod_high, 0, 2, 4, 6, 8, 10, 12, 14);
            const R prod_odd  = wasm_i16x8_shuffle(prod_low, prod_high, 1, 3, 5, 7, 9, 11, 13, 15);
            return wasm_i16x8_add(prod_even, prod_odd);
    #endif
        }

        /// Compute 4-element dot product of [u7x16] and [i8x16] then accumulate into [i32x4].
        static FORCE_INLINE void dot4_u7i8_accum(R &acc, R a, R b)
        {
    #ifdef USE_WASM_SIMD_RELAXED
            acc = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(b, a, acc);
    #else
            const R prod_low  = wasm_i16x8_extmul_low_i8x16(a, b);
            const R prod_high = wasm_i16x8_extmul_high_i8x16(a, b);
            const R psum_low  = wasm_i32x4_extadd_pairwise_i16x8(prod_low);
            const R psum_high = wasm_i32x4_extadd_pairwise_i16x8(prod_high);
            const R psum_even = wasm_i32x4_shuffle(psum_low, psum_high, 0, 2, 4, 6);
            const R psum_odd  = wasm_i32x4_shuffle(psum_low, psum_high, 1, 3, 5, 7);
            const R psum      = wasm_i32x4_add(psum_even, psum_odd);
            acc               = wasm_i32x4_add(acc, psum);
    #endif
        }

        /// Compute 4-element dot product of [i8x16] and [i8x16] then accumulate into [i32x4].
        static FORCE_INLINE void dot4_i8i8_accum(R &acc, R a, R b)
        {
            const R prod_low  = wasm_i16x8_extmul_low_i8x16(a, b);
            const R prod_high = wasm_i16x8_extmul_high_i8x16(a, b);
            const R psum_low  = wasm_i32x4_extadd_pairwise_i16x8(prod_low);
            const R psum_high = wasm_i32x4_extadd_pairwise_i16x8(prod_high);
            const R psum_even = wasm_i32x4_shuffle(psum_low, psum_high, 0, 2, 4, 6);
            const R psum_odd  = wasm_i32x4_shuffle(psum_low, psum_high, 1, 3, 5, 7);
            const R psum      = wasm_i32x4_add(psum_even, psum_odd);
            acc               = wasm_i32x4_add(acc, psum);
        }
    };
#endif

    template <>
    struct VecOp<int8_t, SCALAR> : VecOp<int8_t, SSE>
    {
        static FORCE_INLINE R dot2_u7i8(R a, R b)
        {
            simde__m128i_private r_;
            simde__m128i_private a_ = simde__m128i_to_private(a), b_ = simde__m128i_to_private(b);

            r_.i16[0] = a_.i8[0] * b_.i8[0] + a_.i8[1] * b_.i8[1];
            r_.i16[1] = a_.i8[2] * b_.i8[2] + a_.i8[3] * b_.i8[3];
            r_.i16[2] = a_.i8[4] * b_.i8[4] + a_.i8[5] * b_.i8[5];
            r_.i16[3] = a_.i8[6] * b_.i8[6] + a_.i8[7] * b_.i8[7];
            r_.i16[4] = a_.i8[8] * b_.i8[8] + a_.i8[9] * b_.i8[9];
            r_.i16[5] = a_.i8[10] * b_.i8[10] + a_.i8[11] * b_.i8[11];
            r_.i16[6] = a_.i8[12] * b_.i8[12] + a_.i8[13] * b_.i8[13];
            r_.i16[7] = a_.i8[14] * b_.i8[14] + a_.i8[15] * b_.i8[15];

            return simde__m128i_from_private(r_);
        }

        /// Compute 4-element dot product of [u7x16] and [i8x16] then accumulate into [i32x4].
        static FORCE_INLINE void dot4_u7i8_accum(R &acc, R a, R b)
        {
            simde__m128i_private r_ = simde__m128i_to_private(acc);
            simde__m128i_private a_ = simde__m128i_to_private(a), b_ = simde__m128i_to_private(b);

            r_.i32[0] +=
                +(int32_t)a_.i8[0] * (int32_t)b_.i8[0] + (int32_t)a_.i8[1] * (int32_t)b_.i8[1]
                + (int32_t)a_.i8[2] * (int32_t)b_.i8[2] + (int32_t)a_.i8[3] * (int32_t)b_.i8[3];
            r_.i32[1] +=
                +(int32_t)a_.i8[4] * (int32_t)b_.i8[4] + (int32_t)a_.i8[5] * (int32_t)b_.i8[5]
                + (int32_t)a_.i8[6] * (int32_t)b_.i8[6] + (int32_t)a_.i8[7] * (int32_t)b_.i8[7];
            r_.i32[2] +=
                +(int32_t)a_.i8[8] * (int32_t)b_.i8[8] + (int32_t)a_.i8[9] * (int32_t)b_.i8[9]
                + (int32_t)a_.i8[10] * (int32_t)b_.i8[10] + (int32_t)a_.i8[11] * (int32_t)b_.i8[11];
            r_.i32[3] +=
                +(int32_t)a_.i8[12] * (int32_t)b_.i8[12] + (int32_t)a_.i8[13] * (int32_t)b_.i8[13]
                + (int32_t)a_.i8[14] * (int32_t)b_.i8[14] + (int32_t)a_.i8[15] * (int32_t)b_.i8[15];

            acc = simde__m128i_from_private(r_);
        }

        static FORCE_INLINE void dot4_i8i8_accum(R &acc, R a, R b) { dot4_u7i8_accum(acc, a, b); }
    };

    template <>
    struct VecOp<int16_t, SSE> : VecOpSISSE
    {
        typedef int16_t       T;
        static FORCE_INLINE R set1(T a) { return simde_mm_set1_epi16(a); }
        static FORCE_INLINE R add(R a, R b) { return simde_mm_add_epi16(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return simde_mm_sub_epi16(a, b); }
        static FORCE_INLINE R mulhi(R a, R b) { return simde_mm_mulhi_epi16(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm_min_epi16(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm_max_epi16(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return simde_mm_avg_epu16(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return simde_mm_srai_epi16(a, i);
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
        static FORCE_INLINE R sub(R a, R b) { return simde_mm256_sub_epi16(a, b); }
        static FORCE_INLINE R mulhi(R a, R b) { return simde_mm256_mulhi_epi16(a, b); }
        static FORCE_INLINE R min(R a, R b) { return simde_mm256_min_epi16(a, b); }
        static FORCE_INLINE R max(R a, R b) { return simde_mm256_max_epi16(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return simde_mm256_avg_epu16(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return simde_mm256_srai_epi16(a, i);
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
        static FORCE_INLINE R sub(R a, R b) { return _mm512_sub_epi16(a, b); }
        static FORCE_INLINE R mulhi(R a, R b) { return _mm512_mulhi_epi16(a, b); }
        static FORCE_INLINE R min(R a, R b) { return _mm512_min_epi16(a, b); }
        static FORCE_INLINE R max(R a, R b) { return _mm512_max_epi16(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return _mm512_avg_epu16(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return _mm512_srai_epi16(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return _mm512_slli_epi16(a, i);
        }
        static FORCE_INLINE R dot2(R a, R b) { return _mm512_madd_epi16(a, b); }
    };
#endif

#ifdef USE_NEON
    template <>
    struct VecOp<int16_t, NEON>
    {
        typedef int16_t       T;
        typedef int16x8_t     R;
        static FORCE_INLINE R setzero() { return vdupq_n_s16(0); }
        static FORCE_INLINE R set1(T a) { return vdupq_n_s16(a); }
        static FORCE_INLINE R add(R a, R b) { return vaddq_s16(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return vsubq_s16(a, b); }
        static FORCE_INLINE R mulhi(R a, R b)
        {
            int16x4_t a3210  = vget_low_s16(a);
            int16x4_t b3210  = vget_low_s16(b);
            int32x4_t ab3210 = vmull_s16(a3210, b3210);  // 3333222211110000
            int32x4_t ab7654 = vmull_high_s16(a, b);
            return vuzp2q_s16(vreinterpretq_s16_s32(ab3210), vreinterpretq_s16_s32(ab7654));
        }
        static FORCE_INLINE R min(R a, R b) { return vminq_s16(a, b); }
        static FORCE_INLINE R max(R a, R b) { return vmaxq_s16(a, b); }
        static FORCE_INLINE R avg(R a, R b)
        {
            return vreinterpretq_s16_u16(
                vrhaddq_u16(vreinterpretq_u16_s16(a), vreinterpretq_u16_s16(b)));
        }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return vshrq_n_s16(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return vreinterpretq_s16_u16(vshlq_n_u16(vreinterpretq_u16_s16(a), i));
        }
        static FORCE_INLINE int32x4_t dot2(R a, R b)
        {
            int32x4_t low  = vmull_s16(vget_low_s16(a), vget_low_s16(b));
            int32x4_t high = vmull_high_s16(a, b);
            return vpaddq_s32(low, high);
        }
    };
#endif

#ifdef USE_WASM_SIMD
    template <>
    struct VecOp<int16_t, WASM_SIMD> : VecOpSIWASM_SIMD
    {
        typedef int16_t       T;
        static FORCE_INLINE R set1(T a) { return wasm_i16x8_splat(a); }
        static FORCE_INLINE R add(R a, R b) { return wasm_i16x8_add(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return wasm_i16x8_sub(a, b); }
        static FORCE_INLINE R mulhi(R a, R b)
        {
            const R lo = wasm_i32x4_extmul_low_i16x8(a, b);
            const R hi = wasm_i32x4_extmul_high_i16x8(a, b);
            return wasm_i16x8_shuffle(lo, hi, 1, 3, 5, 7, 9, 11, 13, 15);
        }
        static FORCE_INLINE R min(R a, R b) { return wasm_i16x8_min(a, b); }
        static FORCE_INLINE R max(R a, R b) { return wasm_i16x8_max(a, b); }
        static FORCE_INLINE R avg(R a, R b) { return wasm_u16x8_avgr(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return wasm_i16x8_shr(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return wasm_i16x8_shl(a, i);
        }
        static FORCE_INLINE R dot2(R a, R b) { return wasm_i32x4_dot_i16x8(a, b); }
    };
#endif

    template <>
    struct VecOp<int16_t, SCALAR> : VecOp<int16_t, SSE>
    {};

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

#ifdef USE_NEON
    template <>
    struct VecOp<int32_t, NEON>
    {
        typedef int32_t       T;
        typedef int32x4_t     R;
        static FORCE_INLINE R setzero() { return vdupq_n_s32(0); }
        static FORCE_INLINE R set1(T a) { return vdupq_n_s32(a); }
        static FORCE_INLINE R add(R a, R b) { return vaddq_s32(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return vsubq_s32(a, b); }
        static FORCE_INLINE R min(R a, R b) { return vminq_s32(a, b); }
        static FORCE_INLINE R max(R a, R b) { return vmaxq_s32(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return vshrq_n_s32(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return vreinterpretq_s32_u32(vshlq_n_u32(vreinterpretq_u32_s32(a), i));
        }
        static FORCE_INLINE T reduceadd(R a) { return vaddvq_s32(a); }

        /// Horizontal sum [i32x4] of 4 groups into one [i32x4].
        static FORCE_INLINE R hsum4(R sum0, R sum1, R sum2, R sum3)
        {
            sum0 = vpaddq_s32(sum0, sum1);
            sum2 = vpaddq_s32(sum2, sum3);
            sum0 = vpaddq_s32(sum0, sum2);
            return sum0;
        }
    };
#endif

#ifdef USE_WASM_SIMD
    template <>
    struct VecOp<int32_t, WASM_SIMD> : VecOpSIWASM_SIMD
    {
        typedef int32_t       T;
        static FORCE_INLINE R set1(T a) { return wasm_i32x4_splat(a); }
        static FORCE_INLINE R add(R a, R b) { return wasm_i32x4_add(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return wasm_i32x4_sub(a, b); }
        static FORCE_INLINE R min(R a, R b) { return wasm_i32x4_min(a, b); }
        static FORCE_INLINE R max(R a, R b) { return wasm_i32x4_max(a, b); }
        template <int i>
        static FORCE_INLINE R srai(R a)
        {
            return wasm_i32x4_shr(a, i);
        }
        template <int i>
        static FORCE_INLINE R slli(R a)
        {
            return wasm_i32x4_shl(a, i);
        }
        static FORCE_INLINE T reduceadd(R a)
        {
            auto hi64  = wasm_i32x4_shuffle(a, a, 2, 3, 0, 1);
            auto sum64 = wasm_i32x4_add(a, hi64);
            auto hi32  = wasm_i32x4_shuffle(sum64, sum64, 1, 0, 3, 2);
            auto sum32 = wasm_i32x4_add(sum64, hi32);
            return wasm_i32x4_extract_lane(sum32, 0);
        }

        /// Horizontal sum [i32x4] of 4 groups into one [i32x4].
        static FORCE_INLINE R hsum4(R sum0, R sum1, R sum2, R sum3)
        {
            sum0 = wasm_i32x4_add(wasm_i32x4_shuffle(sum0, sum1, 0, 2, 4, 6),
                                  wasm_i32x4_shuffle(sum0, sum1, 1, 3, 5, 7));
            sum2 = wasm_i32x4_add(wasm_i32x4_shuffle(sum2, sum3, 0, 2, 4, 6),
                                  wasm_i32x4_shuffle(sum2, sum3, 1, 3, 5, 7));
            sum0 = wasm_i32x4_add(wasm_i32x4_shuffle(sum0, sum2, 0, 2, 4, 6),
                                  wasm_i32x4_shuffle(sum0, sum2, 1, 3, 5, 7));
            return sum0;
        }
    };
#endif

    template <>
    struct VecOp<int32_t, SCALAR> : VecOp<int32_t, SSE>
    {
        static FORCE_INLINE T reduceadd(R a)
        {
            simde__m128i_private a_ = simde__m128i_to_private(a);
            return a_.i32[0] + a_.i32[1] + a_.i32[2] + a_.i32[3];
        }

        /// Horizontal sum [i32x4] of 4 groups into one [i32x4].
        static FORCE_INLINE R hsum4(R sum0, R sum1, R sum2, R sum3)
        {
            simde__m128i_private r_;
            simde__m128i_private sum0_ = simde__m128i_to_private(sum0);
            simde__m128i_private sum1_ = simde__m128i_to_private(sum1);
            simde__m128i_private sum2_ = simde__m128i_to_private(sum2);
            simde__m128i_private sum3_ = simde__m128i_to_private(sum3);

            r_.i32[0] = sum0_.i32[0] + sum0_.i32[1] + sum0_.i32[2] + sum0_.i32[3];
            r_.i32[1] = sum1_.i32[0] + sum1_.i32[1] + sum1_.i32[2] + sum1_.i32[3];
            r_.i32[2] = sum2_.i32[0] + sum2_.i32[1] + sum2_.i32[2] + sum2_.i32[3];
            r_.i32[3] = sum3_.i32[0] + sum3_.i32[1] + sum3_.i32[2] + sum3_.i32[3];

            return simde__m128i_from_private(r_);
        }
    };

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
        static FORCE_INLINE R min(R a, R b) { return _mm512_min_ps(a, b); }
        static FORCE_INLINE R max(R a, R b) { return _mm512_max_ps(a, b); }
        static FORCE_INLINE R fmadd(R a, R b, R c) { return _mm512_fmadd_ps(a, b, c); }
        static FORCE_INLINE T reduceadd(R a) { return _mm512_reduce_add_ps(a); }
    };
#endif

#ifdef USE_NEON
    template <>
    struct VecOp<float, NEON>
    {
        typedef float         T;
        typedef float32x4_t   R;
        static FORCE_INLINE R setzero() { return vdupq_n_f32(0.0f); }
        static FORCE_INLINE R set1(T a) { return vdupq_n_f32(a); }
        static FORCE_INLINE R add(R a, R b) { return vaddq_f32(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return vsubq_f32(a, b); }
        static FORCE_INLINE R mul(R a, R b) { return vmulq_f32(a, b); }
        static FORCE_INLINE R min(R a, R b) { return vminq_f32(a, b); }
        static FORCE_INLINE R max(R a, R b) { return vmaxq_f32(a, b); }
        static FORCE_INLINE R fmadd(R a, R b, R c) { return vfmaq_f32(c, b, a); }
        static FORCE_INLINE T reduceadd(R a) { return vaddvq_f32(a); }
    };
#endif

#ifdef USE_WASM_SIMD
    template <>
    struct VecOp<float, WASM_SIMD>
    {
        typedef float         T;
        typedef v128_t        R;
        static FORCE_INLINE R setzero() { return wasm_f32x4_const(0.f, 0.f, 0.f, 0.f); }
        static FORCE_INLINE R set1(T a) { return wasm_f32x4_splat(a); }
        static FORCE_INLINE R add(R a, R b) { return wasm_f32x4_add(a, b); }
        static FORCE_INLINE R sub(R a, R b) { return wasm_f32x4_sub(a, b); }
        static FORCE_INLINE R mul(R a, R b) { return wasm_f32x4_mul(a, b); }
        static FORCE_INLINE R min(R a, R b) { return wasm_f32x4_min(a, b); }
        static FORCE_INLINE R max(R a, R b) { return wasm_f32x4_max(a, b); }
        static FORCE_INLINE R fmadd(R a, R b, R c)
        {
    #ifdef USE_WASM_SIMD_RELAXED
            return wasm_f32x4_relaxed_madd(a, b, c);
    #else
            return wasm_f32x4_add(wasm_f32x4_mul(a, b), c);
    #endif
        }
        static FORCE_INLINE T reduceadd(R a)
        {
            R shuf = wasm_i32x4_shuffle(a, a, 1, 1, 3, 3);  // broadcast elements 3,1 to 2,0
            R sums = wasm_f32x4_add(a, shuf);
            shuf   = wasm_i32x4_shuffle(sums, sums, 2, 3, 2, 3);  // high half -> low half
            sums   = wasm_f32x4_add(sums, shuf);
            return wasm_f32x4_extract_lane(sums, 0);
        }
    };
#endif

    template <>
    struct VecOp<float, SCALAR> : VecOp<float, SSE>
    {
        static FORCE_INLINE T reduceadd(R a)
        {
            simde__m128_private a_ = simde__m128_to_private(a);
            return (a_.f32[0] + a_.f32[1]) + (a_.f32[2] + a_.f32[3]);
        }
    };

}  // namespace detail

}  // namespace Evaluation::simd
