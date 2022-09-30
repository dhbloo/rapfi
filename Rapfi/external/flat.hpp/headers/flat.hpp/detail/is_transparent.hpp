/*******************************************************************************
 * This file is part of the "https://github.com/blackmatov/flat.hpp"
 * For conditions of distribution and use, see copyright notice in LICENSE.md
 * Copyright (C) 2019-2021, by Matvey Cherevko (blackmatov@gmail.com)
 ******************************************************************************/

#pragma once

#include <type_traits>

namespace flat_hpp::detail
{
    template < typename T, typename U, typename = void >
    struct is_transparent
    : std::false_type {};

    template < typename T, typename U >
    struct is_transparent<T, U, std::void_t<typename T::is_transparent>>
    : std::true_type {};

    template < typename T, typename U >
    inline constexpr bool is_transparent_v = is_transparent<T, U>::value;
}
