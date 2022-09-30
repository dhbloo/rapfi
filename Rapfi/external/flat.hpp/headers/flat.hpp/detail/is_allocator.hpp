/*******************************************************************************
 * This file is part of the "https://github.com/blackmatov/flat.hpp"
 * For conditions of distribution and use, see copyright notice in LICENSE.md
 * Copyright (C) 2019-2021, by Matvey Cherevko (blackmatov@gmail.com)
 ******************************************************************************/

#pragma once

#include <type_traits>
#include <utility>

namespace flat_hpp::detail
{
    template < typename Allocator, typename = void >
    struct is_allocator : std::false_type {};

    template < typename Allocator >
    struct is_allocator<Allocator, std::void_t<
        typename Allocator::value_type,
        decltype(std::declval<Allocator&>().deallocate(
            std::declval<Allocator&>().allocate(std::size_t{1}),
            std::size_t{1}))
    >> : std::true_type {};
}
