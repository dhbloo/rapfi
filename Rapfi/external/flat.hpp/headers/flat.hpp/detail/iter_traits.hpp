/*******************************************************************************
 * This file is part of the "https://github.com/blackmatov/flat.hpp"
 * For conditions of distribution and use, see copyright notice in LICENSE.md
 * Copyright (C) 2019-2021, by Matvey Cherevko (blackmatov@gmail.com)
 ******************************************************************************/

#pragma once

#include <iterator>
#include <type_traits>

namespace flat_hpp::detail
{
    template < typename InputIter >
    using iter_value_type = typename std::iterator_traits<InputIter>::value_type;

    template < typename InputIter >
    using iter_key_type = std::remove_const_t<typename iter_value_type<InputIter>::first_type>;

    template < typename InputIter >
    using iter_mapped_type = typename iter_value_type<InputIter>::second_type;
}
