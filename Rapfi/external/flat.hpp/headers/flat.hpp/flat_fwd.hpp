/*******************************************************************************
 * This file is part of the "https://github.com/blackmatov/flat.hpp"
 * For conditions of distribution and use, see copyright notice in LICENSE.md
 * Copyright (C) 2019-2021, by Matvey Cherevko (blackmatov@gmail.com)
 ******************************************************************************/

#pragma once

#include <algorithm>
#include <cassert>
#include <functional>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/eq_compare.hpp"
#include "detail/is_allocator.hpp"
#include "detail/is_sorted.hpp"
#include "detail/is_transparent.hpp"
#include "detail/iter_traits.hpp"
#include "detail/pair_compare.hpp"

namespace flat_hpp
{
    struct sorted_range_t {};
    inline constexpr sorted_range_t sorted_range = sorted_range_t();

    struct sorted_unique_range_t : public sorted_range_t {};
    inline constexpr sorted_unique_range_t sorted_unique_range = sorted_unique_range_t();

    template < typename Key
             , typename Compare = std::less<Key>
             , typename Container = std::vector<Key> >
    class flat_set;

    template < typename Key
             , typename Compare = std::less<Key>
             , typename Container = std::vector<Key> >
    class flat_multiset;

    template < typename Key
             , typename Value
             , typename Compare = std::less<Key>
             , typename Container = std::vector<std::pair<Key, Value>> >
    class flat_map;

    template < typename Key
             , typename Value
             , typename Compare = std::less<Key>
             , typename Container = std::vector<std::pair<Key, Value>> >
    class flat_multimap;
}
