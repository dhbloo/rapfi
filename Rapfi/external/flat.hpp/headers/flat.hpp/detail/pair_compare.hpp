/*******************************************************************************
 * This file is part of the "https://github.com/blackmatov/flat.hpp"
 * For conditions of distribution and use, see copyright notice in LICENSE.md
 * Copyright (C) 2019-2021, by Matvey Cherevko (blackmatov@gmail.com)
 ******************************************************************************/

#pragma once

#include "is_transparent.hpp"

namespace flat_hpp::detail
{
    template < typename Pair, typename Compare >
    class pair_compare : public Compare {
    public:
        pair_compare() = default;

        pair_compare(const Compare& compare)
        : Compare(compare) {}

        bool operator()(
            const typename Pair::first_type& l,
            const typename Pair::first_type& r) const
        {
            return Compare::operator()(l, r);
        }

        bool operator()(const Pair& l, const Pair& r) const {
            return Compare::operator()(l.first, r.first);
        }

        bool operator()(
            const typename Pair::first_type& l,
            const Pair& r) const
        {
            return Compare::operator()(l, r.first);
        }

        bool operator()(
            const Pair& l,
            const typename Pair::first_type& r) const
        {
            return Compare::operator()(l.first, r);
        }

        template < typename K >
        std::enable_if_t<
            is_transparent_v<Compare, K>,
            bool>
        operator()(const K& l, const Pair& r) const {
            return Compare::operator()(l, r.first);
        }

        template < typename K >
        std::enable_if_t<
            is_transparent_v<Compare, K>,
            bool>
        operator()(const Pair& l, const K& r) const {
            return Compare::operator()(l.first, r);
        }
    };
}
