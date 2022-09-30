/*******************************************************************************
 * This file is part of the "https://github.com/blackmatov/flat.hpp"
 * For conditions of distribution and use, see copyright notice in LICENSE.md
 * Copyright (C) 2019-2021, by Matvey Cherevko (blackmatov@gmail.com)
 ******************************************************************************/

#pragma once

namespace flat_hpp::detail
{
    template < typename Compare >
    class eq_compare : public Compare {
    public:
        eq_compare() = default;

        eq_compare(const Compare& compare)
        : Compare(compare) {}

        template < typename L, typename R >
        bool operator()(const L& l, const R& r) const {
            return !Compare::operator()(l, r)
                && !Compare::operator()(r, l);
        }
    };
}
