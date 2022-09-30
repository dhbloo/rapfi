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

#include "searchcommon.h"

namespace Search {

bool BalanceMoveLessComparator::operator()(const RootMove &a, const RootMove &b) const
{
    return a.value != b.value
               ? balancedValue(a.value, bias) > balancedValue(b.value, bias)
               : balancedValue(a.previousValue, bias) > balancedValue(b.previousValue, bias);
}

}  // namespace Search
