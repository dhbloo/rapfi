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

#include "time.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// -------------------------------------------------
// String helpers

/// Strip leading and trailing space characters (only `' '`, not arbitrary whitespace).
std::string &trimInplace(std::string &s);
std::string &upperInplace(std::string &s);
std::string &replaceAll(std::string &str, std::string_view from, std::string_view to);

/// Split `s` on any of the characters in `delims`. Empty fields between consecutive delimiters
/// are dropped unless `includeEmpty` is true.
/// @note The returned views point into `s`; the caller must keep the source string alive for
///     as long as the views are used. Do not pass a temporary as `s`.
std::vector<std::string_view>
split(std::string_view s, std::string_view delims = "\n", bool includeEmpty = false);

/// Human-readable formatters: "3ms", "12s", "5min", "2h" / "950", "12K", "3M", "1G", "5T" /
/// "812K nps", etc. Magnitudes are chosen so the result is at most four significant digits.
std::string timeText(Time time);
std::string nodesText(uint64_t nodes);
std::string speedText(uint64_t nodesPerSecond);

// -------------------------------------------------
// String encoding conversion (no-op on non-Windows targets)

/// Convert a string from a legacy (non-UTF-8) codepage to UTF-8. `codepage == 0` selects the
/// system's active codepage. No-op on non-Windows targets. Callers pass the database's
/// configured codepage explicitly so this helper stays free of any config dependency.
std::string LegacyFileCPToUTF8(std::string str, uint16_t codepage = 0);

/// Convert a string from the console input codepage to UTF-8.
std::string ConsoleCPToUTF8(std::string str);

/// Convert a UTF-8 string to the console output codepage.
std::string UTF8ToConsoleCP(std::string utf8str);
