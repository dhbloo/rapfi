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

#include "../core/types.h"
#include "../search/opening.h"

#include <stdexcept>
#include <string_view>
#include <vector>

namespace Command {

/// Parse rule from string. Throws std::invalid_argument if not matched.
/// Throws std::invalid_argument if ruleStr is not valid.
Rule parseRule(std::string_view ruleStr);

/// Dataset type
enum class DatasetType { SimpleBinary, PackedBinary, KatagoNumpy };
enum class DataWriterType {
    PlainText,
    SimpleBinary,
    SimpleBinaryLZ4,
    PackedBinary,
    PackedBinaryLZ4,
    Numpy
};

/// Parse dataset type from string. Throws std::invalid_argument if not matched.
DatasetType parseDatasetType(std::string dsType);
/// Parse data writer type from string. Throws std::invalid_argument if not matched.
DataWriterType parseDataWriterType(std::string dwType);

/// Parse a position from the position string, eg 'h8h7j6'
/// Throws std::invalid_argument if posStr is not correct.
/// @return The parsed pos sequence.
std::vector<Pos> parsePositionString(std::string_view posStr, int boardWidth, int boardHeight);

}  // namespace Command

#ifndef NO_COMMAND_MODULES

// forward declaration
namespace cxxopts {
class ParseResult;
class Options;
}

namespace Command {

/// Add options regarding playing/thinking to the options object.
void addPlayOptions(cxxopts::Options &options);

/// Add opengen options to the options object.
void addOpengenOptions(cxxopts::Options &options, const Opening::OpeningGenConfig& defaultCfg);

/// Parse opengen config from arguments.
/// Throws std::invalid_argument if arguments are not correct.
Opening::OpeningGenConfig parseOpengenConfig(const cxxopts::ParseResult &result);

}  // namespace Command

#endif
