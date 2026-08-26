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

#include "core/pos.h"
#include "core/types.h"

#include <cstddef>
#include <iosfwd>
#include <memory>
#include <string>

/// MsgMode represents the message mode that controls how messages are outputed in search.
enum class MsgMode { NONE, BRIEF, NORMAL, UCILIKE };

namespace Config {

extern const std::string InternalConfig;

// -------------------------------------------------
// General options

/// GeneralConfig groups the engine-wide knobs read from the "[general]" TOML
/// table. One mutable global instance (GeneralCfg) holds the live
/// configuration; readGeneral fills it (the three string-enum keys hard-reset
/// to their defaults on unknown values), and runtime mutation (gomocup
/// toggles, benchmark save/restore) is plain field assignment or
/// whole-struct copy.
struct GeneralConfig
{
    /// Should we reload config file before searching each move.
    bool reloadConfigEachMove = false;
    /// Should we clear hash after each time config file is loaded.
    bool clearHashAfterConfigLoaded = true;
    /// Default number of threads if not specified (0 means max hardware concurrency).
    size_t defaultThreadNum = 1;
    /// Message output mode.
    MsgMode messageMode = MsgMode::BRIEF;
    /// Coordinate conversion mode for protocol I/O.
    CoordConversionMode ioCoordMode = CoordConversionMode::NONE;
    /// Default candidate range mode if not specified when creating board.
    CandidateRange defaultCandidateRange = CandidateRange::SQUARE3_LINE4;
    /// Memory reserved for stuff other than hash table in max_memory option.
    size_t memoryReservedMB[RULE_NB] = {0};
    /// Default hash table size (zero for not setting).
    size_t defaultTTSizeKB = 0;
};

/// The global live general configuration.
extern GeneralConfig GeneralCfg;

// -------------------------------------------------
// Config loading & exporting

/// Loads TOML config from a text stream.
/// @param configStream Input config text stream.
/// @return True if config is loaded successfully.
bool loadConfig(std::istream &configStream);

/// Load a LZ4 compressed classical evaluation model from a binary stream.
/// Legacy payloads use the compiled move-score blend rows; newer payloads may
/// append the versioned compact context-policy extension.
/// @return Returns true if loaded successfully, otherwise returns false.
bool loadModel(std::istream &inStream);

/// Exports the current classical evaluation model to a binary stream. A
/// versioned extension is appended when context-policy rows differ from their
/// legacy defaults.
void exportModel(std::ostream &outStream);

}  // namespace Config
