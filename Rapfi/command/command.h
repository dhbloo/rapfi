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

#include <filesystem>
#include <string>

namespace Command {

namespace CommandLine {
    /// Path to the binary executable directory.
    extern std::filesystem::path binaryDirectory;

    /// Initialize the command line module with the startup arguments.
    void init(int argc, char *argv[]);
}  // namespace CommandLine

// -------------------------------------------------
// Config loading

/// Global path of the current config file to load from.
extern std::filesystem::path configPath;

/// Whether to allow fallback to internal config if the specified file is not found.
extern bool allowInternalConfig;

/// loadConfig() trys to load config according to the following order:
/// 1. Load from the current config path. Config file is determined by this order:
///    - If config path is an absolute path, it will be used directly.
///    - Otherwise, it will be first resolved from the current working directory.
///    - If the path can not be found in the current working directory,
///      it will be resolved from the binary executable directory.
/// 2. If the external config failed to load, and allowInternalConfig is true, it will
///    try to load from the internal config string. Internal config is stored in
///    the executable when the program is built with it.
bool loadConfig();

/// getModelFullPath() trys to resolve the correct path for the model file.
/// Model file is determined by this order:
///    - If model path is an absolute path, it will be returned directly.
///    - Otherwise, it will be first resolved from the current working directory.
///    - If not found, it will be first resolved to the directory of the current config file.
///    - If not found, it will be resolved from the binary executable directory.
/// If the model file is not found in any places, it will return the original path.
std::filesystem::path getModelFullPath(std::filesystem::path modelPath);

/// loadModelFromFile() trys to load model from modelPath.
bool loadModelFromFile(std::filesystem::path modelPath);

// -------------------------------------------------
// Command modules entry

void gomocupLoop();
void benchmark();
void opengen(int argc, char *argv[]);
void tuning(int argc, char *argv[]);
void selfplay(int argc, char *argv[]);
void dataprep(int argc, char *argv[]);
void database(int argc, char *argv[]);

}  // namespace Command
