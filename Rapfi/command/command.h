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

    /// Get the default config path, which is the "config.toml" under
    /// the current working directory or the binary executable directory.
    std::filesystem::path getDefaultConfigPath();
}  // namespace CommandLine

// -------------------------------------------------
// Config loading

/// Global path of the config file.
extern std::filesystem::path configPath;

/// Whether to allow fallback to internal config if the specified file is not found.
extern bool allowInternalConfig;

/// loadConfig() trys to load config according to the following order:
/// 1. Load from the current config path. If config file exists but fails to load,
///    it will not continue to load other config.
/// 2. Try to load from the default config path, which is the "config.toml" in the
///    current working directory or the binary executable directory.
/// 3. If the above two steps fail, and allowInternalConfig is true, it will
///    try to load from the internal config string. Internal config is only available
///    when the program is built with it.
bool loadConfig();

/// getModelFullPath() first trys to find the right model from the modelPath.
/// Model path can be absolute, relative from current working directory or
/// relative from config file directory.
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
