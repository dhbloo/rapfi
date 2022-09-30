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

namespace Database {
class DBStorage;
}

namespace Command {

namespace CommandLine {
    extern std::filesystem::path binaryDirectory;
    void                         init(int argc, char *argv[]);
    std::filesystem::path        getDefaultConfigPath();
}  // namespace CommandLine

// -------------------------------------------------
// Config loading

extern std::filesystem::path configPath;
extern std::filesystem::path overrideModelPath;
extern bool                  allowInternalConfigFallback;

bool                  loadConfig();
std::filesystem::path getModelFullPath(std::filesystem::path modelPath);
bool                  loadModelFromFile(std::filesystem::path modelPath);

// -------------------------------------------------
// Gomocup protocol

namespace GomocupProtocol {
    extern "C" bool gomocupLoopOnce();
}

void gomocupLoop();

// -------------------------------------------------
// Benchmark

void benchmark();

// -------------------------------------------------
// Opening generation

void opengen(int argc, char *argv[]);

// -------------------------------------------------
// Tuning

void tuning(int argc, char *argv[]);

// -------------------------------------------------
// Selfplay

void selfplay(int argc, char *argv[]);

// -------------------------------------------------
// Data preparation

void dataprep(int argc, char *argv[]);

// -------------------------------------------------
// Database operation

void database(int argc, char *argv[]);

}  // namespace Command
