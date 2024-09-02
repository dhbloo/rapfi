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

#include "command.h"

#include "../config.h"
#include "../core/iohelper.h"
#include "../search/searchthread.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#ifdef _WIN32
    #include <direct.h>
    #define GETCWD _getcwd
#else
    #include <unistd.h>
    #define GETCWD getcwd
#endif

namespace Command {

namespace CommandLine {

    std::filesystem::path binaryDirectory;

    void init(int argc, char *argv[])
    {
        (void)argc;
#ifdef _WIN32
        wchar_t wpath[1024] = {0};
        GetModuleFileNameW(NULL, wpath, MAX_PATH);
        binaryDirectory = wpath;
        binaryDirectory.remove_filename();
#else
        std::string argv0;             // path+name of the executable binary, as given by argv[0]
        std::string binaryPath;        // path of the executable
        std::string workingDirectory;  // path of the working directory

        // extract the path+name of the executable binary
        argv0 = argv[0];

        // extract the working directory
        workingDirectory = "";
        char  buff[40000];
        char *cwd = GETCWD(buff, 40000);
        if (cwd)
            workingDirectory = cwd;

        const std::string pathSeparator = "/";

        // extract the binary directory path from argv0
        binaryPath = argv0;
        size_t pos = binaryPath.find_last_of("\\/");
        if (pos == std::string::npos)
            binaryPath = "." + pathSeparator;
        else
            binaryPath.resize(pos + 1);

        // pattern replacement: "./" at the start of path is replaced by the working directory
        if (binaryPath.find("." + pathSeparator) == 0)
            binaryPath.replace(0, 1, workingDirectory);

        binaryDirectory = binaryPath;
#endif
    }

    std::filesystem::path getDefaultConfigPath()
    {
        // First, check if there is a config at current working dir.
        if (std::filesystem::exists("config.toml"))
            return "config.toml";
        // If not, we choose the one at current binary dir.
        else
            return binaryDirectory / "config.toml";
    }

}  // namespace CommandLine

std::filesystem::path configPath;

bool allowInternalConfig = true;

bool loadConfig()
{
    bool          success = false;
    std::ifstream configFile(configPath);
    if (configFile.is_open()) {
        MESSAGEL("Load config from " << configPath);
        success = Config::loadConfig(configFile);
    }
    else {
        std::filesystem::path defaultConfigPath = CommandLine::getDefaultConfigPath();
        configFile.open(defaultConfigPath);
        if (configFile.is_open()) {
            MESSAGEL("Load config from " << defaultConfigPath);
            success = Config::loadConfig(configFile);
        }
        else if (allowInternalConfig) {
            if (!Config::InternalConfig.empty()) {
                std::istringstream internalConfig(Config::InternalConfig);
                success = Config::loadConfig(internalConfig);
            }
            else
                ERRORL("This version is not built with an internal config. "
                       "Must specify an external config!");
        }
    }

    if (success && Config::ClearHashAfterConfigLoaded)
        Search::Threads.clear(true);

    return success;
}

std::filesystem::path getModelFullPath(std::filesystem::path modelPath)
{
    // First try to open from current working directory
    if (std::filesystem::exists(modelPath)) {
        return modelPath;
    }

    // If not succeeded, try to open from config directory if path is relative
    if (modelPath.is_relative()) {
        auto modelPathinBinaryDir = CommandLine::binaryDirectory / modelPath;
        if (std::filesystem::exists(modelPathinBinaryDir))
            return modelPathinBinaryDir;
    }

    return modelPath;
}

bool loadModelFromFile(std::filesystem::path modelPath)
{
    modelPath = getModelFullPath(modelPath);
    std::ifstream modelFile(modelPath, std::ios::binary);

    if (modelFile.is_open()) {
        if (Config::loadModel(modelFile))
            return true;
        else
            ERRORL("Failed to load model from ["
                   << modelPath << "]. Please check if model binary file is correct.");
    }
    else
        ERRORL("Unable to open model file: " << modelPath);

    return false;
}

}  // namespace Command
