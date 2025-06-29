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
#include "../core/utils.h"
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

}  // namespace CommandLine

std::filesystem::path configPath = "config.toml";

std::filesystem::path resolvedConfigPath;

bool allowInternalConfig = true;

bool loadConfig()
{
    // Absolute path will be used directly
    if (configPath.is_absolute())
        resolvedConfigPath = configPath;
    // Try resolve relative path from the current working directory
    else if (std::filesystem::exists(configPath))
        resolvedConfigPath = configPath;
    // If can not be found in current directory, try to resolve from the binary directory
    else if (std::filesystem::exists(CommandLine::binaryDirectory / configPath))
        resolvedConfigPath = CommandLine::binaryDirectory / configPath;
    // Otherwise, we will try to load from the internal config if allowed.
    else
        resolvedConfigPath.clear();

    bool success = false;
    // Try to load from the resolved config path.
    if (!resolvedConfigPath.empty()) {
        std::ifstream configFile(resolvedConfigPath);
        MESSAGEL("Load config from " << pathToConsoleString(resolvedConfigPath));
        success = Config::loadConfig(configFile);
    }
    // Fallback to internal config if the external config failed to load.
    else if (allowInternalConfig) {
        if (!Config::InternalConfig.empty()) {
            std::istringstream internalConfig(Config::InternalConfig);
            success = Config::loadConfig(internalConfig);
        }
        else
            ERRORL("This version is not built with an internal config. "
                   "Must specify an external config!");
    }

    if (success && Config::ClearHashAfterConfigLoaded)
        Search::Threads.clear(true);

    return success;
}

std::filesystem::path getModelFullPath(std::filesystem::path modelPath)
{
    // First check if the modelPath is absolute
    if (modelPath.is_absolute())
        return modelPath;

    // Then check if the modelPath is relative to the current working directory
    if (std::filesystem::exists(modelPath))
        return modelPath;

    // If not found, and we did load from external config, check if the modelPath
    // is relative to the config file directory
    if (!resolvedConfigPath.empty()) {
        auto modelPathinConfigDir = resolvedConfigPath.parent_path() / modelPath;
        if (std::filesystem::exists(modelPathinConfigDir))
            return modelPathinConfigDir;
    }

    // If not found, check if the modelPath is relative to the binary directory
    auto modelPathinBinaryDir = CommandLine::binaryDirectory / modelPath;
    if (std::filesystem::exists(modelPathinBinaryDir))
        return modelPathinBinaryDir;

    // If still not found, just return the original modelPath
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
