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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

#ifndef _WIN32
    #include <unistd.h>
#endif

namespace Command {

namespace CommandLine {

    std::filesystem::path binaryDirectory;
    std::filesystem::path executablePath;

    namespace {

#ifdef _WIN32
        std::filesystem::path resolveExecutablePath(const char *)
        {
            std::vector<wchar_t> pathBuffer(256);
            while (pathBuffer.size() <= 32768) {
                DWORD length = GetModuleFileNameW(
                    nullptr, pathBuffer.data(), static_cast<DWORD>(pathBuffer.size()));
                if (length == 0)
                    throw std::runtime_error("unable to resolve the executable path");
                if (length < pathBuffer.size())
                    return std::filesystem::path(pathBuffer.data(), pathBuffer.data() + length);
                pathBuffer.resize(pathBuffer.size() * 2);
            }
            throw std::runtime_error("executable path exceeds the Windows path limit");
        }
#else
        std::filesystem::path resolveExecutablePath(const char *argv0)
        {
#ifdef __EMSCRIPTEN__
            // Emscripten uses a synthetic program name that has no backing file.
            return std::filesystem::absolute(argv0);
#else
#ifdef __linux__
            std::vector<char> procPath(256);
            while (procPath.size() <= (1U << 20)) {
                ssize_t length = readlink("/proc/self/exe", procPath.data(), procPath.size());
                if (length < 0)
                    break;
                if (static_cast<size_t>(length) < procPath.size())
                    return std::filesystem::path(procPath.data(),
                                                 procPath.data() + length);
                procPath.resize(procPath.size() * 2);
            }
#endif

            std::filesystem::path argument(argv0);
            std::error_code       ec;
            if (argument.has_parent_path()) {
                if (!std::filesystem::is_regular_file(argument, ec)
                    || access(argument.c_str(), X_OK) != 0)
                    throw std::runtime_error("argv[0] does not name an executable file");
                auto resolved = std::filesystem::weakly_canonical(
                    std::filesystem::absolute(argument), ec);
                if (ec)
                    throw std::runtime_error("unable to canonicalize the executable path");
                return resolved;
            }

            if (const char *pathEnv = std::getenv("PATH")) {
                std::string pathList(pathEnv);
                if (pathList.empty() || pathList.back() == ':')
                    pathList += '.';
                std::stringstream paths(pathList);
                std::string       directory;
                while (std::getline(paths, directory, ':')) {
                    std::filesystem::path candidate =
                        (directory.empty() ? std::filesystem::current_path()
                                           : std::filesystem::path(directory))
                        / argument;
                    if (!std::filesystem::is_regular_file(candidate, ec)
                        || access(candidate.c_str(), X_OK) != 0) {
                        ec.clear();
                        continue;
                    }
                    auto resolved = std::filesystem::weakly_canonical(candidate, ec);
                    if (!ec)
                        return resolved;
                }
            }

            throw std::runtime_error("unable to resolve the executable path through PATH");
#endif
        }
#endif

    }  // namespace

    void init(int argc, char *argv[])
    {
        (void)argc;
        executablePath  = resolveExecutablePath(argv[0]);
        binaryDirectory = executablePath.parent_path();
    }

}  // namespace CommandLine

std::filesystem::path configPath = "config.toml";

std::filesystem::path resolvedConfigPath;

bool allowInternalConfig = true;

bool loadConfig()
{
    // Absolute path will be used directly
    if (configPath.is_absolute()) {
        resolvedConfigPath = configPath;
    }
    // Try resolve relative path from the current working directory
    else if (std::filesystem::exists(configPath))
        resolvedConfigPath = std::filesystem::absolute(configPath);
    // If can not be found in current directory, try to resolve from the binary directory
    else if (std::filesystem::exists(CommandLine::binaryDirectory / configPath))
        resolvedConfigPath =
            std::filesystem::absolute(CommandLine::binaryDirectory / configPath);
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

    if (success && Config::GeneralCfg.clearHashAfterConfigLoaded)
        Search::Engine.clear(true);

    return success;
}

std::filesystem::path getConfigFullPath()
{
    return resolvedConfigPath;
}

std::filesystem::path getModelFullPath(std::filesystem::path modelPath)
{
    // First check if the modelPath is absolute
    if (modelPath.is_absolute())
        return modelPath;

    // Then check if the modelPath is relative to the current working directory
    if (std::filesystem::exists(modelPath))
        return std::filesystem::absolute(modelPath);

    // If not found, and we did load from external config, check if the modelPath
    // is relative to the config file directory
    if (!resolvedConfigPath.empty()) {
        auto modelPathinConfigDir = resolvedConfigPath.parent_path() / modelPath;
        if (std::filesystem::exists(modelPathinConfigDir))
            return std::filesystem::absolute(modelPathinConfigDir);
    }

    // If not found, check if the modelPath is relative to the binary directory
    auto modelPathinBinaryDir = CommandLine::binaryDirectory / modelPath;
    if (std::filesystem::exists(modelPathinBinaryDir))
        return std::filesystem::absolute(modelPathinBinaryDir);

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
