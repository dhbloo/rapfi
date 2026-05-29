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
#include <vector>

/// Build a filesystem::path from a console-encoded multi-byte string. Used for the path
/// arguments that arrive through the piskvork text protocol.
std::filesystem::path pathFromConsoleString(const std::string &path);

/// Render a filesystem::path back into a console-encoded multi-byte string.
std::string pathToConsoleString(const std::filesystem::path &path);

/// All regular files under `dirpath` (recursive) whose extension matches one of `extensions`.
std::vector<std::string> listAllFilesInDirRecursively(const std::string              &dirpath,
                                                      const std::vector<std::string> &extensions);

/// Expand a list of paths: directories are recursively walked and filtered by `extensions`;
/// plain file paths are passed through unchanged.
std::vector<std::string> makeFileListFromPathList(const std::vector<std::string> &paths,
                                                  const std::vector<std::string> &extensions);

/// Ensure directory `dirpath` exists, creating it (and parents) if necessary.
/// If `raiseException` is true, filesystem errors propagate as exceptions; otherwise the
/// return value reports whether the directory exists on completion.
bool ensureDir(std::string dirpath, bool raiseException = true);
