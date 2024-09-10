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

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

// -------------------------------------------------
// Chrono time type forward declaration

/// Time represents a time point/period value in milliseconds
typedef int64_t Time;

Time now();

// -------------------------------------------------
// Math/Template helper functions

template <typename ElemType, int Size>
constexpr int arraySize(ElemType (&arr)[Size])
{
    return Size;
}

template <class T>
constexpr T power(T base, unsigned exponent)
{
    return (exponent == 0) ? 1
           : (exponent % 2 == 0)
               ? power(base, exponent / 2) * power(base, exponent / 2)
               : base * power(base, (exponent - 1) / 2) * power(base, (exponent - 1) / 2);
}

constexpr bool isPowerOfTwo(uint64_t x)
{
    return (x & (x - 1)) == 0;
}

constexpr uint64_t floorLog2(uint64_t x)
{
    return x == 1 ? 0 : 1 + floorLog2(x >> 1);
}

/// Returns the nearest power of two less than x
constexpr uint64_t floorPowerOfTwo(uint64_t x)
{
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return x ^ (x >> 1);
}

/// Returns combination number, (n + m - 1)! / ((n - 1)! * m!)
constexpr uint32_t combineNumber(uint32_t n, uint32_t m)
{
    uint64_t r = 1;
    for (unsigned int i = n; i < m + n; i++)
        r *= i;
    for (unsigned int i = 2; i <= m; i++)
        r /= i;
    return (uint32_t)r;
}

// -------------------------------------------------
// String helper functions

std::string &trimInplace(std::string &s);
std::string &upperInplace(std::string &s);
std::string &replaceAll(std::string &str, std::string_view from, std::string_view to);
std::vector<std::string_view>
split(std::string_view s, std::string_view delims = "\n", bool includeEmpty = false);

std::string timeText(Time time);
std::string nodesText(uint64_t nodes);
std::string speedText(uint64_t nodesPerSecond);

// -------------------------------------------------
// Container helpers

template <typename Container, typename ElemType>
bool contains(const Container &c, ElemType elem)
{
    for (auto it = c.cbegin(); it != c.end(); it++) {
        if (*it == elem)
            return true;
    }
    return false;
}

template <class T, size_t Size, size_t... Sizes>
struct MultiDimNativeArray
{
    using Nested = typename MultiDimNativeArray<T, Sizes...>::type;
    using type   = Nested[Size];
};

template <class T, size_t Size>
struct MultiDimNativeArray<T, Size>
{
    using type = T[Size];
};

/// Type alias to a multi-dimentional native array.
template <class T, size_t... Sizes>
using MDNativeArray = typename MultiDimNativeArray<T, Sizes...>::type;

// -------------------------------------------------
// Fast Pesudo Random Number Generator

/// PRNG struct is a fast generator based on SplitMix64
/// See <https://xoroshiro.di.unimi.it/splitmix64.c>
class PRNG
{
public:
    typedef uint64_t result_type;

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }

    PRNG(uint64_t seed = now()) : x(seed) {}

    uint64_t operator()()
    {
        uint64_t z = (x += 0x9e3779b97f4a7c15);
        z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z          = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

private:
    uint64_t x;
};

// -------------------------------------------------
// String encoding conversion

/// Convert a string from the config's legacy file codepage to utf-8 encoding.
std::string LegacyFileCPToUTF8(std::string str);

/// Convert a string from the console input codepage to utf-8 encoding.
std::string ConsoleCPToUTF8(std::string str);

/// Convert a string from utf-8 encoding to the console output codepage.
std::string UTF8ToConsoleCP(std::string utf8str);

// -------------------------------------------------
// File system-related helpers

/// Convert a filesystem path from a multi-byte string using console input codepage.
std::filesystem::path pathFromConsoleString(const std::string &path);

/// Convert a path back to multi-byte string using console output codepage.
std::string pathToConsoleString(const std::filesystem::path &path);

/// Make a file name list by iterating all files in a directory path (recursively).
/// @param dirpath Directory path
/// @param extensions A list of extension to filter files in directory
/// @return A list of all matched filenames
std::vector<std::string> listAllFilesInDirRecursively(const std::string              &dirpath,
                                                      const std::vector<std::string> &extensions);

/// Make a file name list from a list of paths (file name or directory path).
/// @param paths A list of filenames and directory paths
/// @param extensions A list of extension to filter files in directory
/// @return A list of all matched filenames
std::vector<std::string> makeFileListFromPathList(const std::vector<std::string> &paths,
                                                  const std::vector<std::string> &extensions);

/// Ensure existance of a directory path. Directory is created if not exists.
/// If raiseException is true, when failed to create, an filesystem::filesystem_error is raised.
/// @return Whether directory exists (or created) if raiseException is false.
bool ensureDir(std::string dirpath, bool raiseException = true);

// -------------------------------------------------
// Engine Version

/// Returns the engine major/minor/revision version numbers.
std::tuple<int, int, int> getVersionNumbers();

/// Returns the engine version information.
std::string getVersionInfo();

/// Returns the engine information.
std::string getEngineInfo();
