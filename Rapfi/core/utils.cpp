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

#include "filesystem.h"
#include "string.h"
#include "time.h"

#include <algorithm>
#include <chrono>

#if defined(_WIN32)
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

// -------------------------------------------------
// Time

Time now()
{
    static_assert(sizeof(Time) == sizeof(std::chrono::milliseconds::rep), "Time should be 64 bits");

    auto dur = std::chrono::steady_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
}

// -------------------------------------------------
// String helpers

std::string &trimInplace(std::string &s)
{
    if (s.empty())
        return s;

    s.erase(0, s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
    return s;
}

std::string &upperInplace(std::string &s)
{
    // Cast to unsigned char first: passing a negative char to ::toupper is undefined behaviour.
    std::transform(s.begin(), s.end(), s.begin(), [](char c) {
        return static_cast<char>(::toupper(static_cast<unsigned char>(c)));
    });
    return s;
}

std::string &replaceAll(std::string &str, std::string_view from, std::string_view to)
{
    if (from.empty())  // An empty needle would match at every position and loop forever.
        return str;

    size_t start = 0;
    while ((start = str.find(from, start)) != std::string::npos) {
        str.replace(start, from.length(), to);
        start += to.length();  // Skip past `to` in case `to` contains `from`.
    }
    return str;
}

std::vector<std::string_view> split(std::string_view s, std::string_view delims, bool includeEmpty)
{
    std::vector<std::string_view> output;
    size_t                        first = 0;

    while (first < s.size()) {
        const auto second = s.find_first_of(delims, first);

        if (includeEmpty || first != second)
            output.emplace_back(s.substr(first, second - first));

        if (second == std::string_view::npos)
            break;

        first = second + 1;
    }

    return output;
}

// -------------------------------------------------
// Human-readable formatters

std::string timeText(Time time)
{
    if (time < 10000)
        return std::to_string(time) + "ms";
    else if (time < 1000000)
        return std::to_string(time / 1000) + "s";
    else if (time < 360000000)
        return std::to_string(time / 60000) + "min";
    else
        return std::to_string(time / 3600000) + "h";
}

std::string nodesText(uint64_t nodes)
{
    if (nodes < 10000)
        return std::to_string(nodes);
    else if (nodes < 10000000)
        return std::to_string(nodes / 1000) + "K";
    else if (nodes < 100000000000)
        return std::to_string(nodes / 1000000) + "M";
    else if (nodes < 100000000000000)
        return std::to_string(nodes / 1000000000) + "G";
    else
        return std::to_string(nodes / 1000000000000) + "T";
}

std::string speedText(uint64_t nodesPerSecond)
{
    if (nodesPerSecond < 100000)
        return std::to_string(nodesPerSecond);
    else if (nodesPerSecond < 100000000)
        return std::to_string(nodesPerSecond / 1000) + "K";
    else
        return std::to_string(nodesPerSecond / 1000000) + "M";
}

// -------------------------------------------------
// String encoding conversion
//
// All public conversions go through the same pair of helpers: multi-byte -> wide and
// wide -> multi-byte using a caller-supplied Windows codepage. On non-Windows targets these
// are no-ops (the project's text inputs and outputs are already UTF-8 there).

#if defined(_WIN32)

namespace {

std::wstring mbToWide(const std::string &mb, UINT codepage)
{
    if (mb.empty())
        return {};

    int wideLen = MultiByteToWideChar(codepage, 0, mb.c_str(), (int)mb.length(), nullptr, 0);
    if (wideLen <= 0)
        return {};

    std::wstring wide(wideLen, L'\0');
    if (!MultiByteToWideChar(codepage, 0, mb.c_str(), (int)mb.length(), wide.data(), wideLen))
        return {};

    return wide;
}

std::string wideToMb(const std::wstring &wide, UINT codepage)
{
    if (wide.empty())
        return {};

    int mbLen = WideCharToMultiByte(codepage,
                                    0,
                                    wide.c_str(),
                                    (int)wide.length(),
                                    nullptr,
                                    0,
                                    nullptr,
                                    nullptr);
    if (mbLen <= 0)
        return {};

    std::string mb(mbLen, '\0');
    if (!WideCharToMultiByte(codepage,
                             0,
                             wide.c_str(),
                             (int)wide.length(),
                             mb.data(),
                             mbLen,
                             nullptr,
                             nullptr))
        return {};

    return mb;
}

}  // namespace

#endif  // _WIN32

std::string LegacyFileCPToUTF8(std::string str, uint16_t codepage)
{
#if defined(_WIN32)
    UINT cp = codepage;
    if (cp == 0)  // 0 means "use the system's active codepage".
        cp = GetACP();
    return wideToMb(mbToWide(str, cp), CP_UTF8);
#else
    (void)codepage;
    return str;
#endif
}

std::string ConsoleCPToUTF8(std::string str)
{
#if defined(_WIN32)
    return wideToMb(mbToWide(str, GetConsoleCP()), CP_UTF8);
#else
    return str;
#endif
}

std::string UTF8ToConsoleCP(std::string utf8str)
{
#if defined(_WIN32)
    return wideToMb(mbToWide(utf8str, CP_UTF8), GetConsoleOutputCP());
#else
    return utf8str;
#endif
}

// -------------------------------------------------
// Filesystem helpers

std::filesystem::path pathFromConsoleString(const std::string &path)
{
#if defined(_WIN32)
    // Round-trip through wide chars to preserve non-ASCII characters that the OEM/console
    // codepage cannot represent in std::string.
    return mbToWide(path, GetConsoleCP());
#else
    return path;
#endif
}

std::string pathToConsoleString(const std::filesystem::path &path)
{
#if defined(_WIN32)
    return wideToMb(path.wstring(), GetConsoleOutputCP());
#else
    return path.string();
#endif
}

std::vector<std::string> listAllFilesInDirRecursively(const std::string              &dirpath,
                                                      const std::vector<std::string> &extensions)
{
    auto inExts = [&](std::filesystem::path extension) {
        for (const auto &ext : extensions) {
            if (ext == extension)
                return true;
        }
        return false;
    };

    std::vector<std::string> filenames;
    for (auto &p : std::filesystem::recursive_directory_iterator(pathFromConsoleString(dirpath))) {
        if (p.is_regular_file() && inExts(p.path().extension()))
            filenames.push_back(pathToConsoleString(p.path()));
    }
    return filenames;
}

std::vector<std::string> makeFileListFromPathList(const std::vector<std::string> &paths,
                                                  const std::vector<std::string> &extensions)
{
    std::vector<std::string> filenames;
    for (const auto &path : paths) {
        if (std::filesystem::is_directory(pathFromConsoleString(path))) {
            auto fn = listAllFilesInDirRecursively(path, extensions);
            filenames.insert(filenames.end(), fn.begin(), fn.end());
        }
        else {
            filenames.push_back(path);
        }
    }
    return filenames;
}

bool ensureDir(std::string dirpath, bool raiseException)
{
    std::filesystem::path path = pathFromConsoleString(dirpath);
    std::error_code       ec;
    // Only short-circuit when a *directory* already exists; a regular file at this path is not
    // a satisfied directory, so fall through and let create_directories report the conflict.
    if (std::filesystem::is_directory(path, ec))
        return true;

    if (raiseException)
        return std::filesystem::create_directories(path);
    else
        return std::filesystem::create_directories(path, ec);
}
