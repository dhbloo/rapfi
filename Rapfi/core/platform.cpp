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

#include "platform.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

#ifdef _WIN32
    #if _WIN32_WINNT < 0x0601
        #undef _WIN32_WINNT
        #define _WIN32_WINNT 0x0601  // Force-pull the API prototypes we need.
    #endif

    #ifndef NOMINMAX
        #define NOMINMAX
    #endif

    #include <windows.h>

// Some processor-group APIs are missing on older Windows versions, so we resolve them at
// runtime via GetProcAddress instead of linking against them directly.
// Match the real Windows API signatures exactly: BOOL return (not bool) and WINAPI calling
// convention, so the function-pointer casts are ABI-correct on every architecture.
extern "C" {
typedef BOOL(WINAPI *fun1_t)(PPROCESSOR_NUMBER, PUSHORT);
typedef BOOL(WINAPI *fun2_t)(USHORT, PGROUP_AFFINITY);
typedef BOOL(WINAPI *fun3_t)(HANDLE, CONST GROUP_AFFINITY *, PGROUP_AFFINITY);
typedef BOOL(WINAPI *fun4_t)(USHORT, PGROUP_AFFINITY, USHORT, PUSHORT);
typedef WORD(WINAPI *fun5_t)();
typedef BOOL(WINAPI *fun6_t)(HANDLE, PGROUP_AFFINITY, USHORT);
}
#endif

#if defined(__linux__) && !defined(__ANDROID__)
    #include <algorithm>
    #include <fstream>
    #include <optional>
    #include <sched.h>
    #include <set>
    #include <sstream>
    #include <string>
    #include <unistd.h>
#endif

#if defined(__APPLE__) || defined(__ANDROID__) || defined(__OpenBSD__) \
    || (defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC) && !defined(_WIN32))
    #define POSIXALIGNEDALLOC
    #include <cstdlib>
#endif

// -------------------------------------------------
// NUMA awareness

namespace Numa {

#if defined(_WIN64)

/// Build a vector mapping each online logical processor (in `[group, number]` order) to a
/// NUMA node id, with processors interleaved across nodes so that the first N threads land
/// on N distinct nodes (rather than filling node 0 first).
///
/// 1. Walk every online processor with `GetNumaProcessorNodeEx` to learn its physical node.
/// 2. Subdivide each node further by processor group, so a node that spans groups counts as
///    several logical "split nodes" (avoids the scheduler biasing toward one group).
/// 3. Emit a round-robin order across split nodes.
static std::vector<int> getThreadIdToNodeMapping()
{
    HMODULE k32 = GetModuleHandle("Kernel32.dll");
    if (!k32)
        return {};

    auto gnpne = reinterpret_cast<fun1_t>(GetProcAddress(k32, "GetNumaProcessorNodeEx"));
    if (!gnpne)
        return {};

    const WORD groupCnt = GetActiveProcessorGroupCount();
    size_t     totalLps = 0;
    for (WORD g = 0; g < groupCnt; ++g)
        totalLps += GetActiveProcessorCount(g);

    std::map<int, std::vector<int>>        buckets;  // split-node id -> list of CPU indices
    std::map<std::pair<USHORT, WORD>, int> splitId;  // (node, group) -> split-node id
    int                                    nextSplitId = 0;

    int cpuIndex = 0;  // global, monotonically increasing
    for (WORD g = 0; g < groupCnt; ++g) {
        const DWORD lpInGroup = GetActiveProcessorCount(g);
        for (DWORD p = 0; p < lpInGroup; ++p, ++cpuIndex) {
            PROCESSOR_NUMBER pn {g, static_cast<BYTE>(p), 0};
            USHORT           node = USHRT_MAX;
            if (!gnpne(&pn, &node) || node == USHRT_MAX)
                continue;  // skip offline / unknown

            auto key = std::make_pair(node, g);
            auto it  = splitId.find(key);
            if (it == splitId.end())
                it = splitId.emplace(key, nextSplitId++).first;

            buckets[it->second].push_back(cpuIndex);
        }
    }

    if (buckets.empty())
        return {};

    std::vector<int> mapping;
    mapping.reserve(totalLps);
    for (bool more = true; more;) {
        more = false;
        for (auto &[nodeId, cpus] : buckets)
            if (!cpus.empty()) {
                mapping.push_back(nodeId);
                cpus.pop_back();
                more = true;
            }
    }

    return mapping;
}

NumaNodeId bindThisThread(std::size_t idx)
{
    static const std::vector<int> groups = getThreadIdToNodeMapping();
    const int                     node   = idx < groups.size() ? groups[idx] : -1;
    if (node < 0)
        return DefaultNumaNodeId;

    HMODULE k32    = GetModuleHandle("Kernel32.dll");
    auto    gnpmex = reinterpret_cast<fun2_t>(GetProcAddress(k32, "GetNumaNodeProcessorMaskEx"));
    auto    stga   = reinterpret_cast<fun3_t>(GetProcAddress(k32, "SetThreadGroupAffinity"));
    auto    gnpm2  = reinterpret_cast<fun4_t>(GetProcAddress(k32, "GetNumaNodeProcessorMask2"));
    auto    gmpgc  = reinterpret_cast<fun5_t>(GetProcAddress(k32, "GetMaximumProcessorGroupCount"));
    auto    stscsm = reinterpret_cast<fun6_t>(GetProcAddress(k32, "SetThreadSelectedCpuSetMasks"));

    HANDLE hThread = GetCurrentThread();

    // Preferred path on Windows 11+: SetThreadSelectedCpuSetMasks accepts affinity that spans
    // processor groups.
    if (stscsm && gnpm2 && gmpgc) {
        const USHORT                      groupCount = gmpgc();
        std::unique_ptr<GROUP_AFFINITY[]> ga(new GROUP_AFFINITY[groupCount]);
        USHORT                            returned = 0;

        if (gnpm2(node, ga.get(), groupCount, &returned) && returned) {
            if (stscsm(hThread, ga.get(), returned)) {
                SwitchToThread();  // let the scheduler honour the new affinity
                return static_cast<NumaNodeId>(node);
            }
            // Fall through to the legacy path on failure.
        }
    }

    // Legacy path (Win 7 .. Win 10): one processor group per call.
    if (gnpmex && stga) {
        GROUP_AFFINITY ga {};
        if (gnpmex(node, &ga) && stga(hThread, &ga, nullptr)) {
            SwitchToThread();
            return static_cast<NumaNodeId>(node);
        }
    }

    return DefaultNumaNodeId;
}

#elif defined(__linux__) && !defined(__ANDROID__)

/// Read a kernel "index list" file (e.g. `"0,2-3"`) into a vector of integers.
/// Returns nullopt if the file is missing or empty.
static std::optional<std::vector<int>> readIndexListFromFile(std::string path)
{
    std::ifstream in(path);
    if (!in)
        return std::nullopt;

    std::string s {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
    s.erase(std::remove_if(s.begin(), s.end(), [](int ch) { return std::isspace(ch); }), s.end());
    if (s.empty())
        return std::nullopt;

    std::vector<int>   out;
    std::istringstream iss(s);
    std::string        token;

    while (std::getline(iss, token, ',')) {
        auto dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
            int lo = std::atoi(token.substr(0, dash_pos).c_str());
            int hi = std::atoi(token.substr(dash_pos + 1).c_str());
            for (int v = lo; v <= hi; ++v)
                out.push_back(v);
        }
        else if (!token.empty()) {
            out.push_back(std::atoi(token.c_str()));
        }
    }

    return {out};
}

using CpuIndex  = int;
using NumaTable = std::vector<std::vector<CpuIndex>>;  // node -> list of CPU indices

/// Read the NUMA topology from `/sys/devices/system/node/` and return a per-node list of CPU
/// indices. When `respectAffinity` is true, CPUs outside the calling thread's affinity mask
/// are filtered out. Falls back to a single all-CPU node if /sys is unreadable.
static NumaTable buildNumaTable(bool respectAffinity)
{
    const int onlineCpus = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));

    std::set<CpuIndex> affinity;
    if (respectAffinity) {
        cpu_set_t cur;
        if (sched_getaffinity(0, sizeof(cur), &cur) == 0)
            for (CpuIndex c = 0; c < onlineCpus; ++c)
                if (CPU_ISSET(c, &cur))
                    affinity.insert(c);
    }

    auto allowed = [&](CpuIndex c) { return !respectAffinity || affinity.count(c) == 1; };

    NumaTable tbl;
    bool      fallback = false;

    auto nodeIndices = readIndexListFromFile("/sys/devices/system/node/online");
    if (!nodeIndices)
        fallback = true;
    else {
        // Pre-allocate to the maximum node index to keep table[n] valid even when node ids
        // are sparse.
        int maxNode = *std::max_element(nodeIndices->begin(), nodeIndices->end());
        tbl.resize(maxNode + 1);

        for (int n : *nodeIndices) {
            std::string path = "/sys/devices/system/node/node" + std::to_string(n) + "/cpulist";
            auto        cpuIndices = readIndexListFromFile(path);
            if (!cpuIndices) {
                fallback = true;
                break;
            }

            for (int c : *cpuIndices)
                if (allowed(c))
                    tbl[n].push_back(c);
        }
    }

    if (fallback || tbl.empty()) {
        tbl.assign(1, {});
        for (CpuIndex c = 0; c < onlineCpus; ++c)
            if (allowed(c))
                tbl[0].push_back(c);
    }

    auto newEnd =
        std::remove_if(tbl.begin(), tbl.end(), [](const auto &cpus) { return cpus.empty(); });
    tbl.erase(newEnd, tbl.end());

    for (auto &v : tbl)
        std::sort(v.begin(), v.end());

    return tbl;
}

NumaNodeId bindThisThread(std::size_t idx)
{
    static const NumaTable numaTable = buildNumaTable(true);

    if (numaTable.empty())
        return DefaultNumaNodeId;

    const NumaNodeId node = static_cast<NumaNodeId>(idx % numaTable.size());
    const auto      &cpus = numaTable[node];
    if (cpus.empty())
        return DefaultNumaNodeId;

    cpu_set_t *mask = CPU_ALLOC(cpus.back() + 1);
    if (!mask)
        return DefaultNumaNodeId;

    const std::size_t masksz = CPU_ALLOC_SIZE(cpus.back() + 1);
    CPU_ZERO_S(masksz, mask);
    for (CpuIndex c : cpus)
        CPU_SET_S(c, masksz, mask);

    const int rc = sched_setaffinity(0, masksz, mask);
    CPU_FREE(mask);

    if (rc != 0)
        return DefaultNumaNodeId;

    sched_yield();  // let the scheduler honour the new mask
    return node;
}

#else

/// No-op on unsupported platforms.
NumaNodeId bindThisThread(size_t)
{
    return DefaultNumaNodeId;
}

#endif

}  // namespace Numa

// -------------------------------------------------
// MemAlloc

namespace MemAlloc {

void *alignedAlloc(size_t alignment, size_t size)
{
#if defined(POSIXALIGNEDALLOC)
    void *mem;
    return posix_memalign(&mem, alignment, size) ? nullptr : mem;
#elif defined(_WIN32)
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

void alignedFree(void *ptr)
{
#if defined(POSIXALIGNEDALLOC)
    free(ptr);
#elif defined(_WIN32)
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

/// Try to allocate large pages on Windows. Requires SeLockMemoryPrivilege; returns nullptr
/// if the privilege cannot be obtained or large pages are otherwise unavailable, in which
/// case `alignedLargePageAlloc` falls back to a regular page-aligned allocation.
///
/// Adapted from Stockfish: https://github.com/official-stockfish/Stockfish/pull/2656/commits/
/// 1c53ec970bb77c05d81c266fb22e5db6b1a14ff5
static void *alignedLargePageAllocWindows(size_t size)
{
#ifndef _WIN64
    (void)size;
    return nullptr;
#else
    const size_t largePageSize = GetLargePageMinimum();
    if (!largePageSize)
        return nullptr;

    HANDLE hProcessToken {};
    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                          &hProcessToken))
        return nullptr;

    void *mem = nullptr;
    LUID  luid {};
    if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid)) {
        TOKEN_PRIVILEGES tp {};
        TOKEN_PRIVILEGES prevTp {};
        DWORD            prevTpLen = 0;

        tp.PrivilegeCount           = 1;
        tp.Privileges[0].Luid       = luid;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

        // AdjustTokenPrivileges can return success while leaving the privilege un-enabled;
        // the GetLastError check is what actually confirms the request was honoured.
        if (AdjustTokenPrivileges(hProcessToken,
                                  FALSE,
                                  &tp,
                                  sizeof(TOKEN_PRIVILEGES),
                                  &prevTp,
                                  &prevTpLen)
            && GetLastError() == ERROR_SUCCESS) {
            size = (size + largePageSize - 1) & ~size_t(largePageSize - 1);
            mem  = VirtualAlloc(NULL,
                               size,
                               MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);

            AdjustTokenPrivileges(hProcessToken, FALSE, &prevTp, 0, NULL, NULL);
        }
    }

    CloseHandle(hProcessToken);
    return mem;
#endif
}

void *alignedLargePageAlloc(size_t size)
{
#ifdef _WIN32
    void *mem = alignedLargePageAllocWindows(size);

    if (!mem)
        mem = VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    else {
        // Announce success once per process so the user can confirm large pages took effect.
        // Written directly (not via the iohelper MESSAGE macros) to keep this allocator layer
        // free of any dependency on the protocol output module.
        static bool init = []() {
            std::printf("MESSAGE Large page memory allocation enabled.\n");
            std::fflush(stdout);
            return true;
        }();
    }

    return mem;
#else
    #if defined(__linux__)
    constexpr size_t alignment = 2 * 1024 * 1024;  // assume 2MB huge pages
    #else
    constexpr size_t alignment = 4096;  // assume small (4KiB) pages
    #endif

    size      = ((size + alignment - 1) / alignment) * alignment;
    void *mem = alignedAlloc(alignment, size);
    #if defined(MADV_HUGEPAGE)
    madvise(mem, size, MADV_HUGEPAGE);
    #endif

    return mem;
#endif
}

void alignedLargePageFree(void *ptr)
{
#ifdef _WIN32
    if (ptr && !VirtualFree(ptr, 0, MEM_RELEASE)) {
        // Written directly to stdout (where the protocol's ERROR lines go) to keep this
        // allocator layer free of any dependency on the protocol output module.
        DWORD err = GetLastError();
        std::printf("ERROR Failed to free large page memory. Error code: 0x%lx\n", err);
        std::fflush(stdout);
        std::exit(EXIT_FAILURE);
    }
#else
    alignedFree(ptr);
#endif
}

}  // namespace MemAlloc
