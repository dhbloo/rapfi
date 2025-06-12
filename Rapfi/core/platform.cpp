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

#include "iohelper.h"

#include <cassert>
#include <map>
#include <vector>

#ifdef _WIN32
    #if _WIN32_WINNT < 0x0601
        #undef _WIN32_WINNT
        #define _WIN32_WINNT 0x0601  // Force to include needed API prototypes
    #endif

    #ifndef NOMINMAX
        #define NOMINMAX
    #endif

    #include <windows.h>
// The needed Windows API for processor groups could be missed from old Windows
// versions, so instead of calling them directly (forcing the linker to resolve
// the calls at compile time), try to load them at runtime. To do this we need
// first to define the corresponding function pointers.
extern "C" {
typedef bool (*fun1_t)(const PPROCESSOR_NUMBER, PUSHORT);
typedef bool (*fun2_t)(USHORT, PGROUP_AFFINITY);
typedef bool (*fun3_t)(HANDLE, CONST GROUP_AFFINITY *, PGROUP_AFFINITY);
typedef bool (*fun4_t)(USHORT, PGROUP_AFFINITY, USHORT, PUSHORT);
typedef WORD (*fun5_t)();
typedef bool (*fun6_t)(HANDLE, PGROUP_AFFINITY, USHORT);
}
#endif

#if defined(__linux__) && !defined(__ANDROID__)
    #include <stdlib.h>
    #include <sys/mman.h>
#endif

#if defined(__APPLE__) || defined(__ANDROID__) || defined(__OpenBSD__) \
    || (defined(__GLIBCXX__) && !defined(_GLIBCXX_HAVE_ALIGNED_ALLOC) && !defined(_WIN32))
    #define POSIXALIGNEDALLOC
    #include <stdlib.h>
#endif

// -------------------------------------------------

namespace Numa {

#if defined(_WIN64)

/// getThreadIdToNodeMapping() build once-per-process vector that maps every
/// logical processor ID (0 … onlineCPUCount-1, enumerated in [group, number] order)
/// to the NUMA-node the CPU belongs to.
///
/// 1. Query each online logical processor with GetNumaProcessorNodeEx() to learn
///    its NUMA node.  Optionally split nodes that span several processor groups.
/// 2. Re-order the CPUs *round-robin* by node, so the first N threads each land
///    on a different node, the next N again, …  This makes the engine fill all
///    NUMA nodes evenly instead of exhausting node-0 first.
/// 3. Return a vector<int> threads → node-id.
std::vector<int> getThreadIdToNodeMapping()
{
    HMODULE k32 = GetModuleHandle("Kernel32.dll");
    if (!k32)
        return {};

    auto gnpne = reinterpret_cast<fun1_t>(GetProcAddress(k32, "GetNumaProcessorNodeEx"));
    if (!gnpne)
        return {};

    // enumerate CPUs
    const WORD groupCnt = GetActiveProcessorGroupCount();
    size_t     totalLps = 0;
    for (WORD g = 0; g < groupCnt; ++g)
        totalLps += GetActiveProcessorCount(g);

    // buckets[node-id] -> list of CPUs that belong to that (possibly split) node
    std::map<int, std::vector<int>>        buckets;  // ordered by node-id
    std::map<std::pair<USHORT, WORD>, int> splitId;  // (node,group) -> split-id
    int                                    nextSplitId = 0;

    int cpuIndex = 0;  // global, monotonically increasing
    for (WORD g = 0; g < groupCnt; ++g) {
        const DWORD lpInGroup = GetActiveProcessorCount(g);
        for (DWORD p = 0; p < lpInGroup; ++p, ++cpuIndex) {
            PROCESSOR_NUMBER pn {g, static_cast<BYTE>(p), 0};
            USHORT           node = USHRT_MAX;
            if (!gnpne(&pn, &node) || node == USHRT_MAX)
                continue;  // skip offline / unknown

            // split physical node by processor-group to avoid scheduler bias
            auto key = std::make_pair(node, g);
            auto it  = splitId.find(key);
            if (it == splitId.end())
                it = splitId.emplace(key, nextSplitId++).first;

            const int splitNodeId = it->second;
            buckets[splitNodeId].push_back(cpuIndex);
        }
    }

    if (buckets.empty())
        return {};  // nothing usable

    // build round-robin order
    std::vector<int> mapping;
    mapping.reserve(totalLps);

    for (bool still = true; still;) {
        still = false;
        for (auto &[nodeId, cpus] : buckets)
            if (!cpus.empty()) {
                mapping.push_back(nodeId);  // take one CPU of this node
                cpus.pop_back();            // remove it
                still = true;               // at least one bucket not empty
            }
    }

    return mapping;  // size == #online logical processors
}

/// bindThisThread() set the group affinity of the current thread, and returns the
/// numa node id for the thread. It uses the best_node() function to determine
/// the best node id for the thread with index idx.

// ----------------------------------------------------------------------------
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

    // 1. Preferred: Windows-11 API  (affinity may span groups)
    if (stscsm && gnpm2 && gmpgc) {
        const USHORT                      groupCount = gmpgc();  // max groups
        std::unique_ptr<GROUP_AFFINITY[]> ga(new GROUP_AFFINITY[groupCount]);
        USHORT                            returned = 0;

        if (gnpm2(node, ga.get(), groupCount, &returned) && returned) {
            if (stscsm(hThread, ga.get(), returned)) {
                SwitchToThread();  // let scheduler apply it
                return static_cast<NumaNodeId>(node);
            }
            /* if the call failed we fall through to old API */
        }
    }

    // 2. Fallback: old one-group API  (Win-7 … Win-10)
    if (gnpmex && stga) {
        GROUP_AFFINITY ga {};
        if (gnpmex(node, &ga) && stga(hThread, &ga, nullptr)) {
            SwitchToThread();
            return static_cast<NumaNodeId>(node);
        }
    }

    return DefaultNumaNodeId;  // nothing worked → scheduler decides
}

#elif defined(__linux__) && !defined(__ANDROID__)

NumaNodeId bindThisThread(size_t idx)
{
    // TODO: Implement Linux NUMA binding logic
    (void)idx;  // suppress unused-parameter compiler warning
}

#else

/// Do no-op and return the default numa node id for unsupported platforms.
NumaNodeId bindThisThread(size_t)
{
    return DefaultNumaNodeId;
}

#endif

}  // namespace Numa

// -------------------------------------------------

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

/// Allocating large page on Windows OS needs to change the privileges of current process,
/// If we failed to acquire the privilege level, or current processor does not support
/// large pages, this function would return nullptr, thus fallback to our alignedAlloc().
/// Code is borrowed from
/// https://github.com/official-stockfish/Stockfish/pull/2656/commits/1c53ec970bb77c05d81c266fb22e5db6b1a14ff5
void *alignedLargePageAllocWindows(size_t size)
{
#ifndef _WIN64
    (void)size;  // suppress unused-parameter compiler warning
    return nullptr;
#else
    HANDLE hProcessToken {};
    LUID   luid {};
    void  *mem = nullptr;

    const size_t largePageSize = GetLargePageMinimum();
    if (!largePageSize)
        return nullptr;

    // We need SeLockMemoryPrivilege, so try to enable it for the process
    if (!OpenProcessToken(GetCurrentProcess(),
                          TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY,
                          &hProcessToken))
        return nullptr;

    if (LookupPrivilegeValue(NULL, SE_LOCK_MEMORY_NAME, &luid)) {
        TOKEN_PRIVILEGES tp {};
        TOKEN_PRIVILEGES prevTp {};
        DWORD            prevTpLen = 0;

        tp.PrivilegeCount           = 1;
        tp.Privileges[0].Luid       = luid;
        tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;

        // Try to enable SeLockMemoryPrivilege. Note that even if AdjustTokenPrivileges()
        // succeeds, we still need to query GetLastError() to ensure that the privileges
        // were actually obtained.
        if (AdjustTokenPrivileges(hProcessToken,
                                  FALSE,
                                  &tp,
                                  sizeof(TOKEN_PRIVILEGES),
                                  &prevTp,
                                  &prevTpLen)
            && GetLastError() == ERROR_SUCCESS) {
            // Round up size to full pages and allocate
            size = (size + largePageSize - 1) & ~size_t(largePageSize - 1);
            mem  = VirtualAlloc(NULL,
                               size,
                               MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);

            // Privilege no longer needed, restore previous state
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
    // Try to allocate large pages
    void *mem = alignedLargePageAllocWindows(size);

    // Fall back to regular, page aligned, allocation if necessary
    if (!mem)
        mem = VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    else {
        static bool _init = []() {
            MESSAGEL("Large page memory allocation enabled.");
            return true;
        }();
    }

    return mem;
#else
    #if defined(__linux__)
    constexpr size_t alignment = 2 * 1024 * 1024;  // assumed 2MB page size
    #else
    constexpr size_t alignment = 4096;  // assumed small page size
    #endif

    // round up to multiples of alignment
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
        DWORD err = GetLastError();
        ERRORL("Failed to free large page memory. Error code: 0x" << std::hex << err << std::dec);
        std::exit(EXIT_FAILURE);
    }
#else
    alignedFree(ptr);
#endif
}

}  // namespace MemAlloc
