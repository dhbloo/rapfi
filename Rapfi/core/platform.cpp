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
typedef bool (*fun1_t)(LOGICAL_PROCESSOR_RELATIONSHIP,
                       PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                       PDWORD);
typedef bool (*fun2_t)(USHORT, PGROUP_AFFINITY);
typedef bool (*fun3_t)(HANDLE, CONST GROUP_AFFINITY *, PGROUP_AFFINITY);
typedef bool (*fun4_t)(USHORT, PGROUP_AFFINITY, USHORT, PUSHORT);
typedef WORD (*fun5_t)();
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

namespace WinProcGroup {

#ifndef _WIN32

void bindThisThread(size_t) {}

#else

/// best_node() retrieves logical processor information using Windows specific
/// API and returns the best node id for the thread with index idx. Original
/// code from Texel by Peter Österlund.

int best_node(size_t idx)
{
    int   threads      = 0;
    int   nodes        = 0;
    int   cores        = 0;
    DWORD returnLength = 0;
    DWORD byteOffset   = 0;

    // Early exit if the needed API is not available at runtime
    HMODULE k32  = GetModuleHandle("Kernel32.dll");
    auto    fun1 = (fun1_t)(void (*)())GetProcAddress(k32, "GetLogicalProcessorInformationEx");
    if (!fun1)
        return -1;

    // First call to GetLogicalProcessorInformationEx() to get returnLength.
    // We expect the call to fail due to null buffer.
    if (fun1(RelationAll, nullptr, &returnLength))
        return -1;

    // Once we know returnLength, allocate the buffer
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *buffer, *ptr;
    ptr = buffer = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)malloc(returnLength);

    // Second call to GetLogicalProcessorInformationEx(), now we expect to succeed
    if (!fun1(RelationAll, buffer, &returnLength)) {
        free(buffer);
        return -1;
    }

    while (byteOffset < returnLength) {
        if (ptr->Relationship == RelationNumaNode)
            nodes++;

        else if (ptr->Relationship == RelationProcessorCore) {
            cores++;
            threads += (ptr->Processor.Flags == LTP_PC_SMT) ? 2 : 1;
        }

        assert(ptr->Size);
        byteOffset += ptr->Size;
        ptr = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(((char *)ptr) + ptr->Size);
    }

    free(buffer);

    std::vector<int> groups;

    // Run as many threads as possible on the same node until core limit is
    // reached, then move on filling the next node.
    for (int n = 0; n < nodes; n++)
        for (int i = 0; i < cores / nodes; i++)
            groups.push_back(n);

    // In case a core has more than one logical processor (we assume 2) and we
    // have still threads to allocate, then spread them evenly across available
    // nodes.
    for (int t = 0; t < threads - cores; t++)
        groups.push_back(t % nodes);

    // If we still have more threads than the total number of logical processors
    // then return -1 and let the OS to decide what to do.
    return idx < groups.size() ? groups[idx] : -1;
}

/// bindThisThread() set the group affinity of the current thread

void bindThisThread(size_t idx)
{
    // Use only local variables to be thread-safe
    int node = best_node(idx);

    if (node == -1)
        return;

    // Early exit if the needed API are not available at runtime
    HMODULE k32  = GetModuleHandle("Kernel32.dll");
    auto    fun2 = (fun2_t)(void (*)())GetProcAddress(k32, "GetNumaNodeProcessorMaskEx");
    auto    fun3 = (fun3_t)(void (*)())GetProcAddress(k32, "SetThreadGroupAffinity");
    auto    fun4 = (fun4_t)(void (*)())GetProcAddress(k32, "GetNumaNodeProcessorMask2");
    auto    fun5 = (fun5_t)(void (*)())GetProcAddress(k32, "GetMaximumProcessorGroupCount");

    if (!fun2 || !fun3)
        return;

    if (!fun4 || !fun5) {
        GROUP_AFFINITY affinity;
        if (fun2(node, &affinity))                         // GetNumaNodeProcessorMaskEx
            fun3(GetCurrentThread(), &affinity, nullptr);  // SetThreadGroupAffinity
    }
    else {
        // If a numa node has more than one processor group, we assume they are
        // sized equal and we spread threads evenly across the groups.
        USHORT elements, returnedElements;
        elements                 = fun5();  // GetMaximumProcessorGroupCount
        GROUP_AFFINITY *affinity = (GROUP_AFFINITY *)malloc(elements * sizeof(GROUP_AFFINITY));
        if (fun4(node, affinity, elements, &returnedElements))  // GetNumaNodeProcessorMask2
            fun3(GetCurrentThread(),
                 &affinity[idx % returnedElements],
                 nullptr);  // SetThreadGroupAffinity
        free(affinity);
    }
}

#endif

}  // namespace WinProcGroup

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
