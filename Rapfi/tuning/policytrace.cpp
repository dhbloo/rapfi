/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "policytrace.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <type_traits>

namespace Tuning {
namespace {

    constexpr std::array<uint8_t, 8> TraceMagic        = {'R', 'F', 'P', 'N', 'A', 'T', '0', '0'};
    constexpr std::array<uint8_t, 8> FooterMagic       = {'R', 'F', 'P', 'N', 'E', 'N', 'D', '0'};
    constexpr size_t                 HeaderBytes       = TraceMagic.size();
    constexpr size_t                 FooterBytes       = FooterMagic.size() + 2 * sizeof(uint64_t);
    constexpr size_t                 EventFixedBytes   = 101;
    constexpr size_t                 CandidateBytes    = 77;
    constexpr size_t                 MaximumFrameBytes = sizeof(uint32_t) + EventFixedBytes
                                         + MAX_MOVES * (2 * sizeof(uint16_t) + CandidateBytes)
                                         + sizeof(uint32_t);
    constexpr uint16_t MissingOrdinal = std::numeric_limits<uint16_t>::max();

    class Crc32
    {
    public:
        void add(const void *data, size_t size)
        {
            const auto *bytes = static_cast<const uint8_t *>(data);
            for (size_t i = 0; i < size; i++)
                state = (state >> 8) ^ table()[(state ^ bytes[i]) & 0xffU];
        }

        uint32_t value() const { return state ^ 0xffffffffU; }

    private:
        static const std::array<uint32_t, 256> &table()
        {
            static const std::array<uint32_t, 256> values = [] {
                std::array<uint32_t, 256> result {};
                for (uint32_t i = 0; i < result.size(); i++) {
                    uint32_t value = i;
                    for (int bit = 0; bit < 8; bit++)
                        value = (value >> 1) ^ (0xedb88320U & (0U - (value & 1U)));
                    result[i] = value;
                }
                return result;
            }();
            return values;
        }

        uint32_t state = 0xffffffffU;
    };

    template <typename UInt>
    void appendUnsigned(std::vector<uint8_t> &bytes, UInt value)
    {
        static_assert(std::is_unsigned<UInt>::value, "serialized integer must be unsigned");
        for (size_t i = 0; i < sizeof(UInt); i++)
            bytes.push_back(static_cast<uint8_t>(value >> (i * 8)));
    }

    template <typename Signed>
    void appendSigned(std::vector<uint8_t> &bytes, Signed value)
    {
        using UInt = std::make_unsigned_t<Signed>;
        appendUnsigned(bytes, static_cast<UInt>(value));
    }

    void appendFloat(std::vector<uint8_t> &bytes, float value)
    {
        uint32_t bits;
        static_assert(sizeof(bits) == sizeof(value));
        std::memcpy(&bits, &value, sizeof(bits));
        appendUnsigned(bytes, bits);
    }

    template <typename UInt>
    UInt takeUnsigned(const uint8_t *&cursor, const uint8_t *end)
    {
        static_assert(std::is_unsigned<UInt>::value, "serialized integer must be unsigned");
        if (size_t(end - cursor) < sizeof(UInt))
            throw std::runtime_error("truncated policy trace frame");
        UInt value = 0;
        for (size_t i = 0; i < sizeof(UInt); i++)
            value |= UInt(cursor[i]) << (i * 8);
        cursor += sizeof(UInt);
        return value;
    }

    template <typename Signed>
    Signed takeSigned(const uint8_t *&cursor, const uint8_t *end)
    {
        using UInt = std::make_unsigned_t<Signed>;
        return static_cast<Signed>(takeUnsigned<UInt>(cursor, end));
    }

    float takeFloat(const uint8_t *&cursor, const uint8_t *end)
    {
        const uint32_t bits = takeUnsigned<uint32_t>(cursor, end);
        float          value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    void readExact(std::istream &in, void *data, size_t size, const char *section)
    {
        in.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
        if (in.gcount() != static_cast<std::streamsize>(size))
            throw std::runtime_error(std::string("truncated policy trace ") + section);
    }

    bool validEventEnum(const PolicyTraceEvent &event)
    {
        return event.split <= PolicyTraceSplit::Validation && event.rule < RULE_NB
               && (event.sideToMove == BLACK || event.sideToMove == WHITE)
               && event.domain <= PolicyTraceDomain::Qvcf
               && event.nodeType <= PolicyTraceNodeType::NonPv && event.bestBound <= BOUND_EXACT
               && event.completion == PolicyTraceCompletion::Complete;
    }

    void validateEvent(const PolicyTraceEvent &event)
    {
        auto invalid = [&](const char *field) {
            throw std::runtime_error(std::string("invalid policy trace event ") + field);
        };
        if (!validEventEnum(event))
            invalid("enum");
        if (event.boardSize < 5 || event.boardSize > MAX_BOARD_SIZE)
            invalid("board size");
        if (event.identity.gameId == 0 || event.identity.rootSearchId == 0
            || event.identity.nodeSequence == 0)
            invalid("identity");
        if (event.openingFamily == 0)
            invalid("opening family");
        if (event.ply < 0 || size_t(event.ply) > event.history.size())
            invalid("ply/history");
        if (!std::isfinite(event.depth))
            invalid("depth");
        if (event.history.size() > MAX_MOVES)
            invalid("history size");
        if (event.candidates.empty() || event.candidates.size() > MAX_MOVES)
            invalid("candidate count");
        if (event.policyContext >= 6)
            invalid("policy context");
        if (event.totalNodes == 0)
            invalid("node count");
        if (event.ttMove != Pos::NONE && !event.ttMove.isInBoard(event.boardSize, event.boardSize))
            invalid("TT move");
        if (event.singularExcludedMove != Pos::NONE
            && !event.singularExcludedMove.isInBoard(event.boardSize, event.boardSize))
            invalid("singular excluded move");
        if (event.bestMove != Pos::NONE
            && !event.bestMove.isInBoard(event.boardSize, event.boardSize))
            invalid("best move");
        if (event.rootEligibleMoves.size() > MAX_MOVES
            || (event.nodeType != PolicyTraceNodeType::Root && !event.rootEligibleMoves.empty()))
            invalid("root eligible moves");
        std::set<int> rootMoves;
        for (Pos move : event.rootEligibleMoves)
            if (!move.isInBoard(event.boardSize, event.boardSize)
                || !rootMoves.insert(int(move)).second)
                invalid("root eligible move");
        for (Pos move : event.history)
            if (move != Pos::PASS && !move.isInBoard(event.boardSize, event.boardSize))
                throw std::runtime_error("invalid policy trace history move");

        std::set<int>     moves;
        std::vector<bool> selectionOrdinals(event.candidates.size());
        std::vector<bool> policyOrdinals(event.candidates.size());
        size_t            returnedCandidates = 0;
        size_t            cutoffCandidates   = 0;
        uint16_t          cutoffOrdinal      = MissingOrdinal;
        for (size_t i = 0; i < event.candidates.size(); i++) {
            const PolicyTraceCandidate &candidate = event.candidates[i];
            if (!candidate.move.isInBoard(event.boardSize, event.boardSize)
                || candidate.generatedOrdinal != i
                || candidate.selectionOrdinal >= event.candidates.size()
                || (candidate.policyOrdinal != MissingOrdinal
                    && candidate.policyOrdinal >= event.candidates.size())
                || candidate.filter > PolicyTraceFilter::RootExcluded
                || candidate.disposition == PolicyTraceDisposition::Pending
                || candidate.disposition > PolicyTraceDisposition::Aborted
                || candidate.selfPattern >= PATTERN4_NB || candidate.opponentPattern >= PATTERN4_NB
                || !std::isfinite(candidate.plannedReduction) || candidate.searchCount > 4
                || candidate.bound > BOUND_EXACT)
                throw std::runtime_error("invalid policy trace candidate fields");
            for (size_t searchIndex = 0; searchIndex < candidate.searchCount; searchIndex++)
                if (!std::isfinite(candidate.searchDepths[searchIndex]))
                    throw std::runtime_error("invalid policy trace candidate search depth");
            if (!moves.insert(int(candidate.move)).second
                || selectionOrdinals[candidate.selectionOrdinal])
                throw std::runtime_error("duplicate policy trace candidate or selection ordinal");
            const bool singularExcluded = candidate.move == event.singularExcludedMove;
            const bool rootExcluded     = event.nodeType == PolicyTraceNodeType::Root
                                      && !std::count(event.rootEligibleMoves.begin(),
                                                     event.rootEligibleMoves.end(),
                                                     candidate.move);
            if ((singularExcluded && candidate.filter != PolicyTraceFilter::SingularExcluded)
                || (rootExcluded && candidate.filter != PolicyTraceFilter::RootExcluded)
                || (!singularExcluded && !rootExcluded
                    && (candidate.filter == PolicyTraceFilter::SingularExcluded
                        || candidate.filter == PolicyTraceFilter::RootExcluded)))
                throw std::runtime_error("policy trace search-filter snapshot mismatch");
            selectionOrdinals[candidate.selectionOrdinal] = true;
            if ((candidate.filter == PolicyTraceFilter::None)
                != (candidate.policyOrdinal != MissingOrdinal))
                throw std::runtime_error("policy trace candidate eligibility mismatch");
            if (candidate.policyOrdinal != MissingOrdinal) {
                if (policyOrdinals[candidate.policyOrdinal])
                    throw std::runtime_error("duplicate policy trace returned ordinal");
                policyOrdinals[candidate.policyOrdinal] = true;
                returnedCandidates++;
            }

            const bool searched =
                candidate.disposition == PolicyTraceDisposition::SearchedFailLow
                || candidate.disposition == PolicyTraceDisposition::SearchedAlphaImprovement
                || candidate.disposition == PolicyTraceDisposition::SearchedCutoff;
            const bool pruned =
                candidate.disposition == PolicyTraceDisposition::MoveCountPruned
                || candidate.disposition == PolicyTraceDisposition::TrivialPruned
                || candidate.disposition == PolicyTraceDisposition::PolicyPruned
                || candidate.disposition == PolicyTraceDisposition::DistractDefensePruned;
            if ((searched || pruned
                 || candidate.disposition == PolicyTraceDisposition::DatabaseBlocked)
                && candidate.moveCount == 0)
                throw std::runtime_error("policy trace consumed candidate has zero move count");
            if (searched
                && (candidate.value == VALUE_NONE || candidate.bound == BOUND_NONE
                    || candidate.nodes == 0 || candidate.searchCount == 0))
                throw std::runtime_error("policy trace searched candidate has no result");
            if (candidate.disposition == PolicyTraceDisposition::DatabaseBlocked
                && candidate.searchCount == 0)
                throw std::runtime_error(
                    "policy trace database-blocked candidate was not searched");
            if (candidate.disposition == PolicyTraceDisposition::SearchedCutoff) {
                cutoffCandidates++;
                cutoffOrdinal = candidate.policyOrdinal;
            }
        }
        for (size_t i = 0; i < returnedCandidates; i++)
            if (!policyOrdinals[i])
                throw std::runtime_error("policy trace returned ordinals are not contiguous");
        if (cutoffCandidates > 1 || (cutoffCandidates == 1 && cutoffOrdinal == MissingOrdinal))
            throw std::runtime_error("invalid policy trace cutoff ordering");
        for (const PolicyTraceCandidate &candidate : event.candidates) {
            if (candidate.disposition == PolicyTraceDisposition::PostCutoffCensored
                && (cutoffCandidates != 1 || candidate.policyOrdinal == MissingOrdinal
                    || candidate.policyOrdinal <= cutoffOrdinal))
                throw std::runtime_error("invalid policy trace post-cutoff ordering");
            if (cutoffCandidates == 1 && candidate.policyOrdinal != MissingOrdinal
                && candidate.policyOrdinal > cutoffOrdinal
                && candidate.disposition != PolicyTraceDisposition::PostCutoffCensored)
                throw std::runtime_error("policy trace consumed a candidate after cutoff");
        }
    }

    std::vector<uint8_t> encodeHeader()
    {
        return {TraceMagic.begin(), TraceMagic.end()};
    }

    void validateHeader(const std::array<uint8_t, HeaderBytes> &bytes)
    {
        if (!std::equal(TraceMagic.begin(), TraceMagic.end(), bytes.begin()))
            throw std::runtime_error("invalid policy trace magic");
    }

    uint64_t mix64(uint64_t value)
    {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    std::vector<uint8_t> encodeEvent(const PolicyTraceEvent &event)
    {
        validateEvent(event);
        const size_t payloadBytes = EventFixedBytes + event.history.size() * sizeof(uint16_t)
                                    + event.rootEligibleMoves.size() * sizeof(uint16_t)
                                    + event.candidates.size() * CandidateBytes;
        const size_t frameBytes = sizeof(uint32_t) + payloadBytes + sizeof(uint32_t);
        if (frameBytes > MaximumFrameBytes)
            throw std::length_error("policy trace frame exceeds maximum size");

        std::vector<uint8_t> frame;
        frame.reserve(frameBytes);
        appendUnsigned(frame, static_cast<uint32_t>(frameBytes));
        appendUnsigned(frame, event.identity.gameId);
        appendUnsigned(frame, event.identity.rootSearchId);
        appendUnsigned(frame, event.identity.threadId);
        appendUnsigned(frame, event.identity.nodeSequence);
        appendUnsigned(frame, event.openingFamily);
        appendUnsigned(frame, static_cast<uint8_t>(event.split));
        appendUnsigned(frame, static_cast<uint8_t>(event.rule));
        appendUnsigned(frame, event.boardSize);
        appendUnsigned(frame, static_cast<uint8_t>(event.sideToMove));
        appendUnsigned(frame, static_cast<uint8_t>(event.domain));
        appendUnsigned(frame, static_cast<uint8_t>(event.nodeType));
        appendUnsigned(frame, event.policyContext);
        appendUnsigned(frame, uint8_t(event.p3Active));
        appendUnsigned(frame, uint8_t(event.cutNode));
        appendSigned(frame, event.ply);
        appendFloat(frame, event.depth);
        appendSigned(frame, event.alpha);
        appendSigned(frame, event.beta);
        appendUnsigned(frame, event.positionKey);
        appendSigned(frame, static_cast<int16_t>(int(event.ttMove)));
        appendSigned(frame, static_cast<int16_t>(int(event.singularExcludedMove)));
        appendUnsigned(frame, static_cast<uint16_t>(event.rootEligibleMoves.size()));
        appendUnsigned(frame, static_cast<uint16_t>(event.history.size()));
        appendUnsigned(frame, static_cast<uint16_t>(event.candidates.size()));
        appendUnsigned(frame, event.totalNodes);
        appendUnsigned(frame, event.elapsedTicks);
        appendSigned(frame, static_cast<int16_t>(int(event.bestMove)));
        appendSigned(frame, event.bestValue);
        appendUnsigned(frame, static_cast<uint8_t>(event.bestBound));
        appendUnsigned(frame, static_cast<uint8_t>(event.completion));

        for (Pos move : event.rootEligibleMoves)
            appendSigned(frame, static_cast<int16_t>(int(move)));
        for (Pos move : event.history)
            appendSigned(frame, static_cast<int16_t>(int(move)));
        for (const PolicyTraceCandidate &candidate : event.candidates) {
            appendSigned(frame, static_cast<int16_t>(int(candidate.move)));
            appendUnsigned(frame, candidate.generatedOrdinal);
            appendUnsigned(frame, candidate.selectionOrdinal);
            appendUnsigned(frame, candidate.policyOrdinal);
            appendUnsigned(frame, static_cast<uint8_t>(candidate.filter));
            appendUnsigned(frame, static_cast<uint8_t>(candidate.disposition));
            appendUnsigned(frame, static_cast<uint8_t>(candidate.selfPattern));
            appendUnsigned(frame, static_cast<uint8_t>(candidate.opponentPattern));
            appendSigned(frame, candidate.positionScore);
            appendSigned(frame, candidate.p3Residual);
            appendSigned(frame, candidate.mainHistory);
            appendSigned(frame, candidate.counterMove);
            appendSigned(frame, candidate.finalScore);
            appendSigned(frame, candidate.teacherLogit);
            appendSigned(frame, candidate.alphaBefore);
            appendUnsigned(frame, candidate.moveCount);
            appendFloat(frame, candidate.plannedReduction);
            for (Depth searchDepth : candidate.searchDepths)
                appendFloat(frame, searchDepth);
            appendUnsigned(frame, candidate.searchCount);
            appendSigned(frame, candidate.value);
            appendUnsigned(frame, static_cast<uint8_t>(candidate.bound));
            appendUnsigned(frame, candidate.nodes);
            appendUnsigned(frame, uint8_t(candidate.alphaChanged));
        }
        if (frame.size() + sizeof(uint32_t) != frameBytes)
            throw std::logic_error("policy trace frame size mismatch");

        Crc32 crc;
        crc.add(frame.data(), frame.size());
        appendUnsigned(frame, crc.value());
        return frame;
    }

    PolicyTraceEvent decodeEvent(const std::vector<uint8_t> &frame)
    {
        if (frame.size() < sizeof(uint32_t) + EventFixedBytes + sizeof(uint32_t)
            || frame.size() > MaximumFrameBytes)
            throw std::runtime_error("invalid policy trace frame size");

        Crc32 crc;
        crc.add(frame.data(), frame.size() - sizeof(uint32_t));
        const uint8_t *storedCursor = frame.data() + frame.size() - sizeof(uint32_t);
        const uint8_t *storedEnd    = frame.data() + frame.size();
        if (crc.value() != takeUnsigned<uint32_t>(storedCursor, storedEnd))
            throw std::runtime_error("policy trace frame CRC32 mismatch");

        const uint8_t *cursor = frame.data();
        const uint8_t *end    = frame.data() + frame.size() - sizeof(uint32_t);
        if (takeUnsigned<uint32_t>(cursor, end) != frame.size())
            throw std::runtime_error("policy trace frame length mismatch");

        PolicyTraceEvent event {};
        event.identity.gameId       = takeUnsigned<uint64_t>(cursor, end);
        event.identity.rootSearchId = takeUnsigned<uint64_t>(cursor, end);
        event.identity.threadId     = takeUnsigned<uint32_t>(cursor, end);
        event.identity.nodeSequence = takeUnsigned<uint64_t>(cursor, end);
        event.openingFamily         = takeUnsigned<uint64_t>(cursor, end);
        event.split         = static_cast<PolicyTraceSplit>(takeUnsigned<uint8_t>(cursor, end));
        event.rule          = static_cast<Rule>(takeUnsigned<uint8_t>(cursor, end));
        event.boardSize     = takeUnsigned<uint8_t>(cursor, end);
        event.sideToMove    = static_cast<Color>(takeUnsigned<uint8_t>(cursor, end));
        event.domain        = static_cast<PolicyTraceDomain>(takeUnsigned<uint8_t>(cursor, end));
        event.nodeType      = static_cast<PolicyTraceNodeType>(takeUnsigned<uint8_t>(cursor, end));
        event.policyContext = takeUnsigned<uint8_t>(cursor, end);
        event.p3Active      = bool(takeUnsigned<uint8_t>(cursor, end));
        event.cutNode       = bool(takeUnsigned<uint8_t>(cursor, end));
        event.ply           = takeSigned<int16_t>(cursor, end);
        event.depth         = takeFloat(cursor, end);
        event.alpha         = takeSigned<int32_t>(cursor, end);
        event.beta          = takeSigned<int32_t>(cursor, end);
        event.positionKey   = takeUnsigned<uint64_t>(cursor, end);
        event.ttMove        = Pos(takeSigned<int16_t>(cursor, end));
        event.singularExcludedMove       = Pos(takeSigned<int16_t>(cursor, end));
        const uint16_t rootEligibleCount = takeUnsigned<uint16_t>(cursor, end);
        const uint16_t historyCount      = takeUnsigned<uint16_t>(cursor, end);
        const uint16_t candidateCount    = takeUnsigned<uint16_t>(cursor, end);
        event.totalNodes                 = takeUnsigned<uint64_t>(cursor, end);
        event.elapsedTicks               = takeUnsigned<uint64_t>(cursor, end);
        event.bestMove                   = Pos(takeSigned<int16_t>(cursor, end));
        event.bestValue                  = takeSigned<int32_t>(cursor, end);
        event.bestBound                  = static_cast<Bound>(takeUnsigned<uint8_t>(cursor, end));
        event.completion = static_cast<PolicyTraceCompletion>(takeUnsigned<uint8_t>(cursor, end));
        if (rootEligibleCount > MAX_MOVES || historyCount > MAX_MOVES || candidateCount > MAX_MOVES)
            throw std::runtime_error("policy trace vector count exceeds maximum");
        const size_t expectedRemaining =
            (size_t(rootEligibleCount) + size_t(historyCount)) * sizeof(uint16_t)
            + size_t(candidateCount) * CandidateBytes;
        if (size_t(end - cursor) != expectedRemaining)
            throw std::runtime_error("policy trace vector size mismatch");

        event.rootEligibleMoves.reserve(rootEligibleCount);
        for (size_t i = 0; i < rootEligibleCount; i++)
            event.rootEligibleMoves.emplace_back(takeSigned<int16_t>(cursor, end));
        event.history.reserve(historyCount);
        for (size_t i = 0; i < historyCount; i++)
            event.history.emplace_back(takeSigned<int16_t>(cursor, end));
        event.candidates.reserve(candidateCount);
        for (size_t i = 0; i < candidateCount; i++) {
            PolicyTraceCandidate candidate {};
            candidate.move             = Pos(takeSigned<int16_t>(cursor, end));
            candidate.generatedOrdinal = takeUnsigned<uint16_t>(cursor, end);
            candidate.selectionOrdinal = takeUnsigned<uint16_t>(cursor, end);
            candidate.policyOrdinal    = takeUnsigned<uint16_t>(cursor, end);
            candidate.filter = static_cast<PolicyTraceFilter>(takeUnsigned<uint8_t>(cursor, end));
            candidate.disposition =
                static_cast<PolicyTraceDisposition>(takeUnsigned<uint8_t>(cursor, end));
            candidate.selfPattern      = static_cast<Pattern4>(takeUnsigned<uint8_t>(cursor, end));
            candidate.opponentPattern  = static_cast<Pattern4>(takeUnsigned<uint8_t>(cursor, end));
            candidate.positionScore    = takeSigned<int32_t>(cursor, end);
            candidate.p3Residual       = takeSigned<int32_t>(cursor, end);
            candidate.mainHistory      = takeSigned<int32_t>(cursor, end);
            candidate.counterMove      = takeSigned<int32_t>(cursor, end);
            candidate.finalScore       = takeSigned<int32_t>(cursor, end);
            candidate.teacherLogit     = takeSigned<int32_t>(cursor, end);
            candidate.alphaBefore      = takeSigned<int32_t>(cursor, end);
            candidate.moveCount        = takeUnsigned<uint16_t>(cursor, end);
            candidate.plannedReduction = takeFloat(cursor, end);
            for (Depth &searchDepth : candidate.searchDepths)
                searchDepth = takeFloat(cursor, end);
            candidate.searchCount  = takeUnsigned<uint8_t>(cursor, end);
            candidate.value        = takeSigned<int32_t>(cursor, end);
            candidate.bound        = static_cast<Bound>(takeUnsigned<uint8_t>(cursor, end));
            candidate.nodes        = takeUnsigned<uint64_t>(cursor, end);
            candidate.alphaChanged = bool(takeUnsigned<uint8_t>(cursor, end));
            event.candidates.push_back(candidate);
        }
        if (cursor != end)
            throw std::runtime_error("trailing policy trace frame bytes");
        validateEvent(event);
        return event;
    }

}  // namespace

uint64_t policyOpeningFamily(const std::vector<Pos> &opening)
{
    uint64_t family = 1469598103934665603ULL;
    for (Pos move : opening) {
        const uint16_t bits = static_cast<uint16_t>(static_cast<int16_t>(int(move)));
        family ^= static_cast<uint8_t>(bits);
        family *= 1099511628211ULL;
        family ^= static_cast<uint8_t>(bits >> 8);
        family *= 1099511628211ULL;
    }
    return family ? family : 1;
}

PolicyTraceSplit assignPolicyTraceSplit(uint64_t openingFamily,
                                        uint64_t splitSalt,
                                        uint16_t validationPermille)
{
    if (splitSalt == 0 || validationPermille == 0 || validationPermille >= 1000)
        throw std::invalid_argument("invalid policy split settings");
    return mix64(openingFamily ^ splitSalt) % 1000 < validationPermille
               ? PolicyTraceSplit::Validation
               : PolicyTraceSplit::Train;
}

struct PolicyTraceWriter::Impl
{
    static constexpr size_t BufferCapacity = 4 * 1024 * 1024;

    explicit Impl(const std::filesystem::path &path)
        : finalPath(path)
    {
        temporaryPath = finalPath;
        temporaryPath += ".tmp";
        if (std::filesystem::exists(finalPath) || std::filesystem::exists(temporaryPath))
            throw std::runtime_error("policy trace output already exists");

        out.open(temporaryPath, std::ios::binary);
        if (!out)
            throw std::runtime_error("unable to open temporary policy trace output");
        buffer.reserve(BufferCapacity);
        const std::vector<uint8_t> header = encodeHeader();
        writeBody(header);
    }

    ~Impl()
    {
        if (!published) {
            out.close();
            std::error_code ignored;
            std::filesystem::remove(temporaryPath, ignored);
        }
    }

    void flushBuffer()
    {
        if (buffer.empty())
            return;
        out.write(reinterpret_cast<const char *>(buffer.data()),
                  static_cast<std::streamsize>(buffer.size()));
        if (!out)
            throw std::runtime_error("failed to write policy trace body");
        buffer.clear();
    }

    void writeBody(const std::vector<uint8_t> &bytes)
    {
        if (bytes.size() > BufferCapacity - buffer.size())
            flushBuffer();
        buffer.insert(buffer.end(), bytes.begin(), bytes.end());
    }

    std::filesystem::path                                        finalPath;
    std::filesystem::path                                        temporaryPath;
    std::ofstream                                                out;
    std::vector<uint8_t>                                         buffer;
    uint64_t                                                     events     = 0;
    uint64_t                                                     candidates = 0;
    bool                                                         finalized  = false;
    bool                                                         published  = false;
};

PolicyTraceWriter::PolicyTraceWriter(const std::filesystem::path &path)
    : impl(std::make_unique<Impl>(path))
{}

PolicyTraceWriter::~PolicyTraceWriter() = default;

void PolicyTraceWriter::write(const PolicyTraceEvent &event)
{
    if (!impl || impl->finalized)
        throw std::logic_error("policy trace writer is finalized");
    const std::vector<uint8_t> frame = encodeEvent(event);
    if (event.candidates.size() > std::numeric_limits<uint64_t>::max() - impl->candidates)
        throw std::overflow_error("policy trace candidate count overflow");
    impl->writeBody(frame);
    impl->events++;
    impl->candidates += event.candidates.size();
}

PolicyTraceSummary PolicyTraceWriter::finalize()
{
    if (!impl || impl->finalized)
        throw std::logic_error("policy trace writer is finalized");

    std::vector<uint8_t> footer;
    footer.reserve(FooterBytes);
    footer.insert(footer.end(), FooterMagic.begin(), FooterMagic.end());
    appendUnsigned(footer, impl->events);
    appendUnsigned(footer, impl->candidates);
    if (footer.size() != FooterBytes)
        throw std::logic_error("policy trace footer size mismatch");
    impl->flushBuffer();
    impl->out.write(reinterpret_cast<const char *>(footer.data()),
                    static_cast<std::streamsize>(footer.size()));
    impl->out.flush();
    if (!impl->out)
        throw std::runtime_error("failed to finalize policy trace output");
    impl->out.close();
    impl->finalized = true;

    return {impl->events, impl->candidates};
}

void PolicyTraceWriter::publish()
{
    if (!impl || !impl->finalized || impl->published)
        throw std::logic_error("policy trace is not ready to publish");
    if (std::filesystem::exists(impl->finalPath))
        throw std::runtime_error("policy trace output appeared before publish");
    std::filesystem::rename(impl->temporaryPath, impl->finalPath);
    impl->published = true;
}

PolicyTraceSession::PolicyTraceSession(const std::filesystem::path &path)
    : writer(path)
{}

void PolicyTraceSession::beginGame(uint64_t         newGameId,
                                   uint64_t         newOpeningFamily,
                                   PolicyTraceSplit newSplit)
{
    if (newGameId == 0)
        throw std::invalid_argument("policy trace game ID must be nonzero");
    gameId        = newGameId;
    openingFamily = newOpeningFamily;
    split         = newSplit;
    rootSearchId  = 0;
    nodeSequences.clear();
    gameActive = true;
    rootActive = false;
}

void PolicyTraceSession::beginRootSearch(uint64_t newRootSearchId)
{
    if (!gameActive || newRootSearchId == 0)
        throw std::logic_error("invalid policy trace root-search lifecycle");
    rootSearchId = newRootSearchId;
    nodeSequences.clear();
    rootActive = true;
}

PolicyTraceEventContext PolicyTraceSession::beginEvent(uint32_t threadId)
{
    if (!rootActive)
        throw std::logic_error("policy trace event started outside a root search");
    return {{gameId, rootSearchId, threadId, 0}, openingFamily, split};
}

void PolicyTraceSession::commit(PolicyTraceEvent event)
{
    if (!rootActive || event.identity.gameId != gameId
        || event.identity.rootSearchId != rootSearchId || event.openingFamily != openingFamily
        || event.split != split || event.identity.nodeSequence != 0)
        throw std::logic_error("policy trace event does not belong to the active search");
    if (event.identity.threadId >= nodeSequences.size())
        nodeSequences.resize(size_t(event.identity.threadId) + 1);
    event.identity.nodeSequence = ++nodeSequences[event.identity.threadId];
    writer.write(event);
}

PolicyTraceSummary PolicyTraceSession::finalize()
{
    rootActive = false;
    gameActive = false;
    rethrowFailure();
    return writer.finalize();
}

void PolicyTraceSession::publish()
{
    rethrowFailure();
    writer.publish();
}

void PolicyTraceSession::captureFailure(std::exception_ptr newFailure) noexcept
{
    if (!newFailure)
        return;
    try {
        std::lock_guard<std::mutex> lock(failureMutex);
        if (!failure)
            failure = std::move(newFailure);
    }
    catch (...) {
    }
}

void PolicyTraceSession::rethrowFailure() const
{
    std::exception_ptr captured;
    {
        std::lock_guard<std::mutex> lock(failureMutex);
        captured = failure;
    }
    if (captured)
        std::rethrow_exception(captured);
}

PolicyTraceSummary readPolicyTrace(const std::filesystem::path &path,
                                   const std::function<void(const PolicyTraceEvent &)> &consumer)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("unable to open policy trace " + path.string());

    std::array<uint8_t, HeaderBytes> header;
    readExact(in, header.data(), header.size(), "header");
    validateHeader(header);

    uint64_t                         events = 0, candidates = 0;
    std::array<uint8_t, FooterBytes> footer {};
    for (;;) {
        std::array<uint8_t, sizeof(uint32_t)> prefix;
        readExact(in, prefix.data(), prefix.size(), "frame prefix");
        if (std::equal(prefix.begin(), prefix.end(), FooterMagic.begin())) {
            std::copy(prefix.begin(), prefix.end(), footer.begin());
            readExact(in,
                      footer.data() + prefix.size(),
                      footer.size() - prefix.size(),
                      "footer");
            break;
        }

        const uint8_t *prefixCursor = prefix.data();
        const uint8_t *prefixEnd    = prefix.data() + prefix.size();
        const uint32_t frameBytes   = takeUnsigned<uint32_t>(prefixCursor, prefixEnd);
        if (frameBytes < sizeof(uint32_t) + EventFixedBytes + sizeof(uint32_t)
            || frameBytes > MaximumFrameBytes)
            throw std::runtime_error("invalid policy trace frame length");
        std::vector<uint8_t> frame(frameBytes);
        std::copy(prefix.begin(), prefix.end(), frame.begin());
        readExact(in, frame.data() + prefix.size(), frame.size() - prefix.size(), "frame");

        PolicyTraceEvent event = decodeEvent(frame);
        if (event.candidates.size() > std::numeric_limits<uint64_t>::max() - candidates)
            throw std::overflow_error("policy trace candidate count overflow");
        candidates += event.candidates.size();
        events++;
        if (consumer)
            consumer(event);
    }

    if (!std::equal(FooterMagic.begin(), FooterMagic.end(), footer.begin()))
        throw std::runtime_error("invalid policy trace footer");
    const uint8_t *footerCursor = footer.data() + FooterMagic.size();
    const uint8_t *footerEnd    = footer.data() + footer.size();
    if (takeUnsigned<uint64_t>(footerCursor, footerEnd) != events
        || takeUnsigned<uint64_t>(footerCursor, footerEnd) != candidates)
        throw std::runtime_error("policy trace footer count mismatch");
    if (footerCursor != footerEnd || in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("trailing bytes after policy trace footer");

    return {events, candidates};
}

}  // namespace Tuning
