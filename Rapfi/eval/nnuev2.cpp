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

/* NNUEv2 Reference: https://github.com/hzyhhzy/gomoku_nnue
 *   Training:
 * https://github.com/hzyhhzy/gomoku_nnue/blob/e13e7daf8a49c578b1fab5ea0e55c3cfd30c5efe/train_pytorch_v2/model.py#L838
 *   Inference:
 * https://github.com/hzyhhzy/gomoku_nnue/blob/e13e7daf8a49c578b1fab5ea0e55c3cfd30c5efe/nnue1/Eva_nnuev2.cpp
 */

#include "nnuev2.h"

#include "../core/platform.h"
#include "../game/board.h"
#include "simdops.h"
#include "weightloader.h"

#include <cstring>

namespace {

using namespace Evaluation::nnuev2;

constexpr auto Power3 = []() {
    auto pow3 = std::array<int, 16> {};
    for (size_t i = 0; i < pow3.size(); i++)
        pow3[i] = power(3, i);
    return pow3;
}();

constexpr int DX[4] = {1, 0, 1, 1};
constexpr int DY[4] = {0, 1, 1, -1};

/// DPower3[oldColor][newColor]
constexpr int DPower3[4][4] = {
    {0, 1, 0, -1},
    {-1, 0, 0, -2},
    {0, 0, 0, 0},
    {1, 2, 0, 0},
};

static Evaluation::WeightRegistry<Weight> NNUEv2WeightRegistry;

struct TextWeightLoader : WeightLoader<Weight>
{
    std::unique_ptr<Weight> load(std::istream &in)
    {
        auto w = std::make_unique<Weight>();

        // clear map
        std::memset(w->mapping, 0, sizeof(int16_t) * ShapeNum * FeatureNum);

        std::string modelname;
        in >> modelname;
        if (modelname != "v2") {
            ERRORL("Wrong model type:" << modelname);
            return nullptr;
        }

        int param;
        in >> param;
        if (param != GroupSize) {
            ERRORL("Wrong group size:" << param);
            return nullptr;
        }
        in >> param;
        if (param != MLPChannel) {
            ERRORL("Wrong mlp channel:" << param);
            return nullptr;
        }

        std::string varname;

        // mapping
        in >> varname;
        if (varname != "mapping") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        int shapeN;
        in >> shapeN;
        for (int i = 0; i < shapeN; i++) {
            int shapeID;
            in >> shapeID;
            for (int j = 0; j < FeatureNum; j++)
                in >> w->mapping[shapeID][j];
        }

        // g1lr_w
        in >> varname;
        if (varname != "g1lr_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->g1lr_w[i];

        // h1conv_w
        in >> varname;
        if (varname != "h1conv_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int j = 0; j < 6; j++)
            for (int i = 0; i < GroupSize; i++)
                in >> w->h1conv_w[j][i];

        // h1conv_b
        in >> varname;
        if (varname != "h1conv_b") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->h1conv_b[i];

        // h1lr1_w
        in >> varname;
        if (varname != "h1lr1_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->h1lr1_w[i];

        // h1lr2_w
        in >> varname;
        if (varname != "h1lr2_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->h1lr2_w[i];

        // h3lr_w
        in >> varname;
        if (varname != "h3lr_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->h3lr_w[i];

        // h3lr_b
        in >> varname;
        if (varname != "h3lr_b") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->h3lr_b[i];

        // trunkconv1_w
        in >> varname;
        if (varname != "trunkconv1_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int j = 0; j < TrunkConv1GroupSize; j++)
            for (int i = 0; i < GroupSize; i++)
                in >> w->trunkconv1_w[j][i];

        // trunkconv1_b
        in >> varname;
        if (varname != "trunkconv1_b") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->trunkconv1_b[i];

        // trunklr1_w
        in >> varname;
        if (varname != "trunklr1_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->trunklr1_w[i];

        // trunkconv2_w
        in >> varname;
        if (varname != "trunkconv2_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int j = 0; j < 3; j++)
            for (int i = 0; i < GroupSize; i++)
                in >> w->trunkconv2_w[j][i];

        // trunklr2p_w
        in >> varname;
        if (varname != "trunklr2p_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->trunklr2p_w[i];

        // trunklr2p_b
        in >> varname;
        if (varname != "trunklr2p_b") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->trunklr2p_b[i];

        // trunklr2v_w
        in >> varname;
        if (varname != "trunklr2v_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->trunklr2v_w[i];

        // trunklr2v_b
        in >> varname;
        if (varname != "trunklr2v_b") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->trunklr2v_b[i];

        // policy_linear_w
        in >> varname;
        if (varname != "policy_linear_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->policy_linear_w[i];

        // scale_policyInv
        in >> varname;
        if (varname != "scale_policyInv") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        in >> w->scale_policy_inv;

        // scale_beforemlpInv
        in >> varname;
        if (varname != "scale_beforemlpInv") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        in >> w->scale_before_mlp_inv;

        // valuelr_w
        in >> varname;
        if (varname != "valuelr_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->valuelr_w[i];

        // valuelr_b
        in >> varname;
        if (varname != "valuelr_b") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < GroupSize; i++)
            in >> w->valuelr_b[i];

        // mlp_w1
        in >> varname;
        if (varname != "mlp_w1") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int j = 0; j < GroupSize; j++)
            for (int i = 0; i < MLPChannel; i++)
                in >> w->mlp_w1[j][i];

        // mlp_b1
        in >> varname;
        if (varname != "mlp_b1") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < MLPChannel; i++)
            in >> w->mlp_b1[i];

        // mlp_w2
        in >> varname;
        if (varname != "mlp_w2") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int j = 0; j < MLPChannel; j++)
            for (int i = 0; i < MLPChannel; i++)
                in >> w->mlp_w2[j][i];

        // mlp_b2
        in >> varname;
        if (varname != "mlp_b2") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < MLPChannel; i++)
            in >> w->mlp_b2[i];

        // mlp_w3
        in >> varname;
        if (varname != "mlp_w3") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int j = 0; j < MLPChannel; j++)
            for (int i = 0; i < MLPChannel; i++)
                in >> w->mlp_w3[j][i];

        // mlp_b3
        in >> varname;
        if (varname != "mlp_b3") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < MLPChannel; i++)
            in >> w->mlp_b3[i];

        // mlpfinal_w
        in >> varname;
        if (varname != "mlpfinal_w") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int j = 0; j < MLPChannel; j++)
            for (int i = 0; i < 3; i++)
                in >> w->mlpfinal_w[j][i];

        // mlpfinal_b
        in >> varname;
        if (varname != "mlpfinal_b") {
            ERRORL("Wrong parameter name:" << varname);
            return nullptr;
        }
        for (int i = 0; i < 3; i++)
            in >> w->mlpfinal_b[i];

        for (int i = 0; i < 5; i++) {
            w->mlpfinal_w_padding0[i] = 0;
            w->mlpfinal_w_padding1[i] = 0;
        }

        return std::move(w);
    }
};

}  // namespace

namespace Evaluation::nnuev2 {

Accumulator::Accumulator(int boardSize)
    : boardSize(boardSize)
    , numCells(boardSize * boardSize)
    , boardSizeScale(1.0f / (boardSize * boardSize))
    , trunkUpToDate(false)
{
    DEF_BATCH256(int16_t, GroupSize, RegWidth, GroupBatches);

    shapeTable = new MDNativeArray<uint32_t, 4>[numCells];
    g2         = new MDNativeArray<int16_t, 4, GroupSize>[numCells];
    g1sum      = new MDNativeArray<int16_t, GroupSize>[numCells];
    trunk      = new MDNativeArray<int16_t, GroupSize>[numCells];
    h1m        = new int16_t[(boardSize + 10) * (boardSize + 10) * 6 * RegWidth];
    h3         = new int16_t[numCells * RegWidth];
}

Accumulator::~Accumulator()
{
    delete[] shapeTable;
    delete[] g2;
    delete[] g1sum;
    delete[] trunk;
    delete[] h1m;
    delete[] h3;
}

void Accumulator::initShapeTable()
{
    // Clear shape table
    std::fill_n(&shapeTable[0][0], 4 * numCells, 0);
    // Init shape table
    for (int thick = 1; thick <= 5; thick++) {
        for (int i = 0; i < boardSize; i++) {
            int c = 0;
            for (int j = 0; j < thick; j++)
                c += Power3[11 - j];
            shapeTable[(boardSize - 6 + thick) + i * boardSize][0] = c;
            shapeTable[i + (boardSize - 6 + thick) * boardSize][1] = c;
            shapeTable[(boardSize - 6 + thick) + i * boardSize][2] = c;
            shapeTable[i + (boardSize - 6 + thick) * boardSize][2] = c;
            shapeTable[(boardSize - 6 + thick) + i * boardSize][3] = c;
            shapeTable[i + (6 - 1 - thick) * boardSize][3]         = c;
        }
    }

    for (int thick = 1; thick <= 5; thick++) {
        for (int i = 0; i < boardSize; i++) {
            int c = 2 * Power3[11];
            for (int j = 0; j < thick - 1; j++)
                c += Power3[j];
            shapeTable[(6 - 1 - thick) + i * boardSize][0]         = c;
            shapeTable[i + (6 - 1 - thick) * boardSize][1]         = c;
            shapeTable[(6 - 1 - thick) + i * boardSize][2]         = c;
            shapeTable[i + (6 - 1 - thick) * boardSize][2]         = c;
            shapeTable[(6 - 1 - thick) + i * boardSize][3]         = c;
            shapeTable[i + (boardSize - 6 + thick) * boardSize][3] = c;
        }
    }

    for (int a = 1; a <= 5; a++)
        for (int b = 1; b <= 5; b++) {
            int c = 3 * Power3[11];
            for (int i = 0; i < a - 1; i++)
                c += Power3[10 - i];
            for (int i = 0; i < b - 1; i++)
                c += Power3[i];
            shapeTable[(boardSize - 6 + a) + (5 - b) * boardSize][2]             = c;
            shapeTable[(boardSize - 6 + a) * boardSize + (5 - b)][2]             = c;
            shapeTable[(5 - b) + (5 - a) * boardSize][3]                         = c;
            shapeTable[(boardSize - 6 + a) + (boardSize - 6 + b) * boardSize][3] = c;
        }
}

void Accumulator::clear(const Weight &w)
{
    initShapeTable();

    for (int i = 0; i < numCells; i++) {
        DEF_BATCH256(int16_t, GroupSize, RegWidth, GroupBatches);
        for (int b = 0; b < GroupBatches; b++) {
            auto g1sumNew = simde_mm256_setzero_si256();
            for (int dir = 0; dir < 4; dir++) {
                auto *mapPtr = w.mapping[shapeTable[i][dir]] + b * RegWidth;
                // g2 update
                auto neww = simde_mm256_loadu_si256(mapPtr + GroupSize);
                simde_mm256_storeu_si256(g2[i][dir] + b * RegWidth, neww);
                // g1 update
                auto g1  = simde_mm256_loadu_si256(mapPtr);
                g1sumNew = simde_mm256_add_epi16(g1sumNew, g1);
            }
            simde_mm256_storeu_si256(g1sum[i] + b * RegWidth, g1sumNew);
        }
    }

    trunkUpToDate = false;
}

void Accumulator::update(const Weight &w, Color oldColor, Color newColor, int x, int y)
{
    struct OnePointChange
    {
        int8_t   x;
        int8_t   y;
        int16_t  index;
        int16_t  dir;
        uint32_t oldShape;
        uint32_t newShape;
    } changeTable[4 * 11];
    int changeCount = 0;
    int dPower3     = DPower3[oldColor][newColor];

    // Update shape table and record changes
    for (int dir = 0; dir < 4; dir++) {
        for (int dist = -5; dist <= 5; dist++) {
            int xi = x - dist * DX[dir];
            int yi = y - dist * DY[dir];

            // less-branch test: xi < 0 || xi >= boardSize || yi < 0 || yi >= boardSize
            if ((xi | (boardSize - 1 - xi) | yi | (boardSize - 1 - yi)) < 0)
                continue;

            OnePointChange &c        = changeTable[changeCount++];
            c.x                      = xi;
            c.y                      = yi;
            c.dir                    = dir;
            c.index                  = boardSize * yi + xi;
            c.oldShape               = shapeTable[c.index][dir];
            c.newShape               = c.oldShape + dPower3 * Power3[dist + 5];
            shapeTable[c.index][dir] = c.newShape;
            assert(0 <= c.newShape && c.newShape < ShapeNum);
        }
    }

    for (int i = 0; i < changeCount; i++) {
        const OnePointChange &c = changeTable[i];

        DEF_BATCH256(int16_t, GroupSize, RegWidth, GroupBatches);
        for (int b = 0; b < GroupBatches; b++) {
            // g2 update
            auto g2w = simde_mm256_loadu_si256(w.mapping[c.newShape] + b * RegWidth + GroupSize);
            simde_mm256_storeu_si256(g2[c.index][c.dir] + b * RegWidth, g2w);

            // g1 update
            auto  oldw     = simde_mm256_loadu_si256(w.mapping[c.oldShape] + b * RegWidth);
            auto  neww     = simde_mm256_loadu_si256(w.mapping[c.newShape] + b * RegWidth);
            auto *g1sumPtr = g1sum[c.index] + b * RegWidth;
            auto  sumw     = simde_mm256_loadu_si256(g1sumPtr);
            sumw           = simde_mm256_sub_epi16(sumw, oldw);
            sumw           = simde_mm256_add_epi16(sumw, neww);
            simde_mm256_storeu_si256(g1sumPtr, sumw);
        }
    }

    trunkUpToDate = false;
}

void Accumulator::updateTrunk(const Weight &w)
{
    DEF_BATCH256(int16_t, GroupSize, RegWidth, GroupBatches);
    const int h1mNumBytes = sizeof(int16_t) * (boardSize + 10) * (boardSize + 10) * 6 * RegWidth;

    // 一直到trunk计算完毕，不同batch之间都没有交互,所以放在最外层
    for (int batch = 0; batch < GroupBatches; batch++) {
        int offset = batch * RegWidth;

        // 这个数组太大，就不直接int16_t[(BS + 10) * (BS + 10)][6][16]了
        // 完整的卷积是先乘再相加，此处是相乘但还没相加。h1m沿一条线相加得到h1c。
        // 加了5层padding方便后续处理
        std::memset(h1m, 0, h1mNumBytes);
        //-------------------------------------------------------------------------------
        // g1 prelu和h1conv的乘法部分
        auto g1lr_w    = simde_mm256_loadu_si256(w.g1lr_w + offset);
        auto h1conv_w0 = simde_mm256_loadu_si256(w.h1conv_w[0] + offset);
        auto h1conv_w1 = simde_mm256_loadu_si256(w.h1conv_w[1] + offset);
        auto h1conv_w2 = simde_mm256_loadu_si256(w.h1conv_w[2] + offset);
        auto h1conv_w3 = simde_mm256_loadu_si256(w.h1conv_w[3] + offset);
        auto h1conv_w4 = simde_mm256_loadu_si256(w.h1conv_w[4] + offset);
        auto h1conv_w5 = simde_mm256_loadu_si256(w.h1conv_w[5] + offset);

        for (int y = 0; y < boardSize; y++) {
            for (int x = 0; x < boardSize; x++) {
                int      loc1    = y * boardSize + x;                   // 原始loc
                int      loc2    = (y + 5) * (boardSize + 10) + x + 5;  // padding后的loc
                int16_t *h1mBase = h1m + loc2 * 6 * RegWidth;

                auto g1 = simde_mm256_loadu_si256(g1sum[loc1] + offset);
                auto h1 = simde_mm256_max_epi16(g1, simde_mm256_mulhrs_epi16(g1, g1lr_w));
                simde_mm256_storeu_si256(h1mBase + 0 * RegWidth,
                                         simde_mm256_mulhrs_epi16(h1, h1conv_w0));
                simde_mm256_storeu_si256(h1mBase + 1 * RegWidth,
                                         simde_mm256_mulhrs_epi16(h1, h1conv_w1));
                simde_mm256_storeu_si256(h1mBase + 2 * RegWidth,
                                         simde_mm256_mulhrs_epi16(h1, h1conv_w2));
                simde_mm256_storeu_si256(h1mBase + 3 * RegWidth,
                                         simde_mm256_mulhrs_epi16(h1, h1conv_w3));
                simde_mm256_storeu_si256(h1mBase + 4 * RegWidth,
                                         simde_mm256_mulhrs_epi16(h1, h1conv_w4));
                simde_mm256_storeu_si256(h1mBase + 5 * RegWidth,
                                         simde_mm256_mulhrs_epi16(h1, h1conv_w5));
            }
        }

        //-------------------------------------------------------------------------------
        auto h1lr1_w  = simde_mm256_loadu_si256(w.h1lr1_w + offset);
        auto h1lr2_w  = simde_mm256_loadu_si256(w.h1lr2_w + offset);
        auto h1conv_b = simde_mm256_loadu_si256(w.h1conv_b + offset);
        auto h3lr_b   = simde_mm256_loadu_si256(w.h3lr_b + offset);

        for (int y = 0; y < boardSize; y++) {
            for (int x = 0; x < boardSize; x++) {
                int      loc1    = y * boardSize + x;                   // 原始loc
                int      loc2    = (y + 5) * (boardSize + 10) + x + 5;  // padding后的loc
                int16_t *h1mBase = h1m + loc2 * 6 * RegWidth;
                auto     h2sum   = h3lr_b;

                const int dloc2s[4] = {
                    1,
                    boardSize + 10,
                    boardSize + 10 + 1,
                    -boardSize - 10 + 1,
                };
                for (int dir = 0; dir < 4; dir++) {
                    const int dloc2 = dloc2s[dir];

                    // 把所有需要的全都load出来
                    auto h1cm5 = simde_mm256_loadu_si256(h1mBase - 5 * RegWidth * (6 * dloc2 - 1));
                    auto h1cm4 = simde_mm256_loadu_si256(h1mBase - 4 * RegWidth * (6 * dloc2 - 1));
                    auto h1cm3 = simde_mm256_loadu_si256(h1mBase - 3 * RegWidth * (6 * dloc2 - 1));
                    auto h1cm2 = simde_mm256_loadu_si256(h1mBase - 2 * RegWidth * (6 * dloc2 - 1));
                    auto h1cm1 = simde_mm256_loadu_si256(h1mBase - 1 * RegWidth * (6 * dloc2 - 1));
                    auto h1c0  = simde_mm256_loadu_si256(h1mBase);
                    auto h1c1  = simde_mm256_loadu_si256(h1mBase + 1 * RegWidth * (6 * dloc2 + 1));
                    auto h1c2  = simde_mm256_loadu_si256(h1mBase + 2 * RegWidth * (6 * dloc2 + 1));
                    auto h1c3  = simde_mm256_loadu_si256(h1mBase + 3 * RegWidth * (6 * dloc2 + 1));
                    auto h1c4  = simde_mm256_loadu_si256(h1mBase + 4 * RegWidth * (6 * dloc2 + 1));
                    auto h1c5  = simde_mm256_loadu_si256(h1mBase + 5 * RegWidth * (6 * dloc2 + 1));

                    // 11个h1c和h1conv_b全部相加，使用“二叉树”式加法
                    h1cm5 = simde_mm256_adds_epi16(h1cm5, h1conv_b);
                    h1cm3 = simde_mm256_adds_epi16(h1cm3, h1cm4);
                    h1cm1 = simde_mm256_adds_epi16(h1cm1, h1cm2);
                    h1c1  = simde_mm256_adds_epi16(h1c1, h1c0);
                    h1c3  = simde_mm256_adds_epi16(h1c3, h1c2);
                    h1c5  = simde_mm256_adds_epi16(h1c5, h1c4);

                    h1cm5 = simde_mm256_adds_epi16(h1cm5, h1cm3);
                    h1cm1 = simde_mm256_adds_epi16(h1cm1, h1c1);
                    h1c3  = simde_mm256_adds_epi16(h1c3, h1c5);

                    auto h2 = simde_mm256_adds_epi16(h1cm1, h1c3);
                    h2      = simde_mm256_adds_epi16(h1cm5, h2);

                    auto g2t = simde_mm256_loadu_si256(g2[loc1][dir] + offset);
                    h2 = simde_mm256_max_epi16(h2, simde_mm256_mulhrs_epi16(h2, h1lr1_w));  // h1lr1
                    h2 = simde_mm256_adds_epi16(h2, g2t);                                   //+g2
                    h2 = simde_mm256_max_epi16(h2, simde_mm256_mulhrs_epi16(h2, h1lr2_w));  // h1lr2

                    // h2sum=mean(h2)=(h2+h2+h2+h2)/4
                    h2sum = simde_mm256_adds_epi16(h2sum, simde_mm256_srai_epi16(h2, 2));
                }
                // save h3
                simde_mm256_storeu_si256(h3 + loc1 * RegWidth, h2sum);
            }
        }

        //-------------------------------------------------------------------------------
        int16_t *trunk1 = h1m;  // trunkconv2前的trunk，padding=1. (Reuse memory from h1m)
        std::memset(trunk1, 0, sizeof(int16_t) * (boardSize + 2) * (boardSize + 2) * RegWidth);

        auto h3lr_w = simde_mm256_loadu_si256(w.h3lr_w + offset);
        static_assert(TrunkConv1GroupSize == 4, "Only Trunkconv1GroupSize == 4 is supported!");
        auto trunkconv1_b  = simde_mm256_loadu_si256(w.trunkconv1_b + offset);
        auto trunkconv1_w0 = simde_mm256_loadu_si256(w.trunkconv1_w[0] + offset);
        auto trunkconv1_w1 = simde_mm256_loadu_si256(w.trunkconv1_w[1] + offset);
        auto trunkconv1_w2 = simde_mm256_loadu_si256(w.trunkconv1_w[2] + offset);
        auto trunkconv1_w3 = simde_mm256_loadu_si256(w.trunkconv1_w[3] + offset);
        auto trunklr1_w    = simde_mm256_loadu_si256(w.trunklr1_w + offset);
        h3lr_b             = simde_mm256_loadu_si256(w.h3lr_b + offset);

        for (int y = 0; y < boardSize; y++) {
            for (int x = 0; x < boardSize; x++) {
                int  loc1  = y * boardSize + x;                  // 原始loc
                int  loc2  = (y + 1) * (boardSize + 2) + x + 1;  // padding后的loc
                auto trunk = simde_mm256_loadu_si256(h3 + loc1 * RegWidth);
                // h3lr
                trunk = simde_mm256_max_epi16(trunk, simde_mm256_mulhrs_epi16(trunk, h3lr_w));
                // trunkconv1
                trunk = simde_mm256_adds_epi16(
                    trunkconv1_b,
                    simde_mm256_adds_epi16(
                        simde_mm256_adds_epi16(simde_mm256_mulhrs_epi16(
                                                   simde_mm256_permute4x64_epi64(trunk, 0b00000000),
                                                   trunkconv1_w0),
                                               simde_mm256_mulhrs_epi16(
                                                   simde_mm256_permute4x64_epi64(trunk, 0b01010101),
                                                   trunkconv1_w1)),
                        simde_mm256_adds_epi16(simde_mm256_mulhrs_epi16(
                                                   simde_mm256_permute4x64_epi64(trunk, 0b10101010),
                                                   trunkconv1_w2),
                                               simde_mm256_mulhrs_epi16(
                                                   simde_mm256_permute4x64_epi64(trunk, 0b11111111),
                                                   trunkconv1_w3))));

                // trunklr1
                trunk = simde_mm256_max_epi16(trunk, simde_mm256_mulhrs_epi16(trunk, trunklr1_w));

                // save
                simde_mm256_storeu_si256(trunk1 + loc2 * RegWidth, trunk);
            }
        }

        //-------------------------------------------------------------------------------
        auto trunkconv2_w0 = simde_mm256_loadu_si256(w.trunkconv2_w[0] + offset);
        auto trunkconv2_w1 = simde_mm256_loadu_si256(w.trunkconv2_w[1] + offset);
        auto trunkconv2_w2 = simde_mm256_loadu_si256(w.trunkconv2_w[2] + offset);

        for (int y = 0; y < boardSize; y++) {
            for (int x = 0; x < boardSize; x++) {
                int loc1 = y * boardSize + x;                  // 原始loc
                int loc2 = (y + 1) * (boardSize + 2) + x + 1;  // padding后的loc

                auto trunka = simde_mm256_adds_epi16(
                    simde_mm256_adds_epi16(
                        simde_mm256_loadu_si256(trunk1 + (loc2 - (boardSize + 2)) * RegWidth),
                        simde_mm256_loadu_si256(trunk1 + (loc2 + (boardSize + 2)) * RegWidth)),
                    simde_mm256_adds_epi16(
                        simde_mm256_loadu_si256(trunk1 + (loc2 - 1) * RegWidth),
                        simde_mm256_loadu_si256(trunk1 + (loc2 + 1) * RegWidth)));
                auto trunkb = simde_mm256_adds_epi16(
                    simde_mm256_adds_epi16(
                        simde_mm256_loadu_si256(trunk1 + (loc2 - (boardSize + 2) - 1) * RegWidth),
                        simde_mm256_loadu_si256(trunk1 + (loc2 - (boardSize + 2) + 1) * RegWidth)),
                    simde_mm256_adds_epi16(
                        simde_mm256_loadu_si256(trunk1 + (loc2 + (boardSize + 2) - 1) * RegWidth),
                        simde_mm256_loadu_si256(trunk1 + (loc2 + (boardSize + 2) + 1) * RegWidth)));
                auto trunkc = simde_mm256_loadu_si256(trunk1 + loc2 * RegWidth);

                auto t = simde_mm256_adds_epi16(simde_mm256_mulhrs_epi16(trunka, trunkconv2_w1),
                                                simde_mm256_mulhrs_epi16(trunkb, trunkconv2_w2));
                t      = simde_mm256_adds_epi16(t, simde_mm256_mulhrs_epi16(trunkc, trunkconv2_w0));
                simde_mm256_storeu_si256(trunk[loc1] + offset, t);
            }
        }
    }

    trunkUpToDate = true;
}

std::tuple<float, float, float> Accumulator::evaluateValue(const Weight &w)
{
    if (!trunkUpToDate)
        updateTrunk(w);

    // trunklr2v, sum board
    DEF_BATCH256(int16_t, GroupSize, RegWidth16, GroupBatches16);
    float layer0[GroupSize];
    for (int b = 0; b < GroupBatches16; b++) {
        auto vsum0 = simde_mm256_setzero_si256();
        auto vsum1 = simde_mm256_setzero_si256();

        auto trunklr2v_b = simde_mm256_loadu_si256(w.trunklr2v_b + b * RegWidth16);
        auto trunklr2v_w = simde_mm256_loadu_si256(w.trunklr2v_w + b * RegWidth16);
        for (int i = 0; i < numCells; i++) {
            auto t = simde_mm256_loadu_si256(trunk[i] + b * RegWidth16);
            // trunklr2p
            t     = simde_mm256_adds_epi16(t, trunklr2v_b);
            t     = simde_mm256_max_epi16(t, simde_mm256_mulhrs_epi16(t, trunklr2v_w));
            vsum0 = simde_mm256_add_epi32(
                vsum0,
                simde_mm256_cvtepi16_epi32(simde_mm256_extractf128_si256(t, 0)));
            vsum1 = simde_mm256_add_epi32(
                vsum1,
                simde_mm256_cvtepi16_epi32(simde_mm256_extractf128_si256(t, 1)));
        }

        simde_mm256_storeu_ps(layer0 + b * RegWidth16, simde_mm256_cvtepi32_ps(vsum0));
        simde_mm256_storeu_ps(layer0 + b * RegWidth16 + 8, simde_mm256_cvtepi32_ps(vsum1));
    }

    // scale, valuelr
    DEF_BATCH256(float, GroupSize, RegWidth32, GroupBatches32);
    auto scale = simde_mm256_set1_ps(w.scale_before_mlp_inv * boardSizeScale);
    for (int b = 0; b < GroupBatches32; b++) {
        auto valuelr_b = simde_mm256_loadu_ps(w.valuelr_b + b * RegWidth32);
        auto valuelr_w = simde_mm256_loadu_ps(w.valuelr_w + b * RegWidth32);
        auto v         = simde_mm256_loadu_ps(layer0 + b * RegWidth32);
        v              = simde_mm256_mul_ps(v, scale);
        v              = simde_mm256_add_ps(v, valuelr_b);
        v              = simde_mm256_max_ps(v, simde_mm256_mul_ps(v, valuelr_w));
        simde_mm256_storeu_ps(layer0 + b * RegWidth32, v);
    }

    // linear 1
    float layer1[MLPChannel];
    simd::linearLayer<simd::Activation::Relu>(layer1, layer0, w.mlp_w1, w.mlp_b1);

    // linear 2
    float layer2[MLPChannel];
    simd::linearLayer<simd::Activation::Relu>(layer2, layer1, w.mlp_w2, w.mlp_b2);

    // linear 3
    float layer3[MLPChannel];
    simd::linearLayer<simd::Activation::Relu>(layer3, layer2, w.mlp_w3, w.mlp_b3);

    // linear final
    float value[8];
    simd::linearLayer<simd::Activation::None>(value, layer3, w.mlpfinal_w, w.mlpfinal_b);

    return {value[0], value[1], value[2]};
}

void Accumulator::evaluatePolicy(const Weight &w, PolicyBuffer &policyBuffer)
{
    if (!trunkUpToDate)
        updateTrunk(w);

    float policyFinalScale = w.scale_policy_inv * (1.0f / 32768.0f);

    for (int i = 0; i < numCells; i++) {
        if (!policyBuffer.getComputeFlag(i))
            continue;

        auto psum = simde_mm256_setzero_si256();  // int32
        DEF_BATCH256(int16_t, GroupSize, RegWidth, GroupBatches);
        for (int b = 0; b < GroupBatches; b++) {
            // load
            auto t               = simde_mm256_loadu_si256(trunk[i] + b * RegWidth);
            auto trunklr2p_b     = simde_mm256_loadu_si256(w.trunklr2p_b + b * RegWidth);
            auto trunklr2p_w     = simde_mm256_loadu_si256(w.trunklr2p_w + b * RegWidth);
            auto policy_linear_w = simde_mm256_loadu_si256(w.policy_linear_w + b * RegWidth);

            // trunklr2p
            t = simde_mm256_adds_epi16(t, trunklr2p_b);
            t = simde_mm256_max_epi16(t, simde_mm256_mulhrs_epi16(t, trunklr2p_w));

            // policy linear
            t    = simde_mm256_madd_epi16(t, policy_linear_w);
            psum = simde_mm256_add_epi32(t, psum);
        }

        psum = simde_mm256_hadd_epi32(psum, psum);
        psum = simde_mm256_hadd_epi32(psum, psum);

        int policy      = simde_mm256_extract_epi32(psum, 0) + simde_mm256_extract_epi32(psum, 4);
        policyBuffer(i) = policy * policyFinalScale;
    }
}

NNUEv2Evaluator::NNUEv2Evaluator(int                   boardSize,
                                 Rule                  rule,
                                 std::filesystem::path blackWeightPath,
                                 std::filesystem::path whiteWeightPath)
    : Evaluator(boardSize, rule)
    , weight {nullptr, nullptr}
{
    std::filesystem::path currentWeightPath;
    TextWeightLoader      textLoader;

    CompressedWrapper<StandardHeaderParserWarpper<BinaryPODWeightLoader<Weight>>> binLoader(
        Compressor::Type::LZ4_DEFAULT);
    binLoader.setHeaderValidator([&](StandardHeader header) -> bool {
        constexpr uint32_t ArchHash = ArchHashBase ^ ((MLPChannel << 16) | GroupSize);
        if (header.archHash != ArchHash)
            throw IncompatibleWeightFileError("incompatible architecture in weight file.");

        if (!contains(header.supportedRules, rule))
            throw UnsupportedRuleError(rule);

        if (!contains(header.supportedBoardSizes, boardSize))
            throw UnsupportedBoardSizeError(boardSize);

        if (Config::MessageMode != MsgMode::NONE)
            MESSAGEL("nnuev2: load weight from " << currentWeightPath);
        return true;
    });

    for (const auto &[weightSide, weightPath] : {
             std::make_pair(BLACK, blackWeightPath),
             std::make_pair(WHITE, whiteWeightPath),
         }) {
        bool useTextLoader = weightPath.extension() == ".txt";
        if (useTextLoader)
            MESSAGEL("nnuev2: loading text weight from " << weightPath);
        else
            currentWeightPath = weightPath;

        weight[weightSide] = useTextLoader
                                 ? NNUEv2WeightRegistry.loadWeightFromFile(weightPath, textLoader)
                                 : NNUEv2WeightRegistry.loadWeightFromFile(weightPath, binLoader);
        if (!weight[weightSide])
            throw std::runtime_error("failed to load nnue weight from " + weightPath.string());
    }

    accumulator[BLACK] = std::make_unique<Accumulator>(boardSize);
    accumulator[WHITE] = std::make_unique<Accumulator>(boardSize);

    int numCells = boardSize * boardSize;
    moveCache[BLACK].reserve(numCells);
    moveCache[WHITE].reserve(numCells);
}

NNUEv2Evaluator::~NNUEv2Evaluator()
{
    if (weight[BLACK])
        NNUEv2WeightRegistry.unloadWeight(weight[BLACK]);
    if (weight[WHITE])
        NNUEv2WeightRegistry.unloadWeight(weight[WHITE]);
}

void NNUEv2Evaluator::initEmptyBoard()
{
    moveCache[BLACK].clear();
    moveCache[WHITE].clear();
    accumulator[BLACK]->clear(*weight[BLACK]);
    accumulator[WHITE]->clear(*weight[WHITE]);
}

void NNUEv2Evaluator::beforeMove(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), false);
}

void NNUEv2Evaluator::afterUndo(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), true);
}

ValueType NNUEv2Evaluator::evaluateValue(const Board &board)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate value
    clearCache(self);
    auto [win, loss, draw] = accumulator[self]->evaluateValue(*weight[self]);

    return ValueType(win, loss, draw, true);
}

void NNUEv2Evaluator::evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate policy
    clearCache(self);
    accumulator[self]->evaluatePolicy(*weight[self], policyBuffer);

    policyBuffer.setScoreBias(-100);
}

void NNUEv2Evaluator::clearCache(Color side)
{
    constexpr Color opponentMap[4] = {WHITE, BLACK, WALL, EMPTY};

    for (MoveCache &mc : moveCache[side]) {
        if (side == WHITE) {
            mc.oldColor = opponentMap[mc.oldColor];
            mc.newColor = opponentMap[mc.newColor];
        }

        accumulator[side]->update(*weight[side], mc.oldColor, mc.newColor, mc.x, mc.y);
    }
    moveCache[side].clear();
}

void NNUEv2Evaluator::addCache(Color side, int x, int y, bool isUndo)
{
    Color oldColor = EMPTY;
    Color newColor = side;
    if (isUndo)
        std::swap(oldColor, newColor);

    MoveCache newCache {oldColor, newColor, (int8_t)x, (int8_t)y};

    for (Color c : {BLACK, WHITE}) {
        if (moveCache[c].empty() || !isContraryMove(newCache, moveCache[c].back()))
            moveCache[c].push_back(newCache);
        else
            moveCache[c].pop_back();  // cancel out the last move cache

        assert(moveCache[c].size() < boardSize * boardSize);
    }
}

}  // namespace Evaluation::nnuev2
