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

#include "../core/utils.h"
#include "evaluator.h"

#include <filesystem>

namespace Evaluation::nnuev2 {

using namespace Evaluation;

constexpr uint32_t ArchHashBase        = 0x671d030;
constexpr int      ShapeNum            = 708588;
constexpr int      GroupSize           = 64;
constexpr int      FeatureNum          = GroupSize * 2;
constexpr int      TrunkConv1GroupSize = 4;
constexpr int      MLPChannel          = 64;

struct Weight
{
    // 1  mapf = self.mapping(x), shape=H*W*4*2g
    int16_t mapping[ShapeNum][FeatureNum];

    // 2
    //  g1=mapf[:,:,:self.groupc,:,:]#第一组通道
    //  g2=mapf[:,:,self.groupc:,:,:]#第二组通道

    // 3  h1 = self.g1lr(g1.mean(1)) #四线求和再leakyrelu
    int16_t g1lr_w[GroupSize];

    // 4  h1 = torch.stack(self.h1conv(h1), dim = 1) #沿着一条线卷积

    int16_t h1conv_w[(11 + 1) / 2][GroupSize];  //卷积核是对称的，所以除2
    int16_t h1conv_b[GroupSize];

    // 5  h2 = self.h1lr2(self.h1lr1(h1, dim = 2) + g2, dim = 2)
    int16_t h1lr1_w[GroupSize];
    int16_t h1lr2_w[GroupSize];

    // 6  h3 = h2.mean(1) #最后把四条线整合起来

    // 7  trunk = self.h3lr(h3)
    int16_t h3lr_w[GroupSize];
    int16_t h3lr_b[GroupSize];

    // 8  trunk = self.trunkconv1(trunk)
    int16_t trunkconv1_w[TrunkConv1GroupSize][GroupSize];
    int16_t trunkconv1_b[GroupSize];

    // 9  trunk = self.trunklr1(trunk)
    int16_t trunklr1_w[GroupSize];

    // 10 trunk = self.trunkconv2(trunk)
    int16_t trunkconv2_w[3][GroupSize];  //对称的3x3卷积

    // 11 p = self.trunklr2p(trunk)
    //    v = self.trunklr2v(trunk)
    int16_t trunklr2p_w[GroupSize];
    int16_t trunklr2p_b[GroupSize];
    int16_t trunklr2v_w[GroupSize];
    int16_t trunklr2v_b[GroupSize];

    // 12 p = self.policy_linear(p)
    int16_t policy_linear_w[GroupSize];
    float   scale_policy_inv;

    // 13  v=v.mean((2,3))
    float scale_before_mlp_inv;
    float valuelr_w[GroupSize];
    float valuelr_b[GroupSize];

    // 14  mlp
    float mlp_w1[GroupSize][MLPChannel];  // shape=(inc，outc)，相同的inc对应权重相邻
    float mlp_b1[MLPChannel];
    float mlp_w2[MLPChannel][MLPChannel];
    float mlp_b2[MLPChannel];
    float mlp_w3[MLPChannel][MLPChannel];
    float mlp_b3[MLPChannel];
    float mlpfinal_w[MLPChannel][3];
    float mlpfinal_w_padding0[5];  // mlp_w3在read的时候一次read
                                   // 8个，会read到后续内存mlp_w3[mix6::valueNum-1][2]+5，
    float mlpfinal_b[3];
    float mlpfinal_w_padding1[5];  // mlp_b3在read的时候一次read
                                   // 8个，会read到后续内存mlp_b3[2]+5，
};

struct Accumulator
{
public:
    Accumulator(int boardSize);
    ~Accumulator();

    /// Init accumulator state to empty board.
    void clear(const Weight &w);
    /// Incremental update mix6 network state.
    void update(const Weight &w, Color oldColor, Color newColor, int x, int y);
    /// Update trunk part if value or policy is actually needed.
    void updateTrunk(const Weight &w);
    /// Calculate value of current network state.
    std::tuple<float, float, float> evaluateValue(const Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Weight &w, PolicyBuffer &policyBuffer);

private:
    int   boardSize;
    int   numCells;
    float boardSizeScale;

    //=============================================================
    // NNUEv2 network states

    bool trunkUpToDate;  // Flag for trunk update

    // 1 convert board to shape
    MDNativeArray<uint32_t, 4> *shapeTable;

    // 2  shape到vector  g1无需提取，只缓存g2
    MDNativeArray<int16_t, 4, GroupSize> *g2;

    // 3  g1sum=g1.sum(1), shape=H*W*g
    MDNativeArray<int16_t, GroupSize> *g1sum;

    // 4  h1=self.g1lr(g1sum), shape=HWc
    // MDNativeArray<int16_t, GroupSize> *h1;

    // 后面的部分几乎没法增量计算
    // value头和policy头共享trunk，所以也放在缓存里
    MDNativeArray<int16_t, GroupSize> *trunk;

    // buffers
    int16_t *h1m;  // int16_t[(BS + 10) * (BS + 10)][6][16];
                   // int16_t[(BS + 2) * (BS + 2)][16];
    int16_t *h3;   // int16_t h3[BS*BS][16];
    //=============================================================

    void initShapeTable();
};

class NNUEv2Evaluator : public Evaluator
{
public:
    NNUEv2Evaluator(int                   boardSize,
                    Rule                  rule,
                    std::filesystem::path blackWeightPath,
                    std::filesystem::path whiteWeightPath);
    ~NNUEv2Evaluator();

    void initEmptyBoard();
    void beforeMove(const Board &board, Pos pos);
    void afterUndo(const Board &board, Pos pos);

    ValueType evaluateValue(const Board &board);
    void      evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer);

private:
    struct MoveCache
    {
        Color  oldColor, newColor;
        int8_t x, y;

        friend bool isContraryMove(MoveCache a, MoveCache b)
        {
            bool isSameCoord = a.x == b.x && a.y == b.y;
            bool isContrary  = a.oldColor == b.newColor && a.newColor == b.oldColor;
            return isSameCoord && isContrary;
        }
    };

    /// Clear all caches to sync accumulator state with current board state.
    void clearCache(Color side);
    /// Record new board action, but not update accumulator instantly.
    void addCache(Color side, int x, int y, bool isUndo);

    Weight /* non-owning ptr */ *weight[2];
    std::unique_ptr<Accumulator> accumulator[2];
    std::vector<MoveCache>       moveCache[2];
};

}  // namespace Evaluation::nnuev2
