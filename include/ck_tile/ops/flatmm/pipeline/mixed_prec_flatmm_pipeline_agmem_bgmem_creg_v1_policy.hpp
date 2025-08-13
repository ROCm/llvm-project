// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp"

namespace ck_tile {

struct F16xMXF4FlatmmPipelineAgBgCrPolicy : UniversalFlatmmPipelineAgBgCrPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t KBPerLoad = 32;
    static constexpr index_t N_Pack    = 2; // it's fixed for fp4
    static constexpr index_t K_Pack    = 2; // it's fixed for fp4

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeFp16xF4_ADramTileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = Problem::VectorLoadSize / sizeof(ADataType);
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;

        constexpr index_t M1 = BlockSize / get_warp_size();
        static_assert(M2 != 0, "M2 is zero, which will lead to a division by zero error.");
        static_assert(M1 != 0, "M1 is zero, which will lead to a division by zero error.");
        constexpr index_t M0 = MPerBlock / (M2 * M1);
        static_assert(M0 * M1 * M2 == MPerBlock,
                      "Incorrect M0, M2, M1 configuration! "
                      "M0, M1, M2 must cover whole MPerBlock!");

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<1>,
                                       tuple<sequence<M0, M1, M2>, sequence<K0, K1>>,
                                       tuple<sequence<1>, sequence<1, 2>>,
                                       tuple<sequence<1>, sequence<2, 0>>,
                                       sequence<1, 2>,
                                       sequence<0, 1>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeFp16xF4_DS_WRITE_ATileDistribution()
    {
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;

        static_assert(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>);

        constexpr index_t BlockSize = Problem::kBlockSize;

        constexpr index_t MPerBlock = Problem::BlockGemmShape::kM;
        constexpr index_t KPerBlock = Problem::BlockGemmShape::kK;

        constexpr index_t K1 = Problem::VectorLoadSize / sizeof(ADataType);
        constexpr index_t K0 = KPerBlock / K1;
        constexpr index_t M2 = get_warp_size() / K0;

        constexpr index_t M1 = BlockSize / get_warp_size();
        static_assert(M2 != 0, "M2 is zero, which will lead to a division by zero error.");
        static_assert(M1 != 0, "M1 is zero, which will lead to a division by zero error.");
        constexpr index_t M0 = MPerBlock / (M2 * M1);
        static_assert(M0 * M1 * M2 == MPerBlock,
                      "Incorrect M0, M2, M1 configuration! "
                      "M0, M1, M2 must cover whole MPerBlock!");

        // unmerge K0 to K16_i x K4_1 x K4_2
        // then exchange the order of K4_1 and K4_2
        constexpr index_t XDL_PerKBLoad = 4;
        constexpr index_t K128_Cnt      = K0 / XDL_PerKBLoad / XDL_PerKBLoad;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<1>,
                tuple<sequence<M0, M1, M2>, sequence<K128_Cnt, XDL_PerKBLoad, XDL_PerKBLoad, K1>>,
                tuple<sequence<1>, sequence<1, 2, 2, 2>>,
                tuple<sequence<1>, sequence<2, 0, 2, 1>>,
                sequence<1, 2>,
                sequence<0, 3>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeF16xF4_ALDS_TileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;
        using ADataType = remove_cvref_t<typename Problem::ADataType>;
        using ALayout   = remove_cvref_t<typename Problem::ALayout>;

        static_assert(TileShape::WarpTile::at(I1) == 16, "requires XDL_N == 16");
        static_assert(TileShape::BlockWarps::at(I0) == 1, "requires Wave_M == 1");

        constexpr int Repeat = TileShape::BlockWarps::at(number<1>{});
        constexpr int M0     = TileShape::WarpTile::at(I0);

        constexpr int K_Lane = 64 / TileShape::WarpTile::at(I1); // 4

        constexpr int K2             = TileShape::WarpTile::at(I2) / K_Lane; // 8
        constexpr int XDL_PerThreadK = KBPerLoad / K2;                       // 4
        constexpr int K0             = K_Lane;                               // 4

        return make_static_tile_distribution(
            tile_distribution_encoding<sequence<Repeat>,
                                       tuple<sequence<M0>, sequence<K0, XDL_PerThreadK, K2>>,
                                       tuple<sequence<0>, sequence<2, 1>>,
                                       tuple<sequence<0>, sequence<0, 0>>,
                                       sequence<2>,
                                       sequence<2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeFp4BFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

        static_assert(TileShape::WarpTile::at(I1) == 16, "only for XDL_N == 16");

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t KThdPerWave = WaveSize; // threads cnt in K dim
        constexpr index_t KWavePerBlk = 1;

        constexpr index_t NWavePerBlk = TileShape::BlockWarps::at(number<1>{}); // N_Warp

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<WaveRepeat>,                                 // ?
                tuple<sequence<NWavePerBlk, N_Pack>,                  // second
                                                                      // direction
                      sequence<KWavePerBlk, KThdPerWave, KBPerLoad>>, // first  direction
                // wave in blk,     // thd in wave
                // <M, K>           // <M, K>
                tuple<sequence<0, 1, 2>, sequence<2>>, // which direction
                tuple<sequence<0, 0, 0>, sequence<1>>, // which index
                // <repeat, vec_load>
                sequence<2>,
                sequence<2>>{});
    }

    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeFp4ScaleBFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape; // ck_tile::TileFlatmmShape

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t N_Warp = TileShape::BlockWarps::at(number<1>{});

        constexpr index_t XDLPerBlock = TileShape::kK / TileShape::WarpTile::at(I2);
        constexpr index_t K_Lane      = 64 / TileShape::WarpTile::at(I1);
        constexpr index_t N_Lane      = TileShape::WarpTile::at(I1);

        constexpr index_t NWavePerBlk = N_Warp;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<>,                                       // ?
                tuple<sequence<NWavePerBlk>,                      // second direction
                      sequence<K_Lane, N_Lane, N_Pack * K_Pack>>, // first
                                                                  // direction
                // wave in blk,     // thd in wave
                // <M, K>           // <M, K>
                tuple<sequence<1>, sequence<2, 2>>, // which direction
                tuple<sequence<0>, sequence<0, 1>>, // which index
                // <repeat, vec_load>
                sequence<2>,
                sequence<2>>{});
    }
};

} // namespace ck_tile
