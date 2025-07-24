
// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include <string>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "ck_tile/ops/gemm.hpp"

// GEMM config with 32x132 warp tile
template <typename DataType>
struct FlatmmConfig32
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(DataType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 32;
    static constexpr ck_tile::index_t N_Warp_Tile = 32;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 16 : 32;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 2;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;
};

template <typename DataType>
struct FlatmmConfig32_950 : public FlatmmConfig32<DataType>
{
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 16 : 64;
};

// GEMM config with 16x16 warp tile
template <typename DataType>
struct FlatmmConfig16
{
    static constexpr ck_tile::index_t M_Tile = 128;
    static constexpr ck_tile::index_t N_Tile = 128;
    static constexpr ck_tile::index_t K_Tile = 128 / sizeof(DataType);

    static constexpr ck_tile::index_t M_Warp = 1;
    static constexpr ck_tile::index_t N_Warp = 4;
    static constexpr ck_tile::index_t K_Warp = 1;

    static constexpr ck_tile::index_t M_Warp_Tile = 16;
    static constexpr ck_tile::index_t N_Warp_Tile = 16;
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 32 : 64;

    static constexpr bool kPadM = false;
    static constexpr bool kPadN = false;
    static constexpr bool kPadK = false;

    static constexpr bool TransposeC            = false;
    static constexpr bool UseStructuredSparsity = false;

    static constexpr int kBlockPerCu                = 2;
    static constexpr int TileParitionerGroupNum     = 8;
    static constexpr int TileParitionerM01          = 4;
    static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Default;
    static constexpr ck_tile::index_t NumWaveGroups = 1;
    static constexpr bool DoubleSmemBuffer          = false;
};

template <typename DataType>
struct FlatmmConfig16_950 : public FlatmmConfig16<DataType>
{
    static constexpr ck_tile::index_t N_Tile = 256;
    static constexpr ck_tile::index_t K_Tile = 256 / sizeof(DataType);
    static constexpr ck_tile::index_t K_Warp_Tile = sizeof(DataType) == 2 ? 32 : 128;
    static constexpr int kBlockPerCu                = 2;
};

template <typename ADataType>
struct GemmBasicTypeConfig;

template <>
struct GemmBasicTypeConfig<ck_tile::half_t>
{
    using ADataType   = ck_tile::half_t;
    using BDataType   = ck_tile::half_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <>
struct GemmBasicTypeConfig<ck_tile::bf16_t>
{
    using ADataType   = ck_tile::bf16_t;
    using BDataType   = ck_tile::bf16_t;
    using AccDataType = float;
    using CDataType   = ck_tile::bf16_t;
};
template <>
struct GemmBasicTypeConfig<ck_tile::fp8_t>
{
    using ADataType   = ck_tile::fp8_t;
    using BDataType   = ck_tile::fp8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
    // ToDo: Add more bias config to support different categories of GEMM.
};

template <>
struct GemmBasicTypeConfig<ck_tile::bf8_t>
{
    using ADataType   = ck_tile::bf8_t;
    using BDataType   = ck_tile::bf8_t;
    using AccDataType = float;
    using CDataType   = ck_tile::half_t;
};

template <typename T>
struct DataTypeTraits;

template <>
struct DataTypeTraits<ck_tile::fp8_t>
{
    static constexpr const char* name = "fp8";
};

template <>
struct DataTypeTraits<ck_tile::bf8_t>
{
    static constexpr const char* name = "bf8";
};
template <>
struct DataTypeTraits<float>
{
    static constexpr const char* name = "fp32";
};

template <>
struct DataTypeTraits<double>
{
    static constexpr const char* name = "fp64";
};

template <>
struct DataTypeTraits<ck_tile::half_t>
{
    static constexpr const char* name = "fp16";
};

template <typename T>
struct is_8bit_type
    : std::bool_constant<std::is_same_v<T, ck_tile::fp8_t> || std::is_same_v<T, ck_tile::bf8_t>>
{
};

// template <typename DataType>
// struct GemmConfig
// {
// #if defined(USING_MFMA_16x16x128_F8) //MI350 FP8 16X16
//     static constexpr ck_tile::index_t M_Tile = 128;
//     static constexpr ck_tile::index_t N_Tile = 256;
//     static constexpr ck_tile::index_t K_Tile = 256;

//     static constexpr ck_tile::index_t M_Warp = 1;
//     static constexpr ck_tile::index_t N_Warp = 4;
//     static constexpr ck_tile::index_t K_Warp = 1;

//     static constexpr ck_tile::index_t M_Warp_Tile = 16;
//     static constexpr ck_tile::index_t N_Warp_Tile = 16;
//     static constexpr ck_tile::index_t K_Warp_Tile = 128;
// #elif defined(USING_MFMA_32x32x64_F8) //MI350 FP8 32X32 (need tune)
//     static constexpr ck_tile::index_t M_Tile = 128;
//     static constexpr ck_tile::index_t N_Tile = 128;
//     static constexpr ck_tile::index_t K_Tile = 128;

//     static constexpr ck_tile::index_t M_Warp = 1;
//     static constexpr ck_tile::index_t N_Warp = 4;
//     static constexpr ck_tile::index_t K_Warp = 1;

//     static constexpr ck_tile::index_t M_Warp_Tile = 32;
//     static constexpr ck_tile::index_t N_Warp_Tile = 32;
//     static constexpr ck_tile::index_t K_Warp_Tile = 64;
// #elif defined(USING_MFMA_16x16x32_F16) //MI350 FP16 16X16 (need tune)
//     static constexpr ck_tile::index_t M_Tile = 128;
//     static constexpr ck_tile::index_t N_Tile = 128;
//     static constexpr ck_tile::index_t K_Tile = 128;

//     static constexpr ck_tile::index_t M_Warp = 1;
//     static constexpr ck_tile::index_t N_Warp = 4;
//     static constexpr ck_tile::index_t K_Warp = 1;

//     static constexpr ck_tile::index_t M_Warp_Tile = 16;
//     static constexpr ck_tile::index_t N_Warp_Tile = 16;
//     static constexpr ck_tile::index_t K_Warp_Tile = 32;
// #elif defined(USING_MFMA_32x32x16_F16) //MI350 FP16 32X32 (need tune)
//     static constexpr ck_tile::index_t M_Tile = 128;
//     static constexpr ck_tile::index_t N_Tile = 128;
//     static constexpr ck_tile::index_t K_Tile = 128;

//     static constexpr ck_tile::index_t M_Warp = 1;
//     static constexpr ck_tile::index_t N_Warp = 4;
//     static constexpr ck_tile::index_t K_Warp = 1;

//     static constexpr ck_tile::index_t M_Warp_Tile = 32;
//     static constexpr ck_tile::index_t N_Warp_Tile = 32;
//     static constexpr ck_tile::index_t K_Warp_Tile = 16;
// #elif defined(USING_MFMA_16x16x32_F8) //MI300 FP8 16X16
//     static constexpr ck_tile::index_t M_Tile = 16;
//     static constexpr ck_tile::index_t N_Tile = 64;
//     static constexpr ck_tile::index_t K_Tile = 256;

//     static constexpr ck_tile::index_t M_Warp = 1;
//     static constexpr ck_tile::index_t N_Warp = 4;
//     static constexpr ck_tile::index_t K_Warp = 1;

//     static constexpr ck_tile::index_t M_Warp_Tile = 16;
//     static constexpr ck_tile::index_t N_Warp_Tile = 16;
//     static constexpr ck_tile::index_t K_Warp_Tile = 64;
// #elif defined(USING_MFMA_32x32x16_F8) //MI300 FP8 32X32 (need tune)
//     static constexpr ck_tile::index_t M_Tile = 128;
//     static constexpr ck_tile::index_t N_Tile = 256;
//     static constexpr ck_tile::index_t K_Tile = 128;

//     static constexpr ck_tile::index_t M_Warp = 1;
//     static constexpr ck_tile::index_t N_Warp = 8;
//     static constexpr ck_tile::index_t K_Warp = 1;

//     static constexpr ck_tile::index_t M_Warp_Tile = 32;
//     static constexpr ck_tile::index_t N_Warp_Tile = 32;
//     static constexpr ck_tile::index_t K_Warp_Tile = 32;
// #elif defined(USING_MFMA_16x16x16_F16) //MI300 FP16 16X16 (need tune)
//     static constexpr ck_tile::index_t M_Tile = 128;
//     static constexpr ck_tile::index_t N_Tile = 128;
//     static constexpr ck_tile::index_t K_Tile = 128;

//     static constexpr ck_tile::index_t M_Warp = 1;
//     static constexpr ck_tile::index_t N_Warp = 4;
//     static constexpr ck_tile::index_t K_Warp = 1;

//     static constexpr ck_tile::index_t M_Warp_Tile = 16;
//     static constexpr ck_tile::index_t N_Warp_Tile = 16;
//     static constexpr ck_tile::index_t K_Warp_Tile = 32;
// #elif defined(USING_MFMA_32x32x8_F16) //MI300 FP16 32X32 (need tune)
//     static constexpr ck_tile::index_t M_Tile = 128;
//     static constexpr ck_tile::index_t N_Tile = 128;
//     static constexpr ck_tile::index_t K_Tile = 128;

//     static constexpr ck_tile::index_t M_Warp = 1;
//     static constexpr ck_tile::index_t N_Warp = 4;
//     static constexpr ck_tile::index_t K_Warp = 1;

//     static constexpr ck_tile::index_t M_Warp_Tile = 32;
//     static constexpr ck_tile::index_t N_Warp_Tile = 32;
//     static constexpr ck_tile::index_t K_Warp_Tile = 16;
// #else    
//     static constexpr ck_tile::index_t M_Tile = 128;
//     static constexpr ck_tile::index_t N_Tile = 256;
//     static constexpr ck_tile::index_t K_Tile = 256;

//     static constexpr ck_tile::index_t M_Warp = 1;
//     static constexpr ck_tile::index_t N_Warp = 4;
//     static constexpr ck_tile::index_t K_Warp = 1;

//     static constexpr ck_tile::index_t M_Warp_Tile = 16;
//     static constexpr ck_tile::index_t N_Warp_Tile = 16;
//     static constexpr ck_tile::index_t K_Warp_Tile = 128;
// #endif
// };

template <typename FlatmmConfig,
          typename ADataType,
          typename BDataType,
          typename DsDatatype,
          typename AccDataType,
          typename CDataType,
          typename ALayout,
          typename BLayout,
          typename DsLayout,
          typename ELayout,
          bool persistent,
          typename CDEElementWise>
float flatmm_calc(const ck_tile::FlatmmHostArgs<>& args, const ck_tile::stream_config& s);
