// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/math.hpp"
#include "ck_tile/core/utility/literals.hpp"
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#include "ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp"
#include "ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp"
#include "ck_tile/host.hpp"

// #define disable_tile_gs

namespace ck_tile {

template <class ScaleM = FlatmmScalePointer<-1>, class ScaleN = FlatmmScalePointer<-1>>
struct MoeFlatmmHostArgs : ScaleFlatmmHostArgs<ScaleM, ScaleN, 0>
{
    ck_tile::index_t NumTokens;
    ck_tile::index_t NumExperts;
    ck_tile::index_t TopK;
    const ck_tile::index_t* p_sorted_token_ids;
    const ck_tile::index_t* p_sorted_expert_ids;
    const ck_tile::index_t* p_max_token_id;
    const void* p_sorted_expert_weights;

    CK_TILE_HOST MoeFlatmmHostArgs() noexcept = default;
    CK_TILE_HOST MoeFlatmmHostArgs(const ck_tile::index_t* p_sorted_token_ids_,
                                   const void* p_sorted_expert_weights_,
                                   const ck_tile::index_t* p_sorted_expert_ids_,
                                   const ck_tile::index_t* p_max_token_id_,
                                   const void* a_ptr_,
                                   const void* b_ptr_,
                                   void* c_ptr_,
                                   ck_tile::index_t NumTokens_,
                                   ck_tile::index_t NumExperts_,
                                   ck_tile::index_t TopK_,
                                   ck_tile::index_t k_batch_,
                                   ck_tile::index_t M_,
                                   ck_tile::index_t N_,
                                   ck_tile::index_t K_,
                                   ck_tile::index_t stride_A_,
                                   ck_tile::index_t stride_B_,
                                   ck_tile::index_t stride_C_,
                                   ScaleM scale_m_ = {},
                                   ScaleN scale_n_ = {})
        : ScaleFlatmmHostArgs<ScaleM, ScaleN, 0>(a_ptr_,
                                                 b_ptr_,
                                                 {}, // d_ptr_array
                                                 c_ptr_,
                                                 k_batch_,
                                                 M_,
                                                 N_,
                                                 K_,
                                                 stride_A_,
                                                 stride_B_,
                                                 {}, // d_stride_array
                                                 stride_C_,
                                                 scale_m_,
                                                 scale_n_),
          NumTokens(NumTokens_),
          NumExperts(NumExperts_),
          TopK(TopK_),
          p_sorted_token_ids(p_sorted_token_ids_),
          p_sorted_expert_ids(p_sorted_expert_ids_),
          p_max_token_id(p_max_token_id_),
          p_sorted_expert_weights(p_sorted_expert_weights_)
    {
    }
};

enum class MoeFlatmmKind
{
    kFFN_gemm1_gate_only,
    kFFN_gemm1_gate_up,
    kFFN_gemm2,
};

template <typename TilePartitioner_,
          typename FlatmmPipeline_,
          typename EpiloguePipeline_,
          MoeFlatmmKind kind>
struct MoeFlatmmKernel
{
    using TilePartitioner = remove_cvref_t<TilePartitioner_>;
    using FlatmmPipeline  = remove_cvref_t<FlatmmPipeline_>;
    using BlockGemmShape =
        remove_cvref_t<typename FlatmmPipeline::BlockGemmShape>; // TileFlatmmShape
    using EpiloguePipeline = remove_cvref_t<EpiloguePipeline_>;
    using ALayout          = remove_cvref_t<typename FlatmmPipeline::ALayout>;
    using BLayout          = remove_cvref_t<typename FlatmmPipeline::BLayout>;
    using ELayout          = remove_cvref_t<typename FlatmmPipeline::CLayout>;
    using DsLayout         = remove_cvref_t<typename EpiloguePipeline::DsLayout>;
    using DsDataType       = remove_cvref_t<typename EpiloguePipeline::DsDataType>;
    static constexpr index_t KernelBlockSize  = FlatmmPipeline::BlockSize;
    static constexpr bool UsePersistentKernel = FlatmmPipeline::UsePersistentKernel;

    using ADataType = remove_cvref_t<typename FlatmmPipeline::ADataType>;
    using BDataType = remove_cvref_t<typename FlatmmPipeline::BDataType>;
    // Below type is actually accumulation data type - the output of block GEMM.
    using EDataType = remove_cvref_t<typename EpiloguePipeline::ODataType>;

    using AccDataType = float;

    static constexpr index_t NumDTensor = DsDataType::size();

    static constexpr auto I0 = number<0>();
    static constexpr auto I1 = number<1>();
    static constexpr auto I2 = number<2>();
    static constexpr auto I3 = number<3>();

    static_assert(DsLayout::size() == DsDataType::size(),
                  "The size of DsLayout and DsDataType should be the same");

    static constexpr bool IsInputGemm = kind != MoeFlatmmKind::kFFN_gemm2;

    static constexpr index_t kBlockSize     = EpiloguePipeline::kBlockSize;
    static constexpr index_t kMPerBlock     = EpiloguePipeline::kMPerBlock;
    static constexpr index_t kNPerBlock     = EpiloguePipeline::kNPerBlock;
    static constexpr index_t MWave          = EpiloguePipeline::MWave;
    static constexpr index_t NWave          = EpiloguePipeline::NWave;
    static constexpr index_t MPerXdl        = EpiloguePipeline::MPerXdl;
    static constexpr index_t NPerXdl        = EpiloguePipeline::NPerXdl;
    static constexpr index_t KPerXdl        = EpiloguePipeline::KPerXdl;
    static constexpr index_t isCTransposed  = EpiloguePipeline::isCTransposed;
    static constexpr index_t kMPerIteration = MPerXdl * MWave;
    static constexpr index_t kNPerIteration = NPerXdl * NWave;
    static constexpr index_t kNRepeat       = kNPerBlock / kNPerIteration;

    template <class ScaleM = FlatmmScalePointer<-1>, class ScaleN = FlatmmScalePointer<-1>>
    struct MoeFlatmmKernelArgs
    {
        const ck_tile::index_t* p_sorted_token_ids;
        const ck_tile::index_t* p_sorted_expert_ids;
        const ck_tile::index_t* p_max_token_id;
        const void* p_sorted_expert_weights;
        const void* a_ptr;
        const void* b_ptr;
        void* e_ptr;
        ck_tile::index_t NumTokens;
        ck_tile::index_t TopK;
        ck_tile::index_t M;
        ck_tile::index_t N;
        ck_tile::index_t K;
        ck_tile::index_t stride_A;
        ck_tile::index_t stride_B;
        ck_tile::index_t stride_C;
        ck_tile::index_t k_batch;
        ScaleM scale_m;
        ScaleN scale_n;
    };

    template <class ScaleM = FlatmmScalePointer<-1>, class ScaleN = FlatmmScalePointer<-1>>
    CK_TILE_HOST static constexpr auto
    MakeKernelArgs(const MoeFlatmmHostArgs<ScaleM, ScaleN>& hostArgs)
    {
        return MoeFlatmmKernelArgs<ScaleM, ScaleN>{hostArgs.p_sorted_token_ids,
                                                   hostArgs.p_sorted_expert_ids,
                                                   hostArgs.p_max_token_id,
                                                   hostArgs.p_sorted_expert_weights,
                                                   hostArgs.a_ptr,
                                                   hostArgs.b_ptr,
                                                   hostArgs.e_ptr,
                                                   hostArgs.NumTokens,
                                                   hostArgs.TopK,
                                                   hostArgs.M,
                                                   hostArgs.N,
                                                   hostArgs.K,
                                                   hostArgs.stride_A,
                                                   hostArgs.stride_B,
                                                   hostArgs.stride_C,
                                                   hostArgs.k_batch,
                                                   hostArgs.scale_m,
                                                   hostArgs.scale_n};
    }

    [[nodiscard]] CK_TILE_HOST static const std::string GetName()
    {
        return concat(
            '_', "moe_flatmm", gemm_prec_str<ADataType, BDataType>, FlatmmPipeline::GetName());
    }

    static constexpr auto BlockSize() -> dim3 { return dim3(KernelBlockSize); }

    static constexpr auto GridSize(index_t M, index_t N, index_t KBatch)
    {
        return dim3(TilePartitioner::GridSize(M, N), 1, KBatch);
    }
    template <class ScaleM = FlatmmScalePointer<-1>, class ScaleN = FlatmmScalePointer<-1>>
    static constexpr auto GridSize(const MoeFlatmmKernelArgs<ScaleM, ScaleN>& kargs)
    {
        return dim3(TilePartitioner::GridSize(kargs.M, kargs.N), 1, kargs.k_batch);
    }

    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPingSize()
    {
        return max(FlatmmPipeline::GetSmemSize(), EpiloguePipeline::GetSmemSize());
    }
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemPongSize()
    {
        return FlatmmPipeline::GetSmemSize();
    }

    struct SplitKBatchOffset
    {
        template <class KernelArgs>
        __device__ SplitKBatchOffset(const KernelArgs& kargs, const std::size_t k_id = blockIdx.z)
        {
            constexpr auto K1   = TilePartitioner::BlockGemmShape::WarpTile::at(number<2>{});
            const index_t K_t   = kargs.k_batch * K1;
            const index_t KRead = (kargs.K + K_t - 1) / K_t * K1;

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, ALayout>)
            {
                a_k_split_offset = k_id * KRead;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, ALayout>)
            {
                a_k_split_offset = k_id * KRead * kargs.stride_A;
            }

            if constexpr(std::is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
            {
                b_k_split_offset = k_id * KRead * kargs.stride_B;
            }
            else if constexpr(std::is_same_v<tensor_layout::gemm::ColumnMajor, BLayout>)
            {
                b_k_split_offset = k_id * KRead;
            }

            if(k_id < static_cast<uint32_t>(kargs.k_batch - 1))
            {
                splitted_k = KRead;
            }
            else
            {
                splitted_k = kargs.K - KRead * (kargs.k_batch - 1);
            }
        }

        index_t a_k_split_offset;
        index_t b_k_split_offset;
        index_t splitted_k;
    };

    template <typename KernelArgs>
    CK_TILE_HOST static bool IsSupportedArgument(const KernelArgs& kargs)
    {
        if constexpr(EpiloguePipeline::GetVectorSizeC() % 2 != 0 &&
                     is_any_of<EDataType, fp16_t, bf16_t>::value)
        {
            if(kargs.k_batch != 1)
            {
                std::cerr << "Conditions not met for Kbatch >1 !" << std::endl;
                return false;
            }
        }
        if constexpr(UsePersistentKernel)
        {
            if(kargs.k_batch != 1)
            {
                std::cerr << "Persistent mode doesn't support Kbatch >1 !" << std::endl;
                return false;
            }
        }

        if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.K % TilePartitioner::KPerBlock != 0 && FlatmmPipeline::kPadK == false)
            {
                std::cerr << "Can't support K that is not a multiple of KPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.K % FlatmmPipeline::GetVectorSizeA() != 0)
            {
                std::cerr << "K is not a multiple of vector load size for A tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
            {
                std::cerr << "Can't support M that is not a multiple of MPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.M % FlatmmPipeline::GetVectorSizeA() != 0)
            {
                std::cerr << "M is not a multiple of vector load size for A tensor!" << std::endl;
                return false;
            }
        }

        if constexpr(std::is_same_v<BLayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
            {
                std::cerr << "Can't support N that is not a multiple of NPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.N % FlatmmPipeline::GetVectorSizeB() != 0)
            {
                std::cerr << "N is not a multiple of vector load size for B tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.K % TilePartitioner::KPerBlock != 0 && FlatmmPipeline::kPadK == false)
            {
                std::cerr << "Can't support K that is not a multiple of KPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.K % FlatmmPipeline::GetVectorSizeB() != 0)
            {
                std::cerr << "K is not a multiple of vector load size for B tensor!" << std::endl;
                return false;
            }
        }

        bool DTesnorIsValid = {true};
        static_for<0, NumDTensor, 1>{}([&](auto index) {
            using DiLayout = remove_cvref_t<std::tuple_element_t<index.value, DsLayout>>;
            if(std::is_same_v<DiLayout, ELayout> == false)
            {
                DTesnorIsValid = false;
            }
            if constexpr(std::is_same_v<DiLayout, tensor_layout::gemm::RowMajor>)
            {
                if(kargs.N % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
                {
                    CK_TILE_ERROR("Can't support N for tensor D that is not a multiple of "
                                  "NPerBlock without padding!");
                    DTesnorIsValid = false;
                }
                if(kargs.N % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    CK_TILE_ERROR("N is not a multiple of vector load size for D tensor!");
                    DTesnorIsValid = false;
                }
            }
            else
            {
                if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
                {
                    CK_TILE_ERROR("Can't support M for tensor D that is not a multiple of "
                                  "MPerBlock without padding!");

                    DTesnorIsValid = false;
                }
                if(kargs.M % EpiloguePipeline::GetVectorSizeD(index) != 0)
                {
                    CK_TILE_ERROR("M is not a multiple of vector load size for D tensor!");
                    DTesnorIsValid = false;
                }
            }
        });

        if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
        {
            if(kargs.N % TilePartitioner::NPerBlock != 0 && FlatmmPipeline::kPadN == false)
            {
                std::cerr << "Can't support N that is not a multiple of NPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.N % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                std::cerr << "N is not a multiple of vector load size for C tensor!" << std::endl;
                return false;
            }
        }
        else
        {
            if(kargs.M % TilePartitioner::MPerBlock != 0 && FlatmmPipeline::kPadM == false)
            {
                std::cerr << "Can't support M that is not a multiple of MPerBlock"
                             " without padding!"
                          << std::endl;
                return false;
            }
            if(kargs.M % EpiloguePipeline::GetVectorSizeC() != 0)
            {
                std::cerr << "M is not a multiple of vector load size for C tensor!" << std::endl;
                return false;
            }
        }
        return DTesnorIsValid;
    }

    template <memory_operation_enum DstInMemOp = IsInputGemm ? memory_operation_enum::set
                                                             : memory_operation_enum::atomic_add,
              typename KernelArgs>
    CK_TILE_DEVICE static auto MakeGemmTensorViews(const ADataType* a_ptr,
                                                   const BDataType* b_flat_ptr,
                                                   EDataType* e_ptr,
                                                   const AccDataType* exp_weight_ptr,
                                                   const KernelArgs& kargs,
                                                   const SplitKBatchOffset& splitk_batch_offset)
    {
        // static_assert(!TilePartitioner::BlockGemmShape::PermuteA, "Not implemented!");
        const auto& a_tensor_view = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(IsInputGemm ? kargs.NumTokens : kargs.NumTokens * kargs.TopK,
                               splitk_batch_offset.splitted_k),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global>(
                    a_ptr,
                    make_tuple(splitk_batch_offset.splitted_k,
                               IsInputGemm ? kargs.NumTokens : kargs.NumTokens * kargs.TopK),
                    make_tuple(kargs.stride_A, 1),
                    number<FlatmmPipeline::GetVectorSizeA()>{},
                    number<1>{});
            }
        }();

        index_t kFlatK = FlatmmPipeline::flatKPerWarp * (splitk_batch_offset.splitted_k /
                                                         BlockGemmShape::WarpTile::at(number<2>{}));
        index_t kFlatN = kargs.N * kargs.K / kFlatK;
        const auto& b_flat_tensor_view = [&]() {
            return make_naive_tensor_view<address_space_enum::global>(
                b_flat_ptr,
                make_tuple(kFlatN, kFlatK),
                make_tuple(kFlatK, 1),
                number<FlatmmPipeline::GetVectorSizeB()>{},
                number<1>{});
        }();

        // TODO: enable vector write for C in ColMajor
        const auto& c_tensor_view = [&]() {
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(IsInputGemm ? kargs.NumTokens * kargs.TopK : kargs.NumTokens,
                               kind == MoeFlatmmKind::kFFN_gemm1_gate_up ? kargs.N / 2 : kargs.N),
                    make_tuple(kargs.stride_C, 1),
                    number<EpiloguePipeline::GetVectorSizeC()>{},
                    number<1>{});
            }
            else
            {
                return make_naive_tensor_view<address_space_enum::global, DstInMemOp>(
                    e_ptr,
                    make_tuple(IsInputGemm ? kargs.NumTokens * kargs.TopK : kargs.NumToken,
                               kind == MoeFlatmmKind::kFFN_gemm1_gate_up ? kargs.N / 2 : kargs.N),
                    make_tuple(1, kargs.stride_C),
                    number<1>{},
                    number<1>{});
            }
        }();

        return make_tuple(a_tensor_view, b_flat_tensor_view, c_tensor_view);
    }

    template <typename TensorView>
    CK_TILE_DEVICE static auto MakeGemmPadViews(const TensorView& views)
    {
        const auto& a_pad_view = [&]() {
            const auto& a_tensor_view = views.at(I0);
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::MPerBlock>{},
                                                  number<TilePartitioner::KPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadK>{});
            }
            else
            {
                return pad_tensor_view(a_tensor_view,
                                       make_tuple(number<TilePartitioner::KPerBlock>{},
                                                  number<TilePartitioner::MPerBlock>{}),
                                       sequence<false, FlatmmPipeline::kPadM>{});
            }
        }();

        const auto& b_flat_tensor_view = views.at(I1);

        // TODO vector write in for C in ColMajor
        const auto& c_pad_view = [&]() {
            const auto& c_tensor_view     = views.at(I2);
            constexpr int OutputNPerBlock = kind == MoeFlatmmKind::kFFN_gemm1_gate_up
                                                ? TilePartitioner::NPerBlock / 2
                                                : TilePartitioner::NPerBlock;
            if constexpr(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>)
            {
                return pad_tensor_view(
                    c_tensor_view,
                    make_tuple(number<TilePartitioner::MPerBlock>{}, number<OutputNPerBlock>{}),
                    sequence<false, FlatmmPipeline::kPadN>{});
            }
            else
            {
                return pad_tensor_view(
                    c_tensor_view,
                    make_tuple(number<TilePartitioner::MPerBlock>{}, number<OutputNPerBlock>{}),
                    sequence<FlatmmPipeline::kPadM, false>{});
            }
        }();

        return make_tuple(a_pad_view, b_flat_tensor_view, c_pad_view);
    }

    template <typename PadView>
    CK_TILE_DEVICE static auto
    MakeGemmTileWindows(const PadView& views, [[maybe_unused]] const index_t i_m, const index_t i_n)
    {
        const auto& a_pad_view      = views.at(number<0>{});
        const auto& b_flat_pad_view = views.at(number<1>{});
        const auto& c_pad_view      = views.at(number<2>{});

        const auto& a_block_window = [&]() {
            if constexpr(std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>)
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::MPerBlock>{},
                                                   number<TilePartitioner::KPerBlock>{}),
                                        {i_m, 0}); // NOTE!
            }
            else
            {
                return make_tile_window(a_pad_view,
                                        make_tuple(number<TilePartitioner::KPerBlock>{},
                                                   number<TilePartitioner::MPerBlock>{}),
                                        {0, 0}); // NOTE!
            }
        }();

        const int problem_N_offset = kind == MoeFlatmmKind::kFFN_gemm1_gate_up ? i_n / 2 : i_n;

        const auto& b_flat_block_window = make_tile_window(
            b_flat_pad_view,
            make_tuple(number<FlatmmPipeline::flatNPerWarp>{},
                       number<FlatmmPipeline::flatKPerWarp>{}),
            {static_cast<int>(problem_N_offset / BlockGemmShape::WarpTile::at(I1)), 0});

        constexpr int OutputNPerBlock = kind == MoeFlatmmKind::kFFN_gemm1_gate_up
                                            ? TilePartitioner::NPerBlock / 2
                                            : TilePartitioner::NPerBlock;

        auto c_block_window = make_tile_window(
            c_pad_view,
            make_tuple(number<TilePartitioner::MPerBlock>{}, number<OutputNPerBlock>{}),
            {0, // offset_m is included when construct C-scatter-window offsets
             problem_N_offset});

        return make_tuple(a_block_window, b_flat_block_window, c_block_window);
    }

    template <class ScaleM = FlatmmScalePointer<-1>, class ScaleN = FlatmmScalePointer<-1>>
    CK_TILE_DEVICE void operator()(MoeFlatmmKernelArgs<ScaleM, ScaleN> kargs) const
    {

        const auto [iM, iN]   = TilePartitioner{kargs.M, kargs.N}.GetOutputTileIndex(blockIdx.x);
        const index_t coord_m = __builtin_amdgcn_readfirstlane(iM * TilePartitioner::MPerBlock);
        const index_t coord_n = __builtin_amdgcn_readfirstlane(iN * TilePartitioner::NPerBlock);

        // allocate LDS
        __shared__ char smem_ptr_ping[GetSmemPingSize()];
        __shared__ char smem_ptr_pong[GetSmemPongSize()];

        const index_t expert_id = kargs.p_sorted_expert_ids[iM];

        constexpr auto a_dram_dist = FlatmmPipeline::GetADramTileDistribution();
        const auto a_coord = a_dram_dist.calculate_index(); // 2d thread offset, [i_row, i_col]

        constexpr ck_tile::index_t DramMRepeat =
            decltype(a_dram_dist)::DstrEncode::hs_lengthss_[number<0>{}][number<0>{}];
        statically_indexed_array<ck_tile::index_t, DramMRepeat> a_offsets;

        constexpr index_t token_id_offset = 24;
        constexpr index_t token_id_mask   = (1 << token_id_offset) - 1;

        auto row_to_token_idx = [&](auto row_idx) {
            const index_t fused_token =
                kargs.p_sorted_token_ids[row_idx]; // topk-idx[31:24] + token_idx[23:0]
            index_t gather_token_id = fused_token & token_id_mask;
            if constexpr(!IsInputGemm)
            {
                gather_token_id = gather_token_id * kargs.TopK + (fused_token >> token_id_offset);
            }
            return gather_token_id;
        };

        static_for<0, DramMRepeat, 1>{}([&](auto m0) {
            const auto row_idx =
                coord_m + m0 * (TilePartitioner::MPerBlock / DramMRepeat) + a_coord[I0];
            index_t gather_token_id = row_to_token_idx(row_idx);
            a_offsets[m0]           = std::is_same_v<ALayout, tensor_layout::gemm::RowMajor>
                                          ? gather_token_id * kargs.stride_A
                                          : gather_token_id;
        });

        const SplitKBatchOffset splitk_batch_offset(kargs);
        const index_t expert_stride = __builtin_amdgcn_readfirstlane(kargs.N * kargs.K);

        const ADataType* a_ptr =
            static_cast<const ADataType*>(kargs.a_ptr) + splitk_batch_offset.a_k_split_offset;
        const BDataType* b_flat_ptr = static_cast<const BDataType*>(kargs.b_ptr) +
                                      splitk_batch_offset.b_k_split_offset +
                                      expert_stride * expert_id;
        EDataType* e_ptr = static_cast<EDataType*>(kargs.e_ptr);

        const AccDataType* exp_weight_ptr =
            static_cast<const AccDataType*>(kargs.p_sorted_expert_weights);

        const auto& gemm_tensor_views_tuple = MakeGemmTensorViews(
            a_ptr, b_flat_ptr, e_ptr, exp_weight_ptr, kargs, splitk_batch_offset);
        const auto& gemm_pad_views = MakeGemmPadViews(gemm_tensor_views_tuple);

        auto gemm_tile_windows = MakeGemmTileWindows(gemm_pad_views, coord_m, coord_n);

        const index_t num_loop = TilePartitioner::GetLoopNum(splitk_batch_offset.splitted_k);

        // Run GEMM cooperatively by whole workgroup.
        const auto& a_block_window = gemm_tile_windows.at(number<0>{});
        const auto& b_block_window = gemm_tile_windows.at(number<1>{});

        auto a_gather_block_tile =
            ck_tile::make_tile_scatter_gather(a_block_window.get_bottom_tensor_view(),
                                              a_block_window.get_window_lengths(),
                                              a_block_window.get_window_origin(),
                                              FlatmmPipeline::GetADramTileDistribution(),
                                              a_offsets); // K DRAM tile window for
        auto c_block_tile = FlatmmPipeline{}(a_gather_block_tile,
                                             b_block_window,
                                             number<kind == MoeFlatmmKind::kFFN_gemm1_gate_up>{},
                                             num_loop,
                                             smem_ptr_ping,
                                             smem_ptr_pong);
        using AccTile     = decltype(c_block_tile);

        // Run EpiloguePipeline Pipeline
        auto& c_block_window = gemm_tile_windows.at(number<2>{});
        using ActivationOp   = element_wise::Silu;

        {
            using EpiProblem = typename EpiloguePipeline::Problem;
            using ODataType  = typename EpiloguePipeline::ODataType;
            using CWarpDstr  = typename EpiloguePipeline::CWarpDstr;

            constexpr index_t NumMXdlPerWavePerShuffle = EpiloguePipeline::NumMXdlPerWavePerShuffle;
            constexpr index_t NumNXdlPerWavePerShuffle = EpiloguePipeline::NumNXdlPerWavePerShuffle;
            constexpr index_t MPerIterationShuffle     = EpiloguePipeline::MPerIterationShuffle;
            constexpr index_t NPerIterationShuffle     = EpiloguePipeline::NPerIterationShuffle;

            constexpr index_t EpiVectorSizeC = EpiloguePipeline::GetVectorSizeC();
            constexpr index_t MRepeat        = EpiloguePipeline::MRepeat;
            constexpr index_t NRepeat        = EpiloguePipeline::NRepeat;

            constexpr auto lds_block_desc =
                EpiloguePipeline::template MakeLdsBlockDescriptor<EpiProblem>();
            auto o_lds_block = make_tensor_view<address_space_enum::lds>(
                reinterpret_cast<ODataType*>(smem_ptr_ping), lds_block_desc);

            auto in_lds_window = make_tile_window(
                o_lds_block,
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                {0, 0});

            auto out_lds_window = make_tile_window(
                o_lds_block,
                make_tuple(number<MPerIterationShuffle>{}, number<NPerIterationShuffle>{}),
                {0, 0});

            using SFC = space_filling_curve<
                sequence<kMPerBlock,
                         kind == MoeFlatmmKind::kFFN_gemm1_gate_up ? kNPerBlock / 2 : kNPerBlock>,
                sequence<0, 1>,
                sequence<MPerIterationShuffle, NPerIterationShuffle>>;

            constexpr index_t num_access = SFC::get_num_of_access();

            static_assert(std::is_same_v<ELayout, tensor_layout::gemm::RowMajor>,
                          "Currently, the CShuffle EpiloguePipeline only supports the Row Major "
                          "Output layout");

            using TileEncodingPattern =
                TileDistributionEncodingPattern2D<kBlockSize,
                                                  MPerIterationShuffle,
                                                  NPerIterationShuffle,
                                                  EpiloguePipeline::GetVectorSizeC(),
                                                  tile_distribution_pattern::thread_raked,
                                                  EpiProblem::kNumWaveGroups>;
            constexpr auto dram_tile_distribution =
                TileEncodingPattern::Make2DStaticTileDistribution();

            constexpr auto LdsTileDistr =
                make_static_tile_distribution(EpiloguePipeline::MakeLdsDistributionEncode());

            using LDSTileTensor =
                decltype(make_static_distributed_tensor<AccDataType>(LdsTileDistr));
            LDSTileTensor lds_tile[2];

            constexpr auto c_warp_y_lengths =
                to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
            constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

            constexpr int kM2 = 4;                   // Val
            constexpr int kM1 = (64 / NPerXdl);      // Thr
            constexpr int kM0 = MPerXdl / kM1 / kM2; // Val

            constexpr int ActVectorSize =
                c_warp_y_lengths.product() * NumMXdlPerWavePerShuffle * NumNXdlPerWavePerShuffle;

            const index_t iMWarp = get_warp_id() / NWave;
            const index_t iNWarp = get_warp_id() - iMWarp * NWave;
            const index_t iMLane = get_lane_id() / NPerXdl;
            const index_t iNLane = get_lane_id() % NPerXdl;

            float vec_scale_A[kM0 * kM2 * MRepeat];
            float vec_scale_B[NRepeat];

            float vec_expert_weights[kM0 * kM2 * MRepeat];

            const float* expert_weights = static_cast<const float*>(kargs.p_sorted_expert_weights);

            //===----------------------------------------------------------------------===//
            // Load scales and expert weights
            //===----------------------------------------------------------------------===//
            if constexpr(kind == MoeFlatmmKind::kFFN_gemm1_gate_up)
            {
                static_for<0, NRepeat / 2, 1>{}([&](auto i) {
                    vec_scale_B[i] = kargs.scale_n[expert_id * kargs.N + coord_n / 2 +
                                                   i * NWave * NPerXdl + iNWarp * NPerXdl + iNLane];
                    vec_scale_B[i + NRepeat / 2] =
                        kargs.scale_n[expert_id * kargs.N + kargs.N / 2 + coord_n / 2 +
                                      i * NWave * NPerXdl + iNWarp * NPerXdl + iNLane];
                });
            }
            else
            {
                static_for<0, NRepeat, 1>{}([&](auto i) {
                    vec_scale_B[i] = kargs.scale_n[expert_id * kargs.N + coord_n +
                                                   i * NWave * NPerXdl + iNWarp * NPerXdl + iNLane];
                });
            }

            static_for<0, MRepeat, 1>{}([&](auto i) {
                static_for<0, kM0, 1>{}([&](auto m0) {
                    static_for<0, kM2, 1>{}([&](auto m2) {
                        index_t M2_offset = m2 + iMLane * kM2 + m0 * kM2 * kM1 + iMWarp * MPerXdl +
                                            i * MPerXdl * MWave + coord_m;

                        vec_scale_A[i * kM0 * kM2 + m0 * kM2 + m2] =
                            kargs.scale_m[row_to_token_idx(M2_offset)];
                        if constexpr(!IsInputGemm)
                            vec_expert_weights[i * kM0 * kM2 + m0 * kM2 + m2] =
                                expert_weights[M2_offset];
                    });
                });
            });

            constexpr int UpAccStride = NRepeat / 2;

            //===----------------------------------------------------------------------===//
            // Pingpong process start
            //===----------------------------------------------------------------------===//
            if constexpr(kind == MoeFlatmmKind::kFFN_gemm1_gate_up)
            {
                LDSTileTensor gate_tensor, up_tensor;

                static_assert((NRepeat / NumNXdlPerWavePerShuffle) % 2 == 0);

                gate_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                    merge_sequences(
                        sequence<0 * NumMXdlPerWavePerShuffle, 0 * NumNXdlPerWavePerShuffle>{},
                        c_warp_y_index_zeros),
                    merge_sequences(sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                                    c_warp_y_lengths));
                up_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                    merge_sequences(sequence<0 * NumMXdlPerWavePerShuffle,
                                             0 * NumNXdlPerWavePerShuffle + UpAccStride>{},
                                    c_warp_y_index_zeros),
                    merge_sequences(sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                                    c_warp_y_lengths));

                static_for<0, NumNXdlPerWavePerShuffle, 1>{}([&](auto n_xdl) {
                    static_for<0, NumMXdlPerWavePerShuffle, 1>{}([&](auto m_xdl) {
                        constexpr int acc_xdl_offset =
                            (m_xdl + n_xdl * NumMXdlPerWavePerShuffle) * c_warp_y_lengths.product();
                        static_for<0, kM0, 1>{}([&](auto m0) {
                            static_for<0, kM2, 1>{}([&](auto m2) {
                                gate_tensor.get_thread_buffer()[acc_xdl_offset + m0 * kM2 + m2] *=
                                    vec_scale_A[m_xdl * kM0 * kM2 + m0 * kM2 + m2] *
                                    vec_scale_B[n_xdl];
                                up_tensor.get_thread_buffer()[acc_xdl_offset + m0 * kM2 + m2] *=
                                    vec_scale_A[m_xdl * kM0 * kM2 + m0 * kM2 + m2] *
                                    vec_scale_B[n_xdl + UpAccStride];
                            });
                        });
                    });
                });

                static_for<0, ActVectorSize, 1>{}([&](auto idx) {
                    ActivationOp{}(gate_tensor.get_thread_buffer().at(idx),
                                   gate_tensor.get_thread_buffer().at(idx));
                    lds_tile[0].get_thread_buffer().at(idx) =
                        gate_tensor.get_thread_buffer().at(idx) *
                        up_tensor.get_thread_buffer().at(idx);
                });
            }
            else
            {
                lds_tile[0].get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                    merge_sequences(
                        sequence<0 * NumMXdlPerWavePerShuffle, 0 * NumNXdlPerWavePerShuffle>{},
                        c_warp_y_index_zeros),
                    merge_sequences(sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                                    c_warp_y_lengths));
                static_for<0, NumNXdlPerWavePerShuffle, 1>{}([&](auto n_xdl) {
                    static_for<0, NumMXdlPerWavePerShuffle, 1>{}([&](auto m_xdl) {
                        constexpr int acc_xdl_offset =
                            (m_xdl + n_xdl * NumMXdlPerWavePerShuffle) * c_warp_y_lengths.product();
                        static_for<0, kM0, 1>{}([&](auto m0) {
                            static_for<0, kM2, 1>{}([&](auto m2) {
                                if constexpr(!IsInputGemm)
                                    lds_tile[0]
                                        .get_thread_buffer()[acc_xdl_offset + m0 * kM2 + m2] *=
                                        vec_expert_weights[m_xdl * kM0 * kM2 + m0 * kM2 + m2];
                                lds_tile[0].get_thread_buffer()[acc_xdl_offset + m0 * kM2 + m2] *=
                                    vec_scale_A[m_xdl * kM0 * kM2 + m0 * kM2 + m2] *
                                    vec_scale_B[n_xdl];
                            });
                        });
                    });
                });
                if constexpr(IsInputGemm)
                {
                    static_for<0, ActVectorSize, 1>{}([&](auto idx) {
                        ActivationOp{}(lds_tile[0].get_thread_buffer().at(idx),
                                       lds_tile[0].get_thread_buffer().at(idx));
                    });
                }
            }

            static_for<0, num_access, 1>{}([&](auto iAccess) {
                constexpr int read_stage  = iAccess % 2;
                constexpr int write_stage = read_stage ^ 1;

                block_sync_lds();
                constexpr auto idx_y_start      = SFC::get_index(number<iAccess.value>{});
                constexpr auto idx_y_start_next = SFC::get_index(number<iAccess.value + 1>{});

                constexpr auto mIter = number<idx_y_start.at(number<0>{}) / MPerIterationShuffle>{};
                constexpr auto nIter = number<idx_y_start.at(number<1>{}) / NPerIterationShuffle>{};

                constexpr auto mIter_next =
                    number<idx_y_start_next.at(number<0>{}) / MPerIterationShuffle>{};
                constexpr auto nIter_next =
                    number<idx_y_start_next.at(number<1>{}) / NPerIterationShuffle>{};

                const auto c_warptile_in_tensor_casted = cast_tile<ODataType>(lds_tile[read_stage]);

                store_tile(in_lds_window, c_warptile_in_tensor_casted);

                if constexpr(iAccess < num_access - 1)
                {
                    if constexpr(kind == MoeFlatmmKind::kFFN_gemm1_gate_up)
                    {
                        LDSTileTensor gate_tensor, up_tensor;

                        gate_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter_next * NumMXdlPerWavePerShuffle,
                                                     nIter_next * NumNXdlPerWavePerShuffle>{},
                                            c_warp_y_index_zeros),
                            merge_sequences(
                                sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                                c_warp_y_lengths));
                        up_tensor.get_thread_buffer() = c_block_tile.get_y_sliced_thread_data(
                            merge_sequences(
                                sequence<mIter_next * NumMXdlPerWavePerShuffle,
                                         nIter_next * NumNXdlPerWavePerShuffle + UpAccStride>{},
                                c_warp_y_index_zeros),
                            merge_sequences(
                                sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                                c_warp_y_lengths));

                        static_for<0, NumNXdlPerWavePerShuffle, 1>{}([&](auto n_xdl) {
                            static_for<0, NumMXdlPerWavePerShuffle, 1>{}([&](auto m_xdl) {
                                constexpr int acc_xdl_offset =
                                    (m_xdl + n_xdl * NumMXdlPerWavePerShuffle) *
                                    c_warp_y_lengths.product();
                                static_for<0, kM0, 1>{}([&](auto m0) {
                                    static_for<0, kM2, 1>{}([&](auto m2) {
                                        gate_tensor
                                            .get_thread_buffer()[acc_xdl_offset + m0 * kM2 + m2] *=
                                            vec_scale_A[mIter_next * NumMXdlPerWavePerShuffle *
                                                            kM0 * kM2 +
                                                        m_xdl * kM0 * kM2 + m0 * kM2 + m2] *
                                            vec_scale_B[nIter_next * NumNXdlPerWavePerShuffle +
                                                        n_xdl];
                                        up_tensor
                                            .get_thread_buffer()[acc_xdl_offset + m0 * kM2 + m2] *=
                                            vec_scale_A[mIter_next * NumMXdlPerWavePerShuffle *
                                                            kM0 * kM2 +
                                                        m_xdl * kM0 * kM2 + m0 * kM2 + m2] *
                                            vec_scale_B[nIter_next * NumNXdlPerWavePerShuffle +
                                                        n_xdl + UpAccStride];
                                    });
                                });
                            });
                        });
                        static_for<0, ActVectorSize, 1>{}([&](auto idx) {
                            ActivationOp{}(gate_tensor.get_thread_buffer().at(idx),
                                           gate_tensor.get_thread_buffer().at(idx));
                            lds_tile[write_stage].get_thread_buffer().at(idx) =
                                gate_tensor.get_thread_buffer().at(idx) *
                                up_tensor.get_thread_buffer().at(idx);
                        });
                    }
                    else
                    {
                        lds_tile[write_stage].get_thread_buffer() =
                            c_block_tile.get_y_sliced_thread_data(
                                merge_sequences(sequence<mIter_next * NumMXdlPerWavePerShuffle,
                                                         nIter_next * NumNXdlPerWavePerShuffle>{},
                                                c_warp_y_index_zeros),
                                merge_sequences(
                                    sequence<NumMXdlPerWavePerShuffle, NumNXdlPerWavePerShuffle>{},
                                    c_warp_y_lengths));
                        static_for<0, NumNXdlPerWavePerShuffle, 1>{}([&](auto n_xdl) {
                            static_for<0, NumMXdlPerWavePerShuffle, 1>{}([&](auto m_xdl) {
                                constexpr int acc_xdl_offset =
                                    (m_xdl + n_xdl * NumMXdlPerWavePerShuffle) *
                                    c_warp_y_lengths.product();
                                static_for<0, kM0, 1>{}([&](auto m0) {
                                    static_for<0, kM2, 1>{}([&](auto m2) {
                                        if constexpr(!IsInputGemm)
                                            lds_tile[write_stage]
                                                .get_thread_buffer()[acc_xdl_offset + m0 * kM2 +
                                                                     m2] *= vec_expert_weights
                                                [mIter_next * NumMXdlPerWavePerShuffle * kM0 * kM2 +
                                                 m_xdl * kM0 * kM2 + m0 * kM2 + m2];
                                        lds_tile[write_stage]
                                            .get_thread_buffer()[acc_xdl_offset + m0 * kM2 + m2] *=
                                            vec_scale_A[mIter_next * NumMXdlPerWavePerShuffle *
                                                            kM0 * kM2 +
                                                        m_xdl * kM0 * kM2 + m0 * kM2 + m2] *
                                            vec_scale_B[nIter_next * NumNXdlPerWavePerShuffle +
                                                        n_xdl];
                                    });
                                });
                            });
                        });
                        if constexpr(IsInputGemm)
                        {
                            static_for<0, ActVectorSize, 1>{}([&](auto idx) {
                                ActivationOp{}(lds_tile[write_stage].get_thread_buffer().at(idx),
                                               lds_tile[write_stage].get_thread_buffer().at(idx));
                            });
                        }
                    }
                }

                constexpr int MPerThread = TileEncodingPattern::Y2;
                statically_indexed_array<index_t, MPerThread> offsets;

                auto c_coord = dram_tile_distribution.calculate_index();
                static_for<0, MPerThread, 1>{}([&](auto m0) {
                    auto row_idx = coord_m + mIter * MPerIterationShuffle + c_coord[0] + m0;
                    auto fused_token =
                        kargs.p_sorted_token_ids[row_idx]; // topk-idx[31:24] + token_idx[23:0]

                    index_t scatter_token_id = fused_token & token_id_mask;
                    if constexpr(IsInputGemm)
                    {
                        scatter_token_id =
                            scatter_token_id * kargs.TopK + (fused_token >> token_id_offset);
                    }
                    offsets[m0] = scatter_token_id * kargs.stride_C;
                });

                block_sync_lds();

                auto c_out_tensor =
                    load_tile(make_tile_window(out_lds_window, dram_tile_distribution));

                auto c_scatter_tile_window =
                    make_tile_scatter_gather(c_block_window.get_bottom_tensor_view(),
                                             c_block_window.get_window_lengths(),
                                             c_block_window.get_window_origin(),
                                             dram_tile_distribution,
                                             offsets);

                if constexpr(!IsInputGemm ||
                             EpiloguePipeline::MemoryOperation == memory_operation_enum::atomic_add)
                {
                    c_scatter_tile_window.update(c_out_tensor);
                }
                else
                {
                    c_scatter_tile_window.store(c_out_tensor);
                }
                if constexpr(iAccess != num_access - 1)
                {
                    constexpr auto step = SFC::get_forward_step(iAccess);
                    // row_offset of out windows has been included in scatter offset
                    move_tile_window(c_block_window, {0, step.at(number<1>{})});
                }
            });
        }
    }
};

} // namespace ck_tile
