#pragma once
#include <cute/tensor.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <unit/gemm/device/default_gemm_configuration.hpp>

namespace v_d {
using BaseConfig = cutlass::gemm::device::DefaultGemmConfigurationToCutlass3Types<
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float>;
using Base = BaseConfig::CollectiveMainloop;

template <int M, int N, int Stages = 3>
using Collective = cutlass::gemm::collective::CollectiveMma<
    cutlass::gemm::MainloopSm80CpAsync<Stages>,
    cute::Shape<cute::C<M>, cute::C<N>, cute::_32>,
    typename Base::ElementA, typename Base::StrideA,
    typename Base::ElementB, typename Base::StrideB,
    typename Base::TiledMma,
    typename Base::GmemTiledCopyA, typename Base::SmemLayoutAtomA,
    typename Base::SmemCopyAtomA, typename Base::TransformA,
    typename Base::GmemTiledCopyB, typename Base::SmemLayoutAtomB,
    typename Base::SmemCopyAtomB, typename Base::TransformB>;
}  // namespace v_d
