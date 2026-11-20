// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#if defined(__CUDACC__)
#define TILEMEGA_DENSE_HD __host__ __device__
#else
#define TILEMEGA_DENSE_HD
#endif

namespace tilemega::backend {

struct Sm90DenseKeyRange {
  int begin,end;
  TILEMEGA_DENSE_HD constexpr bool Contains(int position) const {
    return begin<=position && position<end;
  }
};

TILEMEGA_DENSE_HD constexpr Sm90DenseKeyRange Sm90DensePartitionRange(
    int past,int chunk,int chunks,int new_row_work=0) {
  int total=past+1;
  // Clamp at past so only the last chunk owns the newly appended row.
  long long begin=(static_cast<long long>(total)+new_row_work)*chunk/chunks;
  long long end=(static_cast<long long>(total)+new_row_work)*(chunk+1)/chunks;
  begin=begin<past?begin:past;
  end=chunk==chunks-1?total:(end<past?end:past);
  return {int(begin),int(end)};
}

template<int Chunks>
struct Sm90DenseAttentionPartition {
  static_assert(Chunks>=1 && Chunks<=64);
  TILEMEGA_DENSE_HD static constexpr Sm90DenseKeyRange Range(int past,int chunk,
                                                          int new_row_work=0) {
    return Sm90DensePartitionRange(past,chunk,Chunks,new_row_work);
  }
  TILEMEGA_DENSE_HD static constexpr int MaxKeys(int capacity,int new_row_work=0) {
    return int((static_cast<long long>(capacity)+new_row_work+Chunks-1)/Chunks);
  }
};

}  // namespace tilemega::backend
#undef TILEMEGA_DENSE_HD
