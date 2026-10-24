// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/Sm90DenseEpilogueContract.h>
#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <stdexcept>
using namespace tilemega::codegen::native_dense::epilogue_contract;

template<class F> void Reject(F fn) {
  bool rejected=false;
  try {fn();}catch(std::invalid_argument const&) {rejected=true;}
  assert(rejected);
}
int main() {
  // Two rows:16 norm partials per row and32 output partials per row, each f32.
  std::array<Buffer,2> buffers{{{kInternal,kF32,0,16,0,0,true},
      {kExternal,kF32,0,32,0,0,true}}};
  Gemm stage{0,1,2,128,256,32,true,false,true};
  auto validate=[&] {return ValidateMetadata(buffers.data(),2,&stage,1,2);};
  assert((validate()==std::vector<std::uint64_t>{128,256}));
  auto const valid=buffers;
  for(unsigned id:{0u,1u}) {
    for(unsigned dtype:{kBF16,kI32}) {
      buffers[id].dtype=dtype;Reject(validate);buffers=valid;
    }
    buffers[id].live=false;Reject(validate);buffers=valid;
    buffers[id].per_batch=0;Reject(validate);buffers=valid;
    --buffers[id].per_batch;Reject(validate);buffers=valid;
    buffers[id].per_past=1;Reject(validate);buffers=valid;
    buffers[id].constant=std::numeric_limits<std::uint64_t>::max();
    Reject(validate);buffers=valid;
  }
  stage.norm=2;Reject(validate);stage.norm=0;
  stage.sums=2;Reject(validate);stage.sums=1;
  stage.batch=1;Reject(validate);stage.batch=2;
  stage.norm=stage.sums=kAbsent;
  buffers={{{kInternal,kBF16,0,0,0,0,false},{kExternal,kI32,0,0,0,0,false}}};
  assert((validate()==std::vector<std::uint64_t>{0,0}));
  buffers=valid;stage={0,1,2,128,256,32,true,false,true};

  alignas(16) std::array<unsigned char,256> storage{};storage.fill(0x5a);
  unsigned allocations=0,initializations=0;
  auto allocate=[&](std::size_t bytes)->void* {
    ++allocations;assert(bytes==128);return storage.data();
  };
  auto initialize=[&](void* pointer,std::size_t bytes) {
    ++initializations;assert(pointer==storage.data() && bytes==128);
    std::fill_n(static_cast<unsigned char*>(pointer),bytes,0);
  };
  assert(BindBuffer(buffers[0],2,128,nullptr,allocate,initialize)==storage.data());
  assert(allocations==1 && initializations==1 && storage[0]==0 && storage[127]==0 && storage[128]==0x5a);
  auto forbidden_allocate=[](std::size_t)->void* {throw std::logic_error("external allocated");};
  auto forbidden_initialize=[](void*,std::size_t) {throw std::logic_error("external initialized");};
  storage.fill(0x5a);
  assert(BindBuffer(buffers[1],2,256,storage.data(),forbidden_allocate,forbidden_initialize)==storage.data());
  assert(storage.front()==0x5a && storage.back()==0x5a);
  Reject([&] {BindBuffer(buffers[1],2,256,nullptr,forbidden_allocate,forbidden_initialize);});
  Reject([&] {BindBuffer(buffers[0],2,128,nullptr,
      [](std::size_t)->void* {return nullptr;},forbidden_initialize);});

  // Legacy32 and vector8 are different storage contracts, independent of Op.
  buffers[0]={kInternal,kF32,0,4,0,0,true};
  stage={0,kAbsent,2,128,256,32,false,false,true};
  assert(validate()[0]==32);
  stage.norm=kAbsent;stage.sums=0;stage.residual=true;stage.tile_n=8;
  Reject(validate);stage.tile_n=32;buffers[0].per_batch=8;
  assert(validate()[0]==64);
}
