// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/AttentionPartition.h>
#include <cassert>
#include <initializer_list>

int main() {
  using tilemega::codegen::DecodeAttentionPartition;
  for(int past:{0,31,64,575,1087}) {
    auto p=DecodeAttentionPartition(1088,past,64,true);
    assert(p.blocks==17 && p.extent>0 && p.extent<=64);
    assert((p.live_blocks-1)*p.extent<=past && p.live_blocks*p.extent>past);
    assert(p.live_blocks<=p.blocks);
    int covered=0;
    for(int block=0;block<p.blocks;++block)
      for(int key=block*p.extent;key<(block+1)*p.extent && key<=past;++key)++covered;
    assert(covered==past+1);
  }
  auto old=DecodeAttentionPartition(1088,575,128,false);
  assert(old.blocks==9 && old.extent==128 && old.live_blocks==5);
  assert(DecodeAttentionPartition(1088,575,64,true).live_blocks==17);
  assert(DecodeAttentionPartition(1088,1088,64,true).blocks==0);
}
