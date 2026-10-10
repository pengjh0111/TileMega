// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Backend/ConvIteration.h>
#include <tilemega/Codegen/DmConvRuntime.h>
#include <cassert>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>
namespace tilemega::tests::conv_iteration_test {
int TestConvIteration(int,char**) {
  using namespace tilemega;
  unsigned geometries=0,splits=0,rejections=0;std::uint64_t checked=0,cursors=0,cursor_slots=0;
  for(unsigned c:{3,4,8,16,24,27,32,64,96,144,1024})
    for(unsigned pad:{4,8})for(unsigned r:{1,2,3,7})for(unsigned s:{1,2,3,7})
      for(unsigned tk:{16,32,64,128}) {
    unsigned cp=c<=4?(c+pad-1)/pad*pad:(c+7)/8*8;
    codegen::ConvDesc conv;conv.c=c;conv.r=r;conv.s=s;
    codegen::DmBufferLayout layout;layout.kind=codegen::DmLayout::kNHWC;
    layout.rank=4;layout.logical[3]=c;layout.physical[3]=cp;
    if(cp<tk && tk%cp) {
      bool rejected=false;try{(void)backend::ConvIterationGeometry::Build(conv,layout,tk);}
      catch(std::invalid_argument const&){rejected=true;}
      assert(rejected);++rejections;continue;
    }
    auto geometry=backend::ConvIterationGeometry::Build(conv,layout,tk);
    conv.dilation_h=2;conv.dilation_w=3;
    layout.strides[2]=(cp+7)/8*8;
    layout.strides[1]=(3*s+5)*layout.strides[2];
    for(auto begin:std::set<std::uint64_t>{0,geometry.iterations/2,geometry.iterations-1})
      for(unsigned lane=0;lane<tk;lane+=cp==4?4:8) {
        backend::ConvIterationCursor cursor(geometry,conv,layout,begin,lane);
        for(auto it=begin;it<geometry.iterations;++it) {
          auto expected=geometry.At(it,lane),actual=cursor.Point();
          assert(actual.valid==expected.valid);
          if(actual.valid) {
            assert(actual.r==expected.r && actual.s==expected.s && actual.c==expected.c);
            assert(cursor.a_offset==std::int64_t(expected.r)*2*layout.strides[1]+
                std::int64_t(expected.s)*3*layout.strides[2]+expected.c);
            assert(cursor.b_offset==(std::uint64_t(expected.r)*s+expected.s)*cp+expected.c);
          }
          ++cursor_slots;cursor.Advance();
        }
        ++cursors;
      }
    std::set<std::tuple<unsigned,unsigned,unsigned>> found;
    unsigned logical=0;
    for(std::uint64_t it=0;it<geometry.iterations;++it)for(unsigned lane=0;lane<tk;++lane) {
      auto point=geometry.At(it,lane);++checked;
      if(!point.valid)continue;
      assert(point.r<r && point.s<s && point.c<cp);
      assert(found.emplace(point.r,point.s,point.c).second);
      assert(geometry.Iteration(point.r,point.s,point.c)==it);
      logical+=point.c<c;
    }
    assert(found.size()==std::uint64_t(r)*s*cp && logical==r*s*c);
    // Independent nested filter/channel loops assign the reference iteration.
    for(unsigned rr=0;rr<r;++rr)for(unsigned ss=0;ss<s;++ss)for(unsigned cc=0;cc<cp;++cc) {
      auto expected=cp>=tk?std::uint64_t(rr*s+ss)*((cp+tk-1)/tk)+cc/tk:
          std::uint64_t((rr*s+ss)*cp+cc)/tk;
      assert(geometry.Iteration(rr,ss,cc)==expected);
    }
    for(unsigned split:{2,4,8,16,32})if(geometry.iterations%split==0) {
      auto width=geometry.iterations/split;
      assert(width);std::set<std::tuple<unsigned,unsigned,unsigned>> all;
      for(unsigned j=0;j<split;++j)for(auto const& point:found) {
        auto [rr,ss,cc]=point;
        auto it=geometry.Iteration(rr,ss,cc);
        if(it>=j*width && it<(j+1)*width)assert(all.insert(point).second);
      }
      assert(all==found);++splits;
    }
    assert(!geometry.At(geometry.iterations,0).valid && !geometry.At(0,tk).valid);
    ++geometries;
  }
  codegen::ConvDesc conv;conv.c=24;conv.r=conv.s=3;
  codegen::DmBufferLayout layout;layout.kind=codegen::DmLayout::kNHWC;layout.rank=4;
  layout.logical[3]=layout.physical[3]=24;
  auto g=backend::ConvIterationGeometry::Build(conv,layout,16);
  assert(g.iterations==18 && g.At(8,0).r==1 && g.At(8,0).s==1 &&
      g.At(8,0).c==0 && !g.At(8,24).valid);
  for(auto const& pair:std::vector<std::pair<unsigned,unsigned>>{{3,12},{24,64},{27,28}}) {
    conv.c=pair.first;layout.logical[3]=pair.first;layout.physical[3]=pair.second;
    bool rejected=false;try{(void)backend::ConvIterationGeometry::Build(conv,layout,16);}
    catch(std::invalid_argument const&){rejected=true;}
    assert(rejected);++rejections;
  }
  unsigned launches=0,invalid_launches=0;
  for(unsigned channels:{3,24,64})for(unsigned kernel:{1,2,3,7})
    for(unsigned tk:{16,32,64,128})for(unsigned stride:{1,2}) {
      codegen::ConvDesc c;c.n=2;c.h=9;c.w=11;c.c=channels;c.k=19;
      c.r=c.s=kernel;c.pad_h=c.pad_w=3;c.stride_h=c.stride_w=stride;
      c.p=(c.h+6-kernel)/stride+1;c.q=(c.w+6-kernel)/stride+1;
      codegen::DmBufferLayout l;l.kind=codegen::DmLayout::kNHWC;l.rank=4;
      l.logical[0]=l.physical[0]=2;l.logical[1]=c.h;l.logical[2]=c.w;
      l.logical[3]=channels;l.physical[3]=channels==3?4:channels;
      l.halo_top=l.halo_bottom=l.halo_left=l.halo_right=3;
      l.physical[1]=c.h+6;l.physical[2]=c.w+6;
      l.strides[3]=1;l.strides[2]=l.physical[3];
      l.strides[1]=l.physical[2]*l.strides[2];l.strides[0]=l.physical[1]*l.strides[1];
      if(channels==24 && tk>24) {
        bool rejected=false;
        try {(void)codegen::DmConvRuntime::Build(c,l,2*c.p*c.q,19,channels*kernel*kernel,tk,1);}
        catch(std::invalid_argument const&) {rejected=true;}
        assert(rejected);++invalid_launches;continue;
      }
      auto expected=backend::ConvIterationGeometry::Build(c,l,tk);
      for(unsigned split:{1,2,4,8}) {
        bool rejected=false;
        try {
          auto launch=codegen::DmConvRuntime::Build(c,l,2*c.p*c.q,19,channels*kernel*kernel,tk,split);
          assert(expected.iterations%split==0 && launch.tiles==int(expected.iterations));
          assert(launch.storage_k==launch.tiles*int(tk));++launches;
        }catch(std::invalid_argument const&) {rejected=true;}
        assert(rejected==(expected.iterations%split!=0));
      }
      for(unsigned fault=0;fault<11;++fault) {
        auto bad=c;auto wrong=l;int rows=2*c.p*c.q,k=channels*kernel*kernel;
        switch(fault) {
          case 0:++rows;break;
          case 1:++k;break;
          case 2:++bad.p;break;
          case 3:bad.stride_h=0;break;
          case 4:bad.dilation_w=0;break;
          case 5:wrong.halo_top=0;break;
          case 6:wrong.strides[2]=2;break;
          case 7:wrong.physical[1]=c.h;break;
          case 8:wrong.fill=codegen::DmFill::kNegativeInfinity;break;
          case 9:wrong.strides[1]=1;break;
          case 10:wrong.strides[0]=1;break;
        }
        bool rejected=false;
        try {(void)codegen::DmConvRuntime::Build(bad,wrong,rows,19,k,tk,1);}
        catch(std::invalid_argument const&) {rejected=true;}
        assert(rejected);++invalid_launches;
      }
    }
  std::cout<<"CONV_ITERATION geometries="<<geometries<<" splits="<<splits
      <<" rejected="<<rejections<<" issued_slots="<<checked
      <<" cursors="<<cursors<<" cursor_slots="<<cursor_slots
      <<" runtime_launches="<<launches<<" invalid_launches="<<invalid_launches<<" PASS\n";
  return 0;
}

}
