// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ForwardGeometry.h>
#include <cassert>
#include <map>
using namespace tilemega;using namespace tilemega::solver;
int main(){
 for(int rows:{1,17,49,64,65,196,1024})for(int tm:{64,128})for(int batch:{1,2,4,8,16}){
  Sm90ForwardShape shape;shape.body=Sm90ForwardBody::kGemm;shape.dnn1_geometry=json::Object{{"rows",batch*rows},{"batch",batch}};
  Sm90ForwardStage c;c.tile_m=tm;int parts=Sm90ForwardPoolParts(shape,c);std::map<std::pair<int,int>,int>owners;int visited=0;
  for(int tile=0;tile<(batch*rows+tm-1)/tm;++tile)for(int row=tile*tm;row<(tile+1)*tm && row<batch*rows;++row){
   int image=row/rows,part=tile-(image*rows)/tm;assert(part>=0 && part<parts);auto key=std::make_pair(image,part);
   auto found=owners.find(key);assert(found==owners.end() || found->second==tile);owners[key]=tile;++visited;
  }
  assert(visited==batch*rows);if(rows==49 && tm==64)assert(parts==2);
 }
 for(int width:{7,9,16,17,256})for(int height:{11,16,33,256})for(auto tile:{std::pair<int,int>{8,8},{8,16},{16,16}}){
  Sm90ForwardShape shape;shape.body=Sm90ForwardBody::kDepthwise;shape.dnn1_geometry=json::Object{{"output_width",width},{"output_height",height}};
  Sm90ForwardStage c;c.body=shape.body;c.tile_w=tile.first;c.tile_h=tile.second;int parts=Sm90ForwardPoolParts(shape,c);int visited=0,last=-1;
  for(int y=0;y<height;y+=c.tile_h)for(int x=0;x<width;x+=c.tile_w){int part=(y/c.tile_h)*((width+c.tile_w-1)/c.tile_w)+x/c.tile_w;assert(part==++last && part<parts);visited+=std::min(c.tile_h,height-y)*std::min(c.tile_w,width-x);}
  assert(last+1==parts && visited==width*height);
 }
}
