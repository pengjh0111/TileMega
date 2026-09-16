// SPDX-License-Identifier: BSD-3-Clause
// Lossless sorted adjacency TSV run-length codec, independent of the solver.
#include <iostream>
#include <string>
#include <sstream>
#include <stdexcept>
int main(int argc,char**argv)try{
  if(argc!=2)throw std::runtime_error("encode|decode");
  std::ios::sync_with_stdio(false);std::cin.tie(nullptr);
  std::string header;std::getline(std::cin,header);long p,c;
  if(std::string(argv[1])=="encode"){
    if(header!="producer\tconsumer")throw std::runtime_error("DAG header");
    std::cout<<"producer\tfirst\tlast\n";long producer=-1,first=-1,last=-1;
    while(std::cin>>p){
      if(!(std::cin>>c) || p<0 || c<0)throw std::runtime_error("malformed DAG row");
      if(p==producer && c==last+1){last=c;continue;}
      if(producer>=0){if(p<producer || (p==producer && c<=last))throw std::runtime_error("unsorted/duplicate DAG");std::cout<<producer<<'\t'<<first<<'\t'<<last<<'\n';}
      producer=p;first=last=c;
    }
    if(producer>=0)std::cout<<producer<<'\t'<<first<<'\t'<<last<<'\n';
  }else if(std::string(argv[1])=="decode"){
    if(header!="producer\tfirst\tlast")throw std::runtime_error("RLE header");
    std::cout<<"producer\tconsumer\n";long last;
    while(std::cin>>p){if(!(std::cin>>c>>last) || p<0 || c<0)throw std::runtime_error("malformed RLE row");if(c>last)throw std::runtime_error("invalid range");for(;c<=last;++c)std::cout<<p<<'\t'<<c<<'\n';}
  }else throw std::runtime_error("unknown codec");
  if(!std::cin.eof())throw std::runtime_error("malformed TSV");
}catch(std::exception const&e){std::cerr<<e.what()<<'\n';return 1;}
