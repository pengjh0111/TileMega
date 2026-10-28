#include <tilemega/Target/Ascend/BodyProfileCapture.h>
#include <cstring>
#include <iostream>

static unsigned copies=0;
// CPU transport fixture only; this executable does not link/init ACL runtime.
extern "C" aclError aclrtMemcpy(void* dst,size_t maximum,void const* src,size_t bytes,aclrtMemcpyKind kind) {
  if(kind!=ACL_MEMCPY_DEVICE_TO_HOST || bytes>maximum)return 1;
  std::memcpy(dst,src,bytes);++copies;return ACL_SUCCESS;
}
int main(int argc,char** argv) {
  namespace fs=std::filesystem;
  try {
    if(argc!=2)throw std::invalid_argument("Supply a new AS2 probe output directory");
    fs::path root="/home/d50064638/junhuip/TileMega/runs/as2/probes";
    auto out=fs::weakly_canonical(argv[1]),relative=out.lexically_relative(root);
    if(out!=fs::path(argv[1]) || relative.empty() || *relative.begin()==".." || out==root || fs::exists(out))
      throw std::invalid_argument("Use a new canonical AS2 probe directory");
    fs::create_directory(out);std::vector<uint8_t> data((2<<20)+13);
    for(size_t i=0;i<data.size();++i)data[i]=(i*17)%251;
    auto path=out/"profile.bin";
    tilemega::ascend::llm_host::CaptureBodyProfile(path,data.data(),data.size());
    if(copies!=3 || fs::exists(path) || !fs::exists(path.string()+".archive.json"))return 1;
    try {tilemega::ascend::llm_host::CaptureBodyProfile(path,data.data(),data.size());return 1;}
    catch(std::invalid_argument const&) {}
    if(copies!=3)return 1;
    std::cout<<"{\"status\":\"pass\",\"cpu_transport_fixture\":true,\"acl_initialized\":false,"
      "\"chunks\":3,\"bytes\":"<<data.size()<<",\"raw_file_created\":false,\"overwrite_rejected\":true}\n";
    return 0;
  }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
