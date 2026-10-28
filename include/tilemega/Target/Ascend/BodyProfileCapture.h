// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <acl/acl.h>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <vector>
#if defined(TILEMEGA_LLM_COMPRESSED_PROFILE) && TILEMEGA_LLM_COMPRESSED_PROFILE
#include <memory>
#include <openssl/evp.h>
#include <zlib.h>
#endif

namespace tilemega::ascend::llm_host {
#if defined(TILEMEGA_LLM_COMPRESSED_PROFILE) && TILEMEGA_LLM_COMPRESSED_PROFILE
class ProfileSha256 {
  EVP_MD_CTX* context_=nullptr;
 public:
  ProfileSha256() {
    context_=EVP_MD_CTX_new();
    if(!context_ || EVP_DigestInit_ex(context_,EVP_sha256(),nullptr)!=1) {
      EVP_MD_CTX_free(context_);throw std::runtime_error("Cannot initialize profile SHA256");
    }
  }
  ProfileSha256(ProfileSha256 const&)=delete;
  ProfileSha256& operator=(ProfileSha256 const&)=delete;
  ~ProfileSha256(){EVP_MD_CTX_free(context_);}
  void Update(void const* data,size_t bytes) {
    if(EVP_DigestUpdate(context_,data,bytes)!=1)throw std::runtime_error("Profile SHA256 update failed");
  }
  std::string Finish() {
    unsigned char bytes[EVP_MAX_MD_SIZE];unsigned size=0;
    if(EVP_DigestFinal_ex(context_,bytes,&size)!=1 || size!=32)throw std::runtime_error("Profile SHA256 failed");
    static constexpr char hex[]="0123456789abcdef";std::string out;
    for(unsigned i=0;i<size;++i){out+=hex[bytes[i]>>4];out+=hex[bytes[i]&15];}return out;
  }
};
struct ProfileGzipClose {void operator()(gzFile_s* p) const {if(p)gzclose(p);}};
using ProfileGzip=std::unique_ptr<gzFile_s,ProfileGzipClose>;
#endif

inline void CaptureBodyProfile(std::filesystem::path const& path,uint8_t* device,uint64_t bytes) {
  if(!bytes || std::filesystem::exists(path))throw std::invalid_argument("Use a new nonempty Body profile");
#if defined(TILEMEGA_LLM_COMPRESSED_PROFILE) && TILEMEGA_LLM_COMPRESSED_PROFILE
  const std::string archive=path.string()+".gz",receipt=path.string()+".archive.json";
  if(std::filesystem::exists(archive) || std::filesystem::exists(receipt))
    throw std::invalid_argument("Never overwrite a captured profile archive");
  // The raw payload never occupies disk. Hash the actual D2H chunks before
  // compression, then independently decompress and verify them before success.
  std::vector<uint8_t> chunk(1<<20);ProfileSha256 captured;
  ProfileGzip output(gzopen(archive.c_str(),"wb1x"));
  if(!output)throw std::runtime_error("Cannot create compressed Body profile");
  for(uint64_t offset=0;offset<bytes;offset+=chunk.size()) {
    unsigned n=std::min<uint64_t>(chunk.size(),bytes-offset);
    if(aclrtMemcpy(chunk.data(),n,device+offset,n,ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS)
      throw std::runtime_error("Body profile D2H failed");
    captured.Update(chunk.data(),n);
    if(gzwrite(output.get(),chunk.data(),n)!=int(n))throw std::runtime_error("Body profile gzip write failed");
  }
  if(gzclose(output.release())!=Z_OK)throw std::runtime_error("Body profile gzip close failed");
  const auto source_sha=captured.Finish();ProfileSha256 roundtrip;uint64_t restored=0;
  ProfileGzip input(gzopen(archive.c_str(),"rb"));
  if(!input)throw std::runtime_error("Cannot verify compressed Body profile");
  for(;;) {
    int n=gzread(input.get(),chunk.data(),chunk.size());
    if(n<0)throw std::runtime_error("Body profile gzip verification failed");
    if(!n)break;restored+=n;roundtrip.Update(chunk.data(),n);
  }
  if(gzclose(input.release())!=Z_OK || restored!=bytes || roundtrip.Finish()!=source_sha)
    throw std::runtime_error("Body profile compressed roundtrip changed");
  std::ifstream file(archive,std::ios::binary);ProfileSha256 stored;
  while(file) {file.read((char*)chunk.data(),chunk.size());if(file.gcount())stored.Update(chunk.data(),file.gcount());}
  if(file.bad())throw std::runtime_error("Cannot hash compressed Body profile");
  std::ofstream record(receipt);
  record<<"{\"schema\":\"tilemega.as2.profile_archive.v1\",\"source\":"<<std::quoted(path.string())
    <<",\"archive\":"<<std::quoted(archive)<<",\"bytes\":"<<bytes
    <<",\"source_sha256\":"<<std::quoted(source_sha)<<",\"archive_sha256\":"<<std::quoted(stored.Finish())
    <<",\"roundtrip_verified\":true,\"capture\":\"streamed_D2H_compressed_without_raw_disk_copy\","
      "\"source_removal\":\"none; raw file was never created\"}\n";
  if(!record)throw std::runtime_error("Cannot save compressed Body profile receipt");
#else
  std::vector<uint8_t> data(bytes);
  if(aclrtMemcpy(data.data(),data.size(),device,data.size(),ACL_MEMCPY_DEVICE_TO_HOST)!=ACL_SUCCESS)
    throw std::runtime_error("Body profile D2H failed");
  std::ofstream file(path,std::ios::binary);file.write((char const*)data.data(),data.size());
  if(!file)throw std::runtime_error("Cannot save Body profile");
#endif
}
} // namespace tilemega::ascend::llm_host
