// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Target/Calibration.h>
#include <tilemega/Support/BuildFingerprint.h>
#include <tilemega/Support/Json.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>
#include <filesystem>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

namespace tilemega::commands::calibration_suite {
namespace {
std::string Sha256(std::filesystem::path const& path) {
  auto input=llvm::MemoryBuffer::getFile(path.string());
  if(!input)throw std::runtime_error("cannot hash calibration evidence "+path.string());
  llvm::SHA256 digest;digest.update(input.get()->getBuffer());return llvm::toHex(digest.final(),true);
}
}
int RunSuite(int argc,char** argv) {
  calib::Options options;std::string output,base,sections;
  auto stamps=json::Parse(TILEMEGA_CALIBRATION_STAMPS_JSON);
  for(int i=1;i<argc;++i) {
    std::string arg=argv[i];
    if(arg=="--stamps"){std::cout<<stamps.Dump()<<'\n';return 0;}
    if(arg=="--help") {
      std::cout<<"tilemega calibrate --suite serving --out TARGET [--base TARGET] "
          "[--device N] [--repeats N] [--sections base,bf16,wait,events,hop,inflight,task_bodies]\n";
      return 0;
    }
    if(i+1==argc)throw std::runtime_error("missing calibration suite option value");
    auto value=std::string(argv[++i]);
    if(arg=="--suite" && value=="serving")continue;
    if(arg=="--out")output=value;else if(arg=="--base")base=value;
    else if(arg=="--device")options.device=std::stoi(value);
    else if(arg=="--repeats")options.repeats=std::stoi(value);
    else if(arg=="--sections")sections=value;
    else throw std::runtime_error("unknown calibration suite option: "+arg);
  }
  if(output.empty() || options.repeats<1)throw std::runtime_error("suite requires --out and positive repeats");
  std::set<std::string> requested;
  std::istringstream csv(sections);std::string section;
  while(std::getline(csv,section,',')) {
    if(!stamps.Find(section))throw std::runtime_error("unknown calibration section: "+section);
    requested.insert(section);
  }
  analysis::IslContext context;
  auto probed=TargetSpec::Probe(options.device);
  if(auto status=cudaSetDevice(options.device);status!=cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
  if(base.empty())base=(std::filesystem::path(TILEMEGA_SOURCE_DIR)/"configs/targets"/(probed.arch_tag+".json")).string();
  auto target=TargetSpec::FromJson(base);
  if(target.arch_tag!=probed.arch_tag)throw std::runtime_error("calibration base does not match device architecture");
  target.res=probed.res;target.caps=probed.caps;
  auto directory=std::filesystem::path(output+".sections");
  std::filesystem::create_directories(directory);
  for(auto const* name:{"base","bf16","wait","events","hop","inflight","task_bodies"}) {
    if(!requested.empty() && !requested.count(name))continue;
    std::string current=name;auto raw_path=directory/(current+".tsv");std::ofstream raw(raw_path);
    if(!raw)throw std::runtime_error("cannot open calibration evidence");
    std::cerr<<"calibrate section="<<current<<'\n';
    options.bf16=current!="base";
    if(current=="base") {target.calib={};calib::Run(target,options,raw);}
    else if(current=="bf16") {
      auto f32=target.calib;target.calib={};calib::Run(target,options,raw);
      target.calib_bf16=std::move(target.calib);target.calib=std::move(f32);
    }else {
      if(!target.calib_bf16.calibrated)throw std::runtime_error("section requires the BF16 base calibration");
      if(current=="wait") {
        calib::MeasureServingWaitPolicy(target,options,raw);
        target.calibration_stamps.erase("hop");target.calibration_stamps.erase("events");
      }else if(current=="events")calib::MeasureServingEvents(target,options,raw);
      else if(current=="hop")calib::MeasureServingHop(target,options,raw);
      else if(current=="inflight")calib::MeasureInflight(target,options,raw);
      else if(current=="task_bodies")calib::MeasureServingTaskBodies(target,options,raw);
    }
    raw.close();
    if(current=="events") {
      auto& e=target.event_bf16;e.source=e.task_source=std::filesystem::absolute(raw_path).string();
      e.source_sha256=e.task_source_sha256=Sha256(raw_path);
    }
    target.calibration_stamps[name]=stamps.At(name).AsString("calibration stamp");
    auto temporary=output+".pending";target.ToJson(temporary);std::filesystem::rename(temporary,output);
  }
  std::cout<<target.ToJson()<<'\n';return 0;
}
} // namespace tilemega::commands::calibration_suite
