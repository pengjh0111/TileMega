// SPDX-License-Identifier: BSD-3-Clause
// CPU synthetic child processes only; no CUDA compiler or runtime invocation.
#include "../../tools/commands/Mx5CompileReceipt.h"
#include <cassert>
#include <csignal>
#include <fstream>
#include <iostream>
using namespace tilemega;
namespace r = commands::mx_decode::compile_receipt_v1;
void Save(std::string const& p, std::string const& s) { std::ofstream(p) << s; }
template<class F> void Reject(F f) { bool caught=false; try { f(); } catch(std::exception const&) {caught=true;} assert(caught); }
int main(int argc, char** argv) {
  if (argc > 1 && std::string(argv[1]) == "--synthetic-compiler") {
    std::cout << "CPU synthetic process stdout\n"; std::cerr << "CPU synthetic process stderr\n";
    for(int i=2;i<argc;++i) std::cout << argv[i] << '\n';
    if(std::string(argv[4])!="no-binary") Save(argv[2],"CPU synthetic bytes; never a device artifact\n");
    if(argc>5) { std::ofstream changed(argv[5],std::ios::app); changed<<"mutated by synthetic child"; }
    if(std::string(argv[3])=="signal") { std::raise(SIGTERM); return 3; }
    return std::stoi(argv[3]);
  }
  assert(std::filesystem::path(r::CompilerPath("sh")).is_absolute());
  if (argc == 5 && std::string(argv[1]) == "--capture-synthetic") {
    std::string out=argv[2];
    std::ifstream stream(out+".build_command.txt");
    std::string command((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
    auto args=r::Arguments(command);
    return r::Run(args,out,r::Inputs(out,argv[3],args.at(0),argv[4]));
  }
  auto exe=std::filesystem::read_symlink("/proc/self/exe").string();
  char directory[]="/tmp/mx5_receipt_cpu_XXXXXX"; assert(::mkdtemp(directory));
  std::string root=directory;
  auto words=r::Arguments("'"+exe+"' 'a b' 'it'\\''s' \"double space\"");
  assert((words==std::vector<std::string>{exe,"a b","it's","double space"}));
  for(int trial=0;trial<5;++trial) {
    std::string out=root+"/attempt "+std::to_string(trial),pin=out+".requested.json";
    r::RequireFresh(out);
    Save(pin,"{\"synthetic_fixture\":true}");
    for(char const* suffix:{".mx5_pin.json",".cu",".source.json",".source.json.patch"}) Save(out+suffix,"CPU synthetic input\n");
    std::vector<std::string> args{exe,"--synthetic-compiler",out,trial==2?"signal":trial==1?"7":"0",trial==1?"no-binary":"binary"};
    if(trial==3) args.push_back(pin);
    Save(out+".build_command.txt","CPU test command fixture, argv separately tested\n");
    auto inputs=r::Inputs(out,pin,exe,exe);
    if(trial==3) Reject([&]{r::Run(args,out,inputs);});
    else assert(r::Run(args,out,inputs)==(trial==2?128+SIGTERM:trial==1?7:0));
    auto receipt=json::ParseFile(out+".compile_receipt.json");
    assert(receipt.At("kind").AsString("kind")=="mx5_native_compile_receipt");
    assert(receipt.At("same_compile_mir").AsString("MIR")==r::Hash(out+".ptxas.log"));
    assert(receipt.At("binary_sha256").IsNull()==(trial==1));
    assert(receipt.At("changed_inputs").AsArray("changed").size()==std::size_t(trial==3));
    assert(receipt.At("argv").AsArray("argv").size()==args.size());
    Reject([&]{r::RequireFresh(out);});
    Reject([&]{r::WriteExclusive(out+".compile_receipt.json",json::Object{});});
  }
  std::filesystem::remove_all(root);
  std::cout<<"mx5_compile_receipt: CPU synthetic real-child status/artifact/hash/immutability tests passed\n";
}
