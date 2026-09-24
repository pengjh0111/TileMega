// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <sys/wait.h>
#include <signal.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#include <unistd.h>
#include <cerrno>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace tilemega::solver {
struct IsolatedResult {int status=0;std::string payload;};
// Fork after importing once: immutable L-sem and warm symbolic caches are
// inherited, while ISL/MLIR mutable state is isolated by copy-on-write.
inline std::vector<IsolatedResult> EvaluateIsolated(
    std::vector<std::function<std::string()>> const& jobs,
    std::vector<std::string> const& prefixes) {
  if(jobs.size()!=prefixes.size())throw std::invalid_argument("worker prefix count mismatch");
  for(auto const& prefix:prefixes)
    std::filesystem::create_directories(std::filesystem::path(prefix).parent_path());
  std::vector<pid_t> children;
  std::cout.flush();std::cerr.flush();
  auto parent=getpid();
  for(std::size_t i=0;i<jobs.size();++i) {
    pid_t pid=fork();
    if(pid<0) {
      for(auto child:children){kill(child,SIGTERM);while(waitpid(child,nullptr,0)<0 && errno==EINTR){}}
      throw std::runtime_error("cannot fork isolated candidate evaluation");
    }
    if(pid==0) {
#ifdef __linux__
      // An interrupted compiler must not leave expensive orphan evaluations.
      if(prctl(PR_SET_PDEATHSIG,SIGTERM)!=0 || getppid()!=parent)_exit(125);
#endif
      std::ofstream log(prefixes[i]+".log");std::cerr.rdbuf(log.rdbuf());
      try {
        auto value=jobs[i]();std::ofstream result(prefixes[i]+".result");result<<value;result.close();
        bool ok=bool(result);log.flush();_exit(ok?0:2);
      }catch(std::exception const& e){std::cerr<<e.what()<<'\n';log.flush();_exit(1);}
    }
    children.push_back(pid);
  }
  std::vector<IsolatedResult> result;
  for(std::size_t i=0;i<children.size();++i) {
    int status=0;pid_t waited;
    do {waited=waitpid(children[i],&status,0);}while(waited<0 && errno==EINTR);
    IsolatedResult entry;entry.status=waited<0?127:WIFEXITED(status)?WEXITSTATUS(status):128+WTERMSIG(status);
    if(entry.status==0){std::ifstream input(prefixes[i]+".result");std::ostringstream text;text<<input.rdbuf();entry.payload=text.str();if(!input)entry.status=126;}
    result.push_back(std::move(entry));
  }
  return result;
}
}
