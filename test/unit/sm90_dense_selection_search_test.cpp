// SPDX-License-Identifier: BSD-3-Clause
#include "NativeDense.h"
#include <cassert>
#include <stdexcept>
#include <string>

int main() {
  using tilemega::commands::ValidateNativeSelectionSearch;
  ValidateNativeSelectionSearch(true,true,false);
  for(bool converged:{false,true})for(bool exhausted:{false,true}) {
    ValidateNativeSelectionSearch(false,converged,exhausted);
    if(converged && !exhausted)continue;
    bool rejected=false;
    try {ValidateNativeSelectionSearch(true,converged,exhausted);}
    catch(std::invalid_argument const& error) {
      rejected=std::string(error.what()).find("converged search")!=std::string::npos;
    }
    assert(rejected);
  }
}
