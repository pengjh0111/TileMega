import os
import lit.formats
from lit.llvm import llvm_config

config.name = "TileMegaCG"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir"]
config.test_source_root = os.path.join(config.tilemega_source_dir, "test", "Dialect")
config.test_exec_root = os.path.join(config.tilemega_binary_dir, "test")
llvm_config.with_environment("PATH", [
    os.path.join(config.tilemega_binary_dir, "tools"),
    config.llvm_tools_dir,
], append_path=True)
config.substitutions.append(("tilemega-opt", os.path.join(
    config.tilemega_binary_dir, "tools", "tilemega-opt")))
config.substitutions.append(("FileCheck", os.path.join(
    config.llvm_tools_dir, "FileCheck")))
