// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Codegen/DmDescriptors.h>
#include <mlir/IR/Builders.h>
#include <string>
#include <vector>

namespace tilemega::frontend {
struct ModelPlan;
void ValidateDmModelPlan(ModelPlan const&);
mlir::DictionaryAttr EncodeModelPlan(mlir::Builder&, ModelPlan const&,
                                   std::vector<std::uint8_t> const& written);

// Packed integer attributes follow the declaration order in DmDescriptors.h.
// Enum ranges and word counts are checked on both ingress and codegen egress.
mlir::DenseI64ArrayAttr EncodeDm(mlir::Builder&, codegen::ConvDesc const&);
mlir::DenseI64ArrayAttr EncodeDm(mlir::Builder&, codegen::DmBufferLayout const&);
mlir::DenseI64ArrayAttr EncodeDm(mlir::Builder&, codegen::DmGemmAccess const&);
mlir::DenseI64ArrayAttr EncodeDm(mlir::Builder&, codegen::DmMoeStage const&);
mlir::DictionaryAttr EncodeDm(mlir::Builder&, codegen::DmEpilogueChain const&);
codegen::ConvDesc DecodeDmConv(mlir::Attribute);
codegen::DmBufferLayout DecodeDmLayout(mlir::Attribute);
codegen::DmGemmAccess DecodeDmAccess(mlir::Attribute);
codegen::DmMoeStage DecodeDmMoeStage(mlir::Attribute);
codegen::DmEpilogueChain DecodeDmChain(mlir::Attribute);

std::string EmitDm(codegen::ConvDesc const&);
std::string EmitDm(codegen::DmBufferLayout const&);
std::string EmitDm(codegen::DmGemmAccess const&);
std::string EmitDm(codegen::DmMoeStage const&);
std::string EmitDm(codegen::DmEpilogueChain const&);

}  // namespace tilemega::frontend
