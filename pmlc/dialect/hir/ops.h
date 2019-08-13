// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/StandardTypes.h"

#include "pmlc/dialect/hir/types.h"

namespace pmlc {
namespace dialect {
namespace hir {

using llvm::APInt;
using mlir::ArrayRef;
using mlir::Attribute;
using mlir::Builder;
using mlir::IndexType;
using mlir::IntegerAttr;
using mlir::LogicalResult;
using mlir::NamedAttribute;
using mlir::Op;
using mlir::Operation;
using mlir::OperationState;
using mlir::OpFoldResult;
using mlir::RankedTensorType;
using mlir::Region;
using mlir::StringRef;
using mlir::Type;
using mlir::TypeAttr;
using mlir::Value;

namespace OpTrait = mlir::OpTrait;

#define GET_OP_CLASSES
#include "pmlc/dialect/hir/ops.h.inc"

}  // namespace hir
}  // namespace dialect
}  // namespace pmlc
