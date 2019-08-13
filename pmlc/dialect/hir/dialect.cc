// Copyright 2019, Intel Corporation

#include "mlir/IR/Dialect.h"
#include "pmlc/dialect/hir/ops.h"

namespace pmlc {
namespace dialect {
namespace hir {

class Dialect : public mlir::Dialect {
 public:
  explicit Dialect(mlir::MLIRContext* ctx) : mlir::Dialect("pml_hir", ctx) {
    addTypes<IndexedTensorType>();
    addOperations<
#define GET_OP_LIST
#include "pmlc/dialect/hir/ops.cpp.inc"
        >();
  }
};

static mlir::DialectRegistration<Dialect> EdslOps;

}  // namespace hir
}  // namespace dialect
}  // namespace pmlc
