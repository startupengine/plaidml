// Copyright 2019, Intel Corporation

#pragma once

#include "mlir/IR/Types.h"

namespace pmlc {
namespace dialect {
namespace hir {

using mlir::MLIRContext;
using mlir::Type;
using mlir::TypeStorage;

enum Kinds {
  // An affine is a affine polynomial of indexes over integers
  IndexedTensor = Type::Kind::FIRST_PRIVATE_EXPERIMENTAL_2_TYPE,
  AffineMap,
};

class IndexedTensorType : public Type::TypeBase<IndexedTensorType, Type> {
 public:
  using Base::Base;
  static bool kindof(unsigned kind) {  //
    return kind == Kinds::IndexedTensor;
  }
  static IndexedTensorType get(MLIRContext* context) {  //
    return Base::get(context, Kinds::IndexedTensor);
  }
};

struct AffineMapStorage : public mlir::TypeStorage {
  explicit AffineMapStorage(int64_t ndim) : ndim(ndim) {}

  using KeyTy = int64_t;
  bool operator==(const KeyTy& key) const { return key == ndim; }

  static AffineMapStorage* construct(         //
      mlir::TypeStorageAllocator& allocator,  // NOLINT
      const KeyTy& key) {
    return new (allocator.allocate<AffineMapStorage>()) AffineMapStorage(key);
  }

  int64_t ndim;
};

class AffineMapType : public Type::TypeBase<AffineMapType, Type, AffineMapStorage> {
 public:
  using Base::Base;
  static bool kindof(unsigned kind) {  //
    return kind == Kinds::AffineMap;
  }
  static AffineMapType get(mlir::MLIRContext* context, int64_t ndim) {  //
    return Base::get(context, Kinds::AffineMap, ndim);
  }
};

}  // namespace hir
}  // namespace dialect
}  // namespace pmlc
