// Copyright 2019, Intel Corporation

include "pmlc/dialect/scalar/predicates.td"

def HirDialect : Dialect {
  let name = "pml_hir";
  let cppNamespace = "pmlc::dialect::hir";
}

// Whether a type is a RankedTensorType.
def IsRankedTensorTypePred : CPred<"$_self.isa<RankedTensorType>()">;

class RankedTensorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsRankedTensorTypePred, "ranked-tensor">;

def AnyRankedTensor : RankedTensorOf<[AnyType]>;
def AffineMap : Type<CPred<"$_self.isa<AffineMapType>()">, "affine-map">;

class HirOp<string mnemonic, list<OpTrait> traits = [NoSideEffect]> :
    Op<HirDialect, mnemonic, traits> {}

def PlaceholderOp : HirOp<"var"> {
  let summary = "placeholder operation";
  let results = (outs AnyRankedTensor);
  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, RankedTensorType type", [{
      result->types.push_back(type);
    }]
  >];
}

def DimOp : HirOp<"dim"> {
  let summary = "dimension index operation";
  let arguments = (ins AnyRankedTensor:$tensor, APIntAttr:$dim);
  let results = (outs Index);
  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, Value* tensor, int64_t value", [{
      auto type = builder->getIndexType();
      result->addOperands(tensor);
      result->addAttribute("dim", builder->getIntegerAttr(type, value));
      result->addTypes(type);
    }]
  >];
}

def AffineConstantOp : HirOp<"const_idx">, HasFolder {
  let summary = "affine constant";
  let arguments = (ins APIntAttr:$value);
  let results = (outs Index);
  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, int64_t value", [{
      auto type = builder->getIndexType();
      result->addAttribute("value", builder->getIntegerAttr(type, value));
      result->addTypes(type);
    }]
  >];
}

def AffineIndexOp : HirOp<"idx"> {
  let summary = "affine index";
  let results = (outs Index);
  let builders = [OpBuilder<
    "Builder* builder, OperationState* result", [{
      result->types.push_back(builder->getIndexType());
    }]
  >];
}

def AffineDomainOp : HirOp<"domain"> {
  let summary = "affine domain";
  let results = (outs AnyRankedTensor);
  let regions = (region
    SizedRegion<1>:$body
  );
}

def AffineSourceMapOp : HirOp<"msrc"> {
  let summary = "affine source map";
  let arguments = (ins AnyRankedTensor:$tensor, Variadic<Index>:$idxs);
  let results = (outs AffineMap);
  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, Value* tensor, ArrayRef<Value*> idxs", [{
      auto tensorType = tensor->getType().cast<RankedTensorType>();
      auto mapType = AffineMapType::get(builder->getContext(), tensorType.getRank());
      result->addOperands(tensor);
      result->addOperands(idxs);
      result->addTypes(mapType);
    }]
  >];
}

def AffineSinkMapOp : HirOp<"msink"> {
  let summary = "affine sink map";
  let arguments = (ins Variadic<Index>:$idxs_and_dims);
  let results = (outs AffineMap);
  let builders = [OpBuilder<
    "Builder* builder, OperationState* result, ArrayRef<Value*> idxs, ArrayRef<Value*> dims", [{
      auto mapType = AffineMapType::get(builder->getContext(), idxs.size());
      result->addOperands(idxs);
      result->addOperands(dims);
      result->addTypes(mapType);
    }]
  >];
}

def ConSumMulOp : HirOp<"+(x*y)", [Terminator]> {
  let summary = "+(x * y) contraction";
  let arguments = (ins
    AffineMap:$src1,
    AffineMap:$src2,
    AffineMap:$sink
  );
  let results = (outs AnyRankedTensor);
}

def ConAssignOp : HirOp<"=(x)", [Terminator]> {
  let summary = "contraction operation";
  let arguments = (ins AffineMap:$src, AffineMap:$sink);
  let results = (outs AnyRankedTensor);
}

def CallOp : HirOp<"call"> {
  let summary = "call operation";
}
