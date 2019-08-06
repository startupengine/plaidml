// Copyright 2019, Intel Corporation

include "pmlc/dialect/scalar/predicates.td"

// Whether a type is a RankedTensorType.
def IsRankedTensorTypePred : CPred<"$_self.isa<RankedTensorType>()">;

class RankedTensorOf<list<Type> allowedTypes> :
  ShapedContainerType<allowedTypes, IsRankedTensorTypePred, "ranked-tensor">;

def AnyRankedTensor : RankedTensorOf<[AnyType]>;

def HirDialect : Dialect {
  let name = "pml_hir";
  let cppNamespace = "pmlc::dialect::hir";
}

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
  let arguments = (ins AnyRankedTensor:$tensor, APIntAttr:$index);
  let results = (outs Index);
}

def ContractionOp : HirOp<"con"> {
  let summary = "contraction operation";
  let arguments = (ins Variadic<AnyRankedTensor>:$operands);
  let results = (outs AnyRankedTensor);
}

def CallOp : HirOp<"call"> {
  let summary = "call operation";
}
