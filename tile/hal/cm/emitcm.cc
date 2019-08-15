// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/emitcm.h"

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "base/util/error.h"
#include "tile/lang/exprtype.h"
#include "tile/lang/fpconv.h"

#include "base/util/env.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

static std::map<std::pair<DataType, sem::LimitConst::Which>, std::string> LimitConstLookup = {
    {{DataType::BOOLEAN, sem::LimitConst::MIN}, "0"},        {{DataType::BOOLEAN, sem::LimitConst::MAX}, "0"},
    {{DataType::INT8, sem::LimitConst::MIN}, "SCHAR_MIN"},   {{DataType::INT8, sem::LimitConst::MAX}, "SCHAR_MAX"},
    {{DataType::INT16, sem::LimitConst::MIN}, "SHRT_MIN"},   {{DataType::INT16, sem::LimitConst::MAX}, "SHRT_MAX"},
    {{DataType::INT32, sem::LimitConst::MIN}, "INT_MIN"},    {{DataType::INT32, sem::LimitConst::MAX}, "INT_MAX"},
    {{DataType::INT64, sem::LimitConst::MIN}, "LONG_MIN"},   {{DataType::INT64, sem::LimitConst::MAX}, "LONG_MAX"},
    {{DataType::UINT8, sem::LimitConst::MIN}, "0"},          {{DataType::UINT8, sem::LimitConst::MAX}, "UCHAR_MAX"},
    {{DataType::UINT16, sem::LimitConst::MIN}, "0"},         {{DataType::UINT16, sem::LimitConst::MAX}, "USHRT_MAX"},
    {{DataType::UINT32, sem::LimitConst::MIN}, "0"},         {{DataType::UINT32, sem::LimitConst::MAX}, "UINT_MAX"},
    {{DataType::UINT64, sem::LimitConst::MIN}, "0"},         {{DataType::UINT64, sem::LimitConst::MAX}, "ULONG_MAX"},
    {{DataType::FLOAT32, sem::LimitConst::MIN}, "-FLT_MAX"}, {{DataType::FLOAT32, sem::LimitConst::MAX}, "FLT_MAX"},
    {{DataType::FLOAT64, sem::LimitConst::MIN}, "-DBL_MAX"}, {{DataType::FLOAT64, sem::LimitConst::MAX}, "DBL_MAX"},
};

static std::map<std::string, std::string> FuncNameMap = {{"exp", "_EXP"},   {"log", "_LOG"},    {"sqrt", "_SQRT"},
                                                         {"pow", "_POW"},   {"sin", "_SIN"},    {"cos", "_COS"},
                                                         {"tanh", "_TANH"}, {"round", "_ROUND"}};

std::string Emit::to_string(const sem::LValPtr& v) {
  // TODO move this function to semtree
  auto lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(v);
  if (lookup_lval) return to_string(*lookup_lval.get());
  auto subscript_lval = std::dynamic_pointer_cast<sem::SubscriptLVal>(v);
  if (subscript_lval) return to_string(*subscript_lval.get());
  throw std::runtime_error("Not Supported LValPtr");
}

std::string Emit::to_string(const sem::ExprPtr& e) {
  // TODO move this function to semtree
  auto int_const = std::dynamic_pointer_cast<sem::IntConst>(e);
  if (int_const) return to_string(*int_const.get());
  auto unary_expr = std::dynamic_pointer_cast<sem::UnaryExpr>(e);
  if (unary_expr) return to_string(*unary_expr.get());
  auto binary_expr = std::dynamic_pointer_cast<sem::BinaryExpr>(e);
  if (binary_expr) return to_string(*binary_expr.get());
  auto cond_expr = std::dynamic_pointer_cast<sem::CondExpr>(e);
  if (cond_expr) return to_string(*cond_expr.get());
  auto select_expr = std::dynamic_pointer_cast<sem::SelectExpr>(e);
  if (select_expr) return to_string(*select_expr.get());
  auto clamp_expr = std::dynamic_pointer_cast<sem::ClampExpr>(e);
  if (clamp_expr) return to_string(*clamp_expr.get());
  auto cast_expr = std::dynamic_pointer_cast<sem::CastExpr>(e);
  if (cast_expr) return to_string(*cast_expr.get());
  auto limit_const = std::dynamic_pointer_cast<sem::LimitConst>(e);
  if (limit_const) return to_string(*limit_const.get());
  auto load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(e);
  if (load_expr) return to_string(*load_expr.get());
  auto index_expr = std::dynamic_pointer_cast<sem::IndexExpr>(e);
  if (index_expr) return to_string(*index_expr.get());
  throw std::runtime_error("Not Supported ExprPtr");
}

std::string Emit::to_string(const sem::IntConst& n) { return std::to_string(n.value); }
std::string Emit::to_string(const sem::LookupLVal& n) { return n.name; }
std::string Emit::to_string(const sem::SubscriptLVal& n) {
  return "(" + to_string(n.ptr) + " " + to_string(n.offset) + ")";
}
std::string Emit::to_string(const sem::LoadExpr& n) { return to_string(n.inner); }
std::string Emit::to_string(const sem::UnaryExpr& n) { return "(" + n.op + " " + to_string(n.inner) + ")"; }
std::string Emit::to_string(const sem::BinaryExpr& n) { return "(" + to_string(n.lhs) + " " + to_string(n.rhs) + ")"; }
std::string Emit::to_string(const sem::CondExpr& n) {
  return "(" + to_string(n.tcase) + " " + to_string(n.fcase) + " " + to_string(n.cond) + ")";
}
std::string Emit::to_string(const sem::SelectExpr& n) {
  return "(" + to_string(n.tcase) + " " + to_string(n.fcase) + " " + to_string(n.cond) + ")";
}
std::string Emit::to_string(const sem::ClampExpr& n) {
  return "(" + to_string(n.val) + " " + to_string(n.min) + " " + to_string(n.max) + ")";
}
std::string Emit::to_string(const sem::CastExpr& n) { return to_string(n.val); }
std::string Emit::to_string(const sem::LimitConst& n) {
  if (n.which == sem::LimitConst::ZERO) {
    return "0";
  } else if (n.which == sem::LimitConst::ONE) {
    return "1";
  }
  auto it = LimitConstLookup.find(std::make_pair(n.type, n.which));
  if (it == LimitConstLookup.end()) {
    throw std::runtime_error("Invalid type in LimitConst");
  }
  return it->second;
}
std::string Emit::to_string(const sem::IndexExpr& n) {
  switch (n.type) {
    case sem::IndexExpr::GLOBAL:
      if (one_thread_mode) {
        return "_i" + std::to_string(n.dim);
      }
      return "(cm_local_size(" + std::to_string(n.dim) + ")" + " * cm_group_id(" + std::to_string(n.dim) + ")" +
             " + cm_local_id(" + std::to_string(n.dim) + "))";
    case sem::IndexExpr::GROUP:
      return "cm_group_id(" + std::to_string(n.dim) + ")";
    case sem::IndexExpr::LOCAL:
      if (one_thread_mode) {
        return "_i" + std::to_string(n.dim);
      }
      if (use_global_id) {
        return "cm_local_id(" + std::to_string(n.dim) + ")";
      } else {
        return vector_size + " * cm_local_id(" + std::to_string(n.dim) + ")";
      }
    default:
      throw std::runtime_error("Invalid IndexExpr type");
  }
}

void Emit::Visit(const sem::IntConst& n) { emit(std::to_string(n.value)); }

void Emit::Visit(const sem::LookupLVal& n) { emit(n.name); }

void Emit::Visit(const sem::SubscriptLVal& n) {
  if (in_write_statement) {
    n.ptr->Accept(*this);
    emit(", sizeof(");
    emitType(write_type);
    emit(") * ");
    n.offset->Accept(*this);
    return;
  }

  auto s = GetGlobalVarWithOffset(n);
  if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
    emit(input_replace_map[s]);
    return;
  }

  if (is_sub_group_broadcast_first_val) {
    is_sub_group_broadcast_first_val = false;
    n.ptr->Accept(*this);
    emit("(");
    n.offset->Accept(*this);
    auto is_lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(n.ptr);
    if (large_sparse_vactor.find(is_lookup_lval->name) == large_sparse_vactor.end()) {
      emit(" * ");
      emit(vector_size);
    }
    return;
  }
  n.ptr->Accept(*this);
  auto is_lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(n.ptr);
  if (is_lookup_lval && !in_read_statement &&
      independent_vector.find(is_lookup_lval->name) != independent_vector.end()) {
    emit("(");
    n.offset->Accept(*this);
    if (large_sparse_vactor.find(is_lookup_lval->name) == large_sparse_vactor.end()) {
      emit(" * ");
      emit(vector_size);
    }
    emit(")");
    return;
  }

  if (in_read_statement || vector_stride_map.find(GetLValueName(n.ptr)) == vector_stride_map.end() ||
      vector_stride_map[GetLValueName(n.ptr)] >= 1) {
    emit(".select<");
    emit(vector_size);
    emit(", 1>");
  }
  emit("(");
  n.offset->Accept(*this);
  if (large_sparse_vactor.find(is_lookup_lval->name) == large_sparse_vactor.end()) {
    emit(" * ");
    emit(vector_size);
  }
  emit(")");
}
void Emit::Visit(const sem::UnaryExpr& n) {
  emit("(");
  emit(n.op);
  n.inner->Accept(*this);
  emit(")");
}

void Emit::Visit(const sem::LimitConst& n) {
  if (n.which == sem::LimitConst::ZERO) {
    emit("0");
    return;
  } else if (n.which == sem::LimitConst::ONE) {
    emit("1");
    return;
  }
  auto it = LimitConstLookup.find(std::make_pair(n.type, n.which));
  if (it == LimitConstLookup.end()) {
    throw std::runtime_error("Invalid type in LimitConst");
  }
  emit(it->second);
}

void Emit::Visit(const sem::IfStmt& n) {
  emitTab();
  if (n.iftrue && n.iffalse) {
    emit("if (");
    n.cond->Accept(*this);
    emit(")\n");
    n.iftrue->Accept(*this);
    emitTab();
    emit("else\n");
    n.iffalse->Accept(*this);
  } else if (n.iftrue) {
    emit("if (");
    n.cond->Accept(*this);
    emit(")\n");
    n.iftrue->Accept(*this);
  } else if (n.iffalse) {
    // This code is required since it is possible for n.iftrue to be a nullptr.
    // It needs to stay in place because its possible for verbose logging to
    // print
    // pre-simplified code; this would cause a null pointer to be dereferencd
    // and hence a crash.
    emit("if !(");
    n.cond->Accept(*this);
    emit(")\n");
    n.iffalse->Accept(*this);
  }
}

void Emit::Visit(const sem::WhileStmt& n) {
  emitTab();
  emit("while (");
  n.cond->Accept(*this);
  emit(")\n");
  n.inner->Accept(*this);
}

void Emit::Visit(const sem::ReturnStmt& n) {
  emitTab();
  emit("return");
  if (n.value) {
    emit(" (");
    n.value->Accept(*this);
    emit(")");
  }
  emit(";\n");
}

void Emit::Visit(const sem::SpecialStmt& n) {
  emitTab();
  emit(n.name);
  emit("(");
  for (size_t i = 0; i < n.params.size(); i++) {
    n.params[i]->Accept(*this);
    if (i != n.params.size() - 1) {
      emit(", ");
    }
  }
  emit(");\n");
}

void Emit::Visit(const sem::LoadExpr& n) {
  auto ty = TypeOf(n.inner);
  auto s = GetGlobalVarWithOffset(n.inner);
  if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
    emit(input_replace_map[s]);
    return;
  }
  auto inner = std::dynamic_pointer_cast<sem::SubscriptLVal>(n.inner);
  if (inner && GetGlobalVarWithOffset(*inner).size() > 0) {
    if (!use_global_id) {
      int stride = GetLocalIndexStride(inner->offset);
      if (stride > 1) {
        inner->ptr->Accept(*this);
        emit(", ");
        inner->offset->Accept(*this);
        emit(", ");
        emit("element_offset_");
        emit(std::to_string(stride));
        return;
      }
    }
    inner->ptr->Accept(*this);
    emit(", sizeof(");
    emitType(ty);
    emit(") * ");
    inner->offset->Accept(*this);
  } else {
    n.inner->Accept(*this);
  }
}

void Emit::assign_global_var_to_temp(const sem::ExprPtr& e) {
  auto result_map = GetGlobalLoadExprMap(e);
  for (auto result : result_map) {
    emit("//" + result.second + "\n");
    std::string temp_val = "cm_temp" + std::to_string(temp_var_num);
    temp_var_num++;
    auto type = TypeOf(result.first->inner);
    emitVector(type, vector_size, temp_val);
    emit(";\n");

    auto s = GetGlobalVarWithOffset(result.first->inner);
    if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
      emitTab();
      emit(temp_val);
      emit(" = ");
      emit(input_replace_map[s]);
      emit(";\n");
    } else {
      emitTab();
      emit("_read(");
      in_read_statement = true;
      result.first->Accept(*this);
      emit(", ");
      emit(temp_val);
      emit(");\n");
      in_read_statement = false;
      input_replace_map[result.second] = temp_val;
    }
  }
}

std::string c_dtype(const DataType& dt) {
  std::string base;
  switch (dt) {
    case DataType::BOOLEAN:
      base = "bool";
      break;
    case DataType::INT8:
      base = "char";
      break;
    case DataType::INT16:
      base = "short";
      break;
    case DataType::INT32:
      base = "int";
      break;
    case DataType::INT64:
      base = "long";
      break;
    case DataType::UINT8:
      base = "uchar";
      break;
    case DataType::UINT16:
      base = "ushort";
      break;
    case DataType::UINT32:
      base = "uint";
      break;
    case DataType::UINT64:
      base = "ulong";
      break;
    case DataType::FLOAT16:
      base = "half";
      break;
    case DataType::FLOAT32:
      base = "float";
      break;
    case DataType::FLOAT64:
      base = "double";
      break;
    default:
      throw std::runtime_error("Invalid tile type");
  }
  return base;
}

void Emit::SingleElementWrite(sem::LValPtr lhs, sem::ExprPtr rhs) {
  emitTab();
  auto ty_lhs = TypeOf(lhs);
  switch (ty_lhs.dtype) {
    case DataType::INT8:
      emit("write_single_char(");
      break;
    case DataType::INT32:
      emit("write_single_atomic_uint(");
      break;
    case DataType::UINT8:
      emit("write_single_char(");
      break;
    case DataType::INT16:
    case DataType::UINT16:
      emit("write_single_short(");
      break;
    case DataType::FLOAT16:
      emit("write_single_half(");
      break;
    case DataType::INT64:
      emit("write_single_atomic_long(");
      break;
    case DataType::UINT32:
      emit("write_single_atomic_uint(");
      break;
    case DataType::FLOAT32:
      emit("write_single_atomic_float(");
      break;
    default:
      throw std::runtime_error("CM kernels currently not support datatype:" + c_dtype(ty_lhs.dtype));
  }
  in_write_statement = true;
  write_type = ty_lhs;
  lhs->Accept(*this);
  in_write_statement = false;
  emit(", ");
  rhs->Accept(*this);
  if (IsVector(rhs)) {
    emit("(0)");
  }
  emit(");\n");
}

std::string Emit::GetLValueName(const sem::LValPtr& lv) {
  auto lookup = std::dynamic_pointer_cast<sem::LookupLVal>(lv);
  if (lookup) {
    return lookup->name;
  }

  auto subscript = std::dynamic_pointer_cast<sem::SubscriptLVal>(lv);
  if (subscript) {
    return GetLValueName(subscript->ptr);
  }

  throw error::Unimplemented{"GetLValueName: Not Supported LValue"};
}

void Emit::Visit(const sem::StoreStmt& n) {
  auto ty_lhs = TypeOf(n.lhs);

  auto is_lhs_global = GetGlobalVarWithOffset(n.lhs).size() > 0;
  auto rhs_load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.rhs);
  bool is_rhs_global = (rhs_load_exp != nullptr) && GetGlobalVarWithOffset(rhs_load_exp->inner).size() > 0;

  auto rhs_int_const = std::dynamic_pointer_cast<sem::IntConst>(n.rhs);
  if (is_lhs_global && rhs_int_const) {
    emitTab();
    emit("write_single_atomic_uint(");
    in_write_statement = true;
    write_type = ty_lhs;
    n.lhs->Accept(*this);
    in_write_statement = false;
    emit(", ");
    n.rhs->Accept(*this);
    emit(");\n");
    return;
  }

  if (is_lhs_global && is_rhs_global) {
    assign_global_var_to_temp(n.rhs);

    if (use_global_id) {
      SingleElementWrite(n.lhs, n.rhs);
    } else {
      emitTab();
      emit("_write(");
      in_write_statement = true;
      write_type = ty_lhs;
      n.lhs->Accept(*this);
      in_write_statement = false;
      emit(", ");
      n.rhs->Accept(*this);
      emit(");\n");
    }
  }

  if (is_lhs_global && !is_rhs_global) {
    auto p = std::dynamic_pointer_cast<sem::SubscriptLVal>(n.lhs);
    if (p == nullptr) throw std::runtime_error("StoreStmt lhs is not SubscriptLVal!");

    auto cond_expr = std::dynamic_pointer_cast<sem::CondExpr>(n.rhs);
    if (cond_expr) {
      assign_global_var_to_temp(n.rhs);

      std::string temp_var = "cm_temp" + std::to_string(temp_var_num);
      temp_var_num++;
      emitVector(ty_lhs, vector_size, temp_var);
      emit(";\n");

      emitTab();
      emit(temp_var);
      emit(".");
      n.rhs->Accept(*this);
      emit(";\n");

      emitTab();
      emit("_write(");
      in_write_statement = true;
      write_type = ty_lhs;
      n.lhs->Accept(*this);
      in_write_statement = false;
      emit(", ");
      emit(temp_var);
      emit(");\n");

      return;
    }

    auto binary_expr = std::dynamic_pointer_cast<sem::BinaryExpr>(n.rhs);
    if (binary_expr) {
      assign_global_var_to_temp(n.rhs);

      if (use_global_id) {
        SingleElementWrite(n.lhs, n.rhs);
      } else {
        emitTab();
        emit("_write(");
        in_write_statement = true;
        write_type = ty_lhs;
        n.lhs->Accept(*this);
        in_write_statement = false;
        emit(", ");
        n.rhs->Accept(*this);
        emit(");\n");
      }
      return;
    }

    if (use_global_id) {
      SingleElementWrite(n.lhs, n.rhs);
    } else {
      emitTab();
      emit("_write(");
      in_write_statement = true;
      write_type = ty_lhs;
      n.lhs->Accept(*this);
      in_write_statement = false;
      emit(", ");
      n.rhs->Accept(*this);
      emit(");\n");
    }
  }

  if (!is_lhs_global && is_rhs_global) {
    auto rhs_load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.rhs);
    if (rhs_load_exp) {
      emitTab();
      emit("_read(");
      in_read_statement = true;
      rhs_load_exp->Accept(*this);
      emit(", ");
      n.lhs->Accept(*this);
      emit(");\n");
      in_read_statement = false;

      auto stride = GetLocalIndexStride(rhs_load_exp);
      vector_stride_map[GetLValueName(n.lhs)] = stride;
    }
  }

  if (!is_lhs_global && !is_rhs_global) {
    emitTab();
    n.lhs->Accept(*this);
    auto cond_exp = std::dynamic_pointer_cast<sem::CondExpr>(n.rhs);
    auto select_exp = std::dynamic_pointer_cast<sem::SelectExpr>(n.rhs);
    if (cond_exp || select_exp) {
      emit(".");
      n.rhs->Accept(*this);
      emit(";\n");
      return;
    }

    emit(" = ");
    n.rhs->Accept(*this);
    emit(";\n");
  }
}

void Emit::Visit(const sem::DeclareStmt& n) {
  in_declare_stmt = true;
  sem::Type ty = n.type;
  sem::Type init_type;
  if (n.init) {
    init_type = TypeOf(n.init);
  }

  if (ty.base == sem::Type::VALUE) {
    if (ty.dtype == DataType::FLOAT16 && !cl_khr_fp16_) {
      // ty.dtype = DataType::FLOAT32;
    } else if (ty.dtype == DataType::BOOLEAN) {
      if (n.init) {
        ty.dtype = lang::Promote({init_type}).dtype;
        if (ty.dtype == DataType::BOOLEAN) {
          // If the initializer was booleans, make it INT8.
          ty.dtype = DataType::INT8;
        }
      } else {
        // Assume that this is being initialized from an inter-kernel
        // boolean tensor -- which, in cm, we represent as INT8.
        ty.dtype = DataType::INT8;
      }
    }
  }

  if (n.init) {
    if (ty.base == sem::Type::INDEX) {
      emit("// map=");

      for (auto r : local_index_stride_map) {
        emit(r.first);
        emit(": ");
        emit(std::to_string(r.second));
        emit("    ");
      }
      emit("\n");

      int stride = CalculateLocalIndexStride(n.init);
      local_index_stride_map[n.name] = stride;

      if (stride > 1) {
        std::string vname = "element_offset_" + std::to_string(stride);
        if (element_offset_vector.find(vname) == element_offset_vector.end()) {
          if (!use_global_id) {
            emitTab();
            emit("cm_vector(");
            emit(vname);
            emit(", uint, ");
            emit(vector_size);
            emit(", 0, ");
            emit(std::to_string(stride));
            emit(");\n");
            element_offset_vector.insert(vname);
          }
        }
      }
      emit("// use_global_id=");
      if (use_global_id) {
        emit("True ");
      } else {
        emit("False ");
      }
      emit("\n");
    }

    auto load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.init);
    if (load_exp) {
      if (!IsVector(load_exp->inner) && GetGlobalVarWithOffset(load_exp->inner).size() == 0) {
        emitTab();
        emitType(ty);
        emit(" ");
        emit(n.name);

        if (ty.base == sem::Type::INDEX) {
          if (depend_on_local_id(n.init)) {
            dependent_index.insert(n.name);
          }
        }
        if (n.init) {
          emit(" = ");
          n.init->Accept(*this);
        }
        emit(";\n");

        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      }
      emitVector(ty, vector_size, n.name);
      emit(";\n");

      if (GetGlobalVarWithOffset(load_exp->inner).size() > 0) {
        emitTab();
        emit("_read(");
        in_read_statement = true;
        load_exp->Accept(*this);
        emit(", ");
        emit(n.name);
        emit(");\n");
        in_read_statement = false;
      } else {
        emitTab();
        emit(n.name);
        emit(" = ");
        load_exp->Accept(*this);
        emit(";\n");
      }
      CheckValidType(ty);
      scope_->Bind(n.name, ty);
      in_declare_stmt = false;
      return;
    }

    auto cast_exp = std::dynamic_pointer_cast<sem::CastExpr>(n.init);
    if (cast_exp) {
      auto load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(cast_exp->val);
      if (load_exp) {
        emitVector(ty, vector_size, n.name);
        emit(" = ");
        load_exp->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      }
    }

    auto cond_exp = std::dynamic_pointer_cast<sem::CondExpr>(n.init);
    if (cond_exp) {
      if (IsVector(cond_exp->tcase) || IsVector(cond_exp->fcase) || IsVector(cond_exp->cond)) {
        emitVector(ty, vector_size, n.name);
        emit(";\n");

        emitTab();
        emit(n.name);
        emit(".");
        n.init->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      } else {
        emitTab();
        emitType(ty);
        emit(" ");
        emit(n.name);
        emit(".");
        n.init->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      }
    }

    auto select_exp = std::dynamic_pointer_cast<sem::SelectExpr>(n.init);
    if (select_exp) {
      if (IsVector(cond_exp->tcase) || IsVector(cond_exp->fcase) || IsVector(cond_exp->cond)) {
        emitVector(ty, vector_size, n.name);
        emit(";\n");

        emitTab();
        emit(n.name);
        emit(".");
        n.init->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      } else {
        emitTab();
        emitType(ty);
        emit(" ");
        emit(n.name);
        emit(".");
        n.init->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      }
    }

    auto binary_exp = std::dynamic_pointer_cast<sem::BinaryExpr>(n.init);
    if (binary_exp) {
      if (IsVector(binary_exp->lhs) || IsVector(binary_exp->rhs)) {
        if (binary_exp->op == ">" || binary_exp->op == "<" || binary_exp->op == ">=" || binary_exp->op == "<=" ||
            binary_exp->op == "==" || binary_exp->op == "!=") {
          emitVector("char", vector_size, n.name);
          ty.dtype = DataType::INT8;
          scope_->Bind(n.name, ty);
        } else {
          emitVector(ty, vector_size, n.name);
          scope_->Bind(n.name, ty);
        }

        emit(" = ");
        n.init->Accept(*this);
        emit(";\n");
        CheckValidType(ty);
        // scope_->Bind(n.name, ty);
        in_declare_stmt = false;
        return;
      }
    }

    auto call_expr = std::dynamic_pointer_cast<sem::CallExpr>(n.init);
    if (call_expr) {
      for (auto val : call_expr->vals) {
        if (IsVector(val)) {
          emitVector(ty, vector_size, n.name);
          emit(" = ");

          call_expr->Accept(*this);
          emit(";\n");
          CheckValidType(ty);
          scope_->Bind(n.name, ty);
          in_declare_stmt = false;
          return;
        }
      }
    }

    auto unary_expr = std::dynamic_pointer_cast<sem::UnaryExpr>(n.init);
    if (unary_expr && IsVector(unary_expr)) {
      emitVector(ty, vector_size, n.name);
      emit(" = ");

      unary_expr->Accept(*this);
      emit(";\n");
      CheckValidType(ty);
      scope_->Bind(n.name, ty);
      in_declare_stmt = false;
      return;
    }
  }

  if (n.type.array) {
    if (n.type.array >= 160) {
      large_sparse_vactor.insert(n.name);
      emitVector(ty, std::to_string(n.type.array), n.name);
      // throw std::runtime_error("cm vector exceeds maximum supported size");
    } else {
      emitVector(ty, std::to_string(n.type.array) + " * " + vector_size, n.name);
    }
    emit(" = ");
    if (n.init) {
      n.init->Accept(*this);
    } else {
      emit("0");
    }
    emit(";\n");

  } else {
    emitTab();
    emitType(ty);
    emit(" ");
    emit(n.name);

    if (ty.base == sem::Type::INDEX) {
      if (depend_on_local_id(n.init)) {
        dependent_index.insert(n.name);
      }
    }
    if (n.init) {
      emit(" = ");
      n.init->Accept(*this);
    }
    emit(";\n");
  }

  CheckValidType(ty);
  scope_->Bind(n.name, ty);
  in_declare_stmt = false;
}

void Emit::Visit(const sem::BinaryExpr& n) {
  auto ty_lhs = TypeOf(n.lhs);
  auto ty_rhs = TypeOf(n.rhs);
  auto ty = lang::Promote({ty_lhs, ty_rhs});
  emit("(");
  n.lhs->Accept(*this);  // added temporally
  emit(" ");
  emit(n.op);
  emit(" ");
  n.rhs->Accept(*this);
  emit(")");
}

void Emit::Visit(const sem::CondExpr& n) {
  auto type = TypeOf(n.cond);

  emit("merge(");
  n.tcase->Accept(*this);
  emit(", ");
  n.fcase->Accept(*this);
  emit(", ");
  if (type.dtype != DataType::INT8) {
    auto load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.cond);
    if (load_exp && IsVector(load_exp->inner)) {
      emit("vector<char,");
      emit(vector_size);
      emit(">(");
      n.cond->Accept(*this);
      emit(")");
    } else {
      n.cond->Accept(*this);
    }
  } else {
    n.cond->Accept(*this);
  }
  emit(")");
}

void Emit::Visit(const sem::SelectExpr& n) {
  auto type = TypeOf(n.cond);

  emit("merge(");
  n.tcase->Accept(*this);
  emit(", ");
  n.fcase->Accept(*this);
  emit(", ");
  if (type.dtype != DataType::INT8) {
    auto load_exp = std::dynamic_pointer_cast<sem::LoadExpr>(n.cond);
    if (load_exp && IsVector(load_exp->inner)) {
      emit("vector<char,");
      emit(vector_size);
      emit(">(");
      n.cond->Accept(*this);
      emit(")");
    } else {
      n.cond->Accept(*this);
    }
  } else {
    n.cond->Accept(*this);
  }
  emit(")");
}

void Emit::Visit(const sem::ClampExpr& n) {
  auto ty_val = TypeOf(n.val);
  auto ty_min = TypeOf(n.min);
  auto ty_max = TypeOf(n.max);

  // Align value dtypes and vector widths.
  sem::Type ty_clamp{sem::Type::VALUE};
  if (ty_val.base == sem::Type::VALUE) {
    ty_clamp.dtype = ty_val.dtype;
  } else {
    ty_clamp.dtype = DataType::INT32;
  }
  if (ty_min.vec_width != 1) {
    ty_clamp.vec_width = ty_min.vec_width;
  } else {
    ty_clamp.vec_width = ty_max.vec_width;
  }

  emit("_cmamp(");
  in_cmamp_function = true;
  auto load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(n.val);
  if (load_expr) {
    auto s = GetGlobalVarWithOffset(load_expr->inner);
    if (s.length() > 0 && input_replace_map.find(s) != input_replace_map.end()) {
      emit(input_replace_map[s]);

      auto subscript_lval = std::dynamic_pointer_cast<sem::SubscriptLVal>(load_expr->inner);
      if (subscript_lval) {
        emit("(_mod(");
        subscript_lval->offset->Accept(*this);
        emit(" , 4))");
      }

      emit(", ");
      n.min->Accept(*this);
      emit(", ");
      n.max->Accept(*this);
      emit(")");
      return;
    }
  }
  n.val->Accept(*this);
  emit(", ");
  n.min->Accept(*this);
  emit(", ");
  n.max->Accept(*this);
  emit(")");
  in_cmamp_function = false;
}

void Emit::Visit(const sem::CastExpr& n) { n.val->Accept(*this); }

void Emit::Visit(const sem::CallExpr& n) {
  if (n.name == "sub_group_broadcast") {
    is_sub_group_broadcast_first_val = true;
    n.vals[0]->Accept(*this);
    is_sub_group_broadcast_first_val = false;
    emit(" + ");
    n.vals[1]->Accept(*this);
    emit(")");
    return;
  }
  auto it = FuncNameMap.find(n.name);
  if (it != FuncNameMap.end()) {
    emit(it->second);
  } else {
    // Assume this is an cm function.
    // TODO: Enumerate the set of callable functions.
    emit(n.name);
  }
  emit("(");
  for (size_t i = 0; i < n.vals.size(); i++) {
    n.vals[i]->Accept(*this);
    if (i != n.vals.size() - 1) {
      emit(", ");
    }
  }
  emit(")");
}

void Emit::Visit(const sem::IndexExpr& n) {
  switch (n.type) {
    case sem::IndexExpr::GLOBAL:
      if (one_thread_mode) {
        emit("_i" + std::to_string(n.dim));
        return;
      }
      emit("(cm_local_size(" + std::to_string(n.dim) + ")");
      emit(" * cm_group_id(" + std::to_string(n.dim) + ")");
      emit(" + cm_local_id(" + std::to_string(n.dim) + "))");
      break;
    case sem::IndexExpr::GROUP:
      emit("cm_group_id(" + std::to_string(n.dim) + ")");
      break;
    case sem::IndexExpr::LOCAL:
      if (one_thread_mode) {
        emit("_i" + std::to_string(n.dim));
        return;
      }
      if (use_global_id) {
        emit("cm_local_id(" + std::to_string(n.dim) + ")");
      } else {
        emit(vector_size);
        emit(" * cm_local_id(" + std::to_string(n.dim) + ")");
      }
      break;
    default:
      throw std::runtime_error("Invalid IndexExpr type");
  }
}

void Emit::Visit(const sem::Block& n) {
  auto previous_scope = scope_;
  lang::Scope<sem::Type> scope{scope_};
  scope_ = &scope;
  EmitC::Visit(n);
  scope_ = previous_scope;
}

void Emit::Visit(const sem::ForStmt& n) {
  auto previous_scope = scope_;
  lang::Scope<sem::Type> scope{scope_};
  scope_ = &scope;
  scope.Bind(n.var, sem::Type{sem::Type::INDEX});
  EmitC::Visit(n);
  scope_ = previous_scope;
}

void Emit::Visit(const sem::BarrierStmt& n) {}

void Emit::Visit(const sem::Function& n) {
  emit("extern \"C\" _GENX_MAIN_ ");

  if (n.subgroup_size) {
    use_global_id = false;
    vector_size = std::to_string(n.subgroup_size);
  } else {
    use_global_id = true;
    vector_size = "4";
  }

  lang::Scope<sem::Type> scope;
  scope_ = &scope;

  one_thread_mode = false;
  for (const auto& p : n.params) {
    auto ty = p.first;
    if (ty.dtype == DataType::BOOLEAN) {
      // Global booleans are stored as INT8.
      ty.dtype = DataType::INT8;
    }
    if (ty.dtype == DataType::INT8 || ty.dtype == DataType::UINT8 || ty.dtype == DataType::INT16 ||
        ty.dtype == DataType::UINT16) {
      one_thread_mode = true;
    }
    CheckValidType(ty);
    scope.Bind(p.second, ty);
    global_params_.insert(p.second);
    vector_params_.insert(p.second);
  }

  emitType(n.ret);
  emit(" ");
  emit(n.name);
  emit("(");
  bool first_param = true;
  for (const auto& p : n.params) {
    if (first_param) {
      first_param = false;
    } else {
      emit(", ");
    }
    emit("SurfaceIndex");
    emit(" ");
    emit(p.second);
  }
  emit(")\n");

  if (one_thread_mode) {
    int lsize0 = ki_.gwork[0];
    int lsize1 = ki_.gwork[1];
    int lsize2 = ki_.gwork[2];
    emit("{\n");
    ++indent_;
    emitTab();
    emit("if(cm_local_id(0) == 0 && cm_group_id(0) == 0){\n");
    ++indent_;
    emitTab();
    emit("for(int _i0=0;_i0<" + std::to_string(lsize0) + ";_i0++){\n");
    ++indent_;
    emitTab();
    emit("for(int _i1=0;_i1<" + std::to_string(lsize1) + ";_i1++){\n");
    ++indent_;
    emitTab();
    emit("for(int _i2=0;_i2<" + std::to_string(lsize2) + ";_i2++){\n");
    n.body->Accept(*this);
    emitTab();
    emit("}\n");
    --indent_;
    emitTab();
    emit("}\n");
    --indent_;
    emitTab();
    emit("}\n");
    --indent_;
    emitTab();
    emit("}\n");
    --indent_;
    emitTab();
    emit("}\n");
  } else {
    n.body->Accept(*this);
  }
  scope_ = nullptr;
}

void Emit::CheckValidType(const sem::Type& ty) {
  if (cl_khr_fp64_) {
    return;
  }
  if (ty.base == sem::Type::TVOID || ty.base == sem::Type::INDEX) {
    return;
  }
  if (ty.dtype == DataType::FLOAT64) {
    throw error::Unimplemented{"The device does not support 64-bit floating-point types"};
  }
}

sem::Type Emit::TypeOf(const sem::ExprPtr& expr) { return lang::ExprType::TypeOf(scope_, cl_khr_fp16_, true, expr); }

sem::Type Emit::TypeOf(const sem::LValPtr& lvalue) {
  return lang::ExprType::TypeOf(scope_, cl_khr_fp16_, true, lvalue);
}

// type is sem::Type
void Emit::emitVector(const sem::Type& type, const std::string& size, const std::string& name) {
  emitTab();
  emit("vector<");
  emitType(type);
  emit(",");
  emit(size);
  emit("> ");
  emit(name);
  vector_params_.insert(name);
}

// type is std::string
void Emit::emitVector(const std::string& type, const std::string& size, const std::string& name) {
  emitTab();
  emit("vector<");
  emit(type);
  emit(",");
  emit(size);
  emit("> ");
  emit(name);
  vector_params_.insert(name);
}

bool Emit::depend_on_local_id(sem::ExprPtr init) {
  auto index_expr = std::dynamic_pointer_cast<sem::IndexExpr>(init);
  if (index_expr) {
    return (index_expr->type == sem::IndexExpr::GLOBAL || index_expr->type == sem::IndexExpr::LOCAL);
  }
  auto binary_expr = std::dynamic_pointer_cast<sem::BinaryExpr>(init);
  if (binary_expr) {
    if (!binary_expr->op.compare("/")) {
      auto index_expr = std::dynamic_pointer_cast<sem::IndexExpr>(binary_expr->lhs);
      auto int_const = std::dynamic_pointer_cast<sem::IntConst>(binary_expr->rhs);
      if (index_expr && int_const) {
        if ((index_expr->type == sem::IndexExpr::GLOBAL || index_expr->type == sem::IndexExpr::LOCAL) &&
            (int_const->value % 16) == 0)
          return false;
      }
    }
    return (depend_on_local_id(binary_expr->lhs) || depend_on_local_id(binary_expr->rhs));
  }
  auto load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(init);
  if (load_expr) {
    auto lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(load_expr->inner);
    if (lookup_lval && (dependent_index.find(lookup_lval->name) != dependent_index.end())) return true;
  }
  return false;
}

std::map<std::shared_ptr<sem::LoadExpr>, std::string> Emit::GetGlobalLoadExprMap(const sem::ExprPtr p) {
  std::map<std::shared_ptr<sem::LoadExpr>, std::string> result;

  auto is_cond_expr = std::dynamic_pointer_cast<sem::CondExpr>(p);
  if (is_cond_expr) {
    auto rcd = GetGlobalLoadExprMap(is_cond_expr->cond);
    result.insert(rcd.begin(), rcd.end());
    auto rt = GetGlobalLoadExprMap(is_cond_expr->tcase);
    result.insert(rt.begin(), rt.end());
    auto rf = GetGlobalLoadExprMap(is_cond_expr->fcase);
    result.insert(rf.begin(), rf.end());
    return result;
  }

  auto is_cast_expr = std::dynamic_pointer_cast<sem::CastExpr>(p);
  if (is_cast_expr) {
    return GetGlobalLoadExprMap(is_cast_expr->val);
  }

  auto is_binary_expr = std::dynamic_pointer_cast<sem::BinaryExpr>(p);
  if (is_binary_expr) {
    auto rl = GetGlobalLoadExprMap(is_binary_expr->lhs);
    result.insert(rl.begin(), rl.end());
    auto rr = GetGlobalLoadExprMap(is_binary_expr->rhs);
    result.insert(rr.begin(), rr.end());
    return result;
  }

  auto is_clamp_expr = std::dynamic_pointer_cast<sem::ClampExpr>(p);
  if (is_clamp_expr) {
    auto r_val = GetGlobalLoadExprMap(is_clamp_expr->val);
    result.insert(r_val.begin(), r_val.end());
    auto r_min = GetGlobalLoadExprMap(is_clamp_expr->min);
    result.insert(r_min.begin(), r_min.end());
    auto r_max = GetGlobalLoadExprMap(is_clamp_expr->max);
    result.insert(r_max.begin(), r_max.end());
    return result;
  }

  auto is_load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(p);
  if (is_load_expr) {
    auto subscript_lval = std::dynamic_pointer_cast<sem::SubscriptLVal>(is_load_expr->inner);
    if (subscript_lval) {
      auto s = GetGlobalVarWithOffset(subscript_lval);
      if (s.length() > 0) {
        result[is_load_expr] = s;
      }
      return result;
    }

    auto s = GetGlobalVarWithOffset(is_load_expr->inner);
    if (s.length() > 0) {
      result[is_load_expr] = s;
    }
  }

  return result;
}
std::string Emit::GetGlobalVarWithOffset(const sem::LValPtr p) {
  auto is_lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(p);
  if (is_lookup_lval) return GetGlobalVarWithOffset(*is_lookup_lval);
  auto is_subscript_lval = std::dynamic_pointer_cast<sem::SubscriptLVal>(p);
  if (is_subscript_lval) return GetGlobalVarWithOffset(*is_subscript_lval);
  return "";
}
std::string Emit::GetGlobalVarWithOffset(const sem::LookupLVal& v) {
  if (global_params_.find(v.name) != global_params_.end()) {
    return v.name;
  }
  return "";
}

std::string Emit::GetGlobalVarWithOffset(const sem::SubscriptLVal& v) {
  auto s = GetGlobalVarWithOffset(v.ptr);
  if (s.size() > 0) {
    return s + " " + to_string(v.offset);
  }
  return "";
}

bool Emit::IsVector(const sem::ExprPtr p) {
  auto is_cast_expr = std::dynamic_pointer_cast<sem::CastExpr>(p);
  if (is_cast_expr) {
    return IsVector(is_cast_expr->val);
  }

  auto is_load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(p);
  if (is_load_expr) {
    return IsVector(is_load_expr->inner);
  }

  auto is_unary_expr = std::dynamic_pointer_cast<sem::UnaryExpr>(p);
  if (is_unary_expr) {
    return IsVector(is_unary_expr->inner);
  }

  auto is_cond_expr = std::dynamic_pointer_cast<sem::CondExpr>(p);
  if (is_cond_expr) {
    return IsVector(is_cond_expr->tcase) || IsVector(is_cond_expr->fcase) || IsVector(is_cond_expr->cond);
  }

  auto is_select_expr = std::dynamic_pointer_cast<sem::SelectExpr>(p);
  if (is_select_expr) {
    return IsVector(is_select_expr->tcase) || IsVector(is_select_expr->fcase) || IsVector(is_select_expr->cond);
  }

  auto binary_expr = std::dynamic_pointer_cast<sem::BinaryExpr>(p);
  if (binary_expr) {
    return IsVector(binary_expr->lhs) || IsVector(binary_expr->rhs);
  }

  return false;
}

bool Emit::IsVector(const sem::LValPtr p) {
  auto is_lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(p);
  if (is_lookup_lval) return IsVector(*is_lookup_lval);
  auto is_subscript_lval = std::dynamic_pointer_cast<sem::SubscriptLVal>(p);
  if (is_subscript_lval) return IsVector(*is_subscript_lval);
  return false;
}

bool Emit::IsVector(const sem::LookupLVal& v) { return vector_params_.find(v.name) != vector_params_.end(); }
bool Emit::IsVector(const sem::SubscriptLVal& v) { return IsVector(v.ptr); }

int Emit::CalculateLocalIndexStride(const sem::ExprPtr p) {
  auto binary_expr = std::dynamic_pointer_cast<sem::BinaryExpr>(p);
  if (binary_expr) {
    if (!binary_expr->op.compare("/")) {
      auto index_expr = std::dynamic_pointer_cast<sem::IndexExpr>(binary_expr->lhs);
      auto int_const = std::dynamic_pointer_cast<sem::IntConst>(binary_expr->rhs);
      if (index_expr && index_expr->type == sem::IndexExpr::LOCAL && int_const && int_const->value == 1) return 1;
    }
    if (!binary_expr->op.compare("*")) {
      auto int_const = std::dynamic_pointer_cast<sem::IntConst>(binary_expr->lhs);
      auto load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(binary_expr->rhs);
      if (int_const && load_expr) {
        return int_const->value * CalculateLocalIndexStride(load_expr);
      }
    }
    return std::max(CalculateLocalIndexStride(binary_expr->lhs), CalculateLocalIndexStride(binary_expr->rhs));
  }
  auto is_load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(p);
  if (is_load_expr) {
    return GetLocalIndexStride(is_load_expr->inner);
  }

  return 0;
}

int Emit::GetLocalIndexStride(const sem::LValPtr p) {
  auto is_lookup_lval = std::dynamic_pointer_cast<sem::LookupLVal>(p);
  if (is_lookup_lval) {
    return local_index_stride_map[is_lookup_lval->name];
  }
  auto is_subscript_lval = std::dynamic_pointer_cast<sem::SubscriptLVal>(p);
  if (is_subscript_lval) {
    return GetLocalIndexStride(is_subscript_lval->offset);
  }

  return 0;
}

int Emit::GetLocalIndexStride(const sem::ExprPtr p) {
  auto binary_expr = std::dynamic_pointer_cast<sem::BinaryExpr>(p);
  if (binary_expr) {
    return std::max(CalculateLocalIndexStride(binary_expr->lhs), CalculateLocalIndexStride(binary_expr->rhs));
  }
  auto is_load_expr = std::dynamic_pointer_cast<sem::LoadExpr>(p);
  if (is_load_expr) {
    return GetLocalIndexStride(is_load_expr->inner);
  }
  return 0;
}

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
