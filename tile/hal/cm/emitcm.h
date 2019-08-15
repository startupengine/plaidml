// Copyright 2017-2018 Intel Corporation.

#pragma once

#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>

#include "tile/lang/emitc.h"
#include "tile/lang/scope.h"
#include "tile/lang/semprinter.h"

// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cm/compiler.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace cm {

class Emit : public lang::EmitC {
 public:
  explicit Emit(bool cl_khr_fp16, bool cl_khr_fp64, lang::KernelInfo ki)
      : cl_khr_fp16_{cl_khr_fp16}, cl_khr_fp64_{cl_khr_fp64}, scope_{nullptr}, ki_{ki} {}

  void Visit(const sem::IntConst&) final;
  void Visit(const sem::LookupLVal&) final;
  void Visit(const sem::SubscriptLVal&) final;
  void Visit(const sem::UnaryExpr&) final;
  void Visit(const sem::LimitConst&) final;
  void Visit(const sem::IfStmt&) final;
  void Visit(const sem::WhileStmt&) final;
  void Visit(const sem::ReturnStmt&) final;
  void Visit(const sem::SpecialStmt&) final;

  void Visit(const sem::LoadExpr&) final;
  void Visit(const sem::StoreStmt&) final;
  void Visit(const sem::DeclareStmt&) final;
  void Visit(const sem::BinaryExpr&) final;
  void Visit(const sem::CondExpr& n) final;
  void Visit(const sem::SelectExpr& n) final;
  void Visit(const sem::ClampExpr& n) final;
  void Visit(const sem::CastExpr&) final;
  void Visit(const sem::CallExpr&) final;
  void Visit(const sem::IndexExpr&) final;
  void Visit(const sem::Block&) final;
  void Visit(const sem::ForStmt&) final;
  void Visit(const sem::BarrierStmt&) final;
  void Visit(const sem::Function&) final;

  static size_t vsize;
  bool use_group_id = false;
  bool use_local_id = false;
  bool use_global_id = true;
  std::string vector_size = "16";

  bool comments_contains_ident = false;
  bool one_thread_mode = false;

 private:
  std::string to_string(const sem::LValPtr& v);
  std::string to_string(const sem::ExprPtr& e);
  std::string to_string(const sem::IntConst& n);
  std::string to_string(const sem::LookupLVal& n);
  std::string to_string(const sem::SubscriptLVal& n);
  std::string to_string(const sem::LoadExpr& n);
  std::string to_string(const sem::UnaryExpr& n);
  std::string to_string(const sem::BinaryExpr& n);
  std::string to_string(const sem::CondExpr& n);
  std::string to_string(const sem::SelectExpr& n);
  std::string to_string(const sem::ClampExpr& n);
  std::string to_string(const sem::CastExpr& n);
  std::string to_string(const sem::LimitConst& n);
  std::string to_string(const sem::IndexExpr& n);

  void CheckValidType(const sem::Type& ty);
  sem::Type TypeOf(const sem::ExprPtr& expr);
  sem::Type TypeOf(const sem::LValPtr& lvalue);

  void emitVector(const sem::Type& type, const std::string& size, const std::string& name);
  void emitVector(const std::string& type, const std::string& size, const std::string& name);
  std::set<std::string> global_params_;
  std::set<std::string> vector_params_;

  std::map<std::shared_ptr<sem::LoadExpr>, std::string> GetGlobalLoadExprMap(const sem::ExprPtr p);
  void SingleElementWrite(sem::LValPtr lhs, sem::ExprPtr rhs);
  void assign_global_var_to_temp(const sem::ExprPtr& e);

  std::string GetGlobalVarWithOffset(const sem::LValPtr p);
  std::string GetGlobalVarWithOffset(const sem::LookupLVal& v);
  std::string GetGlobalVarWithOffset(const sem::SubscriptLVal& v);

  std::string GetGlobalVar(const sem::LValPtr p);
  std::string GetGlobalVar(const sem::LookupLVal& v);
  std::string GetGlobalVar(const sem::SubscriptLVal& v);

  std::string GetLValueName(const sem::LValPtr& lv);
  std::map<std::string, int> vector_stride_map;

  int CalculateLocalIndexStride(const sem::ExprPtr p);
  int GetLocalIndexStride(const sem::LValPtr p);
  int GetLocalIndexStride(const sem::ExprPtr p);
  std::map<std::string, int> local_index_stride_map;
  std::set<std::string> element_offset_vector;

  bool IsVector(const sem::ExprPtr p);
  bool IsVector(const sem::LValPtr p);
  bool IsVector(const sem::LookupLVal& v);
  bool IsVector(const sem::SubscriptLVal& v);
  bool depend_on_local_id(sem::ExprPtr init);
  bool cl_khr_fp16_;
  bool cl_khr_fp64_;
  lang::Scope<sem::Type>* scope_;
  bool is_sub_group_broadcast_first_val = false;
  bool in_read_statement = false;
  bool in_write_statement = false;
  bool in_cmamp_function = false;
  sem::Type write_type;
  bool in_declare_stmt = false;
  int temp_var_num = 0;
  std::map<std::string, std::string> input_replace_map;
  std::set<std::string> dependent_index;
  std::set<std::string> independent_vector;
  std::set<std::string> large_sparse_vactor;
  lang::KernelInfo ki_;
};

}  // namespace cm
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
