// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <isl/aff.h>
#include <isl/set.h>
#include <isl/val.h>
#include <isl/ast.h>
#include <isl/ast_build.h>
#include <isl/id.h>
#include <isl/space.h>
#include <algorithm>
#include <limits>
#include <vector>

namespace tilemega::analysis {
// Interpret ISL's integer expression, not an approximation of the relation.
// Unsupported expressions and overflow fall back to the original ISL evaluator.
struct OracleExpression {
  isl_ast_expr_type type=isl_ast_expr_error;
  isl_ast_expr_op_type op=isl_ast_expr_op_error;
  long value=0;
  std::vector<OracleExpression> args;
  bool valid=false;
  static OracleExpression Read(isl_ast_expr* expr,std::vector<std::string> const& names) {
    OracleExpression out;if(!expr)return out;
    out.type=isl_ast_expr_get_type(expr);
    if(out.type==isl_ast_expr_int) {
      auto* v=isl_ast_expr_get_val(expr);
      if(isl_val_is_int(v)==isl_bool_true) {
        char* s=isl_val_to_str(v);
        try {out.value=std::stol(s);out.valid=true;}catch(...){}
        free(s);
      }
      isl_val_free(v);
    } else if(out.type==isl_ast_expr_id) {
      auto* id=isl_ast_expr_get_id(expr);auto* name=isl_id_get_name(id);
      auto it=std::find(names.begin(),names.end(),name?name:"");
      out.valid=it!=names.end();out.value=it-names.begin();isl_id_free(id);
    } else if(out.type==isl_ast_expr_op) {
      out.op=isl_ast_expr_op_get_type(expr);
      out.valid=out.op>=isl_ast_expr_op_and && out.op<=isl_ast_expr_op_gt;
      for(int i=0;i<isl_ast_expr_op_get_n_arg(expr);++i) {
        out.args.push_back(Read(isl_ast_expr_op_get_arg(expr,i),names));
        out.valid&=out.args.back().valid;
      }
    }
    isl_ast_expr_free(expr);return out;
  }
  bool Eval(std::vector<long> const& parameters,long& result) const {
    if(!valid)return false;
    if(type==isl_ast_expr_int){result=value;return true;}
    if(type==isl_ast_expr_id){result=parameters.at(value);return true;}
    long a=0,b=0;if(args.empty() || !args[0].Eval(parameters,a))return false;
    if(op==isl_ast_expr_op_cond || op==isl_ast_expr_op_select)
      return args.at(a?1:2).Eval(parameters,result);
    if((op==isl_ast_expr_op_and || op==isl_ast_expr_op_and_then) && !a){result=0;return true;}
    if((op==isl_ast_expr_op_or || op==isl_ast_expr_op_or_else) && a){result=1;return true;}
    if(args.size()>1 && !args[1].Eval(parameters,b))return false;
    __int128 wide=0;
    switch(op) {
      case isl_ast_expr_op_and:case isl_ast_expr_op_and_then:result=bool(a)&&bool(b);return true;
      case isl_ast_expr_op_or:case isl_ast_expr_op_or_else:result=bool(a)||bool(b);return true;
      case isl_ast_expr_op_eq:result=a==b;return true;
      case isl_ast_expr_op_le:result=a<=b;return true;
      case isl_ast_expr_op_lt:result=a<b;return true;
      case isl_ast_expr_op_ge:result=a>=b;return true;
      case isl_ast_expr_op_gt:result=a>b;return true;
      case isl_ast_expr_op_min:case isl_ast_expr_op_max:
        result=a;for(std::size_t i=1;i<args.size();++i){if(!args[i].Eval(parameters,b))return false;
          result=op==isl_ast_expr_op_min?std::min(result,b):std::max(result,b);}return true;
      case isl_ast_expr_op_minus:wide=-__int128(a);break;
      case isl_ast_expr_op_add:wide=__int128(a)+b;break;
      case isl_ast_expr_op_sub:wide=__int128(a)-b;break;
      case isl_ast_expr_op_mul:wide=__int128(a)*b;break;
      case isl_ast_expr_op_div:case isl_ast_expr_op_pdiv_q:
        if(!b)return false;wide=__int128(a)/b;break;
      case isl_ast_expr_op_fdiv_q:
        if(!b)return false;wide=__int128(a)/b;if(__int128(a)%b && ((a<0)!=(b<0)))--wide;break;
      case isl_ast_expr_op_pdiv_r:case isl_ast_expr_op_zdiv_r:
        if(!b)return false;wide=__int128(a)%b;break;
      default:return false;
    }
    if(wide<std::numeric_limits<long>::min() || wide>std::numeric_limits<long>::max())return false;
    result=long(wide);return true;
  }
};
struct OracleProgram {
  OracleExpression domain;
  std::vector<OracleExpression> values;
  // All source coordinates are parameters by this point. Keep the exact
  // domain predicate, including holes and parameter-dependent empty fibers.
  static OracleProgram Build(isl_set* domain,std::vector<isl_pw_aff*> const& expressions) {
    domain=isl_set_params(domain);
    OracleProgram out;std::vector<std::string> names;
    for(int i=0;i<isl_set_dim(domain,isl_dim_param);++i) {
      auto* name=isl_set_get_dim_name(domain,isl_dim_param,i);names.emplace_back(name?name:"");
    }
    auto* build=isl_ast_build_from_context(isl_set_universe(isl_set_get_space(domain)));
    out.domain=OracleExpression::Read(isl_ast_build_expr_from_set(build,domain),names);
    for(auto* expression:expressions)
      out.values.push_back(OracleExpression::Read(isl_ast_build_expr_from_pw_aff(build,
          isl_pw_aff_project_domain_on_params(isl_pw_aff_copy(expression))),names));
    isl_ast_build_free(build);return out;
  }
  bool Eval(std::vector<long> const& parameters,bool& nonempty,std::vector<long>& output) const {
    long guard=0;if(!domain.Eval(parameters,guard))return false;
    nonempty=guard!=0;output.clear();if(!nonempty)return true;
    for(auto const& expression:values){long value;if(!expression.Eval(parameters,value))return false;output.push_back(value);}
    return true;
  }
};
}
