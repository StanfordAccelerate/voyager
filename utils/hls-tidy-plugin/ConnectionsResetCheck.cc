#include "ConnectionsResetCheck.h"

#include <map>
#include <set>

#include "clang-tidy/ClangTidyCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace hls {

namespace {

bool is_connections_type(QualType qt) {
  if (qt.isNull()) return false;
  if (qt->isPointerType() || qt->isReferenceType()) {
    qt = qt->getPointeeType();
  }
  qt = qt.getUnqualifiedType();

  const CXXRecordDecl* record = qt->getAsCXXRecordDecl();
  if (!record) return false;

  std::string name = record->getQualifiedNameAsString();
  return name.find("Connections::Combinational") != std::string::npos ||
         name.find("Connections::In") != std::string::npos ||
         name.find("Connections::Out") != std::string::npos;
}

const ValueDecl* find_base_decl(const Expr* expr) {
  if (!expr) return nullptr;
  expr = expr->IgnoreParenImpCasts();

  if (const auto* dre = dyn_cast<DeclRefExpr>(expr)) {
    return dre->getDecl();
  }

  if (const auto* me = dyn_cast<MemberExpr>(expr)) {
    if (const auto* fd = dyn_cast<FieldDecl>(me->getMemberDecl())) {
      return fd;
    }
    return find_base_decl(me->getBase());
  }

  if (const auto* ase = dyn_cast<ArraySubscriptExpr>(expr)) {
    return find_base_decl(ase->getBase());
  }

  if (isa<CXXThisExpr>(expr)) {
    return nullptr;
  }

  return nullptr;
}

class ConnectionsVisitor : public RecursiveASTVisitor<ConnectionsVisitor> {
  ASTContext& context;
  std::map<const ValueDecl*, SourceLocation>& used_connections;
  std::set<const ValueDecl*>& reset_connections;
  const FunctionDecl* current_function;

 public:
  ConnectionsVisitor(ASTContext& ctx,
                     std::map<const ValueDecl*, SourceLocation>& used,
                     std::set<const ValueDecl*>& reset,
                     const FunctionDecl* func)
      : context(ctx),
        used_connections(used),
        reset_connections(reset),
        current_function(func) {}

  bool VisitCallExpr(CallExpr* call) {
    if (const auto* member_call = dyn_cast<CXXMemberCallExpr>(call)) {
      const CXXMethodDecl* method = member_call->getMethodDecl();
      if (method) {
        std::string method_name = method->getNameAsString();
        if (method_name == "Reset" || method_name == "ResetRead" ||
            method_name == "ResetWrite") {
          const Expr* object_expr = member_call->getImplicitObjectArgument();
          if (object_expr && is_connections_type(object_expr->getType())) {
            const ValueDecl* decl = find_base_decl(object_expr);
            if (decl) {
              reset_connections.insert(decl);
              return true;
            }
          }
        }
      }
    }

    if (const auto* member_call = dyn_cast<CXXMemberCallExpr>(call)) {
      const Expr* object_expr = member_call->getImplicitObjectArgument();
      if (object_expr && is_connections_type(object_expr->getType())) {
        const ValueDecl* decl = find_base_decl(object_expr);
        if (decl && used_connections.find(decl) == used_connections.end()) {
          used_connections[decl] = call->getBeginLoc();
        }
      }
    }

    for (const Expr* arg : call->arguments()) {
      if (!arg) continue;
      const Expr* arg_clean = arg->IgnoreParenImpCasts();
      if (is_connections_type(arg_clean->getType())) {
        const ValueDecl* decl = find_base_decl(arg_clean);
        if (decl && used_connections.find(decl) == used_connections.end()) {
          used_connections[decl] = arg->getBeginLoc();
        }
      }
    }

    return true;
  }
};

}  // end anonymous namespace

ConnectionsResetCheck::ConnectionsResetCheck(StringRef name,
                                             ClangTidyContext* context)
    : ClangTidyCheck(name, context) {}

void ConnectionsResetCheck::registerMatchers(MatchFinder* finder) {
  finder->addMatcher(
      functionDecl(isDefinition(), hasDescendant(callExpr(
                                       callee(functionDecl(hasName("wait"))))))
          .bind("func_with_wait"),
      this);
}

void ConnectionsResetCheck::check(const MatchFinder::MatchResult& result) {
  const auto* func = result.Nodes.getNodeAs<FunctionDecl>("func_with_wait");
  if (!func || !func->hasBody()) return;

  ASTContext* context = result.Context;
  const Stmt* body = func->getBody();

  std::map<const ValueDecl*, SourceLocation> used_connections;
  std::set<const ValueDecl*> reset_connections;

  ConnectionsVisitor visitor(*context, used_connections, reset_connections,
                             func);
  visitor.TraverseStmt(const_cast<Stmt*>(body));

  for (const auto& pair : used_connections) {
    const ValueDecl* decl = pair.first;
    const SourceLocation loc = pair.second;

    if (reset_connections.find(decl) == reset_connections.end()) {
      diag(loc,
           "connection '%0' is used in function '%1' with wait(), but is not "
           "reset with Reset/ResetRead/ResetWrite")
          << decl->getNameAsString() << func->getNameAsString();

      diag(decl->getLocation(), "declaration of '%0' here", DiagnosticIDs::Note)
          << decl->getNameAsString();
    }
  }
}

}  // namespace hls
}  // namespace tidy
}  // namespace clang
