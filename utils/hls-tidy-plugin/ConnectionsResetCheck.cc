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

bool isConnectionsType(QualType QT) {
  if (QT.isNull()) return false;
  if (QT->isPointerType() || QT->isReferenceType()) {
    QT = QT->getPointeeType();
  }
  QT = QT.getUnqualifiedType();

  const CXXRecordDecl *Record = QT->getAsCXXRecordDecl();
  if (!Record) return false;

  std::string Name = Record->getQualifiedNameAsString();
  return Name.find("Connections::Combinational") != std::string::npos ||
         Name.find("Connections::In") != std::string::npos ||
         Name.find("Connections::Out") != std::string::npos;
}

const ValueDecl *findBaseDecl(const Expr *E) {
  if (!E) return nullptr;
  E = E->IgnoreParenImpCasts();

  if (const auto *DRE = dyn_cast<DeclRefExpr>(E)) {
    return DRE->getDecl();
  }

  if (const auto *ME = dyn_cast<MemberExpr>(E)) {
    if (const auto *FD = dyn_cast<FieldDecl>(ME->getMemberDecl())) {
      return FD;
    }
    return findBaseDecl(ME->getBase());
  }

  if (const auto *ASE = dyn_cast<ArraySubscriptExpr>(E)) {
    return findBaseDecl(ASE->getBase());
  }

  if (isa<CXXThisExpr>(E)) {
    return nullptr;
  }

  return nullptr;
}

class ConnectionsVisitor : public RecursiveASTVisitor<ConnectionsVisitor> {
  ASTContext &Context;
  std::map<const ValueDecl *, SourceLocation> &UsedConnections;
  std::set<const ValueDecl *> &ResetConnections;
  const FunctionDecl *CurrentFunction;

 public:
  ConnectionsVisitor(ASTContext &Ctx,
                     std::map<const ValueDecl *, SourceLocation> &Used,
                     std::set<const ValueDecl *> &Reset,
                     const FunctionDecl *Func)
      : Context(Ctx),
        UsedConnections(Used),
        ResetConnections(Reset),
        CurrentFunction(Func) {}

  bool VisitCallExpr(CallExpr *Call) {
    if (const auto *MemberCall = dyn_cast<CXXMemberCallExpr>(Call)) {
      const CXXMethodDecl *Method = MemberCall->getMethodDecl();
      if (Method) {
        std::string MethodName = Method->getNameAsString();
        if (MethodName == "Reset" || MethodName == "ResetRead" ||
            MethodName == "ResetWrite") {
          const Expr *ObjectExpr = MemberCall->getImplicitObjectArgument();
          if (ObjectExpr && isConnectionsType(ObjectExpr->getType())) {
            const ValueDecl *Decl = findBaseDecl(ObjectExpr);
            if (Decl) {
              ResetConnections.insert(Decl);
              return true;
            }
          }
        }
      }
    }

    if (const auto *MemberCall = dyn_cast<CXXMemberCallExpr>(Call)) {
      const Expr *ObjectExpr = MemberCall->getImplicitObjectArgument();
      if (ObjectExpr && isConnectionsType(ObjectExpr->getType())) {
        const ValueDecl *Decl = findBaseDecl(ObjectExpr);
        if (Decl && UsedConnections.find(Decl) == UsedConnections.end()) {
          UsedConnections[Decl] = Call->getBeginLoc();
        }
      }
    }

    for (const Expr *Arg : Call->arguments()) {
      if (!Arg) continue;
      const Expr *ArgClean = Arg->IgnoreParenImpCasts();
      if (isConnectionsType(ArgClean->getType())) {
        const ValueDecl *Decl = findBaseDecl(ArgClean);
        if (Decl && UsedConnections.find(Decl) == UsedConnections.end()) {
          UsedConnections[Decl] = Arg->getBeginLoc();
        }
      }
    }

    return true;
  }
};

}  // end anonymous namespace

ConnectionsResetCheck::ConnectionsResetCheck(StringRef Name,
                                             ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void ConnectionsResetCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      functionDecl(isDefinition(), hasDescendant(callExpr(
                                       callee(functionDecl(hasName("wait"))))))
          .bind("funcWithWait"),
      this);
}

void ConnectionsResetCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("funcWithWait");
  if (!Func || !Func->hasBody()) return;

  ASTContext *Context = Result.Context;
  const Stmt *Body = Func->getBody();

  std::map<const ValueDecl *, SourceLocation> usedConnections;
  std::set<const ValueDecl *> resetConnections;

  ConnectionsVisitor Visitor(*Context, usedConnections, resetConnections, Func);
  Visitor.TraverseStmt(const_cast<Stmt *>(Body));

  for (const auto &Pair : usedConnections) {
    const ValueDecl *Decl = Pair.first;
    const SourceLocation Loc = Pair.second;

    if (resetConnections.find(Decl) == resetConnections.end()) {
      diag(Loc,
           "Connection '%0' is used in function '%1' with wait(), but is not "
           "reset with Reset/ResetRead/ResetWrite")
          << Decl->getNameAsString() << Func->getNameAsString();

      diag(Decl->getLocation(), "declaration of '%0' here", DiagnosticIDs::Note)
          << Decl->getNameAsString();
    }
  }
}

}  // namespace hls
}  // namespace tidy
}  // namespace clang