#include "NestedUnrollCheck.h"

#include <iostream>

#include "clang-tidy/ClangTidyCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace hls {

namespace {

bool hasUnrollPragma(SourceLocation Loc, const SourceManager &SM) {
  if (Loc.isMacroID()) Loc = SM.getExpansionLoc(Loc);

  FileID FID = SM.getFileID(Loc);
  unsigned Line = SM.getSpellingLineNumber(Loc);

  // Only look at the line immediately before
  if (Line == 1) return false;

  SourceLocation PrevLineStart = SM.translateLineCol(FID, Line - 1, 1);
  if (PrevLineStart.isInvalid()) return false;

  SourceLocation LoopStart =
      Lexer::getLocForEndOfToken(PrevLineStart, 0, SM, LangOptions());

  StringRef LineText = Lexer::getSourceText(
      CharSourceRange::getCharRange(PrevLineStart, Loc), SM, LangOptions(), 0);

  // Strip whitespace
  LineText = LineText.trim();

  return LineText.starts_with("#pragma hls_unroll yes");
}

class UnrollVisitor : public RecursiveASTVisitor<UnrollVisitor> {
 public:
  UnrollVisitor(ASTContext &Context, ClangTidyCheck *Check)
      : Context(Context), Check(Check) {}

  bool TraverseStmt(Stmt *S) {
    if (!S) return true;

    if (auto *FS = dyn_cast<ForStmt>(S)) {
      bool isUnrolled =
          hasUnrollPragma(FS->getBeginLoc(), Context.getSourceManager());
      bool wasInsideUnrolled = insideUnrolled;

      if (wasInsideUnrolled && !isUnrolled) {
        Check->diag(FS->getBeginLoc(),
                    "Loop is nested inside a '#pragma hls_unroll yes' loop but "
                    "is missing its own unroll pragma");
      }

      insideUnrolled = isUnrolled || insideUnrolled;
      RecursiveASTVisitor::TraverseStmt(S);
      insideUnrolled = wasInsideUnrolled;
      return true;
    }

    return RecursiveASTVisitor::TraverseStmt(S);
  }

 private:
  ASTContext &Context;
  ClangTidyCheck *Check;
  bool insideUnrolled = false;
};

}  // namespace

HLSUnrollNestedCheck::HLSUnrollNestedCheck(StringRef Name,
                                           ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void HLSUnrollNestedCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(functionDecl(isDefinition()).bind("func"), this);
}

void HLSUnrollNestedCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("func");
  if (!Func || !Func->hasBody()) return;

  UnrollVisitor Visitor(*Result.Context, this);
  Visitor.TraverseDecl(const_cast<FunctionDecl *>(Func));
}

}  // namespace hls
}  // namespace tidy
}  // namespace clang
