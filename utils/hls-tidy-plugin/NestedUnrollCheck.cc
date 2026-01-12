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

bool has_unroll_pragma(SourceLocation loc, const SourceManager& sm) {
  if (loc.isMacroID()) loc = sm.getExpansionLoc(loc);

  FileID fid = sm.getFileID(loc);
  unsigned line = sm.getSpellingLineNumber(loc);

  // Only look at the line immediately before
  if (line == 1) return false;

  SourceLocation prev_line_start = sm.translateLineCol(fid, line - 1, 1);
  if (prev_line_start.isInvalid()) return false;

  SourceLocation loop_start =
      Lexer::getLocForEndOfToken(prev_line_start, 0, sm, LangOptions());

  StringRef line_text =
      Lexer::getSourceText(CharSourceRange::getCharRange(prev_line_start, loc),
                           sm, LangOptions(), 0);

  // Strip whitespace
  line_text = line_text.trim();

  return line_text.starts_with("#pragma hls_unroll yes");
}

class UnrollVisitor : public RecursiveASTVisitor<UnrollVisitor> {
 public:
  UnrollVisitor(ASTContext& context, ClangTidyCheck* check)
      : context(context), check(check) {}

  bool TraverseStmt(Stmt* stmt) {
    if (!stmt) return true;

    if (auto* for_stmt = dyn_cast<ForStmt>(stmt)) {
      bool is_unrolled = has_unroll_pragma(for_stmt->getBeginLoc(),
                                           context.getSourceManager());
      bool was_inside_unrolled = inside_unrolled;

      if (was_inside_unrolled && !is_unrolled) {
        check->diag(for_stmt->getBeginLoc(),
                    "loop is nested inside a '#pragma hls_unroll yes' loop but "
                    "is missing its own unroll pragma");
      }

      inside_unrolled = is_unrolled || inside_unrolled;
      RecursiveASTVisitor::TraverseStmt(stmt);
      inside_unrolled = was_inside_unrolled;
      return true;
    }

    return RecursiveASTVisitor::TraverseStmt(stmt);
  }

 private:
  ASTContext& context;
  ClangTidyCheck* check;
  bool inside_unrolled = false;
};

}  // namespace

HLSUnrollNestedCheck::HLSUnrollNestedCheck(StringRef name,
                                           ClangTidyContext* context)
    : ClangTidyCheck(name, context) {}

void HLSUnrollNestedCheck::registerMatchers(MatchFinder* finder) {
  finder->addMatcher(functionDecl(isDefinition()).bind("func"), this);
}

void HLSUnrollNestedCheck::check(const MatchFinder::MatchResult& result) {
  const auto* func = result.Nodes.getNodeAs<FunctionDecl>("func");
  if (!func || !func->hasBody()) return;

  UnrollVisitor visitor(*result.Context, this);
  visitor.TraverseDecl(const_cast<FunctionDecl*>(func));
}

}  // namespace hls
}  // namespace tidy
}  // namespace clang
