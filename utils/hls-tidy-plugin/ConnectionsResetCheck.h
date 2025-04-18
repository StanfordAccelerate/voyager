#ifndef LLVM_CLANG_TOOLS_EXTRA_TIDY_CONNECTIONSRESETCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_TIDY_CONNECTIONSRESETCHECK_H

#include "clang-tidy/ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace hls {

class ConnectionsResetCheck : public ClangTidyCheck {
 public:
  ConnectionsResetCheck(llvm::StringRef Name, ClangTidyContext *Context);

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

}  // namespace hls
}  // namespace tidy
}  // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_TIDY_CONNECTIONSRESETCHECK_H
