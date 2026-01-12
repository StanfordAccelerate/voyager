#ifndef LLVM_CLANG_TOOLS_EXTRA_TIDY_CONNECTIONS_RESET_CHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_TIDY_CONNECTIONS_RESET_CHECK_H

#include "clang-tidy/ClangTidyCheck.h"

namespace clang {
namespace tidy {
namespace hls {

class ConnectionsResetCheck : public ClangTidyCheck {
 public:
  ConnectionsResetCheck(llvm::StringRef name, ClangTidyContext* context);

  void registerMatchers(ast_matchers::MatchFinder* finder) override;
  void check(const ast_matchers::MatchFinder::MatchResult& result) override;
};

}  // namespace hls
}  // namespace tidy
}  // namespace clang

#endif  // LLVM_CLANG_TOOLS_EXTRA_TIDY_CONNECTIONS_RESET_CHECK_H
