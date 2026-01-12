#include "ConnectionsResetCheck.h"
#include "NestedUnrollCheck.h"
#include "clang-tidy/ClangTidy.h"
#include "clang-tidy/ClangTidyModuleRegistry.h"

using namespace clang::tidy;

namespace {

class CatapultHlsTidyModule : public ClangTidyModule {
 public:
  void addCheckFactories(ClangTidyCheckFactories& Factories) override {
    Factories.registerCheck<clang::tidy::hls::ConnectionsResetCheck>(
        "hls-connections-reset-check");
    Factories.registerCheck<clang::tidy::hls::HLSUnrollNestedCheck>(
        "hls-nested-unroll-check");
  }
};

}  // namespace

// Register the module
static ClangTidyModuleRegistry::Add<CatapultHlsTidyModule> X(
    "catapult-hls-tidy-module", "Adds checks for Catapult HLS code.");
