cmake_minimum_required(VERSION 3.13)
project(CatapultHlsChecks LANGUAGES CXX)

# Use conda-provided LLVM/Clang
set(LLVM_DIR "$ENV{CONDA_PREFIX}/lib/cmake/llvm")
set(Clang_DIR "$ENV{CONDA_PREFIX}/lib/cmake/clang")

find_package(LLVM REQUIRED CONFIG)
find_package(Clang REQUIRED CONFIG)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CLANG_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

add_library(CatapultHlsChecks MODULE
  ConnectionsResetCheck.cc
  NestedUnrollCheck.cc
  CatapultHlsTidyModule.cc
)

target_link_libraries(CatapultHlsChecks
  PRIVATE
    clangAST
    clangASTMatchers
    clangBasic
    clangTidy
)

set_target_properties(CatapultHlsChecks PROPERTIES
  CXX_STANDARD 17
)