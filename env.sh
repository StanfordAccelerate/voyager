#!/bin/bash

export PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CODEGEN_DIR=test/compiler

# Pip inside conda environment may be shared by multiple worktrees
# Prefer this checkout over the shared environment's editable installation
export PYTHONPATH="$PROJECT_ROOT/voyager-compiler/src${PYTHONPATH:+:$PYTHONPATH}"
