# IDE Setup
This file describes the process of setting up the codebase to work with various
IDEs.

## VS Code + Microsoft C/C++

The C/C++ extension from Microsoft has worked out of the box.

## VS Code + Clangd

To use clangd, you need to generate a `compile_commands.json` file to tell the
tool the exact commands used to build your code. You can generate it with
`compiledb`, which can be installed with `pip` inside the `conda` environment:
```bash
pip install compiledb
```

Once `compiledb` is installed, use it in conjunction with a `make` command to
have it generate the `compile_commands.json`. For example, to generate
`compile_commands.json` for `TestRunner` on a 32x32 array with MXINT8, run
```bash
DATATYPE=MXINT8 IC_DIMENSION=32 OC_DIMENSION=32 compiledb make TestRunner
```

Please refer to `run_regression.py` for the exact `make` commands for different
targets.

The environment variables provided affect how clangd parses your code and
displays it in the editor. You should use the same configuration to generate the
`compile_commands.json` as the intended hardware architecture.
