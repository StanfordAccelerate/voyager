"""Shared Verilator unittest runner for CIM SystemVerilog suites."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
UNITTEST_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
CIM_SRC_DIR = REPO_ROOT / "src" / "cim"
BUILD_ROOT = UNITTEST_DIR / "build" / "verilog"
DEFAULT_INIT_SEED_COUNT = 8
DEFAULT_BASE_SEED = 0x1234ABCD
VERILATOR_X_FLAGS = (
    "--x-initial",
    "unique",
    "--x-assign",
    "unique",
    "--x-initial-edge",
)
WARNING_PREFIXES = ("%Warning", "Warning:")
COLOR_CODES = {
    "SEED": "\033[36m",
    "CONFIG": "\033[36m",
    "BUILD": "\033[34m",
    "CACHED": "\033[35m",
    "RUN": "\033[36m",
    "PASS": "\033[32m",
    "FAIL": "\033[31m",
    "WARN": "\033[33m",
    "LOG": "\033[33m",
    "SUMMARY": "\033[1m",
}
COLOR_RESET = "\033[0m"


# Result from one captured external command
@dataclass(frozen=True)
class CommandResult:
    log_path: Path
    output: str


# Result from preparing one simulator binary
@dataclass(frozen=True)
class BuildResult:
    obj_dir: Path
    cached: bool
    log_path: Path | None
    warnings: tuple[str, ...]


# Result from one case including all requested runtime seeds
@dataclass(frozen=True)
class CaseResult:
    name: str
    passed: bool
    illegal: bool
    cached: bool
    elapsed_s: float
    run_count: int
    warnings: tuple[str, ...]
    failed_stage: str | None = None
    failed_log: Path | None = None
    error: str | None = None


# Suite-specific context passed to run callbacks
@dataclass(frozen=True)
class RunContext:
    case: Any
    suite: "SuiteConfig"
    obj_dir: Path
    init_seed_count: int
    base_seed: int
    waveform_dir: Path | None


# Immutable suite definition consumed by the shared runner
@dataclass(frozen=True)
class SuiteConfig:
    name: str
    description: str
    top_module: str
    cases: tuple[Any, ...]
    source_files: tuple[Path, ...]
    include_dirs: tuple[Path, ...]
    build_inputs: tuple[Path, ...]
    tb_text: Callable[[Any], str]
    validate_case: Callable[[Any], None]
    run_fields: Callable[[Any, int, int], dict[str, object]]
    run_case: Callable[[RunContext], int]
    case_run_count: Callable[[Any, int], int]
    is_illegal: Callable[[Any], bool] = lambda _case: False


# Exception used to preserve subprocess failure details for the summary
class CommandError(RuntimeError):
    def __init__(self, stage: str, cmd: list[str], returncode: int, log_path: Path) -> None:
        self.stage = stage
        self.cmd = cmd
        self.returncode = returncode
        self.log_path = log_path
        super().__init__(
            f"{stage} failed with exit code {returncode}: {shlex.join(cmd)}"
        )


# Thread-safe status printer with optional ANSI colors
class Printer:
    def __init__(
        self,
        use_color: bool,
        max_case_name_len: int,
        case_count: int,
        default_sim_run_count: int,
    ) -> None:
        self.use_color = use_color
        self.lock = threading.Lock()
        self.max_case_name_len = max_case_name_len
        self.summary_count_width = max(2, len(str(max(1, case_count))))
        self.sim_run_width = max(3, len(str(max(1, default_sim_run_count))))
        self.field_value_widths = {
            "base": 10,
            "elapsed": 7,
            "first_init_seed": 10,
            "impl": len("macro1"),
            "init_seed": 10,
            "init_seed_count": max(2, len(str(DEFAULT_INIT_SEED_COUNT))),
            "kind": len("illegal"),
            "stage": len("runner"),
            "stimulus_seed": 10,
            "warnings": len("summary"),
            "built": self.summary_count_width,
            "cached": self.summary_count_width,
            "cases": self.summary_count_width,
            "failed": self.summary_count_width,
            "illegal": self.summary_count_width,
            "jobs": self.summary_count_width,
            "passed": self.summary_count_width,
            "sim_runs": self.sim_run_width,
            "rc": 3,
        }

    # Print one status line without interleaving parallel workers
    def event(self, status: str, message: str) -> None:
        label = f"{status:<7}"
        if self.use_color and status in COLOR_CODES:
            label = f"{COLOR_CODES[status]}{label}{COLOR_RESET}"
        with self.lock:
            print(f"{label} {message}", flush=True)

    # Print aligned key-value fields without interleaving parallel workers
    def event_fields(self, status: str, **fields: object) -> None:
        self.event(status, self.format_fields(fields))

    # Print a captured log without interleaving parallel workers
    def raw(self, text: str) -> None:
        with self.lock:
            print(text, end="" if text.endswith("\n") else "\n", flush=True)

    # Format the case column as a positional field after the status label
    def format_case(self, value: object) -> str:
        return str(value).ljust(self.max_case_name_len)

    # Format one key-value field with stable value widths for scan-friendly output
    def format_field(self, key: str, value: object) -> str:
        text_value = str(value)
        width = self.field_value_widths.get(key)
        if width is None:
            return f"{key}={text_value}"
        return f"{key}={text_value}".ljust(len(key) + 1 + width)

    # Format a sequence of key-value fields into aligned columns
    def format_fields(self, fields: dict[str, object]) -> str:
        formatted: list[str] = []
        if "case" in fields:
            formatted.append(self.format_case(fields["case"]))
        for key, value in fields.items():
            if key != "case":
                formatted.append(self.format_field(key, value))
        return "  ".join(formatted).rstrip()


# Parse decimal or hex seeds from the command line
def parse_seed(value: str) -> int:
    seed = int(value, 0)
    if seed < 0:
        raise argparse.ArgumentTypeError("seed must be non-negative")
    return seed & 0xFFFFFFFF


# Parse positive integer command-line counts
def parse_positive_int(value: str) -> int:
    parsed = int(value, 0)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


# Resolve whether ANSI colors should be emitted
def use_color_for(mode: str) -> bool:
    if mode == "always":
        return True
    if mode == "never":
        return False
    return sys.stdout.isatty()


# Return a path relative to the repo root when possible
def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# Write text only when content changes so mtimes do not force rebuilds
def write_text_if_changed(path: Path, text: str) -> bool:
    if path.exists() and path.read_text() == text:
        return False
    path.write_text(text)
    return True


# Derive a stable 31-bit seed from the run seed, case name, and stream label
def stable_seed(base_seed: int, case: Any, label: str, index: int = 0) -> int:
    payload = f"{base_seed:08x}:{case.name}:{label}:{index}".encode()
    seed = int.from_bytes(hashlib.blake2s(payload, digest_size=4).digest(), "big") & 0x7FFFFFFF
    return seed or 1


# Return the Verilator initialization seed for one replay stream
def init_seed_for(base_seed: int, case: Any, seed_idx: int) -> int:
    return stable_seed(base_seed, case, "init", seed_idx)


# Return the SV stimulus seed for one generated case
def stimulus_seed_for(base_seed: int, case: Any) -> int:
    return stable_seed(base_seed, case, "stimulus")


# Build the Verilator command for one generated testbench
def verilator_cmd(
    suite: SuiteConfig,
    tb: Path,
    obj_dir: Path,
    enable_waveform: bool,
) -> list[str]:
    include_args = [f"-I{include_dir}" for include_dir in suite.include_dirs]
    trace_args = ["--trace"] if enable_waveform else []
    return [
        "verilator",
        "-sv",
        "--binary",
        "--timing",
        *trace_args,
        "-Wno-ZERODLY",
        *VERILATOR_X_FLAGS,
        *include_args,
        *[str(source) for source in suite.source_files],
        str(tb),
        "--top-module",
        suite.top_module,
        "--Mdir",
        str(obj_dir),
    ]


# Capture a command's combined stdout and stderr into a log file
def run_command_to_log(cmd: list[str], cwd: Path, log_path: Path, stage: str) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = completed.stdout or ""
    log_text = (
        f"$ {shlex.join(cmd)}\n"
        f"# cwd: {cwd}\n"
        f"# started: {started}\n"
        f"# exit_code: {completed.returncode}\n\n"
        f"{output}"
    )
    log_path.write_text(log_text)
    if completed.returncode != 0:
        raise CommandError(stage, cmd, completed.returncode, log_path)
    return CommandResult(log_path=log_path, output=log_text)


# Return the installed Verilator version string
def get_verilator_version() -> str:
    if shutil.which("verilator") is None:
        raise SystemExit("verilator is required to run these unit tests")
    completed = subprocess.run(
        ["verilator", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return (completed.stdout or "").strip()


# Pull Verilator warning headers out of a build log
def warning_lines(log_text: str) -> tuple[str, ...]:
    warnings: list[str] = []
    for line in log_text.splitlines():
        stripped = line.strip()
        if stripped.startswith(WARNING_PREFIXES):
            warnings.append(stripped)
    return tuple(warnings)


# Construct the build manifest used to decide whether a simulator is current
def build_manifest(
    suite: SuiteConfig,
    tb_text: str,
    cmd: list[str],
    verilator_version: str,
    enable_waveform: bool,
) -> dict[str, object]:
    return {
        "suite": suite.name,
        "top": suite.top_module,
        "waveform": enable_waveform,
        "verilator_version": verilator_version,
        "verilator_command": cmd,
        "testbench": tb_text,
        "inputs": {
            str(path): {
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in suite.build_inputs
        },
    }


# Read a JSON manifest if it exists and is well formed
def read_manifest(path: Path) -> dict[str, object] | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


# Generate or reuse the simulator binary for one parameterized test case
def build_case(
    suite: SuiteConfig,
    case: Any,
    verilator_version: str,
    force_rebuild: bool,
    enable_waveform: bool,
    printer: Printer,
) -> BuildResult:
    suite.validate_case(case)

    case_dir = BUILD_ROOT / suite.name / case.name
    obj_dir = case_dir / "obj"
    logs_dir = case_dir / "logs"
    manifest_path = case_dir / "build_manifest.json"
    binary = obj_dir / f"V{suite.top_module}"
    tb = case_dir / f"{suite.top_module}.sv"
    tb_text = suite.tb_text(case)
    cmd = verilator_cmd(suite, tb, obj_dir, enable_waveform)
    manifest = build_manifest(suite, tb_text, cmd, verilator_version, enable_waveform)

    case_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_text_if_changed(tb, tb_text)

    if not force_rebuild and binary.exists() and read_manifest(manifest_path) == manifest:
        printer.event_fields("CACHED", case=case.name, simulator=display_path(binary))
        return BuildResult(obj_dir=obj_dir, cached=True, log_path=None, warnings=())

    if force_rebuild and obj_dir.exists():
        shutil.rmtree(obj_dir)

    log_path = logs_dir / "verilator.log"
    printer.event_fields("BUILD", case=case.name, log=display_path(log_path))
    result = run_command_to_log(cmd, REPO_ROOT, log_path, "build")
    write_text_if_changed(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    return BuildResult(
        obj_dir=obj_dir,
        cached=False,
        log_path=log_path,
        warnings=warning_lines(result.output),
    )


# Return the waveform plusarg for one run when waveform dumping is enabled
def waveform_plusargs(context: RunContext, log_name: str) -> list[str]:
    if context.waveform_dir is None:
        return []

    context.waveform_dir.mkdir(parents=True, exist_ok=True)
    waveform_path = context.waveform_dir / f"{Path(log_name).stem}.vcd"
    return [f"+waveform={waveform_path}"]


# Run a simulator binary with common Verilator plusargs and suite-specific plusargs
def run_sim(
    context: RunContext,
    plusargs: list[str],
    log_name: str,
) -> CommandResult:
    log_path = context.obj_dir.parent / "logs" / log_name
    return run_command_to_log(
        [
            str(context.obj_dir / f"V{context.suite.top_module}"),
            *plusargs,
            *waveform_plusargs(context, log_name),
        ],
        REPO_ROOT,
        log_path,
        "run",
    )


# Run the common replayable init-seed matrix for one case
def run_seed_matrix(context: RunContext, extra_plusargs: list[str] | None = None) -> int:
    stimulus_seed = stimulus_seed_for(context.base_seed, context.case)
    logs_dir = context.obj_dir.parent / "logs"
    width = max(2, len(str(context.init_seed_count - 1)))
    for seed_idx in range(context.init_seed_count):
        init_seed = init_seed_for(context.base_seed, context.case, seed_idx)
        log_path = logs_dir / f"run_{seed_idx:0{width}d}.log"
        run_command_to_log(
            [
                str(context.obj_dir / f"V{context.suite.top_module}"),
                f"+verilator+seed+{init_seed}",
                "+verilator+rand+reset+2",
                f"+rng_seed={stimulus_seed:08x}",
                *(extra_plusargs or []),
                *waveform_plusargs(context, log_path.name),
            ],
            REPO_ROOT,
            log_path,
            "run",
        )
    return context.init_seed_count


# Run one collected/one-off simulator invocation for a case
def run_single(
    context: RunContext,
    log_name: str,
    extra_plusargs: list[str] | None = None,
) -> int:
    stimulus_seed = stimulus_seed_for(context.base_seed, context.case)
    init_seed = init_seed_for(context.base_seed, context.case, 0)
    run_sim(
        context,
        [
            f"+verilator+seed+{init_seed}",
            "+verilator+rand+reset+2",
            f"+rng_seed={stimulus_seed:08x}",
            *(extra_plusargs or []),
        ],
        log_name,
    )
    return 1


# Generate, compile, and execute one parameterized SystemVerilog test case
def run_case(
    suite: SuiteConfig,
    case: Any,
    init_seed_count: int,
    base_seed: int,
    verilator_version: str,
    force_rebuild: bool,
    enable_waveform: bool,
    printer: Printer,
) -> CaseResult:
    started = time.perf_counter()
    try:
        build = build_case(
            suite,
            case,
            verilator_version,
            force_rebuild,
            enable_waveform,
            printer,
        )
        context = RunContext(
            case=case,
            suite=suite,
            obj_dir=build.obj_dir,
            init_seed_count=init_seed_count,
            base_seed=base_seed,
            waveform_dir=(build.obj_dir.parent / "waves") if enable_waveform else None,
        )
        printer.event_fields("RUN", case=case.name, **suite.run_fields(case, init_seed_count, base_seed))
        run_count = suite.run_case(context)

        elapsed_s = time.perf_counter() - started
        pass_fields: dict[str, object] = {
            "case": case.name,
            "sim_runs": run_count,
            "elapsed": f"{elapsed_s:.2f}s",
        }
        if context.waveform_dir is not None:
            pass_fields["waveforms"] = display_path(context.waveform_dir)
        printer.event_fields("PASS", **pass_fields)
        return CaseResult(
            name=case.name,
            passed=True,
            illegal=suite.is_illegal(case),
            cached=build.cached,
            elapsed_s=elapsed_s,
            run_count=run_count,
            warnings=build.warnings,
        )
    except CommandError as exc:
        elapsed_s = time.perf_counter() - started
        printer.event_fields(
            "FAIL",
            case=case.name,
            stage=exc.stage,
            rc=exc.returncode,
            log=display_path(exc.log_path),
            elapsed=f"{elapsed_s:.2f}s",
        )
        return CaseResult(
            name=case.name,
            passed=False,
            illegal=suite.is_illegal(case),
            cached=False,
            elapsed_s=elapsed_s,
            run_count=0,
            warnings=(),
            failed_stage=exc.stage,
            failed_log=exc.log_path,
            error=str(exc),
        )
    except Exception as exc:
        elapsed_s = time.perf_counter() - started
        printer.event_fields(
            "FAIL",
            case=case.name,
            stage="runner",
            error=exc,
            elapsed=f"{elapsed_s:.2f}s",
        )
        return CaseResult(
            name=case.name,
            passed=False,
            illegal=suite.is_illegal(case),
            cached=False,
            elapsed_s=elapsed_s,
            run_count=0,
            warnings=(),
            failed_stage="runner",
            error=str(exc),
        )


# Dump captured logs for failing cases after parallel work settles
def dump_failure_logs(results: list[CaseResult], printer: Printer) -> None:
    for result in results:
        if result.passed or result.failed_log is None:
            continue
        printer.event_fields("LOG", case=result.name, log=display_path(result.failed_log))
        try:
            printer.raw(result.failed_log.read_text(errors="replace"))
        except FileNotFoundError:
            printer.raw(f"<missing log: {result.failed_log}>\n")


# Print warning summaries after builds complete so parallel output stays readable
def report_warnings(results: list[CaseResult], mode: str, printer: Printer) -> None:
    if mode == "never":
        return
    for result in results:
        if not result.warnings:
            continue
        if mode == "summary":
            printer.event_fields(
                "WARN",
                case=result.name,
                warnings=len(result.warnings),
                message=result.warnings[0],
            )
        else:
            for warning in result.warnings:
                printer.event_fields("WARN", case=result.name, message=warning)


# Parse common runner arguments for a suite
def parse_args(suite: SuiteConfig) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=suite.description)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case.name for case in suite.cases],
        help="Run only the named case. Can be passed more than once.",
    )
    parser.add_argument(
        "--init-seed-count",
        type=parse_positive_int,
        default=DEFAULT_INIT_SEED_COUNT,
        help=f"Number of Verilator resetless-state initialization seeds per case. "
             f"Default: {DEFAULT_INIT_SEED_COUNT}.",
    )
    parser.add_argument(
        "--seed",
        type=parse_seed,
        default=DEFAULT_BASE_SEED,
        help=f"Base seed for replayable randomized stimulus. Hex values are accepted. "
             f"Default: 0x{DEFAULT_BASE_SEED:08x}.",
    )
    parser.add_argument(
        "--jobs",
        type=parse_positive_int,
        help="Number of test cases to run in parallel. Default: available CPUs.",
    )
    parser.add_argument(
        "--color",
        choices=["auto", "always", "never"],
        default="auto",
        help="Control ANSI color output. Default: auto.",
    )
    parser.add_argument(
        "--warnings",
        choices=["summary", "full", "never"],
        default="summary",
        help="Control Verilator warning reporting. Default: summary.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild simulator binaries even when the build manifest matches.",
    )
    parser.add_argument(
        "--waveform",
        "--dump-waveform",
        action="store_true",
        help="Dump one VCD waveform per simulator run. Default: disabled.",
    )
    return parser.parse_args()


# Select cases from the optional command-line filters
def selected_cases(suite: SuiteConfig, case_filters: list[str] | None) -> list[Any]:
    if not case_filters:
        return list(suite.cases)
    wanted = set(case_filters)
    return [case for case in suite.cases if case.name in wanted]


# Compute the number of workers for a selected case set
def worker_count_for(args: argparse.Namespace, selected: list[Any]) -> int:
    return min(args.jobs or (os.cpu_count() or 1), len(selected))


# Run selected cases with a thread pool and return results in declaration order
def run_selected_cases(
    suite: SuiteConfig,
    selected: list[Any],
    args: argparse.Namespace,
    verilator_version: str,
    printer: Printer,
) -> list[CaseResult]:
    worker_count = worker_count_for(args, selected)
    config_fields: dict[str, object] = {
        "cases": len(selected),
        "jobs": worker_count,
        "init_seed_count": args.init_seed_count,
        "warnings": args.warnings,
    }
    if args.waveform:
        config_fields["waveform"] = "vcd"
    printer.event_fields("CONFIG", **config_fields)
    results_by_name: dict[str, CaseResult] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                run_case,
                suite,
                case,
                args.init_seed_count,
                args.seed,
                verilator_version,
                args.force_rebuild,
                args.waveform,
                printer,
            ): case
            for case in selected
        }
        for future in as_completed(futures):
            case = futures[future]
            results_by_name[case.name] = future.result()
    return [results_by_name[case.name] for case in selected]


# Run one suite from its command-line entrypoint
def run_suite(suite: SuiteConfig) -> int:
    args = parse_args(suite)
    selected = selected_cases(suite, args.case)
    if not selected:
        raise SystemExit("no cases selected")

    for path in suite.build_inputs:
        if not path.exists():
            raise SystemExit(f"Cannot find {path}")

    default_sim_run_count = sum(suite.case_run_count(case, DEFAULT_INIT_SEED_COUNT) for case in suite.cases)
    max_case_name_len = max(len(case.name) for case in suite.cases)
    printer = Printer(
        use_color_for(args.color),
        max_case_name_len,
        len(suite.cases),
        default_sim_run_count,
    )

    verilator_version = get_verilator_version()
    started = time.perf_counter()
    printer.event_fields("SEED", base=f"0x{args.seed:08x}")
    results = run_selected_cases(suite, selected, args, verilator_version, printer)
    elapsed_s = time.perf_counter() - started

    report_warnings(results, args.warnings, printer)
    dump_failure_logs(results, printer)

    passed = sum(result.passed for result in results)
    failed = len(results) - passed
    illegal_cases = sum(result.illegal for result in results)
    cached = sum(result.cached for result in results)
    built = len(results) - cached
    run_count = sum(result.run_count for result in results)
    printer.event_fields(
        "SUMMARY",
        passed=passed,
        failed=failed,
        built=built,
        cached=cached,
        illegal=illegal_cases,
        sim_runs=run_count,
        elapsed=f"{elapsed_s:.2f}s",
    )
    return 0 if failed == 0 else 1
