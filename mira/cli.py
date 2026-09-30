"""Command-line driver.

    mira run     prog.mira --target coreml --check     # compile, run on random inputs, compare with CPU
    mira emit    prog.mira --stage ir                  # show any compiler stage
    mira bench   prog.mira                             # time every target
    mira check   prog.mira                             # just type-check
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from .backends import TARGETS
from .compiler import CompiledProgram, compile_file
from .errors import MiraError

STAGES = ["tokens", "ast", "ir", "opt", "passes", "partition", "asm", "timeline"]


def _dims(items: list[str]) -> dict[str, int]:
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not v:
            raise SystemExit(f"--dim expects NAME=VALUE, got {it!r}")
        out[k] = int(v)
    return out


def _load_npz(path):
    if not path:
        return {}
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _compile(args, target: str) -> CompiledProgram:
    return compile_file(args.file, target, entry=args.entry, weights=_load_npz(args.weights), dims=_dims(args.dim),
                        seed=args.seed, optimize=not args.O0, fuse=not args.no_fuse, units=args.units,
                        double_buffer=not args.no_double_buffer, precision=args.precision,
                        cost_model=not args.no_cost_model)


def _inputs(prog: CompiledProgram, path, seed: int) -> dict[str, np.ndarray]:
    given = _load_npz(path)
    rng = np.random.default_rng(seed + 1)
    return {v.name: given[v.name] if v.name in given else rng.standard_normal(v.type.shape).astype(v.type.np_dtype)
            for v in prog.graph.inputs}


def _sim_segments(prog: CompiledProgram):
    return [s for s in prog.segments if s.device == "npu-sim"]


def cmd_run(args) -> int:
    prog = _compile(args, args.target)
    feeds = _inputs(prog, args.inputs, args.seed)
    out = prog.run(feeds)
    print(prog.summary())
    total = sum(t for _, t in prog.last_timings)
    print(f"output {list(out.shape)} {out.dtype}   wall time {total * 1e3:.2f} ms")
    with np.printoptions(precision=4, suppress=True, threshold=12, edgeitems=3):
        print(out)
    if args.save:
        np.save(args.save, out)
        print(f"saved to {args.save}")
    if args.check and args.target != "cpu":
        ref = compile_file(args.file, "cpu", entry=args.entry, weights=_load_npz(args.weights),
                           dims=_dims(args.dim), seed=args.seed).run(feeds)
        err = float(np.abs(out.astype(np.float32) - ref.astype(np.float32)).max())
        rel = err / (float(np.abs(ref).max()) + 1e-12)
        ok = rel < args.tol
        print(f"check vs cpu reference: max abs err {err:.3e}, relative {rel:.3e}  ->  {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


def cmd_emit(args) -> int:
    from . import syntax
    from .lexer import tokenize
    from .parser import parse

    src = open(args.file).read()
    if args.stage == "tokens":
        for t in tokenize(src, args.file):
            print(f"{t.loc.line:4d}:{t.loc.col:<4d} {t.kind:<8} {t.text}")
        return 0
    if args.stage == "ast":
        print(syntax.format_module(parse(src, args.file)))
        return 0
    prog = _compile(args, args.target)
    if args.stage == "ir":
        print(prog.pass_log[0][1])
    elif args.stage == "opt":
        print(prog.graph)
    elif args.stage == "passes":
        for name, text in prog.pass_log:
            print(f"// ----- after {name}\n{text}\n")
    elif args.stage == "partition":
        print(prog.placement_report())
        print()
        print(prog.summary())
    elif args.stage in ("asm", "timeline"):
        segs = _sim_segments(prog)
        if not segs:
            print("no npu-sim segments (use --target npu-sim)")
            return 1
        if args.stage == "timeline":
            prog.run(_inputs(prog, args.inputs, args.seed))
        for s in segs:
            print(f"// ===== {s.graph.name}")
            print(s.executable.assembly(args.limit) if args.stage == "asm" else s.executable.timeline(args.limit))
            if args.stage == "timeline":
                print(s.executable.report())
    return 0


def cmd_bench(args) -> int:
    targets = args.targets.split(",")
    feeds = None
    ref = None
    print(f"{'target':<16} {'compile':>9} {'median':>10} {'min':>10}   {'rel err':>9}  segments")
    for t in targets:
        name, _, units = t.partition(":")
        if units:
            args.units = units
        prog = _compile(args, name)
        if feeds is None:
            feeds = _inputs(prog, args.inputs, args.seed)
        out = prog.run(feeds)      # warm-up (and correctness)
        if ref is None:
            ref = compile_file(args.file, "cpu", entry=args.entry, weights=_load_npz(args.weights),
                               dims=_dims(args.dim), seed=args.seed).run(feeds).astype(np.float32)
        times = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            prog.run(feeds)
            times.append(time.perf_counter() - t0)
        rel = float(np.abs(out.astype(np.float32) - ref).max() / (np.abs(ref).max() + 1e-12))
        segs = "+".join(s.device for s in prog.segments)
        print(f"{t:<16} {prog.compile_seconds:8.2f}s {np.median(times) * 1e3:8.3f}ms {min(times) * 1e3:8.3f}ms"
              f"   {rel:9.2e}  {segs}")
        for s in _sim_segments(prog):
            st = s.executable.last_stats
            print(f"{'':<16} simulated MNPU-1 time for {s.graph.name}: {st.cycles / 1e3:,.1f} us "
                  f"({st.macs / st.cycles:,.0f} MAC/cycle)")
    return 0


def cmd_check(args) -> int:
    prog = _compile(args, "cpu")
    ins = ", ".join(f"{v.name}: {v.type}" for v in prog.graph.inputs)
    print(f"ok: {args.file}  inputs ({ins}) -> {prog.graph.outputs[0].type}, {len(prog.graph.compute_ops())} ops")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="mira", description="Mira tensor language compiler")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("file")
        p.add_argument("--target", "-t", default="cpu", choices=TARGETS + ["sim", "ane"])
        p.add_argument("--entry", default="main")
        p.add_argument("--weights", help=".npz with values for const parameters (default: random)")
        p.add_argument("--inputs", help=".npz with runtime inputs (default: random)")
        p.add_argument("--dim", action="append", help="bind a shape symbol or int parameter, e.g. --dim B=32")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--precision", default="auto", choices=["auto", "f16", "none"])
        p.add_argument("--units", default="ne", choices=["ne", "all", "gpu", "cpu"], help="Core ML compute units")
        p.add_argument("-O0", action="store_true", help="disable optimization passes")
        p.add_argument("--no-fuse", action="store_true")
        p.add_argument("--no-double-buffer", action="store_true")
        p.add_argument("--no-cost-model", action="store_true", help="offload every supported op")

    p = sub.add_parser("run", help="compile and run")
    common(p)
    p.add_argument("--check", action="store_true", help="compare against the CPU reference")
    p.add_argument("--tol", type=float, default=2e-2)
    p.add_argument("--save")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("emit", help="print a compiler stage")
    common(p)
    p.add_argument("--stage", "-s", default="opt", choices=STAGES)
    p.add_argument("--limit", type=int, default=60)
    p.set_defaults(fn=cmd_emit)

    p = sub.add_parser("bench", help="benchmark targets")
    common(p)
    p.add_argument("--targets", default="cpu,npu-sim,coreml:cpu,coreml:ne")
    p.add_argument("--iters", type=int, default=20)
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("check", help="type-check only")
    common(p)
    p.set_defaults(fn=cmd_check)

    args = ap.parse_args(argv)
    if args.target in ("sim", "ane"):
        args.target = {"sim": "npu-sim", "ane": "coreml"}[args.target]
    if args.precision == "none":
        args.precision = None
    try:
        return args.fn(args)
    except MiraError as e:
        print(e.render(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
