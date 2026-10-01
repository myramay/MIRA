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
from .compiler import CompiledProgram, DynamicProgram, ShapeSpecialized, compile_file
from .errors import MiraError

STAGES = ["tokens", "ast", "ir", "opt", "passes", "partition", "asm", "timeline", "mlir", "html"]


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


def _shapes(items: list[str]) -> dict[str, tuple[int, ...]]:
    out = {}
    for it in items or []:
        name, _, dims = it.rpartition("=")
        if not name:
            raise SystemExit(f"--shape expects NAME=D0,D1,..., got {it!r}")
        out[name] = tuple(int(d) for d in dims.split(",") if d)
    return out


def _compile(args, target: str) -> CompiledProgram:
    extra = {"input_shapes": _shapes(args.shape)} if args.file.endswith(".onnx") else {}
    prog = compile_file(args.file, target, entry=args.entry, weights=_load_npz(args.weights), dims=_dims(args.dim),
                        seed=args.seed, optimize=not args.O0, fuse=not args.no_fuse, units=args.units,
                        double_buffer=not args.no_double_buffer, precision=args.precision,
                        cost_model=not args.no_cost_model, quantize=args.quantize, sim_fast=args.sim_fast,
                        sim_config=dict(kv.split("=", 1) for kv in args.sim or []), mlir_backend=args.mlir_backend,
                        **extra)
    if isinstance(prog, ShapeSpecialized):
        from .frontends.onnx_import import onnx_inputs
        dyn = [f"{n}={dims}" for n, dims, _ in onnx_inputs(args.file) if None in dims]
        raise MiraError(f"the model has inputs with symbolic dimensions ({'; '.join(dyn)}); pin them with --shape, "
                        f"e.g. --shape {dyn[0].split('=')[0]}=1,...")
    if isinstance(prog, DynamicProgram):
        syms = sorted({d for p in prog.fn.params for d in p.type.dims if isinstance(d, str)}
                      | {p.name for p in prog.fn.params if p.type.is_int} - set(_dims(args.dim)))
        raise MiraError(f"the entry function has runtime dimensions ({', '.join(syms)}); "
                        f"give them values with --dim, e.g. --dim {syms[0]}=8")
    return prog


def _as_tuple(out) -> tuple:
    return out if isinstance(out, tuple) else (out,)


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
    outs = _as_tuple(prog.run(feeds))
    print(prog.summary())
    total = sum(t for _, t in prog.last_timings)
    print(f"wall time {total * 1e3:.2f} ms")
    for i, out in enumerate(outs):
        print(f"output {i}: {list(out.shape)} {out.dtype}")
        with np.printoptions(precision=4, suppress=True, threshold=12, edgeitems=3):
            print(out)
    if args.save:
        np.savez(args.save, *outs) if len(outs) > 1 else np.save(args.save, outs[0])
        print(f"saved to {args.save}")
    if args.check and args.target != "cpu":
        refs = _as_tuple(compile_file(args.file, "cpu", entry=args.entry, weights=_load_npz(args.weights),
                                      dims=_dims(args.dim), seed=args.seed).run(feeds))
        ok = True
        for i, (out, ref) in enumerate(zip(outs, refs)):
            err = float(np.abs(out.astype(np.float32) - ref.astype(np.float32)).max())
            rel = err / (float(np.abs(ref).max()) + 1e-12)
            ok &= rel < args.tol
            print(f"check output {i} vs cpu reference: max abs err {err:.3e}, relative {rel:.3e}  ->  "
                  f"{'PASS' if rel < args.tol else 'FAIL'}")
        return 0 if ok else 1
    return 0


def cmd_emit(args) -> int:
    from . import syntax
    from .lexer import tokenize
    from .parser import parse

    if args.stage in ("tokens", "ast"):
        if args.file.endswith(".onnx"):
            raise MiraError(f"'{args.stage}' is a Mira source stage; ONNX models start at the 'ir' stage")
        src = open(args.file).read()
        if args.stage == "tokens":
            for t in tokenize(src, args.file):
                print(f"{t.loc.line:4d}:{t.loc.col:<4d} {t.kind:<8} {t.text}")
        else:
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
    elif args.stage == "html":
        return emit_html(args, prog)
    elif args.stage == "mlir":
        from .backends.mlir import emit_module, supported
        bad = [f"{op.kind}: {r}" for op in prog.graph.compute_ops() if (r := supported(op))]
        if bad:
            raise MiraError("can't emit the whole program as MLIR: " + "; ".join(sorted(set(bad))))
        print(emit_module(prog.graph), end="")
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


def emit_html(args, prog: CompiledProgram) -> int:
    """Run the program on the simulator and write the interactive timeline page."""
    import copy
    import os
    from .backends.npusim.visualize import collect, render_html

    if args.target != "npu-sim":
        raise MiraError("the html view is for the simulated NPU: add -t sim")
    feeds = _inputs(prog, args.inputs, args.seed)
    prog.run(feeds)
    runs = [prog]
    labels = ["double buffering on" if not args.no_double_buffer else "double buffering off"]
    if args.compare:
        variant = copy.copy(args)
        if args.compare == "no-double-buffer":
            variant.no_double_buffer = not args.no_double_buffer
            labels.append("double buffering off" if variant.no_double_buffer else "double buffering on")
        elif "=" in args.compare:
            variant.sim = list(args.sim or []) + [args.compare]
            labels = ["baseline", args.compare]
        else:
            raise MiraError("--compare takes no-double-buffer or a hardware setting like mxu_dim=64")
        other = _compile(variant, "npu-sim")
        other.run(feeds)
        runs.append(other)
    data = [collect(label, p) for label, p in zip(labels, runs)]
    cfg = data[0]["config"]
    subtitle = (f"{os.path.basename(args.file)} on MNPU-1 · {cfg['mxu']} systolic array · {cfg['sram_kib']} KiB SRAM "
                f"in {cfg['banks']} banks · {cfg['dma_gbs']:g} GB/s DMA · {cfg['clock_ghz']:g} GHz")
    out = args.output or os.path.splitext(os.path.basename(args.file))[0] + ".sim.html"
    with open(out, "w") as f:
        f.write(render_html(data, "MNPU-1 simulation", subtitle))
    n = sum(len(d["ins"]) for d in data)
    print(f"wrote {out} ({os.path.getsize(out) / 1024:,.0f} KiB, {n:,} instructions). Open it with:  open {out}")
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
        out = _as_tuple(prog.run(feeds))[0]      # warm-up (and correctness, on the first output)
        if ref is None:
            ref = _as_tuple(compile_file(args.file, "cpu", entry=args.entry, weights=_load_npz(args.weights),
                                         dims=_dims(args.dim), seed=args.seed).run(feeds))[0].astype(np.float32)
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
        p.add_argument("--target", "-t", default="cpu", choices=TARGETS + ["sim", "ane", "iree"])
        p.add_argument("--entry", default="main")
        p.add_argument("--weights", help=".npz with values for const parameters (default: random)")
        p.add_argument("--inputs", help=".npz with runtime inputs (default: random)")
        p.add_argument("--dim", action="append", help="bind a shape symbol or int parameter, e.g. --dim B=32")
        p.add_argument("--shape", action="append", metavar="NAME=D0,D1,...",
                       help="ONNX models: input shape for an input with symbolic dims, e.g. --shape x=1,3,224,224")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--precision", default="auto", choices=["auto", "f16", "none"])
        p.add_argument("--units", default="ne", choices=["ne", "all", "gpu", "cpu"], help="Core ML compute units")
        p.add_argument("-O0", action="store_true", help="disable optimization passes")
        p.add_argument("--no-fuse", action="store_true")
        p.add_argument("--no-double-buffer", action="store_true")
        p.add_argument("--no-cost-model", action="store_true", help="offload every supported op")
        p.add_argument("--quantize", choices=["int8"], help="store large weights as int8 (per-channel scales)")
        p.add_argument("--sim-fast", action="store_true", help="npu-sim: timing only (outputs from the reference)")
        p.add_argument("--mlir-backend", default="cpu", choices=["cpu", "metal"],
                       help="mlir target: compile with IREE for the CPU or the Mac GPU (Metal)")
        p.add_argument("--sim", action="append", metavar="KEY=VALUE",
                       help="npu-sim hardware parameter, e.g. --sim mxu_dim=64 --sim sram_bytes=1048576")

    p = sub.add_parser("run", help="compile and run")
    common(p)
    p.add_argument("--check", action="store_true", help="compare against the CPU reference")
    p.add_argument("--tol", type=float, default=2e-2)
    p.add_argument("--save")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("emit", help="print a compiler stage")
    common(p)
    p.add_argument("--stage", "-s", default="opt", choices=STAGES)
    p.add_argument("--output", "-o", help="html stage: where to write the page (default: <name>.sim.html)")
    p.add_argument("--compare", metavar="VARIANT",
                   help="html stage: also run a variant, e.g. no-double-buffer or mxu_dim=64, and show both")
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
    if args.target == "iree":
        args.target = "mlir"
    if args.precision == "none":
        args.precision = None
    try:
        return args.fn(args)
    except MiraError as e:
        print(e.render(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
