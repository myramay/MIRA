"""MLIR backend: MIRA IR -> MLIR (upstream dialects) -> IREE -> native code.

This is how production ML compilers are built: instead of hand-writing every
backend, lower to MLIR's standard dialects and reuse the ecosystem.

    MIRA op                      MLIR
    elementwise / where / cast   linalg.generic + arith / math (broadcasting via indexing maps)
    matmul                       linalg.matmul / linalg.batch_matmul (fp32 accumulation)
    conv2d, maxpool2d            tensor.pad + linalg.conv_2d_nchw_fchw / linalg.pooling_nchw_max
    reductions, softmax, norm    linalg.reduce + linalg.generic
    reshape                      tensor.collapse_shape + tensor.expand_shape
    transpose, broadcast         linalg.transpose, linalg.generic
    slice, pad, concat           tensor.extract_slice, tensor.pad, tensor.concat
    int8 weights                 i8 constant + dequantizing linalg.generic
    if / while                   scf.if / scf.while (control flow compiled onto the device)

The emitted text is plain MLIR: `mira emit -s mlir` prints it, and any MLIR tool
(mlir-opt, iree-opt, iree-compile) can consume it. The `mlir` target compiles it
with IREE for the CPU (llvm-cpu) or the Mac GPU (metal-spirv) and runs it.
"""
from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from .. import ir
from ..ops import conv_params
from ..types import TensorType

GELU_C = 0.7978845608028654
GELU_K = 0.044715

UNSUPPORTED = {"sort", "cumprod", "conv2d_grad_input", "conv2d_grad_weight", "maxpool2d_grad", "scatter_add"}
# (qconv2d with groups > 1 never occurs: the W8A8 pass keeps grouped convolutions in floating point)
I32_MAX, I32_MIN = 2**31 - 1, -2**31
REDUCE_ALIGN = 32      # see Emitter.reduce
CONV_LAYOUT = "nchw"   # see Emitter.conv: "nhwc" was slower with IREE 3.11 in our benchmarks
F32_MATH = {"exp", "log", "sqrt", "tanh", "sigmoid", "gelu", "erf", "pow", "div"}   # computed in fp32 for fp16


def is_int(dt: str) -> bool:
    return dt.startswith("i")


def ttype(t: TensorType) -> str:
    return "tensor<" + "".join(f"{d}x" for d in t.shape) + t.dtype + ">"


def scalar_lit(x: float, dtype: str) -> str:
    if is_int(dtype):
        return str(I32_MAX if x == np.inf else I32_MIN if x == -np.inf else int(x))
    x = float(x)
    if np.isinf(x):
        # IEEE bit patterns for +/- infinity
        return {("f32", True): "0x7F800000", ("f32", False): "0xFF800000",
                ("f16", True): "0x7C00", ("f16", False): "0xFC00"}[(dtype, x > 0)]
    return f"{x:.9e}"     # MLIR float literals need digits on both sides of the point


def dense(arr: np.ndarray) -> str:
    """A DenseElementsAttr literal; large tensors are written as a hex blob of their raw bytes."""
    flat = np.ascontiguousarray(arr)
    if flat.size == 1:
        v = flat.ravel()[0]
        if flat.dtype.kind == "i":
            return f"dense<{int(v)}>"
        return f"dense<{scalar_lit(float(v), 'f16' if flat.dtype == np.float16 else 'f32')}>"
    return f'dense<"0x{flat.astype(flat.dtype.newbyteorder("<")).tobytes().hex().upper()}">'


RESOURCE_MIN = 64   # constants bigger than this go into the module's resource section


class Emitter:
    def __init__(self):
        self.lines: list[str] = []
        self.ids = itertools.count()
        self.indent = "    "
        self.resources: dict[str, np.ndarray] = {}

    def const_attr(self, arr: np.ndarray) -> str:
        """Small constants inline; big ones as dense_resource blobs kept out of the code."""
        if arr.size <= RESOURCE_MIN:
            return dense(arr)
        name = f"w{len(self.resources)}"
        self.resources[name] = np.ascontiguousarray(arr)
        return f"dense_resource<{name}>"

    # ----- low-level helpers

    def fresh(self) -> str:
        return f"%t{next(self.ids)}"

    def emit(self, text: str) -> None:
        self.lines.append(self.indent + text)

    def empty(self, t: TensorType) -> str:
        r = self.fresh()
        self.emit(f"{r} = tensor.empty() : {ttype(t)}")
        return r

    def fill(self, t: TensorType, value: float) -> str:
        e = self.empty(t)
        c = self.fresh()
        self.emit(f"{c} = arith.constant {scalar_lit(value, t.dtype)} : {t.dtype}")
        r = self.fresh()
        self.emit(f"{r} = linalg.fill ins({c} : {t.dtype}) outs({e} : {ttype(t)}) -> {ttype(t)}")
        return r

    @staticmethod
    def dims(n: int) -> str:
        return ", ".join(f"d{i}" for i in range(n))

    def bmap(self, shape: tuple[int, ...], out: tuple[int, ...]) -> str:
        """Indexing map that broadcasts `shape` onto `out` (NumPy rules)."""
        n = len(out)
        lead = n - len(shape)
        res = []
        for i, d in enumerate(shape):
            res.append("0" if d == 1 and out[lead + i] != 1 else f"d{lead + i}")
        return f"affine_map<({self.dims(n)}) -> ({', '.join(res)})>"

    def generic(self, out_t: TensorType, operands: list[tuple[str, TensorType]], body) -> str:
        """Elementwise linalg.generic; `body(args) -> (lines, result)` works on scalars."""
        n = out_t.rank
        init = self.empty(out_t)
        maps = [self.bmap(t.shape, out_t.shape) for _, t in operands]
        maps.append(f"affine_map<({self.dims(n)}) -> ({self.dims(n)})>")
        iters = ", ".join(['"parallel"'] * n)
        ins = ", ".join(v for v, _ in operands)
        in_types = ", ".join(ttype(t) for _, t in operands)
        r = self.fresh()
        head = f"{r} = linalg.generic {{indexing_maps = [{', '.join(maps)}], iterator_types = [{iters}]}}"
        if operands:
            head += f" ins({ins} : {in_types})"
        self.emit(head + f" outs({init} : {ttype(out_t)}) {{")
        args = [f"%a{i}" for i in range(len(operands))]
        sig = ", ".join(f"{a}: {t.dtype}" for a, (_, t) in zip(args, operands))
        self.emit(f"^bb0({sig}{', ' if sig else ''}%out: {out_t.dtype}):")
        lines, res = body(args)
        for ln in lines:
            self.emit("  " + ln)
        self.emit(f"  linalg.yield {res} : {out_t.dtype}")
        self.emit(f"}} -> {ttype(out_t)}")
        return r

    def reshape(self, v: str, src: TensorType, dst: TensorType) -> str:
        """Any reshape = flatten to 1-D, then unflatten (MLIR only has collapse/expand of adjacent dims)."""
        if src.shape == dst.shape:
            return v
        flat = TensorType(src.dtype, (src.numel,))
        dims = lambda n: "[" + ", ".join(str(i) for i in range(n)) + "]"   # noqa: E731
        if src.rank == 0:
            r = self.fresh()
            self.emit(f"{r} = tensor.expand_shape {v} [] output_shape [1] : {ttype(src)} into {ttype(flat)}")
            v = r
        elif src.rank > 1:
            r = self.fresh()
            self.emit(f"{r} = tensor.collapse_shape {v} [{dims(src.rank)}] : {ttype(src)} into {ttype(flat)}")
            v = r
        if dst.rank == 0:
            r = self.fresh()
            self.emit(f"{r} = tensor.collapse_shape {v} [] : {ttype(flat)} into {ttype(dst)}")
            return r
        if dst.rank > 1:
            r = self.fresh()
            shape = ", ".join(map(str, dst.shape))
            self.emit(f"{r} = tensor.expand_shape {v} [{dims(dst.rank)}] output_shape [{shape}]"
                      f" : {ttype(flat)} into {ttype(dst)}")
            return r
        return v

    def reduce(self, v: str, t: TensorType, axes: tuple[int, ...], kind: str) -> tuple[str, TensorType]:
        """linalg.reduce over `axes` (keepdims=False) with add, max or min."""
        if kind in ("max", "min"):
            # Workaround: on x86, IREE 3.11 vectorizes a max/min reduction whose length isn't a multiple
            # of the vector width by padding the leftover lanes with 0 instead of the identity, so
            # max([-7.9, -8.8, ...]) of 38 values comes out as -0. Padding the axis ourselves with
            # -inf / +inf to a multiple of 32 leaves IREE nothing to pad. (Found by the fuzzer on CI.)
            hi = [(-d) % REDUCE_ALIGN if i in axes and d > 1 else 0 for i, d in enumerate(t.shape)]
            if any(hi):
                padded = TensorType(t.dtype, tuple(d + h for d, h in zip(t.shape, hi)))
                v = self.pad(v, t, padded, [0] * t.rank, hi, -np.inf if kind == "max" else np.inf)
                t = padded
        out_t = TensorType(t.dtype, tuple(d for i, d in enumerate(t.shape) if i not in axes))
        init = self.fill(out_t, {"add": 0.0, "max": -np.inf, "min": np.inf}[kind])
        r = self.fresh()
        op = ({"add": "arith.addi", "max": "arith.maxsi", "min": "arith.minsi"} if is_int(t.dtype) else
              {"add": "arith.addf", "max": "arith.maximumf", "min": "arith.minimumf"})[kind]
        self.emit(f"{r} = linalg.reduce ins({v} : {ttype(t)}) outs({init} : {ttype(out_t)}) "
                  f"dimensions = [{', '.join(map(str, axes))}]")
        self.emit(f"  (%in: {t.dtype}, %acc: {t.dtype}) {{")
        self.emit(f"    %s = {op} %in, %acc : {t.dtype}")
        self.emit(f"    linalg.yield %s : {t.dtype}")
        self.emit("  }")
        return r, out_t

    def keep(self, v: str, t: TensorType, full: tuple[int, ...], axes: tuple[int, ...]) -> tuple[str, TensorType]:
        kept = TensorType(t.dtype, tuple(1 if i in axes else d for i, d in enumerate(full)))
        return self.reshape(v, t, kept), kept

    def widen(self, v: str, t: TensorType) -> tuple[str, TensorType]:
        """fp16 -> fp32 (reductions and normalizations run in fp32, like the reference semantics)."""
        if t.dtype == "f32":
            return v, t
        wt = t.with_dtype("f32")
        return self.generic(wt, [(v, t)], lambda args: ([f"%r = arith.extf {args[0]} : f16 to f32"], "%r")), wt

    def narrow(self, v: str, t: TensorType, dtype: str) -> str:
        if dtype == t.dtype:
            return v
        return self.generic(t.with_dtype(dtype), [(v, t)],
                            lambda args: ([f"%r = arith.truncf {args[0]} : f32 to {dtype}"], "%r"))

    def const_scalar(self, x: float, dtype: str) -> tuple[list[str], str]:
        n = self.fresh().replace("%t", "%c")
        return [f"{n} = arith.constant {scalar_lit(x, dtype)} : {dtype}"], n

    # ----- scalar bodies

    def widened(self, inner, n_args: int):
        """Run an fp32 scalar body on fp16 values: extend the arguments, compute, round once at the end.

        Matches the reference semantics (fp32 math, rounded to fp16 once per op), and avoids
        compounding fp16 rounding inside multi-step functions like gelu.
        """
        def body(a):
            lines, wide = [], []
            for x in a[:n_args]:
                w = self.fresh().replace("%t", "%s")
                lines.append(f"{w} = arith.extf {x} : f16 to f32")
                wide.append(w)
            more, r = inner(wide)
            out = self.fresh().replace("%t", "%s")
            return lines + more + [f"{out} = arith.truncf {r} : f32 to f16"], out
        return body

    def unary_body(self, kind: str, dt: str):
        if dt == "f16" and kind in F32_MATH:
            return self.widened(self.unary_body(kind, "f32"), 1)

        def body(a):
            x = a[0]
            L: list[str] = []

            def c(val):
                ls, n = self.const_scalar(val, dt)
                L.extend(ls)
                return n

            def op(text):
                n = self.fresh().replace("%t", "%s")
                L.append(f"{n} = {text}")
                return n
            if is_int(dt) and kind in ("relu", "neg", "abs", "sign"):
                zero = c(0)
                if kind == "relu":
                    r = op(f"arith.maxsi {x}, {zero} : {dt}")
                elif kind == "neg":
                    r = op(f"arith.subi {zero}, {x} : {dt}")
                elif kind == "abs":
                    r = op(f"math.absi {x} : {dt}")
                else:
                    pos = op(f"arith.extui {op(f'arith.cmpi sgt, {x}, {zero} : {dt}')} : i1 to {dt}")
                    neg = op(f"arith.extui {op(f'arith.cmpi slt, {x}, {zero} : {dt}')} : i1 to {dt}")
                    r = op(f"arith.subi {pos}, {neg} : {dt}")
                return L, r
            if kind == "relu":
                r = op(f"arith.maximumf {x}, {c(0.0)} : {dt}")
            elif kind == "neg":
                r = op(f"arith.negf {x} : {dt}")
            elif kind in ("exp", "log", "sqrt", "tanh", "erf"):
                r = op(f"math.{kind} {x} : {dt}")
            elif kind == "abs":
                r = op(f"math.absf {x} : {dt}")
            elif kind == "sigmoid":
                e = op(f"math.exp {op(f'arith.negf {x} : {dt}')} : {dt}")
                r = op(f"arith.divf {c(1.0)}, {op(f'arith.addf {c(1.0)}, {e} : {dt}')} : {dt}")
            elif kind == "gelu":
                x3 = op(f"arith.mulf {op(f'arith.mulf {x}, {x} : {dt}')}, {x} : {dt}")
                inner = op(f"arith.addf {x}, {op(f'arith.mulf {c(GELU_K)}, {x3} : {dt}')} : {dt}")
                t = op(f"math.tanh {op(f'arith.mulf {c(GELU_C)}, {inner} : {dt}')} : {dt}")
                r = op(f"arith.mulf {op(f'arith.mulf {c(0.5)}, {x} : {dt}')}, "
                       f"{op(f'arith.addf {c(1.0)}, {t} : {dt}')} : {dt}")
            elif kind == "sign":
                pos = op(f"arith.uitofp {op(f'arith.cmpf ogt, {x}, {c(0.0)} : {dt}')} : i1 to {dt}")
                neg = op(f"arith.uitofp {op(f'arith.cmpf olt, {x}, {c(0.0)} : {dt}')} : i1 to {dt}")
                r = op(f"arith.subf {pos}, {neg} : {dt}")
            elif kind == "copy":
                r = x
            else:
                raise NotImplementedError(kind)
            return L, r
        return body

    def binary_body(self, kind: str, dt: str):
        if dt == "f16" and kind in F32_MATH:
            return self.widened(self.binary_body(kind, "f32"), 2)
        simple = {"add": "arith.addf", "sub": "arith.subf", "mul": "arith.mulf", "div": "arith.divf",
                  "pow": "math.powf", "maximum": "arith.maximumf", "minimum": "arith.minimumf"}
        preds = {"greater": "ogt", "greater_equal": "oge", "equal": "oeq"}

        if is_int(dt):
            simple = {"add": "arith.addi", "sub": "arith.subi", "mul": "arith.muli",
                      "maximum": "arith.maxsi", "minimum": "arith.minsi"}
            preds = {"greater": "sgt", "greater_equal": "sge", "equal": "eq"}

        def body(a):
            x, y = a
            n = self.fresh().replace("%t", "%s")
            if kind in simple:
                return [f"{n} = {simple[kind]} {x}, {y} : {dt}"], n
            b = self.fresh().replace("%t", "%s")
            if is_int(dt):
                return [f"{b} = arith.cmpi {preds[kind]}, {x}, {y} : {dt}", f"{n} = arith.extui {b} : i1 to {dt}"], n
            return [f"{b} = arith.cmpf {preds[kind]}, {x}, {y} : {dt}", f"{n} = arith.uitofp {b} : i1 to {dt}"], n
        return body

    # ----- graphs

    def graph_body(self, g: ir.Graph, env: dict[ir.Value, str]) -> list[str]:
        """Emit g's ops (with g.inputs already bound in env); return SSA names of its outputs."""
        for op in g.ops:
            if op.is_control:
                outs = self.control(op, env)
                env.update(zip(op.results, outs))
            else:
                env[op.result] = self.op(op, env)
        return [env[v] for v in g.outputs]

    def control(self, op: ir.Op, env: dict[ir.Value, str]) -> list[str]:
        types = [r.type for r in op.results]
        res_types = ", ".join(ttype(t) for t in types)
        n = len(types)
        if op.kind == "if":
            flag = self.truth(env[op.inputs[0]], op.inputs[0].type)
            r = self.fresh()
            then, orelse, k = op.attrs["then"], op.attrs["else"], op.attrs["n_then"]
            self.emit(f"{r}{':' + str(n) if n > 1 else ''} = scf.if {flag} -> ({res_types}) {{")
            for g, args in ((then, op.inputs[1:1 + k]), (orelse, op.inputs[1 + k:])):
                sub_env = {ph: env[a] for ph, a in zip(g.inputs, args)}
                saved = self.indent
                self.indent += "  "
                outs = self.graph_body(g, sub_env)
                self.emit(f"scf.yield {', '.join(outs)} : {res_types}")
                self.indent = saved
                if g is then:
                    self.emit("} else {")
            self.emit("}")
            return [f"{r}#{i}" if n > 1 else r for i in range(n)]

        ns, nc = op.attrs["n_state"], op.attrs["n_cond"]
        cond, body = op.attrs["cond"], op.attrs["body"]
        init = [env[v] for v in op.inputs[:ns]]
        ccaps, bcaps = op.inputs[ns:ns + nc], op.inputs[ns + nc:]
        r = self.fresh()
        before = [f"%w{next(self.ids)}" for _ in range(ns)]
        binds = ", ".join(f"{b} = {i}" for b, i in zip(before, init))
        self.emit(f"{r}{':' + str(n) if n > 1 else ''} = scf.while ({binds}) : ({res_types}) -> ({res_types}) {{")
        saved = self.indent
        self.indent += "  "
        sub_env = dict(zip(cond.inputs[:ns], before))
        sub_env.update({ph: env[a] for ph, a in zip(cond.inputs[ns:], ccaps)})
        c = self.graph_body(cond, sub_env)[0]
        flag = self.truth(c, cond.outputs[0].type)
        self.emit(f"scf.condition({flag}) {', '.join(before)} : {res_types}")
        self.indent = saved
        after = [f"%w{next(self.ids)}" for _ in range(ns)]
        self.emit("} do {")
        self.emit(f"^bb0({', '.join(f'{a}: {ttype(t)}' for a, t in zip(after, types))}):")
        self.indent += "  "
        sub_env = dict(zip(body.inputs[:ns], after))
        sub_env.update({ph: env[a] for ph, a in zip(body.inputs[ns:], bcaps)})
        outs = self.graph_body(body, sub_env)
        self.emit(f"scf.yield {', '.join(outs)} : {res_types}")
        self.indent = saved
        self.emit("}")
        return [f"{r}#{i}" if n > 1 else r for i in range(n)]

    def truth(self, v: str, t: TensorType) -> str:
        """i1 truth value of a one-element tensor. The comparison runs on the device and the
        host only reads back an i32 (IREE's host VM has no fp16 scalars)."""
        flag_t = TensorType("i32", t.shape)
        cmp = "arith.cmpi ne" if is_int(t.dtype) else "arith.cmpf une"
        flags = self.generic(flag_t, [(v, t)], lambda args: ([
            f"%z = arith.constant {scalar_lit(0, t.dtype)} : {t.dtype}",
            f"%b = {cmp}, {args[0]}, %z : {t.dtype}", "%i = arith.extui %b : i1 to i32"], "%i"))
        idx = []
        for _ in range(t.rank):
            z = self.fresh()
            self.emit(f"{z} = arith.constant 0 : index")
            idx.append(z)
        s_ = self.fresh()
        self.emit(f"{s_} = tensor.extract {flags}[{', '.join(idx)}] : {ttype(flag_t)}")
        zero = self.fresh()
        self.emit(f"{zero} = arith.constant 0 : i32")
        b = self.fresh()
        self.emit(f"{b} = arith.cmpi ne, {s_}, {zero} : i32")
        return b

    # ----- one op

    def op(self, op: ir.Op, env: dict[ir.Value, str]) -> str:
        k, a = op.kind, op.attrs
        out = op.result.type
        dt = out.dtype
        ins = [(env[v], v.type) for v in op.inputs]
        if k == "const":
            r = self.fresh()
            self.emit(f"{r} = arith.constant {self.const_attr(a['value'])} : {ttype(out)}")
            return r
        if k == "dequantize":
            q = self.fresh()
            qt = TensorType("i8", tuple(a["q"].shape))
            q_type = "tensor<" + "".join(f"{d}x" for d in qt.shape) + "i8>"
            self.emit(f"{q} = arith.constant {self.const_attr(a['q'])} : {q_type}")
            shape = [1] * out.rank
            shape[a["axis"]] = -1
            s = self.fresh()
            st = TensorType(dt, tuple(out.shape[i] if i == a["axis"] else 1 for i in range(out.rank)))
            self.emit(f"{s} = arith.constant {dense(a['scale'].astype(out.np_dtype).reshape(st.shape))} : {ttype(st)}")
            init = self.empty(out)
            n = out.rank
            r = self.fresh()
            idm = f"affine_map<({self.dims(n)}) -> ({self.dims(n)})>"
            iters = ", ".join(['"parallel"'] * n)
            self.emit(f"{r} = linalg.generic {{indexing_maps = [{idm}, {self.bmap(st.shape, out.shape)}, {idm}], "
                      f"iterator_types = [{iters}]}} "
                      f"ins({q}, {s} : tensor<{''.join(f'{d}x' for d in qt.shape)}i8>, {ttype(st)}) "
                      f"outs({init} : {ttype(out)}) {{")
            self.emit(f"^bb0(%q: i8, %sc: {dt}, %o: {dt}):")
            self.emit(f"  %f = arith.sitofp %q : i8 to {dt}")
            self.emit(f"  %m = arith.mulf %f, %sc : {dt}")
            self.emit(f"  linalg.yield %m : {dt}")
            self.emit(f"}} -> {ttype(out)}")
            return r
        if k in ("relu", "neg", "exp", "log", "sqrt", "tanh", "abs", "sigmoid", "gelu", "sign", "erf"):
            return self.generic(out, ins, self.unary_body(k, dt))
        if k in ("add", "sub", "mul", "div", "pow", "maximum", "minimum", "greater", "greater_equal", "equal"):
            return self.generic(out, ins, self.binary_body(k, dt))
        if k == "where":
            cdt = op.inputs[0].type.dtype
            cmp = "arith.cmpi ne" if is_int(cdt) else "arith.cmpf une"

            def body(args):
                c, x, y = args
                return ([f"%z = arith.constant {scalar_lit(0, cdt)} : {cdt}", f"%b = {cmp}, {c}, %z : {cdt}",
                         f"%r = arith.select %b, {x}, {y} : {dt}"], "%r")
            return self.generic(out, ins, body)
        if k == "cast":
            src = op.inputs[0].type.dtype
            if is_int(src) and is_int(dt):
                return ins[0][0]
            conv = ("arith.sitofp" if is_int(src) else "arith.fptosi" if is_int(dt) else
                    "arith.extf" if (src, dt) == ("f16", "f32") else "arith.truncf")
            return self.generic(out, ins, lambda args: ([f"%r = {conv} {args[0]} : {src} to {dt}"], "%r"))
        if k == "gather":
            return self.gather(ins[0], ins[1], out, a["axis"] % ins[0][1].rank)
        if k in ("argmax", "argmin"):
            return self.arg_reduce(ins[0], a["axis"] % ins[0][1].rank, a["keepdims"], k)
        if k == "dynamic_slice":
            (x, xt), (st, stt) = ins
            offs = self.offsets(st, stt, xt.shape, tuple(a["size"]))
            r = self.fresh()
            self.emit(f"{r} = tensor.extract_slice {x}[{', '.join(offs)}] [{', '.join(map(str, a['size']))}] "
                      f"[{', '.join(['1'] * xt.rank)}] : {ttype(xt)} to {ttype(out)}")
            return r
        if k == "dynamic_update_slice":
            (x, xt), (u, ut), (st, stt) = ins
            offs = self.offsets(st, stt, xt.shape, ut.shape)
            r = self.fresh()
            self.emit(f"{r} = tensor.insert_slice {u} into {x}[{', '.join(offs)}] [{', '.join(map(str, ut.shape))}] "
                      f"[{', '.join(['1'] * xt.rank)}] : {ttype(ut)} into {ttype(xt)}")
            return r
        if k == "broadcast":
            return self.generic(out, ins, self.unary_body("copy", dt))
        if k == "reshape":
            return self.reshape(ins[0][0], ins[0][1], out)
        if k == "transpose":
            init = self.empty(out)
            r = self.fresh()
            self.emit(f"{r} = linalg.transpose ins({ins[0][0]} : {ttype(ins[0][1])}) outs({init} : {ttype(out)}) "
                      f"permutation = [{', '.join(map(str, a['perm']))}]")
            return r
        if k == "slice":
            r = self.fresh()
            self.emit(f"{r} = tensor.extract_slice {ins[0][0]}[{', '.join(map(str, a['begin']))}] "
                      f"[{', '.join(map(str, a['size']))}] [{', '.join(['1'] * out.rank)}] : "
                      f"{ttype(ins[0][1])} to {ttype(out)}")
            return r
        if k == "pad":
            return self.pad(ins[0][0], ins[0][1], out, [lo for lo, _ in a["pads"]], [hi for _, hi in a["pads"]],
                            a.get("value", 0.0))
        if k == "concat" and out.rank == 1 and all(t.numel == 1 for _, t in ins):
            # Small vectors of scalars (e.g. a runtime start position [t, 0]) are built with
            # tensor.from_elements: IREE 3.11 miscompiles tensor.concat -> extract -> extract_slice.
            elems = []
            for v, t in ins:
                idx = []
                for _ in range(t.rank):
                    z = self.fresh()
                    self.emit(f"{z} = arith.constant 0 : index")
                    idx.append(z)
                e = self.fresh()
                self.emit(f"{e} = tensor.extract {v}[{', '.join(idx)}] : {ttype(t)}")
                elems.append(e)
            r = self.fresh()
            self.emit(f"{r} = tensor.from_elements {', '.join(elems)} : {ttype(out)}")
            return r
        if k == "concat":
            r = self.fresh()
            self.emit(f"{r} = tensor.concat dim({a['axis'] % out.rank}) {', '.join(v for v, _ in ins)} : "
                      f"({', '.join(ttype(t) for _, t in ins)}) -> {ttype(out)}")
            return r
        if k in ("reduce_sum", "reduce_mean", "reduce_max"):
            x, xt = self.widen(*ins[0])
            axes = tuple(sorted(ax % xt.rank for ax in a["axes"]))
            r, rt = self.reduce(x, xt, axes, "max" if k == "reduce_max" else "add")
            if k == "reduce_mean":
                count = int(np.prod([xt.shape[ax] for ax in axes]))
                r = self.generic(rt, [(r, rt)], lambda args: ([f"%n = arith.constant {scalar_lit(count, 'f32')} : f32",
                                                                f"%r = arith.divf {args[0]}, %n : f32"], "%r"))
            if a["keepdims"]:
                r, rt = self.keep(r, rt, xt.shape, axes)
            return self.narrow(r, rt, dt)
        if k == "softmax":
            x, xt = self.widen(*ins[0])
            ax = (a["axis"] % xt.rank,)
            m, mt = self.keep(*self.reduce(x, xt, ax, "max"), xt.shape, ax)
            e = self.generic(xt, [(x, xt), (m, mt)], lambda args: (
                [f"%d = arith.subf {args[0]}, {args[1]} : f32", "%r = math.exp %d : f32"], "%r"))
            s_, st = self.keep(*self.reduce(e, xt, ax, "add"), xt.shape, ax)
            return self.narrow(self.generic(xt, [(e, xt), (s_, st)], self.binary_body("div", "f32")), xt, dt)
        if k == "layernorm":
            (x, xt), (g, gt), (b, bt) = (self.widen(*i) for i in ins)
            ax = (xt.rank - 1,)
            d = xt.shape[-1]

            def mean_of(v):
                s1, st = self.reduce(v, xt, ax, "add")
                s1 = self.generic(st, [(s1, st)], lambda args: ([f"%n = arith.constant {scalar_lit(d, 'f32')} : f32",
                                                                 f"%r = arith.divf {args[0]}, %n : f32"], "%r"))
                return self.keep(s1, st, xt.shape, ax)
            mu, mt = mean_of(x)
            xc = self.generic(xt, [(x, xt), (mu, mt)], self.binary_body("sub", "f32"))
            sq = self.generic(xt, [(xc, xt), (xc, xt)], self.binary_body("mul", "f32"))
            var, vt = mean_of(sq)
            eps = a["eps"]
            y = self.generic(xt, [(xc, xt), (var, vt), (g, gt), (b, bt)], lambda args: ([
                f"%e = arith.constant {scalar_lit(eps, 'f32')} : f32",
                f"%v = arith.addf {args[1]}, %e : f32", "%s = math.rsqrt %v : f32",
                f"%n = arith.mulf {args[0]}, %s : f32", f"%m = arith.mulf %n, {args[2]} : f32",
                f"%r = arith.addf %m, {args[3]} : f32"], "%r"))
            return self.narrow(y, xt, dt)
        if k in ("qmatmul", "qconv2d"):
            return self.epilogue(op, self.int8_op(op, ins[0], out), out, ins)
        if k == "matmul":
            return self.epilogue(op, self.matmul(ins[0], ins[1], out), out, ins)
        if k == "conv2d":
            return self.epilogue(op, self.conv(ins[0], ins[1], out, a), out, ins)
        if k == "maxpool2d":
            x, xt = ins[0]
            s_ = a["size"]
            init = self.fill(out, -np.inf)
            win = self.fresh()
            self.emit(f"{win} = tensor.empty() : tensor<{s_}x{s_}x{dt}>")
            r = self.fresh()
            self.emit(f"{r} = linalg.pooling_nchw_max {{dilations = dense<1> : tensor<2xi64>, "
                      f"strides = dense<{a['stride']}> : tensor<2xi64>}} ins({x}, {win} : {ttype(xt)}, "
                      f"tensor<{s_}x{s_}x{dt}>) outs({init} : {ttype(out)}) -> {ttype(out)}")
            return r
        raise NotImplementedError(f"MLIR lowering for '{k}'")

    def gather(self, table: tuple[str, TensorType], idx: tuple[str, TensorType], out: TensorType, ax: int) -> str:
        """out[..., i..., ...] = table[..., idx[i...], ...] as a linalg.generic that reads the table directly."""
        (tv, tt), (iv, it) = table, idx
        if it.rank == 0:
            # One position: a dynamic one-row slice. (IREE 3.11 can't vectorize a 0-d index gather.)
            i, ii = self.fresh(), self.fresh()
            self.emit(f"{i} = tensor.extract {iv}[] : {ttype(it)}")
            self.emit(f"{ii} = arith.index_cast {i} : i32 to index")
            offs = ["0"] * tt.rank
            offs[ax] = ii
            sizes = [str(d) for d in tt.shape]
            sizes[ax] = "1"
            row_t = TensorType(tt.dtype, tuple(1 if j == ax else d for j, d in enumerate(tt.shape)))
            r = self.fresh()
            self.emit(f"{r} = tensor.extract_slice {tv}[{', '.join(offs)}] [{', '.join(sizes)}] "
                      f"[{', '.join(['1'] * tt.rank)}] : {ttype(tt)} to {ttype(row_t)}")
            return self.reshape(r, row_t, out)
        n = out.rank
        idx_dims = ", ".join(f"d{ax + j}" for j in range(it.rank))
        maps = [f"affine_map<({self.dims(n)}) -> ({idx_dims})>", f"affine_map<({self.dims(n)}) -> ({self.dims(n)})>"]
        init = self.empty(out)
        r = self.fresh()
        iters = ", ".join(['"parallel"'] * n)
        self.emit(f"{r} = linalg.generic {{indexing_maps = [{', '.join(maps)}], iterator_types = [{iters}]}} "
                  f"ins({iv} : {ttype(it)}) outs({init} : {ttype(out)}) {{")
        self.emit(f"^bb0(%i: i32, %o: {out.dtype}):")
        self.emit("  %ii = arith.index_cast %i : i32 to index")
        coords = []
        for j in range(tt.rank):
            if j == ax:
                coords.append("%ii")
            else:
                pos = j if j < ax else j - 1 + it.rank
                self.emit(f"  %d{j} = linalg.index {pos} : index")
                coords.append(f"%d{j}")
        self.emit(f"  %v = tensor.extract {tv}[{', '.join(coords)}] : {ttype(tt)}")
        self.emit(f"  linalg.yield %v : {out.dtype}")
        self.emit(f"}} -> {ttype(out)}")
        return r

    def arg_reduce(self, x: tuple[str, TensorType], ax: int, keepdims: bool, kind: str) -> str:
        """argmax/argmin = reduce to the extreme value, then the smallest index that attains it.

        The index reduction runs in f32 (exact below 2^24): IREE 3.11's CPU microkernel path fails
        to compile an i32 min-reduction (`linalg.reduce` with `arith.minsi`).
        """
        xv, xt = x
        best, bt = self.keep(*self.reduce(xv, xt, (ax,), "max" if kind == "argmax" else "min"), xt.shape, (ax,))
        cand_t = TensorType("f32", xt.shape)
        eq = "arith.cmpi eq" if is_int(xt.dtype) else "arith.cmpf oeq"

        def body(args):
            return ([f"%e = {eq}, {args[0]}, {args[1]} : {xt.dtype}", f"%p = linalg.index {ax} : index",
                     "%pi = arith.index_cast %p : index to i32", "%pf = arith.sitofp %pi : i32 to f32",
                     "%big = arith.constant 3.0e+38 : f32", "%r = arith.select %e, %pf, %big : f32"], "%r")
        cand = self.generic(cand_t, [(xv, xt), (best, bt)], body)
        r, rt = self.reduce(cand, cand_t, (ax,), "min")
        r = self.generic(rt.with_dtype("i32"), [(r, rt)], lambda args: ([f"%r = arith.fptosi {args[0]} : f32 to i32"],
                                                                         "%r"))
        rt = rt.with_dtype("i32")
        if keepdims:
            r, rt = self.keep(r, rt, xt.shape, (ax,))
        return r

    def offsets(self, start: str, st: TensorType, shape: tuple[int, ...], size: tuple[int, ...]) -> list[str]:
        """Runtime slice offsets read from an i32 vector and clamped into bounds (like the reference)."""
        out = []
        for i, (d, n) in enumerate(zip(shape, size)):
            ci, s, lo, hi, a, b, o = (self.fresh() for _ in range(7))
            self.emit(f"{ci} = arith.constant {i} : index")
            self.emit(f"{s} = tensor.extract {start}[{ci}] : {ttype(st)}")
            self.emit(f"{lo} = arith.constant 0 : i32")
            self.emit(f"{hi} = arith.constant {d - n} : i32")
            self.emit(f"{a} = arith.maxsi {s}, {lo} : i32")
            self.emit(f"{b} = arith.minsi {a}, {hi} : i32")
            self.emit(f"{o} = arith.index_cast {b} : i32 to index")
            out.append(o)
        return out

    def pad(self, v: str, t: TensorType, out: TensorType, lo: list[int], hi: list[int], value: float) -> str:
        r = self.fresh()
        idx = ", ".join(f"%i{i}: index" for i in range(t.rank))
        self.emit(f"{r} = tensor.pad {v} low[{', '.join(map(str, lo))}] high[{', '.join(map(str, hi))}] {{")
        self.emit(f"^bb0({idx}):")
        self.emit(f"  %p = arith.constant {scalar_lit(value, t.dtype)} : {t.dtype}")
        self.emit(f"  tensor.yield %p : {t.dtype}")
        self.emit(f"}} : {ttype(t)} to {ttype(out)}")
        return r

    def matmul(self, x: tuple[str, TensorType], w: tuple[str, TensorType], out: TensorType) -> str:
        (xv, xt), (wv, wt) = x, w
        acc_dt = "f32"     # accumulate in fp32 even for fp16 inputs, like the NPUs do
        if wt.rank == 2:
            m = int(np.prod(xt.shape[:-1]))
            x2 = self.reshape(xv, xt, TensorType(xt.dtype, (m, xt.shape[-1])))
            at = TensorType(acc_dt, (m, wt.shape[1]))
            init = self.fill(at, 0.0)
            r = self.fresh()
            self.emit(f"{r} = linalg.matmul ins({x2}, {wv} : {ttype(TensorType(xt.dtype, (m, xt.shape[-1])))}, "
                      f"{ttype(wt)}) outs({init} : {ttype(at)}) -> {ttype(at)}")
        else:
            batch = out.shape[:-2]
            b = int(np.prod(batch))
            xs = TensorType(xt.dtype, batch + xt.shape[-2:])
            ws = TensorType(wt.dtype, batch + wt.shape[-2:])
            if xt.shape != xs.shape:
                xv = self.generic(xs, [(xv, xt)], self.unary_body("copy", xt.dtype))
            if wt.shape != ws.shape:
                wv = self.generic(ws, [(wv, wt)], self.unary_body("copy", wt.dtype))
            x3t = TensorType(xt.dtype, (b,) + xt.shape[-2:])
            w3t = TensorType(wt.dtype, (b,) + wt.shape[-2:])
            x3 = self.reshape(xv, xs, x3t)
            w3 = self.reshape(wv, ws, w3t)
            at = TensorType(acc_dt, (b, out.shape[-2], out.shape[-1]))
            init = self.fill(at, 0.0)
            r = self.fresh()
            self.emit(f"{r} = linalg.batch_matmul ins({x3}, {w3} : {ttype(x3t)}, {ttype(w3t)}) "
                      f"outs({init} : {ttype(at)}) -> {ttype(at)}")
        final_acc = TensorType(acc_dt, out.shape)
        r = self.reshape(r, at, final_acc)
        if out.dtype != acc_dt:
            r = self.generic(out, [(r, final_acc)], lambda args: ([f"%r = arith.truncf {args[0]} : f32 to {out.dtype}"],
                                                                   "%r"))
        return r

    def conv(self, x: tuple[str, TensorType], w: tuple[str, TensorType], out: TensorType, a: dict) -> str:
        (xv, xt), (wv, wt) = x, w
        (sh, sw), (ph, pw), (dh, dw), groups = conv_params(a)
        if xt.dtype != "f32":
            # Widen fp16 operands explicitly: IREE 3.11's CPU codegen fails on some mixed-precision
            # (f16 x f16 -> f32) convolutions ("slice ... runs out-of-bounds"). Constant weights fold.
            x32, w32 = xt.with_dtype("f32"), wt.with_dtype("f32")
            ext = lambda args: ([f"%r = arith.extf {args[0]} : f16 to f32"], "%r")   # noqa: E731
            xv, xt = self.generic(x32, [(xv, xt)], ext), x32
            wv, wt = self.generic(w32, [(wv, wt)], ext), w32
        if ph or pw:
            n, c, h, wd = xt.shape
            pt = TensorType(xt.dtype, (n, c, h + 2 * ph, wd + 2 * pw))
            xv = self.pad(xv, xt, pt, [0, 0, ph, pw], [0, 0, ph, pw], 0.0)
            xt = pt
        at = TensorType("f32", out.shape)
        attrs = (f"{{dilations = dense<[{dh}, {dw}]> : tensor<2xi64>, "
                 f"strides = dense<[{sh}, {sw}]> : tensor<2xi64>}}")
        if groups == 1 and CONV_LAYOUT == "nhwc":
            # IREE's fast convolution paths are channels-last: transpose in and out (consecutive convs' transposes
            # cancel, and the weight transpose folds into the constant at compile time).
            n, c, h, wd = xt.shape
            o, _, kh, kw = wt.shape
            x_t = TensorType(xt.dtype, (n, h, wd, c))
            w_t = TensorType(wt.dtype, (kh, kw, c, o))
            a_t = TensorType("f32", (n,) + out.shape[2:] + (o,))
            xv, wv = self.transpose(xv, xt, x_t, (0, 2, 3, 1)), self.transpose(wv, wt, w_t, (2, 3, 1, 0))
            init = self.fill(a_t, 0.0)
            r = self.fresh()
            self.emit(f"{r} = linalg.conv_2d_nhwc_hwcf {attrs} ins({xv}, {wv} : {ttype(x_t)}, {ttype(w_t)}) "
                      f"outs({init} : {ttype(a_t)}) -> {ttype(a_t)}")
            r = self.transpose(r, a_t, at, (0, 3, 1, 2))
        elif groups == 1:
            init = self.fill(at, 0.0)
            r = self.fresh()
            self.emit(f"{r} = linalg.conv_2d_nchw_fchw {attrs} ins({xv}, {wv} : {ttype(xt)}, {ttype(wt)}) "
                      f"outs({init} : {ttype(at)}) -> {ttype(at)}")
        else:   # grouped / depthwise: split the channel axes into (group, channels-per-group)
            n, c, h, wd = xt.shape
            o, cg, kh, kw = wt.shape
            xg = TensorType(xt.dtype, (n, groups, c // groups, h, wd))
            wg = TensorType(wt.dtype, (groups, o // groups, cg, kh, kw))
            ag = TensorType("f32", (n, groups, o // groups) + out.shape[2:])
            xv, wv = self.reshape(xv, xt, xg), self.reshape(wv, wt, wg)
            init = self.fill(ag, 0.0)
            r = self.fresh()
            self.emit(f"{r} = linalg.conv_2d_ngchw_gfchw {attrs} ins({xv}, {wv} : {ttype(xg)}, {ttype(wg)}) "
                      f"outs({init} : {ttype(ag)}) -> {ttype(ag)}")
            r = self.reshape(r, ag, at)
        if out.dtype != "f32":
            r = self.generic(out, [(r, at)], lambda args: ([f"%r = arith.truncf {args[0]} : f32 to {out.dtype}"], "%r"))
        return r

    def int8_op(self, op: ir.Op, x: tuple[str, TensorType], out: TensorType) -> str:
        """W8A8: quantize x to i8 with the calibrated scale, multiply i8 x i8 with i32 sums, rescale to float."""
        a = op.attrs
        xv, xt = x
        xs = float(a["x_scale"])
        qt = TensorType("i8", xt.shape)
        dt = xt.dtype
        xq = self.generic(qt, [(xv, xt)], lambda args: ([
            f"%w = arith.extf {args[0]} : {dt} to f32" if dt != "f32" else f"%w = arith.addf {args[0]}, %zero : f32",
            f"%s = arith.constant {scalar_lit(xs, 'f32')} : f32", "%d = arith.divf %w, %s : f32",
            "%r = math.roundeven %d : f32", f"%hi = arith.constant {scalar_lit(127.0, 'f32')} : f32",
            f"%lo = arith.constant {scalar_lit(-127.0, 'f32')} : f32", "%c1 = arith.minimumf %r, %hi : f32",
            "%c2 = arith.maximumf %c1, %lo : f32", "%q = arith.fptosi %c2 : f32 to i8"], "%q")
                          if dt != "f32" else ([
            f"%s = arith.constant {scalar_lit(xs, 'f32')} : f32", f"%d = arith.divf {args[0]}, %s : f32",
            "%r = math.roundeven %d : f32", f"%hi = arith.constant {scalar_lit(127.0, 'f32')} : f32",
            f"%lo = arith.constant {scalar_lit(-127.0, 'f32')} : f32", "%c1 = arith.minimumf %r, %hi : f32",
            "%c2 = arith.maximumf %c1, %lo : f32", "%q = arith.fptosi %c2 : f32 to i8"], "%q"))
        q = a["q"]
        wt = TensorType("i8", tuple(q.shape))
        wv = self.fresh()
        self.emit(f"{wv} = arith.constant {self.const_attr(q)} : {ttype(wt)}")
        if op.kind == "qmatmul":
            m = int(np.prod(xt.shape[:-1]))
            q2 = TensorType("i8", (m, xt.shape[-1]))
            acc_t = TensorType("i32", (m, q.shape[1]))
            init = self.fill(acc_t, 0)
            r = self.fresh()
            self.emit(f"{r} = linalg.matmul ins({self.reshape(xq, qt, q2)}, {wv} : {ttype(q2)}, {ttype(wt)}) "
                      f"outs({init} : {ttype(acc_t)}) -> {ttype(acc_t)}")
            scale_shape = (1, q.shape[1])
        else:
            (sh, sw), (ph, pw), (dh, dw), groups = conv_params(a)
            # IREE 3.11's CPU codegen fails ("slice ... runs out-of-bounds") on integer convolutions, and on any
            # convolution whose input is an elementwise op applied *after* tensor.pad. So: convert the int8
            # values to f32 first (f32 holds these integer products and sums exactly, up to 2^24), then pad,
            # then convolve. The result is the same as an i8 x i8 -> i32 convolution.
            ext = lambda args: (["%e = arith.sitofp " + args[0] + " : i8 to f32"], "%e")   # noqa: E731
            qf, wf = qt.with_dtype("f32"), wt.with_dtype("f32")
            xq, wv, qt, wt = self.generic(qf, [(xq, qt)], ext), self.generic(wf, [(wv, wt)], ext), qf, wf
            if ph or pw:
                n, c, h, wd = xt.shape
                pt = TensorType("f32", (n, c, h + 2 * ph, wd + 2 * pw))
                xq, qt = self.pad(xq, qt, pt, [0, 0, ph, pw], [0, 0, ph, pw], 0.0), pt
            acc_t = TensorType("f32", out.shape)
            init = self.fill(acc_t, 0.0)
            r = self.fresh()
            self.emit(f"{r} = linalg.conv_2d_nchw_fchw {{dilations = dense<[{dh}, {dw}]> : tensor<2xi64>, "
                      f"strides = dense<[{sh}, {sw}]> : tensor<2xi64>}} ins({xq}, {wv} : {ttype(qt)}, {ttype(wt)}) "
                      f"outs({init} : {ttype(acc_t)}) -> {ttype(acc_t)}")
            scale_shape = (1, q.shape[0], 1, 1)
        combined = (np.float64(xs) * a["w_scale"].astype(np.float64)).astype(np.float32).reshape(scale_shape)
        st = TensorType("f32", scale_shape)
        sv = self.fresh()
        self.emit(f"{sv} = arith.constant {self.const_attr(combined)} : {ttype(st)}")
        y_t = TensorType(out.dtype, acc_t.shape)
        conv = "" if out.dtype == "f32" else f"%y = arith.truncf %m : f32 to {out.dtype}"
        to_f = ("%f = arith.sitofp {} : i32 to f32" if acc_t.dtype == "i32" else "%f = arith.addf {}, %fz : f32")
        y = self.generic(y_t, [(r, acc_t), (sv, st)], lambda args: (
            ([] if acc_t.dtype == "i32" else ["%fz = arith.constant 0.0 : f32"]) + [to_f.format(args[0]),
             f"%m = arith.mulf %f, {args[1]} : f32"] + ([conv] if conv else []),
            "%y" if conv else "%m"))
        return self.reshape(y, y_t, out)

    def transpose(self, v: str, t: TensorType, out: TensorType, perm: tuple[int, ...]) -> str:
        init = self.empty(out)
        r = self.fresh()
        self.emit(f"{r} = linalg.transpose ins({v} : {ttype(t)}) outs({init} : {ttype(out)}) "
                  f"permutation = [{', '.join(map(str, perm))}]")
        return r

    def epilogue(self, op: ir.Op, acc: str, out: TensorType, ins) -> str:
        for fn, idx, swapped in op.attrs.get("epilogue", ()):
            if idx is None:
                acc = self.generic(out, [(acc, out)], self.unary_body(fn, out.dtype))
            else:
                pair = [(acc, out), ins[idx]]
                acc = self.generic(out, pair[::-1] if swapped else pair, self.binary_body(fn, out.dtype))
        return acc


def emit_module(g: ir.Graph, func_name: str = "main") -> str:
    """The whole graph as one MLIR module with a single public function."""
    em = Emitter()
    env = {}
    for i, v in enumerate(g.inputs):
        if v.type.rank == 0:   # scalars cross the module boundary as [1] (IREE drops 0-d host arrays)
            env[v] = em.reshape(f"%arg{i}", TensorType(v.type.dtype, (1,)), v.type)
        else:
            env[v] = f"%arg{i}"
    outs = em.graph_body(g, env)
    args = ", ".join(f"%arg{i}: {ttype(TensorType(v.type.dtype, v.type.shape or (1,)))}"
                     for i, v in enumerate(g.inputs))
    res = ", ".join(ttype(v.type) for v in g.outputs)
    head = ["module {", f"  func.func @{func_name}({args}) -> ({res}) {{"]
    body = em.lines + [f"    return {', '.join(outs)} : {res}", "  }", "}"]
    if em.resources:
        # blob format: 4-byte little-endian alignment, then the raw little-endian data
        blobs = [f'      {name}: "0x10000000{arr.astype(arr.dtype.newbyteorder("<")).tobytes().hex().upper()}"'
                 for name, arr in em.resources.items()]
        body += ["", "{-#", "  dialect_resources: {", "    builtin: {", ",\n".join(blobs), "    }", "  }", "#-}"]
    return "\n".join(head + body) + "\n"


def supported(op: ir.Op) -> Optional[str]:
    if op.kind in UNSUPPORTED:
        return f"{op.kind}: no MLIR lowering"
    for _, sub in op.subgraphs():
        for inner in sub.compute_ops():
            reason = supported(inner)
            if reason:
                return f"control flow body contains {reason}"
    return None


# ------------------------------------------------------------------ target: compile with IREE and run

IREE_BACKENDS = {"cpu": ("llvm-cpu", "local-task"), "metal": ("metal-spirv", "metal")}


class MLIRExecutable:
    def __init__(self, g: ir.Graph, backend: str):
        import iree.compiler as ireec
        import iree.runtime as ireert
        from iree.runtime import _binding
        if hasattr(_binding, "disable_leak_checker"):
            _binding.disable_leak_checker()   # otherwise nanobind prints harmless "leaked" noise at exit
        self.g = g
        self.backend = backend
        self.text = emit_module(g)
        target, driver = IREE_BACKENDS[backend]
        # data tiling + microkernels: IREE packs matmul operands into cache-friendly tiles and calls
        # hand-tuned inner loops (about 4x faster than plain codegen for fp16 here)
        extra = (["--iree-llvmcpu-target-cpu=host", "--iree-opt-data-tiling", "--iree-llvmcpu-enable-ukernels=all"]
                 if target == "llvm-cpu" else [])
        try:
            vmfb = ireec.compile_str(self.text, target_backends=[target], extra_args=extra)
        except ireec.CompilerToolError as e:
            raise RuntimeError(f"IREE failed to compile the emitted MLIR:\n{str(e)[:4000]}") from None
        config = ireert.Config(driver)
        self.ctx = ireert.SystemContext(config=config)
        self.ctx.add_vm_module(ireert.VmModule.copy_buffer(self.ctx.instance, vmfb))
        self.fn = self.ctx.modules.module["main"]
        self.vmfb_bytes = len(vmfb)

    def run(self, feeds: dict[ir.Value, np.ndarray]) -> dict[ir.Value, np.ndarray]:
        args = [np.ascontiguousarray(feeds[v], dtype=v.type.np_dtype).reshape(v.type.shape or (1,))
                for v in self.g.inputs]
        res = self.fn(*args)
        res = res if isinstance(res, (tuple, list)) else [res]
        return {v: np.asarray(r.to_host() if hasattr(r, "to_host") else r).reshape(v.type.shape).astype(v.type.np_dtype)
                for v, r in zip(self.g.outputs, res)}

    def report(self) -> str:
        return (f"mlir/iree ({IREE_BACKENDS[self.backend][0]}): {len(self.g.compute_ops())} ops, "
                f"{len(self.text.splitlines())} lines of MLIR, {self.vmfb_bytes / 1024:.0f} KiB module")


class MLIRTarget:
    name = "mlir"
    supports_control_flow = True    # scf.if / scf.while compile onto the device

    def __init__(self, backend: str = "cpu"):
        if backend not in IREE_BACKENDS:
            raise ValueError(f"unknown IREE backend '{backend}' (choose {', '.join(IREE_BACKENDS)})")
        self.backend = backend

    def check(self, op: ir.Op) -> Optional[str]:
        return supported(op)

    def compile(self, g: ir.Graph) -> MLIRExecutable:
        return MLIRExecutable(g, self.backend)
