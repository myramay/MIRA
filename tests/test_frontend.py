import glob

import numpy as np
import pytest

import mira
from mira import syntax
from mira.lexer import tokenize
from mira.parser import parse

EXAMPLES = sorted(glob.glob("examples/*.mira"))


def test_lexer_basics():
    toks = tokenize("let y = x @ w1 ** 2.5e-1 # comment\n")
    assert [(t.kind, t.text) for t in toks] == [
        ("kw", "let"), ("ident", "y"), ("op", "="), ("ident", "x"), ("op", "@"), ("ident", "w1"),
        ("op", "**"), ("float", "2.5e-1"), ("newline", "\\n"), ("eof", "")]


def test_lexer_range_is_not_a_float():
    kinds = [(t.kind, t.text) for t in tokenize("0..8")][:3]
    assert kinds == [("int", "0"), ("op", ".."), ("int", "8")]


def test_newlines_inside_parens_are_ignored():
    toks = tokenize("f(a,\n  b)\n")
    assert [t.kind for t in toks].count("newline") == 1


def test_precedence():
    m = parse("fn main(x: f32[2, 2]) -> f32[2, 2] {\n return -x ** 2 + x @ x * 3\n}\n")
    ret = m.functions["main"].body[0].value
    assert syntax.format_expr(ret) == "-x ** 2 + x @ x * 3"
    assert isinstance(ret, syntax.Binary) and ret.op == "+"
    assert isinstance(ret.lhs, syntax.Unary)             # -(x ** 2)
    assert isinstance(ret.rhs, syntax.Binary) and ret.rhs.op == "*"   # (x @ x) * 3


@pytest.mark.parametrize("path", EXAMPLES)
def test_pretty_print_round_trip(path):
    src = open(path).read()
    once = syntax.format_module(parse(src))
    twice = syntax.format_module(parse(once))
    assert once == twice


def compile_err(src: str) -> str:
    with pytest.raises(mira.MiraError) as ei:
        mira.compile_source(src)
    return ei.value.render(src)


def test_error_shape_mismatch_points_at_operator():
    msg = compile_err("fn main(a: f32[4, 3], b: f32[4, 5]) -> f32[4, 3] {\n  return a + b\n}\n")
    assert "can't be broadcast" in msg and "2:12" in msg


def test_error_matmul_inner_dims():
    msg = compile_err("fn main(a: f32[4, 3], b: f32[4, 5]) -> f32[4, 5] {\n  return a @ b\n}\n")
    assert "inner dimensions don't match" in msg


def test_error_unknown_function_suggests():
    msg = compile_err("fn main(a: f32[4]) -> f32[4] {\n  return rleu(a)\n}\n")
    assert "unknown function 'rleu'" in msg and "did you mean 'relu'" in msg


def test_error_undefined_variable_suggests():
    msg = compile_err("fn main(alpha: f32[4]) -> f32[4] {\n  return relu(alpah)\n}\n")
    assert "undefined name 'alpah'" in msg and "alpha" in msg


def test_error_generic_binding_conflict():
    src = ("fn f(x: f32[N, N]) -> f32[N, N] {\n  return x\n}\n"
           "fn main(a: f32[3, 4]) -> f32[3, 4] {\n  return f(a)\n}\n")
    assert "N is already 3, but got 4" in compile_err(src)


def test_error_recursion():
    src = ("fn f(x: f32[2]) -> f32[2] {\n  return f(x)\n}\n"
           "fn main(a: f32[2]) -> f32[2] {\n  return f(a)\n}\n")
    assert "recursive call" in compile_err(src)


def test_error_return_type():
    assert "signature says" in compile_err("fn main(a: f32[2]) -> f32[3] {\n  return a\n}\n")


def test_error_syntax():
    assert "expected ')'" in compile_err("fn main(a: f32[2]) -> f32[2] {\n  return relu(a\n}\n")


def test_generic_functions_are_specialized_per_call():
    src = """
fn double(x: f32[N]) -> f32[N] {
  return x * 2
}
fn main(a: f32[3], b: f32[5]) -> f32[8] {
  return concat([double(a), double(b)])
}
"""
    p = mira.compile_source(src)
    out = p.run({"a": np.ones(3, np.float32), "b": np.ones(5, np.float32)})
    np.testing.assert_allclose(out, 2 * np.ones(8))


def test_loops_unroll_and_reassign():
    src = """
fn main(x: f32[4]) -> f32[4] {
  let h = x
  for i in 0..5 {
    h = h + i
  }
  return h
}
"""
    out = mira.compile_source(src).run({"x": np.zeros(4, np.float32)})
    np.testing.assert_allclose(out, np.full(4, 0 + 1 + 2 + 3 + 4))


def test_int_params_and_computed_dims():
    src = """
fn halves(x: f32[N], k: int) -> f32[k, N / k] {
  return reshape(x, [k, N / k])
}
fn main(x: f32[12]) -> f32[3, 4] {
  return halves(x, 3)
}
"""
    out = mira.compile_source(src).run({"x": np.arange(12, dtype=np.float32)})
    np.testing.assert_allclose(out, np.arange(12).reshape(3, 4))


def test_entry_dims_from_caller():
    src = "fn main(x: f32[B, 4]) -> f32[B, 4] {\n  return relu(x)\n}\n"
    p = mira.compile_source(src, dims={"B": 7})
    assert p.graph.inputs[0].type.shape == (7, 4)
    with pytest.raises(mira.MiraError, match="dimension 'B' of the entry function"):
        mira.compile_source(src, dynamic=False)
    assert isinstance(mira.compile_source(src), mira.DynamicProgram)


def test_weights_are_baked_in_as_constants():
    src = "fn main(x: f32[2, 3], const w: f32[3, 2]) -> f32[2, 2] {\n  return x @ w\n}\n"
    w = np.arange(6, dtype=np.float32).reshape(3, 2)
    p = mira.compile_source(src, weights={"w": w})
    assert [v.name for v in p.graph.inputs] == ["x"]
    x = np.ones((2, 3), np.float32)
    np.testing.assert_allclose(p.run({"x": x}), x @ w)
