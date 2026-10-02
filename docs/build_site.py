"""Build the live demo site (published to GitHub Pages by .github/workflows/pages.yml).

    python docs/build_site.py site      # writes site/index.html and one interactive page per demo

Every page is a real MNPU-1 simulation run produced by `mira emit ... -s html`, so the site
always matches the current code. Only NumPy is needed (no Core ML, IREE, or PyTorch).
"""
import html
import os
import sys

from mira.cli import main as mira

REPO = "https://github.com/myramay/MIRA"

DEMOS = [
    ("mlp-double-buffering.html", "Double buffering, on vs off",
     "A 3-layer MLP run twice. With double buffering the chip loads the next tile of data while the "
     "matrix unit works on the current one: blue transfers slide underneath orange compute and the run "
     "finishes 1.25× sooner. Without it, the two take turns.",
     ["examples/mlp.mira", "--compare", "no-double-buffer"]),
    ("attention-bigger-mxu.html", "A transformer block on a bigger matrix unit",
     "One transformer encoder block (attention + MLP) on the standard 32×32 systolic array versus a 64×64 one. "
     "Hover the IR-ops lane to see layernorm, the query/key/value matmuls, softmax, and the head reshuffles.",
     ["examples/attention.mira", "--compare", "mxu_dim=64"]),
    ("cnn.html", "A convolutional network, instruction by instruction",
     "A small image classifier with about 16,000 instructions. Convolutions run as implicit GEMM: one small matmul "
     "per kernel tap, fed by strided DMA. Zoom in to see the repeating pattern, and notice the matrix unit is only "
     "~30% busy: the first layer's matmuls are too small for a 32×32 array.",
     ["examples/cnn.mira"]),
    ("resnet-loop.html", "Eight residual blocks, unrolled",
     "A deep residual MLP written with a compile-time loop. Each block's matmul carries its bias, ReLU and residual "
     "add in a fused epilogue, so intermediate results never leave the chip.",
     ["examples/resnet_loop.mira"]),
]

INDEX = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MIRA: a compiler for neural processing units</title>
<style>
  :root { --bg: #f7f7f5; --panel: #fff; --ink: #1d1d1f; --muted: #6b6b70; --line: #e4e4e0; --accent: #2f6fdf;
          color-scheme: light; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #141416; --panel: #1d1d20; --ink: #ececee; --muted: #9c9ca3; --line: #2e2e33; --accent: #6b9cf5;
            color-scheme: dark; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--ink);
         font: 16px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif; }
  main { max-width: 920px; margin: 0 auto; padding: 48px 20px 64px; }
  h1 { font-size: 34px; letter-spacing: -0.02em; margin: 0 0 6px; }
  .lede { font-size: 19px; color: var(--muted); margin: 0 0 22px; max-width: 46em; }
  a { color: var(--accent); }
  .links { display: flex; flex-wrap: wrap; gap: 10px; margin: 0 0 34px; }
  .btn { display: inline-block; padding: 8px 14px; border-radius: 8px; border: 1px solid var(--line);
         background: var(--panel); color: var(--ink); text-decoration: none; font-size: 15px; }
  .btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
  h2 { font-size: 21px; margin: 34px 0 12px; }
  ul.facts { padding-left: 20px; margin: 0; }
  ul.facts li { margin: 4px 0; }
  .demos { display: grid; gap: 14px; }
  .demo { display: block; background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
          padding: 16px 18px; color: inherit; text-decoration: none; transition: border-color .15s; }
  .demo:hover { border-color: var(--accent); }
  .demo h3 { margin: 0 0 4px; font-size: 17px; color: var(--accent); }
  .demo p { margin: 0; color: var(--muted); font-size: 15px; }
  .key { display: inline-flex; align-items: center; gap: 6px; margin-right: 14px; font-size: 14px; color: var(--muted); }
  .sw { width: 11px; height: 11px; border-radius: 3px; display: inline-block; }
  footer { margin-top: 44px; color: var(--muted); font-size: 14px; }
</style>
</head>
<body>
<main>
  <h1>MIRA</h1>
  <p class="lede">A small programming language and compiler for running neural networks on <b>neural processing
  units</b>: the AI chips inside phones and laptops. One program compiles to Apple's Neural Engine, to standard
  MLIR, and to <b>MNPU-1</b>, a simulated NPU you can watch work, instruction by instruction, below.</p>
  <div class="links">
    <a class="btn primary" href="__FIRST__">Open the first demo</a>
    <a class="btn" href="__REPO__">Source on GitHub</a>
    <a class="btn" href="__REPO__/blob/main/docs/HOW_IT_WORKS.md">How it works</a>
  </div>

  <h2>Live simulations</h2>
  <p style="margin:0 0 12px;color:var(--muted)">
    <span class="key"><span class="sw" style="background:#2f6fdf"></span>DMA: moves data on and off the chip</span>
    <span class="key"><span class="sw" style="background:#e08a00"></span>MXU: the matrix-multiply unit</span>
    <span class="key"><span class="sw" style="background:#1a9b6b"></span>VPU: the vector unit</span>
    <br>Scroll to zoom, drag to pan, hover any bar for the exact instruction.</p>
  <div class="demos">
__DEMOS__
  </div>

  <h2>What MIRA does</h2>
  <ul class="facts">
    <li>Runs torchvision's <b>ResNet-18</b> and <b>MobileNetV2</b> entirely on the Apple Neural Engine:
        <b>5.6×</b> and <b>35×</b> faster than PyTorch on the CPU (1.39 ms and 0.48 ms per 224×224 image).</li>
    <li><b>int8</b> two ways: int8 weights (1.8× faster on the ANE for a 50M-parameter MLP), or int8 weights
        <i>and</i> int8 math with calibrated activation scales, on every target.</li>
    <li><b>Trains</b> models on the NPU: <code>grad()</code> differentiates programs at compile time.</li>
    <li>Imports <b>PyTorch models</b> via ONNX (MLPs, CNNs, transformers), matching PyTorch's output.</li>
    <li>Trains a <b>character-level GPT</b> on nursery rhymes in about 20 seconds, then writes them out one
        character at a time with a KV cache inside a data-dependent loop.</li>
    <li>Its fuzzer and tests found real bugs in <b>Apple's Core ML</b> and <b>Google's IREE</b>, with workarounds
        and regression tests in the repo.</li>
  </ul>

  <footer>These pages are rebuilt from the current code on every push. MNPU-1 is a modeled chip, not real hardware:
  its timings come from a cycle model. MIT licensed.</footer>
</main>
</body>
</html>
"""


def build(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cards = []
    for filename, title, blurb, args in DEMOS:
        path = os.path.join(out_dir, filename)
        if mira(["emit", args[0], "-t", "sim", "-s", "html", "-o", path, *args[1:]]) != 0:
            raise SystemExit(f"failed to build {filename}")
        cards.append(f'    <a class="demo" href="{filename}"><h3>{html.escape(title)}</h3>'
                     f'<p>{html.escape(blurb)}</p></a>')
    page = (INDEX.replace("__DEMOS__", "\n".join(cards)).replace("__REPO__", REPO)
            .replace("__FIRST__", DEMOS[0][0]))
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(page)
    print(f"site written to {out_dir}/ ({len(DEMOS)} demos)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "site")
