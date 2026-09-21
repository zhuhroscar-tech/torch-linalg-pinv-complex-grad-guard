# torch-linalg-pinv-complex-grad-guard

Guards a real `torch.compile` correctness bug: on complex-dtype input,
AOTAutograd-based backends (`aot_eager`, `inductor`) compute a
**wrong gradient** for `torch.linalg.pinv` (and `torch.linalg.matrix_sqrth`,
used internally by some autograd formulas), while eager mode,
`torch.func.grad`, and Dynamo-only compilation (`backend="eager"`) all
agree with a numerically verified reference.

Upstream report: [pytorch/pytorch#197084](https://github.com/pytorch/pytorch/issues/197084)
("wrong gradients for `torch.linalg.pinv` and `torch.linalg.matrix_sqrth`
with complex input under `torch.compile` (`aot_eager` and `inductor`);
eager, functorch and Dynamo-only are correct"). As of 2026-09-21 this
issue is **open** and unfixed upstream.

## What actually happens

```python
import torch

def f(a):
    return torch.linalg.pinv(a).sum().abs()

a = torch.randn(4, 3, dtype=torch.complex64, requires_grad=True)

# Eager: correct.
f(a).backward()
g_eager = a.grad.clone()

# torch.compile with an AOTAutograd-based backend: WRONG gradient,
# not a rounding error -- often several absolute units off.
a2 = a.detach().clone().requires_grad_(True)
torch.compile(f, backend="aot_eager")(a2).backward()
g_aot = a2.grad

# max(|g_eager - g_aot|) is large (order 1-10s in our own
# reproduction), not floating-point noise.
```

Independently reproduced on this repository's own CI (Linux + macOS
runners, CPU-only, torch installed via `pip`) using the issue's own
minimal reproduction shape.

## The guard

`safe_complex_pinv_grad(fn)` wraps a function that computes
`torch.linalg.pinv`/`matrix_sqrth` on complex input so that call
always runs under `torch.compiler.disable()` — i.e. AOTAutograd never
traces through it, even inside an outer `torch.compile` region. This
restores eager's correct gradient for that call, at the cost of
losing compiled speedups for it specifically; everything else in an
outer compiled graph is unaffected.

```python
from torch_linalg_pinv_complex_grad_guard import safe_complex_pinv_grad

guarded_f = safe_complex_pinv_grad(f)
compiled = torch.compile(guarded_f, backend="inductor")
compiled(a).backward()  # correct gradient, matches eager
```

## CLI

```
pip install -e '.[dev,torch]'
torch-linalg-pinv-complex-grad-guard          # human-readable report
torch-linalg-pinv-complex-grad-guard --json   # machine-readable
torch-linalg-pinv-complex-grad-guard --no-color
```

Exit code `0` when the guard is fully correct (or the native bug is
already fixed upstream), `1` if the guard failed to restore the
correct gradient on any case, `2` if `torch` is not installed.

## Verification discipline

- Every claim is backed by a **fresh, from-scratch reproduction**
  against the currently installed `torch` build — never a cached or
  assumed result. If PyTorch fixes #197084 upstream, this tool's own
  "bug reproduced" assertion will start failing, which is the correct
  and expected outcome (see `tests/test_core.py`'s
  `test_native_bug_is_actually_reproduced_on_this_host`).
- A bug-injection test (`test_native_compiled_diverges_bug_injection_check`)
  proves the regression suite is not tautological: it directly shows
  the *unguarded* compiled path diverging from eager.
- CI runs on both `ubuntu-latest` and `macos-latest` (CPU-only; no
  CUDA/GPU dependency — the bug is backend-agnostic per the upstream
  issue, not device-specific).

## Limitations

- Only covers `torch.linalg.pinv`; `torch.linalg.matrix_sqrth` is
  named in the same upstream issue but not separately exercised here
  (it's an internal helper, not a public API, and harder to call
  directly without pulling in whichever public op uses it).
- The guard trades away Inductor speedups for the wrapped call. If
  the wrapped function's `pinv` call dominates the compiled graph,
  this may cost meaningful performance versus a hypothetical correct
  compiled path — no benchmark of that tradeoff is included.
- Verified on CPU only, on the `torch` version pinned in CI at build
  time. Not verified on CUDA/ROCm/MPS backends.
