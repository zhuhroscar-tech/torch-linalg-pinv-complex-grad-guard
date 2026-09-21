# torch-linalg-pinv-complex-grad-guard

修复一个真实存在的 `torch.compile` 正确性问题：当输入为复数（complex）类型时，
基于 AOTAutograd 的后端（`aot_eager`、`inductor`）计算的
`torch.linalg.pinv`（以及内部使用的 `torch.linalg.matrix_sqrth`）的
**梯度是错误的**，而 eager 模式、`torch.func.grad` 以及仅使用 Dynamo
的编译（`backend="eager"`）都与经过数值验证的参考结果一致。

上游报告：[pytorch/pytorch#197084](https://github.com/pytorch/pytorch/issues/197084)。
截至 2026-09-21，该问题在上游**仍未修复**。

## 实际现象

```python
import torch

def f(a):
    return torch.linalg.pinv(a).sum().abs()

a = torch.randn(4, 3, dtype=torch.complex64, requires_grad=True)

# Eager：正确
f(a).backward()
g_eager = a.grad.clone()

# 使用基于 AOTAutograd 的后端编译：梯度错误，且不是舍入误差
# （在我们自己的复现中，误差量级可达 1~10 以上）
a2 = a.detach().clone().requires_grad_(True)
torch.compile(f, backend="aot_eager")(a2).backward()
g_aot = a2.grad
```

已在本仓库自己的 CI（Linux + macOS 运行器，仅 CPU，通过 `pip` 安装
torch）上使用上游 issue 自带的最小复现脚本独立复现。

## 修复方案

`safe_complex_pinv_grad(fn)` 包装一个在复数输入上调用
`torch.linalg.pinv`/`matrix_sqrth` 的函数，使该调用始终在
`torch.compiler.disable()` 下执行——即即使外层存在
`torch.compile`，AOTAutograd 也不会追踪该调用。这样可以恢复 eager
模式下正确的梯度，代价是该调用本身失去编译加速；外层计算图的其他部分不受影响。

```python
from torch_linalg_pinv_complex_grad_guard import safe_complex_pinv_grad

guarded_f = safe_complex_pinv_grad(f)
compiled = torch.compile(guarded_f, backend="inductor")
compiled(a).backward()  # 正确的梯度，与 eager 一致
```

## 命令行工具

```
pip install -e '.[dev,torch]'
torch-linalg-pinv-complex-grad-guard          # 人类可读报告
torch-linalg-pinv-complex-grad-guard --json   # 机器可读
torch-linalg-pinv-complex-grad-guard --no-color
```

退出码：修复完全生效（或上游 bug 已修复）为 `0`；修复未能恢复正确梯度为
`1`；未安装 `torch` 为 `2`。

## 验证纪律

- 每一个结论都基于**针对当前安装的 torch 版本从零开始的复现**，从不使用缓存或假设的结果。
- 通过“故意注入 bug”的测试证明回归测试并非同义反复：直接展示未加保护的编译路径确实偏离 eager 结果。
- CI 同时在 `ubuntu-latest` 与 `macos-latest` 上运行（仅 CPU；该 bug
  与后端相关而非设备相关，无需 CUDA/GPU）。

## 局限性

- 仅覆盖 `torch.linalg.pinv`；同一上游 issue 提到的
  `torch.linalg.matrix_sqrth` 未单独测试（它是内部辅助函数，不易直接调用）。
- 该修复会使被包装的调用失去 Inductor 加速；未包含该性能取舍的基准测试。
- 仅在 CPU 上验证，且仅针对 CI 构建时固定的 torch 版本；未在
  CUDA/ROCm/MPS 后端上验证。
