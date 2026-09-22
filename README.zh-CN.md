# torch-linalg-pinv-complex-grad-guard

本仓库已合并到 **[torch-correctness-guards](https://github.com/zhuhroscar-tech/torch-correctness-guards)**。

请改用统一的 umbrella 包：

```bash
pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git
python -m torch_correctness_guards.cli run linalg-pinv-complex-grad
# 或安装命令行入口后：
torch-guard run linalg-pinv-complex-grad
```

迁移后的 guard 仍然跟踪同一个上游问题：[pytorch/pytorch#197084](https://github.com/pytorch/pytorch/issues/197084)。

为减少重复的单用途包，本仓库将归档；历史记录仍保留以便追溯。
