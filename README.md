# torch-linalg-pinv-complex-grad-guard

This repository has been consolidated into **[torch-correctness-guards](https://github.com/zhuhroscar-tech/torch-correctness-guards)**.

Use the umbrella package instead:

```bash
pip install git+https://github.com/zhuhroscar-tech/torch-correctness-guards.git
python -m torch_correctness_guards.cli run linalg-pinv-complex-grad
# or, after installing the console script:
torch-guard run linalg-pinv-complex-grad
```

The migrated guard still tracks the same upstream issue: [pytorch/pytorch#197084](https://github.com/pytorch/pytorch/issues/197084).

This repo is archived to reduce duplicate single-purpose packages; history remains available for provenance.
