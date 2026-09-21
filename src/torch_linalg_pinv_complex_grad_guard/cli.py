"""Command-line interface: run the from-scratch diagnosis of the
torch.compile (AOTAutograd) complex torch.linalg.pinv/matrix_sqrth
wrong-gradient bug (pytorch/pytorch#197084) against the currently
installed torch build, using the shared semantic-color design
system.
"""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-linalg-pinv-complex-grad-guard",
        description=(
            "Diagnose whether the currently installed torch build's "
            "AOTAutograd-based backends (aot_eager, inductor) compute a "
            "WRONG gradient for torch.linalg.pinv on complex-dtype "
            "input (pytorch/pytorch#197084) -- and verify "
            "safe_complex_pinv_grad() restores eager's correct "
            "gradient. Never trusts a cached or previously-reported "
            "result, always re-runs the repro on THIS host's actual "
            "installed torch version."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-linalg-pinv-complex-grad-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_correct"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields([("torch version", report["torch_version"])])

    if report["any_native_bug"]:
        print(status_headline(style, "fail", "AOTAutograd complex pinv/matrix_sqrth wrong-gradient bug reproduced on this host (pytorch#197084)"))
    else:
        print(status_headline(style, "info", "no complex pinv gradient divergence reproduced on this host's installed torch build"))

    if report["guard_fully_correct"]:
        print(status_headline(style, "ok", "safe_complex_pinv_grad() restores eager's correct gradient on every case"))
    else:
        print(status_headline(style, "fail", "guard did NOT restore the correct gradient on at least one case"))

    section("cases (shape, seed -> aot_eager/inductor/guarded max-abs-diff vs eager)")
    for c in report["cases"]:
        native_flag = "WRONG" if (c["aot_eager_diverges"] or c["inductor_diverges"]) else "ok"
        guard_flag = "guard-ok" if c["guarded_matches_eager"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"shape={tuple(c['shape'])} seed={c['seed']}",
                    f"aot_eager_diff={c['max_abs_diff_aot_eager']:.3e}  "
                    f"inductor_diff={c['max_abs_diff_inductor']:.3e}  "
                    f"native={native_flag:6s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_correct"] else 1


if __name__ == "__main__":
    sys.exit(main())
