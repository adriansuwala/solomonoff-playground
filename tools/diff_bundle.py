#!/usr/bin/env python3
"""Compare the rebuilt data bundle against the committed one, field by field.

WHY THIS EXISTS INSTEAD OF `git diff --exit-code`
-------------------------------------------------
CI originally rebuilt public/data/ and required a byte-identical result. That
worked on the machine that generated the bundle and failed on the GitHub runner,
for a reason worth writing down so nobody "simplifies" it back.

The power-law fits in scaling.json come from the authors' vendored
curve_fit/minimize (vendor/spp/figures/table2_scaling_exponents/scaling_analysis.py).
Those routines iterate a numerical Jacobian and finish inside a tolerance, so the
trailing bits of the fitted amplitude/exponent are decided by the linear algebra
underneath them -- and scipy's manylinux wheels bundle OpenBLAS, which selects a
kernel at RUNTIME from the CPU's feature set. The machine that generated the
bundle (Xeon Gold 5412U, Sapphire Rapids, AVX-512) and the ubuntu-24.04 runner
therefore sum in a different order and land on a different last bit.

Two floating-point stacks agreeing bit-for-bit is a property of two identical
CPUs, not of a correct build. Pinning numpy and scipy fixes the *versions*; it
cannot fix the *microarchitecture*. So this check enforces the things that are
actually portable:

  * identical key sets at every level      -> catches a changed builder shape
  * identical array lengths                -> catches dropped or added rows
  * identical strings, labels and flags    -> caught EXACTLY, no tolerance
  * numbers within a relative tolerance    -> catches real drift

A stale bundle, a hand-edited JSON file, a renamed corpus or a builder that
computes the wrong quantity all move values by O(1) or break the structure
outright. The default tolerance sits between those and optimiser noise by four
orders of magnitude.

TWO TOLERANCES, AND WHY THE LOOSE ONE IS NARROW
----------------------------------------------
The one thing that will not hold to 1e-6 is the power-law fit: measured drift
across CPUs is ~5e-4, which is the failure this file exists to prevent CI
reporting as routine. So the fit fields get their own, looser tolerance and
everything else keeps the tight one.

The loose tolerance is scoped by PATH, not applied globally, and that scoping is
what keeps the check worth running:

  * only leaves under a `fit` object -- alpha, amplitude, floor, rmse
  * 520 leaves, which is 0.1% of the bundle's 784,920 numeric leaves
  * the other 99.9% are still compared at 1e-6, and strings and key sets are
    still compared exactly

A blanket 1e-4 would have been simpler and nearly worthless: it would have
loosened the check for every measured loss and compute count in the bundle, not
just the numbers an optimiser touched.

WHY THE FIT IS ILL-CONDITIONED (and why "converge harder" did not fix it)
-----------------------------------------------------------------------
The stored amplitude is A at compute = 1, extrapolated roughly 31 orders of
magnitude below the smallest compute anyone measured, via
`amplitude = A_ref * compute_scale**alpha`. Amplitude therefore scales as
`compute_scale**alpha` with compute_scale = 1e15, so

    d(amplitude)/amplitude  ~=  alpha * ln(1e15) * d(alpha)/alpha

with alpha ~ 2. The exponent multiplies a ~1e-7 converged error by ~34, giving
~1e-5..1e-4. That is conditioning, not convergence: tightening the optimiser
past 1e-14 does not move it, because the exponent has already settled. See
docs/decisions.md D7 and D8.

Measured, on this bundle, perturbing the inputs by 1e-13 (a proxy for a
different BLAS summation order):

    alpha      worst 7.8e-05      floor     worst 4.9e-05
    amplitude  worst 5.2e-04      rmse      worst 3.4e-04

Note what is NOT loose: the fitted loss at the largest observed compute is
stable to ~1e-11, and that is the quantity the page's charts actually draw.

The `--rtol` knob and the reported worst delta are printed on every run so the
margin stays visible instead of becoming a mystery.

Usage:
    python tools/diff_bundle.py [--rtol 1e-6] [--fit-rtol 1e-3] [--ref HEAD] [path ...]

Exit codes:
    0  bundle reproduces within tolerance
    1  drift outside tolerance, or structural change
    2  a bundle file is missing from the tree or from the commit
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS = ["public/data"]

MAX_REPORTED = 12  # a CI log is not a place to paste two 3 MB JSON files

# Tolerance for the power-law fit scalars, which are the only values in the
# bundle an iterative optimiser produced and the only ones that will not agree
# across two CPUs. Applied by PATH so the 784,200 measured leaves keep the tight
# tolerance -- see the module docstring for why this is not a global loosening.
DEFAULT_FIT_RTOL = 1e-3


def is_fit_leaf(path: str) -> bool:
    """True for a numeric scalar that is a direct member of a `fit` object.

    Every `fit` in the bundle is an object holding alpha/amplitude/floor/rmse
    (104 of them, all dicts), so the shape to recognise is `....fit.<scalar>`.
    The component before the last has to be exactly "fit", which keeps a corpus
    named "fit", a `fitness` field, and anything nested deeper inside a fit out
    of the loosened set.
    """
    parts = [p for p in path.replace("[", ".").replace("]", "").split(".")
             if p and p != "$"]
    return len(parts) >= 2 and parts[-2] == "fit"


def committed_bytes(path: str, ref: str) -> bytes | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                       capture_output=True)
    return r.stdout if r.returncode == 0 else None


def walk(old, new, path="$"):
    """Yield (path, old, new) for every leaf that differs, plus structural breaks."""
    if isinstance(old, bool) or isinstance(new, bool):
        if type(old) is not type(new) or old != new:
            yield (path, old, new)
        return
    if isinstance(old, (int, float)) and isinstance(new, (int, float)):
        if old != new:
            yield (path, old, new)
        return
    if isinstance(old, dict) and isinstance(new, dict):
        missing = sorted(set(old) - set(new))
        added = sorted(set(new) - set(old))
        if missing or added:
            yield (path + ".<keys>",
                   {"missing": missing[:5], "added": added[:5]}, "key set changed")
        # Keep descending into the shared keys. Bailing out here would let a
        # structural break mask numeric drift further down the same subtree,
        # which is the case an operator most wants to see.
        for k in old:
            if k in new:
                yield from walk(old[k], new[k], f"{path}.{k}")
        return
    if isinstance(old, list) and isinstance(new, list):
        if len(old) != len(new):
            yield (path, f"len={len(old)}", f"len={len(new)}")
        # zip stops at the shorter list, so a length break reports alongside
        # whatever else changed in the rows the two sides still share.
        for i, (o, n) in enumerate(zip(old, new)):
            yield from walk(o, n, f"{path}[{i}]")
        return
    if old != new:
        yield (path, old, new)


def rel_delta(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    if scale == 0.0:
        return 0.0
    return abs(a - b) / scale


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=None,
                    help="files or dirs to compare (default: public/data)")
    ap.add_argument("--rtol", type=float, default=1e-6,
                    help="relative tolerance for measured numeric leaves "
                         "(default 1e-6)")
    ap.add_argument("--fit-rtol", type=float, default=DEFAULT_FIT_RTOL,
                    help=f"relative tolerance for the power-law fit scalars "
                         f"under a `fit` object (default {DEFAULT_FIT_RTOL:g}). "
                         f"These are the only bundle values an iterative "
                         f"optimiser produced and the only ones that cannot be "
                         f"reproduced across CPUs.")
    ap.add_argument("--ref", default="HEAD", help="committed ref to compare against")
    args = ap.parse_args()

    roots = args.paths or DEFAULT_PATHS
    files: list[Path] = []
    for r in roots:
        p = ROOT / r
        if p.is_dir():
            files.extend(sorted(p.rglob("*.json")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"error: no such path: {r}", file=sys.stderr)
            return 2
    if not files:
        print("error: no bundle files found", file=sys.stderr)
        return 2

    failures = 0
    print(f"comparing {len(files)} bundle file(s) against {args.ref} "
          f"(rtol={args.rtol:g})")

    for f in files:
        rel = f.relative_to(ROOT).as_posix()
        raw_new = f.read_bytes()
        raw_old = committed_bytes(rel, args.ref)
        if raw_old is None:
            print(f"  FAIL {rel}: not present in {args.ref} (uncommitted bundle file)")
            failures += 1
            continue
        if raw_old == raw_new:
            print(f"  ok   {rel}: byte-identical")
            continue
        try:
            old, new = json.loads(raw_old), json.loads(raw_new)
        except json.JSONDecodeError as e:
            print(f"  FAIL {rel}: rebuilt file is not valid JSON ({e})")
            failures += 1
            continue

        numeric, structural = [], []
        for path, o, n in walk(old, new):
            (numeric if isinstance(o, (int, float))
             and isinstance(n, (int, float))
             and not isinstance(o, bool) and not isinstance(n, bool)
             else structural).append((path, o, n))

        # Split the numeric leaves by tolerance rather than applying one number
        # to all of them. A fit leaf gets `--fit-rtol`; every other numeric leaf
        # keeps `--rtol`. `bad` is what fails the run.
        fit_bad, tight_bad, fit_worst, tight_worst = [], [], 0.0, 0.0
        for p, o, n in numeric:
            d = rel_delta(float(o), float(n))
            if is_fit_leaf(p):
                fit_worst = max(fit_worst, d)
                if d > args.fit_rtol:
                    fit_bad.append((p, o, n, d))
            else:
                tight_worst = max(tight_worst, d)
                if d > args.rtol:
                    tight_bad.append((p, o, n, d))
        bad = fit_bad + tight_bad
        worst = max(fit_worst, tight_worst)

        # Report both tolerances on a passing run too. A gate whose margin is
        # invisible is a gate nobody trusts, and "how close was this?" is the
        # question an operator asks when a fit starts drifting.
        note = (f"{len(numeric)} numeric leaf/leaves differ, worst relative "
                f"delta {worst:.3e}")
        if len(numeric):
            note += (f" (fit {fit_worst:.3e} of {args.fit_rtol:g}, "
                     f"measured {tight_worst:.3e} of {args.rtol:g})")

        if not bad and not structural:
            print(f"  ok   {rel}: {note}")
            continue

        failures += 1
        print(f"  FAIL {rel}: {note}")
        for p, o, n in structural[:MAX_REPORTED]:
            print(f"         structural  {p}: {o!r} -> {n!r}")
        for p, o, n, d in sorted(bad, key=lambda t: -t[3])[:MAX_REPORTED]:
            tag = "fit     " if is_fit_leaf(p) else "measured"
            print(f"         numeric     {tag} {p}: {o!r} -> {n!r} (rel {d:.3e})")
        if len(structural) + len(bad) > MAX_REPORTED:
            print(f"         ... {len(structural) + len(bad) - MAX_REPORTED} more")

    print()
    if failures:
        print(f"FAIL: {failures} of {len(files)} file(s) do not reproduce within "
              f"rtol={args.rtol:g}.")
        print("If the deltas here are ~1e-12 and confined to fitted amplitude/"
              "exponent, that is OpenBLAS picking a different kernel on a "
              "different CPU, not a broken builder.")
        print("Regenerating the bundle is only the right fix if the values are "
              "wrong by more than the tolerance.")
        return 1
    print(f"PASS: bundle reproduces from the committed builder "
          f"(rtol={args.rtol:g}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
