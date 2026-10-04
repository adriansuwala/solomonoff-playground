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
outright. Last-bit optimiser noise moves them by ~1e-12. The default tolerance
sits between those by four orders of magnitude.

The `--rtol` knob and the reported worst delta are printed on every run so the
margin stays visible instead of becoming a mystery.

Usage:
    python tools/diff_bundle.py [--rtol 1e-6] [--ref HEAD] [path ...]

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
                    help="relative tolerance for numeric leaves (default 1e-6)")
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

        bad = [(p, o, n) for p, o, n in numeric
               if rel_delta(float(o), float(n)) > args.rtol]
        worst = max((rel_delta(float(o), float(n)) for _, o, n in numeric),
                    default=0.0)
        note = (f"{len(numeric)} numeric leaf/leaves differ, worst relative "
                f"delta {worst:.3e}")

        if not bad and not structural:
            print(f"  ok   {rel}: {note} (within tolerance)")
            continue

        failures += 1
        print(f"  FAIL {rel}: {note}")
        for p, o, n in structural[:MAX_REPORTED]:
            print(f"         structural  {p}: {o!r} -> {n!r}")
        for p, o, n in sorted(bad, key=lambda t: -rel_delta(float(t[1]), float(t[2])))[:MAX_REPORTED]:
            print(f"         numeric     {p}: {o!r} -> {n!r} "
                  f"(rel {rel_delta(float(o), float(n)):.3e})")
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
