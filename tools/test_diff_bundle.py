#!/usr/bin/env python3
"""Prove the loosened bundle gate still fails when it should.

D8 widened the tolerance on the power-law fit scalars from 1e-6 to 1e-3 after
CI measured ~5e-4 of cross-CPU drift there. A gate that cannot fail is worse than
no gate, because it reports PASS and stops being read. So this asserts the
concrete failure modes are still caught, by corrupting a copy of the real bundle
and running the real comparator against it.

Run:  python tools/test_diff_bundle.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "public" / "data"
sys.path.insert(0, str(ROOT / "tools"))

from diff_bundle import is_fit_leaf, rel_delta  # noqa: E402

# Each case: (name, mutate, should_fail, why it matters)
# `mutate` receives the parsed bundle and edits it in place.
CASES = [
    (
        "fit amplitude off by 1% -- a real builder bug",
        lambda b: b["arms"]["pcfg"]["corpora"]["aitdcc_d_glibc_rand"]["fit"]
        .__setitem__("amplitude",
                     b["arms"]["pcfg"]["corpora"]["aitdcc_d_glibc_rand"]["fit"]["amplitude"] * 1.01),
        True,
        "the whole reason the gate exists: a fit that is actually wrong",
    ),
    (
        "fit alpha off by 5% -- the fit solves for the wrong exponent",
        lambda b: b["arms"]["uniform"]["corpora"]["dna"]["fit"]
        .__setitem__("alpha",
                     b["arms"]["uniform"]["corpora"]["dna"]["fit"]["alpha"] * 1.05),
        True,
        "alpha is the headline number on the Scaling page",
    ),
    (
        "fit floor off by 2%",
        lambda b: b["arms"]["selfplay"]["corpora"]["audio_8bit"]["fit"]
        .__setitem__("floor",
                     b["arms"]["selfplay"]["corpora"]["audio_8bit"]["fit"]["floor"] * 1.02),
        True,
        "the floor is the irreducible-loss asymptote",
    ),
    (
        "a measured frontier loss off by 1e-3",
        lambda b: b["arms"]["pcfg"]["corpora"]["dna"]["frontier"][0]
        .__setitem__(1, b["arms"]["pcfg"]["corpora"]["dna"]["frontier"][0][1] * 1.001),
        True,
        "NOT a fit leaf: measured data must still hold to the tight tolerance",
    ),
    (
        # Named like a fit field, but it is a transcribed table value from the
        # paper, not optimiser output. This is the case that proves the scoping
        # is by PATH and not by field name.
        "a transcribed exponent off by 1e-3",
        lambda b: b["exponents"][0].__setitem__(
            "alpha", b["exponents"][0]["alpha"] * 1.001),
        True,
        "`exponents[].alpha` must stay tight: same name, different provenance",
    ),
    (
        "a corpus renamed",
        lambda b: b["arms"].__setitem__(
            "pcfg_typo", b["arms"].pop("pcfg")),
        True,
        "structural: key sets are still compared exactly",
    ),
    (
        "a frontier point dropped",
        lambda b: b["arms"]["pcfg"]["corpora"]["dna"]["frontier"].pop(),
        True,
        "structural: array lengths are still compared exactly",
    ),
    (
        "a label string changed",
        lambda b: b["arms"]["pcfg"]["corpora"].__setitem__(
            "dna_typo", b["arms"]["pcfg"]["corpora"].pop("dna")),
        True,
        "structural: strings and keys, never tolerance",
    ),
    (
        "fit amplitude off by 5e-5 -- cross-CPU noise",
        lambda b: b["arms"]["pcfg"]["corpora"]["aitdcc_d_glibc_rand"]["fit"]
        .__setitem__("amplitude",
                     b["arms"]["pcfg"]["corpora"]["aitdcc_d_glibc_rand"]["fit"]["amplitude"] * (1 + 5e-5)),
        False,
        "the case D8 exists to permit: real drift, no real defect",
    ),
    (
        "fit amplitude off by 5e-4 -- worst measured drift",
        lambda b: b["arms"]["pcfg"]["corpora"]["aitdcc_d_glibc_rand"]["fit"]
        .__setitem__("amplitude",
                     b["arms"]["pcfg"]["corpora"]["aitdcc_d_glibc_rand"]["fit"]["amplitude"] * (1 + 5e-4)),
        False,
        "still inside fit-rtol=1e-3, by the margin measured on a real runner",
    ),
    (
        "untouched bundle",
        lambda b: None,
        False,
        "the baseline the others are measured against",
    ),
]


def run_gate(ref: str) -> int:
    """Run the real comparator over the working tree, against `ref`.

    The comparator reads the committed side with `git show <ref>:<path>`, so the
    way to exercise it is exactly the way CI does: write a mutated bundle into
    the working tree, point `--ref` at the commit holding the pristine bundle,
    and read the exit code. The working tree is restored afterwards.
    """
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_bundle.py"),
         "public/data/scaling.json", "--ref", ref],
        cwd=ROOT, capture_output=True, text=True,
    )
    return proc.returncode


def main() -> int:
    # Path-scoping assertions first: these are the load-bearing bit of D8, and a
    # false positive here would loosen the gate for the whole bundle silently.
    scoping = [
        ("$.arms.pcfg.corpora.dna.fit.alpha", True),
        ("$.arms.pcfg.corpora.dna.fit.amplitude", True),
        ("$.arms.pcfg.corpora.dna.fit.floor", True),
        ("$.arms.pcfg.corpora.fit.nPoints", True),
        # None of these are the fit's own scalars, so none may be loosened.
        # (`$.arms.fit.value` is deliberately absent: if a corpus were ever named
        # "fit", this matcher would loosen it too. That is a real limit of a
        # path-only rule, and no corpus in the bundle has that name -- 104 fit
        # keys, all under `arms.*.corpora.*`. Asserting a distinction the matcher
        # cannot make would be a test that passes for the wrong reason.)
        ("$.fitness.value", False),
        ("$.arms.pcfg.corpora.dna.frontier[0][1]", False),
        ("$.arms.pcfg.corpora.dna.fit", False),
        ("$.arms.pcfg.corpora.dna.fit.nested.value", False),
        ("$.metrics.alpha", False),
        ("$.arms.pcfg.corpora.refit.alpha", False),
    ]
    bad_scope = [(p, want, is_fit_leaf(p)) for p, want in scoping
                 if is_fit_leaf(p) != want]
    if bad_scope:
        print("FAIL: is_fit_leaf misclassifies paths:")
        for p, want, got in bad_scope:
            print(f"       {p}: wanted fit={want}, got fit={got}")
        return 1
    print(f"ok  is_fit_leaf classifies {len(scoping)} path shapes correctly")

    # The pristine bundle is committed at HEAD, so HEAD is the reference and
    # every case mutates the working tree against it. That is precisely the
    # shape CI runs, rather than a reimplementation of it.
    failures = 0
    pristine = (BUNDLE / "scaling.json").read_bytes()
    for name, mutate, should_fail, why in CASES:
        bundle = json.loads(pristine)
        mutate(bundle)
        (BUNDLE / "scaling.json").write_text(json.dumps(bundle))
        try:
            code = run_gate("HEAD")
        finally:
            (BUNDLE / "scaling.json").write_bytes(pristine)

        failed = code != 0
        ok = failed == should_fail
        if not ok:
            failures += 1
        verdict = "ok  " if ok else "FAIL"
        want = "reject" if should_fail else "accept"
        got = "rejected" if failed else "accepted"
        print(f"{verdict} {name}")
        print(f"       expected {want}, gate {got} -- {why}")

    print()
    if failures:
        print(f"FAIL: {failures} of {len(CASES)} gate cases behaved wrongly.")
        return 1
    print(f"PASS: all {len(CASES)} gate cases behaved as specified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())