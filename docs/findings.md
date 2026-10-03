# Findings

Defects found while building this app, severity-ordered, each with a live
reproducer. These are kept because the reproducers are runnable, so the document
doubles as a regression check.

Run `npm run data:verify` to re-check items 1–4; they are wired into it.

---

## F1 — A silently swallowed exception produced an empty central claim

**Severity.** High. Would have shipped the paper's headline result as nothing.

**What happened.** In `tools/build_data.py` the corpus catalogue is a list of
4-tuples:

```python
CORPORA = [("dclm", "text (dclm)", "Text", True), ...]
#             key     paper_label    group   in_table2
```

The frontier loop unpacked it as:

```python
for _, _, corpus, _ in P.CORPORA:      # WRONG
```

which binds `corpus` to the fourth element — the `in_table2` **boolean** — so
every `compute_frontier(data, corpus)` call raised `KeyError`, and the bare
`except (ValueError, KeyError): continue` swallowed it. The build "succeeded" and
emitted `scaling.json` with **zero frontiers and zero exponent fits** across every
arm: the paper's central claim rendered as an empty object.

The same class of error was in the Table 2 loop (`for key, label, group, in2 in
P.CORPORA` shadowed nothing but read as if it were correct).

**Why it survived.** No assertion covered "the scaling bundle is non-empty", and
the type checker's own diagnostic pointed at the *other* loop, not this one.

**Fix.** Bind the tuple by name, and distinguish "corpus absent from this source"
(normal — the three arms cover different rung ranges) from "corpus present but its
frontier cannot be built" (a real defect, now raised).

**Reproducer.** `tools/verify_data.py` check
`Table 2 rows all have a computed exponent`, plus per-source
`at least one corpus has a fit`.

---

## F2 — Table 2's exponents were computed from the wrong file

**Severity.** High. Would have produced a plausible, entirely wrong table.

**What happened.** Table 2 was first computed from the `fig2` CSVs. Every row was
wrong, and none of them looked wrong:

| corpus | paper | traj JSON | fig2 CSV |
|---|---|---|---|
| text (dclm) | 0.123 | **0.123** | 0.080 |
| CIFAR-10 (planar) | 0.145 | **0.145** | 0.149 |
| audio 8-bit PCM | 0.260 | **0.260** | 0.241 |
| DNA (8-symbol) | 0.435 | **0.435** | 0.243 |
| MIDI (Mutopia) | 0.249 | **0.249** | 0.186 |

The CSVs are per-K trajectory slices with fewer seeds behind some points, so their
compute-optimal frontiers differ. Neither file is a substitute for the other.

**Fix.** Both are kept, and the bundle labels which claim each backs: `ladder` for
Table 2 and Figure 1, `arms` for Figure 2. Recorded as `docs/decisions.md` D3.

**Reproducer.** `tools/verify_data.py` section [1] parses `exponents_table.tex` and
compares all ten rows.

---

## F3 — The verifier parsed Table 5's header as data

**Severity.** Medium. Produced 10 convincing-looking failures that were the
verifier's fault, not the bundle's.

**What happened.** `reward_arms.tex` has a header row,
`dataset & None & uniform & ...`, and ten data rows. The parser accepted any line
with eight `&`-separated cells, so it consumed the header and labelled every
column with header text. Worse, the header's last cell is `negate \\` — the
literal word `negate` — so the column mapping was off by one for the entire table.

A second bug in the same function: `re.sub(r"\\[a-zA-Z]+", "", c)` ran *before*
unwrapping `\textbf{...}`, turning `\textbf{10.62}` into `{10.62}`, and a third
issue stripped the row terminator `\\` from the final cell so every `negate` value
parsed as absent.

**Fix.** Require the first cell to look like a row label, unwrap `\textbf` first,
and drop the `\\` terminator explicitly.

**Reproducer.** `tools/verify_data.py` section [2]; it prints the parsed row labels
so a mis-parse is visible immediately.

---

## F4 — Table 5 was hand-transcribed from a mangled PDF table

**Severity.** High (prevented).

**What happened.** The first draft of `tools/paper_refs.py` carried a
`TABLE5_VALUES` dict transcribed from the PDF's Table 5. It had nine values per row
against seven column headers, with `last_step` and `negate` misaligned — and the
values looked individually plausible, which is the dangerous part.

**Fix.** Removed. All seven arms are now derived at build time following the
authors' own `make_table.py`, which reads three different files: the five ablations
from `frontier_traj_reward.json`, `none` from `frontier_traj_perk.json`, and
`uniform` from `uniform_frontier_perk.csv`. Running their script regenerates
`reward_arms.tex` byte-identically, so the derivation is exact.

**Reproducer.** `tools/verify_data.py` section [2] diffs every derived cell against
the committed `.tex`.

---

## F5 — A log axis cannot represent zero

**Severity.** Low, but silent.

**What happened.** `ChartFrame` hardcoded a log-scaled x-axis, correct for
effective compute but unable to represent 0. The local-measurement chart plots
against self-play round index starting at 0, so `log10(0)` is undefined and the
ticks rendered as `NaN`.

**Fix.** `xScale?: 'log' | 'linear'` prop, defaulting to log. Tick labels switch to
plain numbers when linear.

---

## F6 — Operator precedence spread a number instead of an array

**Severity.** Low.

**What happened.** `Math.max(...points.length - 1, 1)` parses as
`Math.max(...(points.length - 1), 1)` — spreading the *result* of a subtraction,
which is a number, not an iterable. Runtime `TypeError`.

**Fix.** `Math.max(points.length - 1, 1)`.

---

## Notes on the test suite itself

Two checks in `verify_data.py` were wrong in ways that *looked* like data
corruption, and one tolerance was too tight:

- Comparing exponents at `round(alpha, 3) == printed` failed on `audio_8bit`,
  where the underlying fit is `0.2605` and the authors printed `0.260`. The fit
  sits on the rounding midpoint, so the paper's printed value is one of two
  equally valid renderings. Tolerance is now a half-ulp window, which still
  rejects a genuinely different fit.
- The `negate` column needed all three of the F3 fixes at once, which is why it
  failed identically across all ten rows. A failure that repeats uniformly across
  a whole column is a parser or schema bug, not a data bug — that pattern is worth
  recognising before touching the data.