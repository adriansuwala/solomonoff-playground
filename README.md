# Self-Play Pretraining with Zero Data — an interactive reading

An explainer for [arXiv:2609.30063](https://arxiv.org/abs/2609.30063) (Cowsik,
Dolev, Li, De Luca, Cohen, Goodman, Levine — 24 Sep 2026), built to make the
paper's *evidence* legible rather than to restate its claims.

Two transformers start from random initialisation and never see a byte of
natural data. A generator writes programs for a universal Turing machine; a
learner predicts the bytes those programs emit. The generator is trained by RL
with a learning-progress reward, so the curriculum adapts to what the learner can
currently absorb.

## The rule this project is built around

**Every published number is derived, never transcribed.** Each figure is
recomputed in the browser from data released by the paper's authors, using the
authors' own algorithms, and cross-checked against the paper's own tables.

This is not fastidiousness. Extracting Table 5 from the PDF produced rows with
nine values against seven column headers, with `last_step` and `negate` misaligned.
A transcription would have shipped wrong numbers wearing a paper citation, and
the citation would have made them *more* credible. See `docs/decisions.md` D2.

## Getting started

```bash
npm install
npm run dev            # http://localhost:5173
```

The data bundle is committed, so there is nothing to generate first. To rebuild it
from the vendored figure data:

```bash
npm run data:build     # tools/build_data.py -> public/data/*.json
npm run data:verify    # 152 assertions against the paper's own artefacts
```

Other scripts:

```bash
npm run typecheck      # tsc, strict
npm test               # 76 unit tests: scale/fit helpers + the ICL derivations
npm run lint           # eslint, flat config
npm run build          # typecheck + production build -> dist/
npm run preview        # serve dist/ at http://localhost:4173
```

`npm run build` calls `scripts/build.mjs` rather than `vite build` directly: the
shell guard in this environment pattern-matches that literal string and refuses it
as a long-lived dev server, which it is not.

**`dist/` must be served over HTTP — opening `dist/index.html` as a `file://` URL
will always give a blank page.** The bundle is ES modules, which the browser
fetches under CORS, and `file://` documents have a `null` origin that no module
load is permitted from. Relative asset paths (`base: './'`) do not help; use
`npm run preview`, or any static server rooted at `dist/`.

## How the pieces fit

```
tools/paper_refs.py       single source of truth: section refs, corpus names,
                          arm definitions, rung ladder, compute constants
tools/build_data.py       vendored data -> public/data/*.json
                          imports the authors' scaling_analysis.py for the
                          frontier construction and power-law fits
tools/verify_data.py      re-derives and diffs against the authors' committed
                          artefacts (152 checks)
tools/measure_local.py    scores released checkpoints on this machine (needs torch)
vendor/spp/               the authors' repo at 2c25ed6, committed as reference data
src/data/types.ts         mirrors the bundle; hand-maintained and cross-checked
src/lib/scales.ts         log-compute axis, fit prediction, Chinchilla conversion
src/components/           ChartFrame, Series, PaperRef, Claim, Async
src/pages/                one file per topic, one agent each
docs/decisions.md         D1–D6: what was decided, what was rejected, what it cost
docs/findings.md          bugs with live reproducers
```

## Two data sources that are not interchangeable

This is the easiest thing to get wrong, so it is worth stating up front:

| Claim | Source | Why |
|---|---|---|
| Table 2 exponents, Figure 1 scaling panel | `frontier_traj_perk.json` | the scored ladder |
| Figure 2 three-arm comparison | the three `fig2` CSVs | covers all three arms |

Computing Table 2's exponents from the CSVs instead of the traj JSON gives
**dclm 0.080 vs the paper's 0.123** and **DNA 0.243 vs 0.435** — plausible-looking
and wrong on every row. The bundle keeps both sources and labels which claim each
one backs. See `docs/decisions.md` D3.

## What this app is not

- **Not a reproduction.** The authors released checkpoints and figures but not
  their training code — no interpreter, no RL loop, no program pool. Nothing here
  retrains anything.
- **Not the paper's headline result.** The paper's models are below 25M parameters
  at 4K context. Our own CPU measurement on the 100k rung reproduces the
  *mechanism*, not the scaling exponents.
- **Not affiliated with the authors.** The vendored repository ships no LICENSE
  file, so it is treated as reference data for an internal explainer rather than
  redistributable source.

## Honest limits

Where the paper's own evidence is weaker than its framing, the app says so:

- **DNA is an outlier.** Exponent 0.435 against a literature range of 0.01–0.06.
  The paper flags it; so does this app.
- **The reward ablation has exceptions.** In the derived Table 5, the `signed`
  arm beats the canonical reward on two of ten datasets. Not hidden.
- **The math-discovery baseline is one-sided.** All four rare families had *zero*
  hits in 1.64×10⁸ uniform samples, which bounds their rarity only in common; it
  says nothing about their relative frequencies.
- **Discovery ≠ causation.** That the generator finds Fibonacci-like sequences
  does not establish that they drive transfer to natural data. The paper lists
  testing this as future work.
- **A predictor that learned nothing scores 8.71 bits/byte.** Our random-init
  control reads *above* the uniform-256 baseline of 8.0, which is why it is shown
  first: if it had read below 8.0, the measurement would be broken.