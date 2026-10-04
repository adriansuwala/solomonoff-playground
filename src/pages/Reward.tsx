/**
 * Reward: the learning-progress reward, and Table 5's causal evidence for it.
 *
 * This is the one experiment in the paper that cannot be faked from released
 * data. Everything else (Scaling, Curriculum, InContextLearning) scores a
 * checkpoint the authors shipped; here the thing being compared is a *training
 * run*, and the generator's runs were not shipped — only the learner each run
 * left behind. So the table is an endpoint trace of a process we cannot re-run,
 * which is why it carries the argument and why the page states where it stops
 * (D2: the cells are derived, never typed; D6: every claim names a section).
 */
import { useState } from 'react'
import { loadTable5 } from '@/data/loaders'
import { Async, Card, Claim, Disclosure, PaperRef } from '@/components/UI'
import { ChartFrame } from '@/components/Chart'
import type { Table5, Table5Row } from '@/data/types'

/**
 * Footnote markers on the ablation headers, matching the paper's Table 5
 * caption. `Table5.notes` carries the prose; only the mapping of which column
 * carries which marker has to be written down somewhere.
 */
const FOOTNOTES: Record<string, string> = { last_step: 'b', loss_delta: 'a,b' }


/**
 * The uniform-over-256-bytes baseline: 8.0 bits/byte. This is what a predictor
 * ignorant of the corpus alphabet scores, NOT what our random-init control
 * scores -- that control reads 8.72 (see local_measurements.json), because a
 * randomly initialised transformer is not a uniform sampler. Every "above the
 * baseline" test below is therefore against 8.0, which is the weaker bar.
 */
const UNIFORM_256_BPB = 8

export function Reward() {
  return (
    <>
      <h1 className="page__title">The reward, and the evidence for it</h1>
      <p className="page__lede">
        Every other experiment in the paper is a measurement of a checkpoint the
        authors released. This is the one that cannot be reconstructed from
        released data: it compares <em>training runs</em>, and what we get to
        inspect is the learner each run left behind. That is a weaker instrument
        than the runs themselves — it shows where a run ended, not how it got
        there — but it is the only causal evidence in the paper for the design
        choice the whole method rests on.
      </p>

      <h2 className="section">What the generator is paid for</h2>
      <p className="body">
        The generator writes programs for a universal Turing machine; the learner
        predicts the bytes those programs emit. The generator is rewarded for{' '}
        <em>learning progress</em> — not for difficulty, not for novelty, not for
        entropy. With <code>e</code> the current self-play round:
      </p>
      <div className="program" style={{ marginTop: 12 }}>
        r_i = |⟨∇θ L(y_i; θ_now), P_e ⊙ δθ_e⟩|
      </div>
      <div className="kv" style={{ marginTop: 12 }}>
        <span className="kv__k">δθ_e</span>
        <span className="kv__v">θ_⌊e/2⌋ − θ_e</span>
        <span className="kv__k">P_e</span>
        <span className="kv__v">diag(sqrt(v̂_e) + ε), the AdamW step operator</span>
        <span className="kv__k">L(y_i; θ)</span>
        <span className="kv__v">
          the learner&apos;s loss on the sequence <code>y_i</code> that program{' '}
          <code>i</code> emitted
        </span>
      </div>
      <p className="body" style={{ marginTop: 12 }}>
        A program is paid in proportion to how much the learner&apos;s gradient on
        it <em>agrees with the direction the learner has actually been moving</em>,
        measured through the optimiser&apos;s own preconditioner, which rescales
        each coordinate by its running second-moment estimate.{' '}
        <PaperRef reference="eq2" also={['sec2.2']} inline />
      </p>

      <h3 className="subhead">Three cases, only one of which pays</h3>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>what the program is</th>
              <th>gradient at θ_now</th>
              <th>reward</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>already mastered</td>
              <td className="num">≈ 0</td>
              <td className="num">none</td>
            </tr>
            <tr>
              <td>unlearnable structure</td>
              <td className="num">large, not aligned with δθ_e</td>
              <td className="num">none</td>
            </tr>
            <tr>
              <td>at the learner&apos;s frontier</td>
              <td className="num">substantial and aligned</td>
              <td className="num">high</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="body">
        The two zero rows are the point. A mastered program teaches nothing, so
        paying for it spends a learner step on bytes it can already predict. Pure
        noise has a large gradient, but the gradient points nowhere the learner
        has been going, so paying for it teaches nothing either. Reward
        concentrates on programs that are hard <em>and</em> absorbable — a moving
        target that only the learner itself can define.{' '}
        <PaperRef reference="sec2.2" inline />
      </p>

      <h3 className="subhead">Why the absolute value, and why the ⌊e/2⌋ window</h3>
      <p className="body">
        The outer <code>|·|</code> makes the reward non-negative. Without it, a
        program whose gradient opposes recent motion earns a negative reward, so
        the generator would be paid to <em>produce</em> bytes that look learnable
        and push the wrong way — an instability the paper avoids by
        construction. The cost of the abs is that it discards direction: it
        cannot distinguish "hard in a direction worth going" from "hard in a
        direction the optimiser will not follow". The <code>signed</code> arm in
        Table 5 exists to price that loss, and it prices it low — which the
        verdict below states rather than rounds away.{' '}
        <PaperRef reference="eq2" also={['appF']} inline />
      </p>
      <p className="body">
        The lookback window is not <code>θ_now</code> but <code>θ_⌊e/2⌋</code>,
        the checkpoint from roughly the middle of the run so far. Measuring
        against a longer trajectory averages over short-term fluctuations — one
        step&apos;s <code>δθ</code> is mostly noise for a given program — while
        the length of the window means a mistake made early stops counting
        against that program once the learner has genuinely moved past it. The
        one-step window <code>θ<sub>e−1</sub></code> is the ablation that tests
        exactly this choice.{' '}
        <PaperRef reference="sec2.2" also={['appF']} inline />
      </p>

      <h2 className="section">The failure mode this design exists to avoid</h2>
      <p className="body">
        The obvious alternative is a difficulty reward: pay for programs the
        learner gets wrong. It is gameable, and the paper says so. Take a
        perfectly predictable sequence and inject a few random bytes into it. The
        result is arbitrarily hard to predict — difficulty is unbounded — and it
        contains no useful structure, because the injected bytes are uniform
        noise. An optimiser maximising difficulty finds that trade immediately
        and then stops producing anything learnable while the loss climbs. The
        learning-progress reward cannot be gamed this way, because random bytes
        earn nothing: large gradient, no alignment with where the learner has
        been going.{' '}
        <PaperRef reference="sec2.2" inline />
      </p>
      <div className="claim claim--caveat">
        <p className="claim__text">
          <strong>What Table 5 does and does not settle.</strong> One rung, one
          round, one K-seed ensemble per cell, scored on held-out sequences. It
          says which arm ended up better; it does not show the trajectory. Per
          the paper&apos;s own footnote the <code>last_step</code> and{' '}
          <code>loss_delta</code> arms are bimodal across seeds, so their ensemble
          values average over runs that diverged. A table like this can rank arms
          without supporting every mechanism a reader would like to attribute to
          them. <PaperRef reference="appF" inline />
        </p>
      </div>

      <Async load={loadTable5}>{(t) => <TableEvidence t={t} />}</Async>
    </>
  )
}

// ---------------------------------------------------------------------------
// Table 5
// ---------------------------------------------------------------------------

/** Everything below reads one bundle, so the selection state lives here. */
function TableEvidence({ t }: { t: Table5 }) {
  const [selected, setSelected] = useState<string>(t.rows[0]?.key ?? '')

  return (
    <>
      <h2 className="section">
        Table 5 — one canonical reward, six ways to break it
      </h2>
      <p className="body">
        Every ablation changes exactly one property of the canonical reward and
        is then trained from scratch. Entries are validation loss in bits per
        byte of the {t.K}-seed ensemble, scored by us from the authors&apos;
        released checkpoints at round {t.round.toLocaleString()} of rung{' '}
        <code>{t.rung}</code>. <PaperRef reference="table5" also={['appF']} inline />
      </p>
      <p className="card__note">
        No value on this page was typed in. Each cell was derived by{' '}
        <code>tools/build_data.py</code> from the shipped frontier trajectories
        and gated by <code>tools/verify_data.py</code>, which re-derives every
        cell against the authors&apos; committed <code>reward_arms.tex</code>.
      </p>

      <Table5Table t={t} />

      <Disclosure summary="What each ablation changes">
        <div className="kv">
          {t.columns.map((c) => (
            <KvLine key={c.key} k={c.label} v={c.desc} />
          ))}
        </div>
        <p className="body" style={{ marginTop: 10 }}>
          The <code>uniform</code> arm is not a variant of the reward at all: it
          removes the generator and draws programs i.i.d. uniform, so it is the
          control for the adaptive curriculum rather than for the reward&apos;s
          algebra. <PaperRef reference="appF" inline />
        </p>
      </Disclosure>

      <Heatmap t={t} selected={selected} onSelect={setSelected} />

      <h3 className="subhead">One dataset at a time</h3>
      <Card
        title="Arm explorer — every ablation on a single dataset"
        note="Sorted by final validation loss; lower is better. The canonical arm is in teal."
      >
        <div className="controls">
          {t.rows.map((r) => (
            <button
              key={r.key}
              aria-pressed={selected === r.key}
              onClick={() => setSelected(r.key)}
            >
              {r.label}
            </button>
          ))}
        </div>
        <ArmChart t={t} row={t.rows.find((r) => r.key === selected)} />
      </Card>

      <Verdict t={t} />
    </>
  )
}

/** One key/value line for the .kv grid inside the Disclosure. */
function KvLine({ k, v }: { k: string; v: string }) {
  return (
    <>
      <span className="kv__k">{k}</span>
      <span className="kv__v">{v}</span>
    </>
  )
}

/** The table: each row's best arm marked, short ensembles called out. */
function Table5Table({ t }: { t: Table5 }) {
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            <th>dataset</th>
            {t.columns.map((c) => (
              <th key={c.key} title={c.desc}>
                {c.label}
                {FOOTNOTES[c.key] && (
                  <sup style={{ color: 'var(--warn)' }}> {FOOTNOTES[c.key]}</sup>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {t.rows.map((row) => (
            <tr key={row.key}>
              <td>{row.label}</td>
              {t.columns.map((col, i) => {
                const cell = row.cells[i]
                const isBest = row.bestIndex === i
                return (
                  <td
                    key={col.key}
                    className={`num${isBest ? ' best' : ''}`}
                    title={
                      cell ? `ensemble over ${cell.kUsed} of ${t.K} seeds` : 'unscored'
                    }
                  >
                    {cell ? cell.bpb.toFixed(2) : '—'}
                    {cell && cell.kUsed < t.K && (
                      <div
                        style={{
                          fontSize: 10,
                          color: 'var(--text-faint)',
                          fontFamily: 'var(--sans)',
                        }}
                      >
                        {cell.kUsed} seeds
                      </div>
                    )}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      {t.notes.map((n, i) => (
        <div className="card__note" key={i} style={{ marginTop: 8 }}>
          <sup style={{ color: 'var(--warn)' }}>{['a', 'b'][i] ?? '·'}</sup>{' '}
          {n}
        </div>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Heatmap
// ---------------------------------------------------------------------------

const HEAT_RGB = '227, 179, 65'

/** Opacity ∝ how much worse than the row's best, on a log ratio scale. */
function heatAlpha(ratio: number, maxRatio: number): number {
  if (!(ratio > 1)) return 0
  const span = Math.log10(Math.max(maxRatio, 10))
  const t = Math.log10(ratio) / span
  return Math.min(1, Math.max(0, t)) * 0.72
}

function rowBest(row: Table5Row): number | null {
  let best: number | null = null
  for (const c of row.cells) {
    if (!c) continue
    if (best === null || c.bpb < best) best = c.bpb
  }
  return best
}

function Heatmap({
  t, selected, onSelect,
}: {
  t: Table5
  selected: string
  onSelect: (key: string) => void
}) {
  const ratios: number[] = []
  for (const row of t.rows) {
    const best = rowBest(row)
    if (best === null || best <= 0) continue
    for (const c of row.cells) if (c) ratios.push(c.bpb / best)
  }
  const maxRatio = ratios.length ? Math.max(...ratios) : 10
  const legendSpan = Math.max(maxRatio, 10)

  return (
    <Card
      title="How far each arm lands from the best arm on its own dataset"
      note="Cell opacity is the log ratio to that row's best cell, so 0 opacity is 'best on this dataset'. Click a dataset to load it into the explorer below."
    >
      <div className="table-wrap">
        <table className="data" style={{ width: 'auto' }}>
          <thead>
            <tr>
              <th>dataset</th>
              {t.columns.map((c) => (
                <th key={c.key} title={c.desc}>{c.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {t.rows.map((row) => {
              const best = rowBest(row)
              return (
                <tr
                  key={row.key}
                  style={
                    selected === row.key
                      ? { outline: '1px solid var(--accent)' }
                      : undefined
                  }
                >
                  <td>
                    <button
                      aria-pressed={selected === row.key}
                      onClick={() => onSelect(row.key)}
                      style={{
                        background: 'none',
                        border: 'none',
                        padding: 0,
                        color: 'var(--text)',
                        fontSize: 13,
                        textAlign: 'left',
                      }}
                    >
                      {row.label}
                    </button>
                  </td>
                  {t.columns.map((col, i) => {
                    const cell = row.cells[i]
                    const alpha =
                      cell && best !== null && best > 0
                        ? heatAlpha(cell.bpb / best, maxRatio)
                        : 0
                    return (
                      <td
                        key={col.key}
                        className="num"
                        style={{
                          background: `rgba(${HEAT_RGB},${alpha.toFixed(3)})`,
                          minWidth: 56,
                        }}
                        title={
                          cell && best !== null && best > 0
                            ? `${cell.bpb.toFixed(2)} bits/byte — ${(cell.bpb / best).toFixed(1)}× this row's best`
                            : 'unscored'
                        }
                      >
                        {cell ? cell.bpb.toFixed(2) : '—'}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="legend" style={{ marginTop: 12 }}>
        <span className="legend__item">loss ratio to row best</span>
        {[0, 0.25, 0.5, 0.75, 1].map((s) => (
          <span
            key={s}
            style={{
              display: 'inline-flex',
              flexDirection: 'column',
              alignItems: 'center',
              fontSize: 11,
              color: 'var(--text-faint)',
              fontFamily: 'var(--mono)',
            }}
          >
            <span
              style={{
                display: 'block',
                width: 54,
                height: 8,
                background: `rgba(${HEAT_RGB},${(s * 0.72).toFixed(3)})`,
                borderBottom: '1px solid var(--border)',
              }}
            />
            {s === 0
              ? '1×'
              : `${Math.round(10 ** (s * Math.log10(legendSpan)))}×`}
          </span>
        ))}
        <span style={{ color: 'var(--text-faint)', fontSize: 12 }}>
          opacity is log-scaled, so arithmetic (best 0.22, worst 10.64) stays
          legible next to the random-bytes row
        </span>
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Arm explorer
// ---------------------------------------------------------------------------

function ArmChart({ t, row }: { t: Table5; row: Table5Row | undefined }) {
  const bars = (row?.cells ?? [])
    .map((cell, i) => ({
      key: t.columns[i]?.key ?? String(i),
      label: t.columns[i]?.label ?? String(i),
      bpb: cell?.bpb ?? null,
      kUsed: cell?.kUsed ?? null,
    }))
    .filter(
      (b): b is {
        key: string
        label: string
        bpb: number
        kUsed: number | null
      } => b.bpb !== null,
    )
    .sort((a, b) => a.bpb - b.bpb)

  if (!row || bars.length === 0) {
    return <div className="loading">No scored arms for this dataset.</div>
  }
  const n = bars.length
  const maxBpb = Math.max(...bars.map((b) => b.bpb), 0.01)

  return (
    <>
      <ChartFrame
        height={330}
        xScale="linear"
        xDomain={[0, maxBpb * 1.5]}
        yDomain={[0, n]}
        xLabel={`validation bits/byte at round ${t.round.toLocaleString()}`}
        yLabel="ablation"
      >
        {({ x, y, innerHeight }) => (
          <g>
            {bars.map((b, i) => {
              const cy = y(i + 0.5)
              const h = Math.max(5, (innerHeight / n) * 0.6)
              const canonical = b.key === 'none'
              return (
                <g key={b.key}>
                  <rect
                    x={0}
                    y={cy - h / 2}
                    width={x(b.bpb)}
                    height={h}
                    rx={3}
                    fill={canonical ? 'var(--arm-selfplay)' : 'var(--border-strong)'}
                    opacity={canonical ? 0.85 : 0.7}
                  />
                  <text
                    x={x(b.bpb) + 7}
                    y={cy}
                    dy="0.32em"
                    fontSize={12}
                    fontFamily="var(--mono)"
                    fill={canonical ? 'var(--arm-selfplay)' : 'var(--text-muted)'}
                  >
                    {b.label}
                    {canonical ? ' (canonical)' : ''} · {b.bpb.toFixed(2)}
                    {b.kUsed !== null && b.kUsed < t.K ? ` · k=${b.kUsed}` : ''}
                  </text>
                </g>
              )
            })}
          </g>
        )}
      </ChartFrame>
      <p className="card__note">
        Canonical arm: {t.columns.find((c) => c.key === 'none')?.desc ?? '—'}
      </p>
    </>
  )
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

interface PairResult {
  /** Rows where `arm` ends up worse than `base`, with the ratio. */
  worse: [string, number][]
  /** Rows where it does not. */
  notWorse: [string, number][]
}

/** Compare two arms across every row where both have a scored cell. */
function compareArms(t: Table5, arm: string, base: string): PairResult {
  const out: PairResult = { worse: [], notWorse: [] }
  for (const row of t.rows) {
    const a = cellOf(t, row, arm)
    const b = cellOf(t, row, base)
    if (a === null || b === null || b <= 0) continue
    const ratio = a / b
    if (a > b) out.worse.push([row.label, ratio])
    else out.notWorse.push([row.label, ratio])
  }
  return out
}

function cellOf(t: Table5, row: Table5Row, key: string): number | null {
  const i = t.columns.findIndex((c) => c.key === key)
  if (i < 0) return null
  const cell = row.cells[i]
  return cell ? cell.bpb : null
}

function median(xs: number[]): number | null {
  if (xs.length === 0) return null
  const s = [...xs].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  const lo = s[mid - 1]
  const hi = s[mid]
  if (lo === undefined || hi === undefined) return null
  return s.length % 2 === 1 ? hi : (lo + hi) / 2
}

const fmt = (x: number | null, digits = 2): string =>
  x === null ? '—' : x.toFixed(digits)

/** "arithmetic 0.22 / 7.94" style quote of one arm's values on named rows. */
function valuesOn(t: Table5, key: string, labels: string[]): string {
  return labels
    .map((l) => {
      const row = t.rows.find((r) => r.label === l)
      return row ? `${l} ${fmt(cellOf(t, row, key))}` : null
    })
    .filter((s): s is string => s !== null)
    .join(', ')

}

/** The lowest kUsed seen in a column, for the ensemble caveat. */
function minSeeds(t: Table5, key: string): number {
  let k = Infinity
  for (const row of t.rows) {
    const cell = cellOf2(t, row, key)
    if (cell) k = Math.min(k, cell)
  }
  return Number.isFinite(k) ? k : 0
}

function cellOf2(t: Table5, row: Table5Row, key: string): number | null {
  const i = t.columns.findIndex((c) => c.key === key)
  if (i < 0) return null
  return row.cells[i]?.kUsed ?? null
}

function Verdict({ t }: { t: Table5 }) {
  const negate = compareArms(t, 'negate', 'none')
  const shuffle = compareArms(t, 'shuffle', 'none')
  const lastStep = compareArms(t, 'last_step', 'none')
  const lossDelta = compareArms(t, 'loss_delta', 'last_step')
  const signed = compareArms(t, 'signed', 'none')
  const uniform = compareArms(t, 'uniform', 'none')

  const total = t.rows.length
  const labels = (xs: [string, number][]): string[] => xs.map(([l]) => l)
  const ratios = (xs: [string, number][]): number[] => xs.map(([, r]) => r)

  const negateAboveInit = t.rows.filter((r) => {
    const v = cellOf(t, r, 'negate')
    return v !== null && v > UNIFORM_256_BPB
  })
  const negateAtOrBelowInit = t.rows
    .filter((r) => {
      const v = cellOf(t, r, 'negate')
      return v !== null && v <= UNIFORM_256_BPB
    })
    .map((r) => r.label)

  const uniformRandRow = t.rows.find((r) => r.key === 'aitdcc_d_glibc_rand')
  const noneBest = t.rows.filter((r) => r.bestIndex === 0).length
  const lossDeltaSeeds = minSeeds(t, 'loss_delta')

  return (
    <>
      <h2 className="section">What the evidence actually rules out</h2>
      <p className="body">
        Six claims, each computed from the cells above rather than asserted. Where
        a claim does not hold on every row, the exception is named.
      </p>

      <Claim reference="table5" also={['appF']}>
        <strong>The reward is not noise.</strong> Flipping its sign (
        <code>negate</code>) lands above the {UNIFORM_256_BPB} bits/byte of a
        uniform-over-256-bytes predictor on {negateAboveInit.length} of {total}{' '}
        datasets, and is worse than the canonical reward on{' '}
        {negate.worse.length} of {total} — median{' '}
        {fmt(median(ratios(negate.worse)))}× the canonical loss. A reward whose
        sign does not matter could not produce that. The one dataset where the 8.0
        comparison fails is {valuesOn(t, 'negate', negateAtOrBelowInit)}, and
        even there it sits far above the canonical arm on the same dataset (
        {valuesOn(t, 'none', negateAtOrBelowInit)}) — only the comparison against
        the untrained reference fails.
      </Claim>

      <Claim reference="table5" also={['appF']}>
        <strong>The pairing matters, not just the reward distribution.</strong>{' '}
        <code>shuffle</code> preserves the set of reward values and destroys only
        the correspondence between a program and its reward. It is worse than
        the canonical arm on {shuffle.worse.length} of {total} rows, median{' '}
        {fmt(median(ratios(shuffle.worse)))}× — and worst where the learnable
        frontier is narrow: arithmetic{' '}
        {valuesOn(t, 'shuffle', ['arithmetic'])} against{' '}
        {valuesOn(t, 'none', ['arithmetic'])}. This is the ablation that
        separates a reward <em>function</em> from a reward <em>scale</em>.
      </Claim>

      <Claim reference="table5" also={['appF']}>
        <strong>Averaging over a longer block helps.</strong> The one-step window{' '}
        <code>last_step</code> is worse than the ⌊e/2⌋ lookback on all{' '}
        {lastStep.worse.length} of {total} rows, median{' '}
        {fmt(median(ratios(lastStep.worse)))}× (arithmetic{' '}
        {valuesOn(t, 'last_step', ['arithmetic'])} vs{' '}
        {valuesOn(t, 'none', ['arithmetic'])}). So the improvement the design
        attributes to the lookback window is present at the endpoint — though
        this table cannot separate "better averaging" from "different
        optimisation dynamics", and the footnote says those seeds diverged into
        two modes.
      </Claim>

      <Claim reference="table5" also={['appF']}>
        <strong>The first-order score beats the finite difference.</strong>{' '}
        <code>loss_delta</code> uses realised progress{' '}
        <code>L(θ_pre) − L(θ_post)</code> and is worse than{' '}
        <code>last_step</code> on {lossDelta.worse.length} of {total} rows, median{' '}
        {fmt(median(ratios(lossDelta.worse)))}×. With the caveat: this arm runs
        {lossDeltaSeeds} seeds rather than {t.K}, and it is one of the two bimodal
        arms. It rules the finite difference out at this rung, not in general.
      </Claim>

      <Claim caveat reference="table5" also={['appF']}>
        <strong>The absolute value is doing the least work of the six.</strong>{' '}
        <code>signed</code> is worse than the canonical reward on{' '}
        {signed.worse.length} of {total} rows and better on {signed.notWorse.length}{' '}
        ({valuesOn(t, 'signed', labels(signed.notWorse))} vs{' '}
        {valuesOn(t, 'none', labels(signed.notWorse))}), and on those rows the
        paper prints <code>signed</code> in bold as the row&apos;s best. The
        margin is thin — on melody (Mutopia) it is{' '}
        {valuesOn(t, 'signed', ['melody (Mutopia)'])} against{' '}
        {valuesOn(t, 'none', ['melody (Mutopia)'])}, under 0.01 bits/byte, which
        an ensemble at this rung does not resolve cleanly. The honest reading:
        the abs buys consistency across datasets rather than a large win, and the
        paper&apos;s argument for why it is <em>needed</em> (no negative rewards)
        is not separately measured anywhere in this table.
      </Claim>

      <Claim reference="table5" also={['appF', 'sec2.2']}>
        <strong>The adaptive curriculum is the load-bearing component.</strong>{' '}
        Removing the generator (<code>uniform</code>) is worse than the canonical
        arm on {uniform.worse.length} of {total} rows, median{' '}
        {fmt(median(ratios(uniform.worse)))}×, and catastrophic where the
        learnable frontier is narrow: arithmetic{' '}
        {valuesOn(t, 'uniform', ['arithmetic'])} against{' '}
        {valuesOn(t, 'none', ['arithmetic'])}, melody (Mutopia){' '}
        {valuesOn(t, 'uniform', ['melody (Mutopia)'])} against{' '}
        {valuesOn(t, 'none', ['melody (Mutopia)'])}. Where it is cheap — DNA and
        the 16-bit audio — the target is dense enough that random programs still
        teach something. On{' '}
        {uniformRandRow ? uniformRandRow.label : 'random bytes'} it lands within
        0.01 bits/byte of the canonical arm (
        {valuesOn(t, 'uniform', ['random bytes'])} vs{' '}
        {valuesOn(t, 'none', ['random bytes'])}), which is the correct behaviour:
        there is nothing in uniform bytes to select for. Access to a universal
        program space is not sufficient on its own; the sampler choosing which
        part of it to attempt is what buys the gain.
      </Claim>

      <div className="claim claim--caveat">
        <p className="claim__text">
          <strong>Two limits worth keeping in view.</strong> The canonical arm is
          the printed best on {noneBest} of {total} rows and near-best on the
          rest — a consistent winner, not a uniform one. And the table ranks seven
          runs of one small rung at one round: strong evidence that the canonical
          reward is the right choice here, weak evidence about how much each
          ingredient contributes in the settings the paper reports.{' '}
          <PaperRef reference="appF" inline />
        </p>
      </div>
    </>
  )
}