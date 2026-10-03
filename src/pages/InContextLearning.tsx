/**
 * InContextLearning: the learner with the gradient switched off.
 *
 * Two pieces of evidence, in order. The first is Figure 4 — exact-match
 * accuracy against the number of in-context demonstrations m, for three
 * learners of matched architecture that differ only in what they were
 * pretrained on. The second is Figure 5, which is the more interesting one:
 * what the model is actually *emitting* on the SUM task as m grows, and how
 * confident it is. Figure 4 says the accuracy goes up; Figure 5 says the
 * model changes strategy to make it go up.
 *
 * Data-shape note, because it shapes a lot of this file. `tools/build_data.py`
 * parses the `m=`/`k=`/`L=` key forms into IclRow.m/IclRow.k, but three of the
 * six released result files use bare positional keys (`sum|8|2`,
 * `assoc|m=8|V=4`, `assoc|extra=2|V=16`). Those rows arrive with m and/or k
 * null. `sweepGrid` and `assocGroups` below recover the sweep coordinates from
 * the row order plus the files' own `ms`/`ks` arrays, and hand back null when
 * the reconstruction does not line up exactly — in which case the page says
 * "not measured" instead of drawing an axis with no data on it. Nothing here
 * invents a number; the reconstruction only recovers indices the harness
 * already recorded.
 */
import { useMemo, useState, type ReactNode } from 'react'
import { loadIcl, loadMeta } from '@/data/loaders'
import {
  Async, Card, Claim, Disclosure, Legend, PaperRef, Stat,
} from '@/components/UI'
import {
  ChartFrame, HoverReadout, Series, formatTick, niceTicks, useTooltip,
} from '@/components/Chart'
import { decadeTicks } from '@/lib/scales'
import type {
  IclArm, IclData, IclFile, IclRow, Meta, SumBehavior,
} from '@/data/types'

// ---------------------------------------------------------------------------
// Arms
// ---------------------------------------------------------------------------

/** Keys as they appear in icl.json, which uses `uniform_prior`, not `uniform`. */
const ARM_ORDER = ['selfplay', 'uniform_prior', 'pcfg'] as const
type IclArmKey = (typeof ARM_ORDER)[number]

const ARM_META: Record<IclArmKey, 'selfplay' | 'uniform' | 'pcfg'> = {
  selfplay: 'selfplay',
  uniform_prior: 'uniform',
  pcfg: 'pcfg',
}

const ARM_COLOR: Record<IclArmKey, string> = {
  selfplay: 'var(--arm-selfplay)',
  uniform_prior: 'var(--arm-uniform)',
  pcfg: 'var(--arm-pcfg)',
}

const ARM_DASH: Record<IclArmKey, string | null> = {
  selfplay: null,
  uniform_prior: '6 4',
  pcfg: '2 3',
}

const ARM_LABEL: Record<IclArmKey, string> = {
  selfplay: 'Self-play',
  uniform_prior: 'Universal prior',
  pcfg: 'PCFG',
}

/** Chance for a single byte under argmax: 1/256 = 0.39%. */
const CHANCE = 1 / 256

const pct = (v: number | null | undefined, digits = 1): string =>
  v === null || v === undefined ? '—' : `${(v * 100).toFixed(digits)}%`

// ---------------------------------------------------------------------------
// Coordinate recovery for the positional-key files
// ---------------------------------------------------------------------------

/** The harness's context budget, and the fixed prompt prefix cost. */
const CONTEXT_BYTES = 4096
const PROMPT_PREFIX = 1

interface Cell { task: string; m: number; k: number }

/**
 * Recover (task, m, k) for a sweep file whose keys were `task|m|k` and whose
 * rows therefore carry null m and k. The harness emitted, for each task in
 * turn, every m x k pair in ascending order, skipping pairs whose prompt would
 * overflow the context. Both facts are checked against the row count before the
 * result is used.
 */
function sweepGrid(file: IclFile | undefined): Cell[] | null {
  if (!file) return null
  const ms = file.ms
  const ks = file.ks
  if (!ms || !ks || ms.length === 0 || ks.length === 0) return null

  const pairs: [number, number][] = []
  for (const m of ms) {
    for (const k of ks) {
      if (PROMPT_PREFIX + m * (k + 2) + k + 1 > CONTEXT_BYTES) continue
      pairs.push([m, k])
    }
  }

  const tasks: string[] = []
  for (const r of file.rows) if (!tasks.includes(r.task)) tasks.push(r.task)

  let cursor = 0
  const grid: Cell[] = []
  for (const task of tasks) {
    const own = file.rows.filter((r) => r.task === task).length
    if (own !== pairs.length) return null
    for (const [m, k] of pairs) {
      grid.push({ task, m, k })
      cursor += 1
    }
  }
  return grid.length === file.rows.length ? grid : null
}

/**
 * The three associative-recall dictionary sizes in icl_v3_results, keyed by
 * `V=`. V was dropped by the flattener, but the file emitted one contiguous,
 * strictly m-increasing block per V, so the blocks recover V by position.
 */
const ASSOC_V = [4, 16, 64] as const
/** The largest dictionary size in the sweep: where the arms actually differ. */
const ASSOC_V_MAX: number = ASSOC_V[2]

interface AssocGroup { v: number; rows: IclRow[] }

function assocGroups(file: IclFile | undefined): AssocGroup[] | null {
  if (!file) return null
  const rows = file.rows.filter((r) => r.task === 'assoc' && r.m !== null)
  const groups: AssocGroup[] = []
  for (const row of rows) {
    const head = groups[groups.length - 1]
    const m = row.m ?? 0
    if (head && m > (head.rows[head.rows.length - 1]?.m ?? Infinity)) {
      head.rows.push(row)
    } else {
      groups.push({ v: ASSOC_V[groups.length] ?? -1, rows: [row] })
    }
  }
  if (groups.length !== ASSOC_V.length) return null
  return groups.some((g) => g.v < 0) ? null : groups
}

// ---------------------------------------------------------------------------
// Series extraction (every metric lookup is guarded: names differ by harness)
// ---------------------------------------------------------------------------

interface Pt { m: number; v: number; p: number | null }

function armOf(icl: IclData, arm: IclArmKey): IclArm | null {
  return icl.arms[arm] ?? null
}

/** Accuracy against m for one arm, one task, one arity, one metric. */
function sweepSeries(
  icl: IclData, arm: IclArmKey, fileName: string,
  task: string, k: number, metric: string,
): Pt[] | null {
  const file = armOf(icl, arm)?.files[fileName]
  const grid = sweepGrid(file)
  if (!file || !grid) return null
  const out: Pt[] = []
  file.rows.forEach((row, i) => {
    const cell = grid[i]
    if (!cell || cell.task !== task || cell.k !== k) return
    const v = row.metrics[metric]
    if (v === undefined) return
    out.push({ m: cell.m, v, p: row.metrics['p_correct'] ?? null })
  })
  if (out.length === 0) return null
  return out.sort((a, b) => a.m - b.m)
}

/** One cell of the main sweep, for the numbers quoted in prose. */
function sweepCell(
  icl: IclData, arm: IclArmKey, task: string, k: number, m: number,
): number | null {
  const series = sweepSeries(icl, arm, 'icl_results', task, k, 'acc')
  if (!series) return null
  return series.find((p) => p.m === m)?.v ?? null
}

function extraSeries(
  icl: IclData, arm: IclArmKey, task: string,
): Pt[] | null {
  const file = armOf(icl, arm)?.files['icl_v3_results']
  if (!file) return null
  const out: Pt[] = []
  for (const row of file.rows) {
    if (row.task !== task || row.m === null) continue
    const v = row.metrics['acc']
    if (v === undefined) continue
    out.push({ m: row.m, v, p: row.metrics['p_correct'] ?? null })
  }
  return out.length > 0 ? out.sort((a, b) => a.m - b.m) : null
}

function assocSeries(
  icl: IclData, arm: IclArmKey, v: number,
): Pt[] | null {
  const groups = assocGroups(armOf(icl, arm)?.files['icl_v3_results'])
  const group = groups?.find((g) => g.v === v)
  if (!group) return null
  const out: Pt[] = []
  for (const row of group.rows) {
    if (row.m === null) continue
    const a = row.metrics['acc']
    if (a === undefined) continue
    out.push({ m: row.m, v: a, p: row.metrics['p_correct'] ?? null })
  }
  return out.length > 0 ? out.sort((a, b) => a.m - b.m) : null
}

/** Accuracy in the largest-m cell of a series, for the numbers quoted in prose. */
function lastOf(pts: Pt[] | null): number | null {
  return pts !== null && pts.length > 0 ? (pts[pts.length - 1]?.v ?? null) : null
}
function assocTop(icl: IclData, arm: IclArmKey, v: number): number | null {
  return lastOf(assocSeries(icl, arm, v))
}
function extraTop(icl: IclData, arm: IclArmKey, task: string): number | null {
  return lastOf(extraSeries(icl, arm, task))
}
/** Best cell anywhere in a task's sweep, for claims phrased as "never above". */
function extraBest(icl: IclData, arm: IclArmKey, task: string): number | null {
  const pts = extraSeries(icl, arm, task)
  return pts !== null && pts.length > 0
    ? Math.max(...pts.map((p) => p.v))
    : null
}

function m0(icl: IclData, arm: IclArmKey, task: string, metric: string): number | null {
  const file = armOf(icl, arm)?.files['icl_m0_results']
  if (!file) return null
  for (const row of file.rows) {
    if (row.task !== task) continue
    const v = row.metrics[metric]
    if (v !== undefined) return v
  }
  return null
}

/** The 4096-trial sum cells for m <= 4, which the main sweep does not cover. */
const SUM_LOW_M = [0, 1, 2, 3, 4]
function sumLowM(icl: IclData, arm: IclArmKey): number[] | null {
  const file = armOf(icl, arm)?.files['icl_sum_lowm_results']
  if (!file) return null
  const sum = file.rows.filter((r) => r.task === 'sum')
  if (sum.length !== SUM_LOW_M.length) return null
  return sum.map((r) => r.metrics['acc'] ?? Number.NaN)
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function InContextLearning() {
  return (
    <>
      <h1 className="page__title">In-context learning, with the gradient off</h1>
      <p className="page__lede">
        Take the learner, point it at a task it was never trained on, and forbid
        it from updating. It sees a handful of demonstrations of an unknown rule
        and has to infer that rule from the context alone. This is the cleanest
        evidence in the paper that self-play taught transferable{' '}
        <em>structure</em> rather than facts about one dataset. It is also one of
        the few places where all three pretraining arms were scored under a
        single identical protocol, so the comparison is apples-to-apples.
      </p>

      <Claim reference="sec3.2" also={['fig4', 'fig1']}>
        On six held-out tasks the self-play learner gets steadily better as
        demonstrations are added, with no gradient step and no fine-tuning of
        any kind. The same architecture pretrained on programs from the fixed
        universal prior, and one pretrained on random PCFG output, are scored
        under an identical protocol.
      </Claim>

      <h2 className="section">The task format, precisely</h2>
      <p className="body">
        Every task uses one byte-level format. A single prefix byte{' '}
        <code>O</code>, then <em>m</em> demonstrations of the form sentinel{' '}
        <code>0</code>, the <em>k</em> input bytes, then <code>f(x)</code>; then a
        query consisting of a sentinel and <em>k</em> fresh input bytes. The
        model's next-byte distribution is read at that position and scored
        greedily. Because the alphabet is 256 bytes, chance is{' '}
        <strong>1/256 = 0.39%</strong> — anything above a few percent is
        structure, and anything at 0.0% is a model that has genuinely failed to
        find the rule.
      </p>
      <Card
        title="The same prompt, written out"
        note="sum task, k = 2, m = 2. Query input 200 and 100; the answer is (200 + 100) mod 256 = 44."
      >
        <div className="program">
          <span className="program__prefix">O</span>
          {'  '}
          <span className="program__op">0</span> 200 100 44
          {'  '}
          <span className="program__op">0</span> 200 100 44
          {'  '}
          <span className="program__op">0</span>{' '}
          <span style={{ color: 'var(--accent)' }}>200 100</span>
          {' → '}
          <span style={{ color: 'var(--warn)' }}>?</span>
        </div>
        <p className="card__note" style={{ marginTop: 10 }}>
          Input bytes are drawn from 1–255; 0 is reserved for the sentinel, so
          the sentinel is unambiguous. Output is a single byte, so the target is
          one greedy argmax over 256 options.{' '}
          <PaperRef reference="appD" inline />
        </p>
      </Card>

      <Async load={loadMeta}>
        {(meta) => (
          <>
            <h2 className="section">What each task is a probe for</h2>
            <p className="body">
              The tasks are chosen so that each one isolates a different
              mechanism. The paper's own framing is the row labels here.
            </p>
            <TaskTable meta={meta} />
          </>
        )}
      </Async>

      <Async load={loadMeta}>
        {(meta) => (
          <Async load={loadIcl}>
            {(icl) => (
              <>
                <h2 className="section">Figure 4 — accuracy against context</h2>
                <SweepExplorer icl={icl} meta={meta} />
                <ExtrasExplorer icl={icl} meta={meta} />
                <M0Note icl={icl} />
              </>
            )}
          </Async>
        )}
      </Async>

      <Async load={loadIcl}>
        {(icl) => <SumBehaviorSection icl={icl} />}
      </Async>

      <CaveatCard />
    </>
  )
}

// ---------------------------------------------------------------------------
// Task table
// ---------------------------------------------------------------------------

/** Capability each task isolates, in the paper's framing. */
const CAPABILITY: Record<string, string> = {
  max: 'comparison across the input',
  min: 'comparison across the input',
  mean: 'arithmetic aggregation',
  sum: 'arithmetic aggregation',
  first: 'fixed-address copying (induction heads)',
  last: 'fixed-address copying (induction heads)',
  assoc: 'contextual search',
  index: 'dynamic indexing',
  palin: 'copying with a position-dependent lag',
  stack: 'simulating a context-free grammar',
  succ: 'applying a fixed rule',
}

/** Rows we can actually plot, and the file each one lives in. */
const MEASURED: Record<string, string> = {
  max: 'icl_results', min: 'icl_results', mean: 'icl_results',
  sum: 'icl_results', first: 'icl_results', last: 'icl_results',
  assoc: 'icl_v3_results', stack: 'icl_v4_results', palin: 'icl_v4_results',
}

function TaskTable({ meta }: { meta: Meta }) {
  const keys = [
    ...Object.keys(meta.iclTasks),
    ...Object.keys(meta.iclExtraTasks),
  ]
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            <th>Task</th>
            <th style={{ textAlign: 'left' }}>What it does</th>
            <th style={{ textAlign: 'left' }}>What it isolates</th>
            <th style={{ textAlign: 'left' }}>In the released bundle</th>
          </tr>
        </thead>
        <tbody>
          {keys.map((key) => {
            const info = meta.iclTasks[key] ?? meta.iclExtraTasks[key]
            if (!info) return null
            const file = MEASURED[key]
            return (
              <tr key={key}>
                <td>{info.label}</td>
                <td style={{ textAlign: 'left', whiteSpace: 'normal' }}>
                  {info.blurb}
                </td>
                <td style={{ textAlign: 'left' }}>{CAPABILITY[key] ?? '—'}</td>
                <td style={{ textAlign: 'left' }}>
                  {file ? <code>{file}</code> : 'not measured'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="card__note">
        Two probes named in §3.2 are <strong>not</strong> in the released Figure 4
        bundle: <code>index</code> (reverse string, dynamic indexing) lives in an{' '}
        <code>icl_v2_results</code> file that was not published alongside the
        three arms, and <code>shift</code> likewise. We say so rather than
        implying the panel exists. <PaperRef reference="appD" inline />
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Figure 4 explorer
// ---------------------------------------------------------------------------

type MetricKey = 'acc' | 'p_correct'

const METRIC_LABEL: Record<MetricKey, string> = {
  acc: 'exact-match accuracy (argmax)',
  p_correct: 'probability mass on the correct byte',
}

function SweepExplorer({ icl, meta }: { icl: IclData; meta: Meta }) {
  const tasks = useMemo(
    () => Object.keys(meta.iclTasks).filter((t) => MEASURED[t] === 'icl_results'),
    [meta],
  )
  const [task, setTask] = useState('sum')
  const [k, setK] = useState(2)
  const [metric, setMetric] = useState<MetricKey>('acc')

  const ks = useMemo(() => {
    const file = armOf(icl, 'selfplay')?.files['icl_results']
    return file?.ks ?? []
  }, [icl])

  const series = useMemo(
    () => ARM_ORDER.map((arm) => ({
      arm, pts: sweepSeries(icl, arm, 'icl_results', task, k, metric),
    })),
    [icl, task, k, metric],
  )
  const plotted = series.filter((s) => s.pts && s.pts.length > 1)
  const blurb = meta.iclTasks[task]

  return (
    <>
      <div className="controls">
        <label className="field">
          <span className="field__label">Task</span>
          <select value={task} onChange={(e) => setTask(e.target.value)}>
            {tasks.map((t) => (
              <option key={t} value={t}>{meta.iclTasks[t]?.label ?? t}</option>
            ))}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Inputs per example (k)</span>
          <select value={k} onChange={(e) => setK(Number(e.target.value))}>
            {ks.map((kk) => <option key={kk} value={kk}>{kk}</option>)}
          </select>
        </label>
        <label className="field">
          <span className="field__label">Metric</span>
          <select
            value={metric}
            onChange={(e) => setMetric(e.target.value as MetricKey)}
          >
            {(['acc', 'p_correct'] as const).map((mk) => (
              <option key={mk} value={mk}>{METRIC_LABEL[mk]}</option>
            ))}
          </select>
        </label>
      </div>

      <Card
        title={`${blurb?.label ?? task} — ${blurb?.blurb ?? ''}`.trim()}
        note={`${armOf(icl, 'selfplay')?.trials ?? '—'} trials per cell, three arms, same prompt, same decoder. Chance is 1/256.`}
      >
        {plotted.length === 0 ? (
          <p className="body">Not measured for this combination.</p>
        ) : (
          <ArmChart
            series={plotted}
            ms={Array.from(new Set(plotted.flatMap(
              (s) => s.pts?.map((p) => p.m) ?? [],
            ))).sort((a, b) => a - b)}
            yMax={Math.max(
              0.1,
              ...plotted.flatMap((s) => s.pts?.map((p) => p.v) ?? []),
            )}
            xLabel="in-context demonstrations m (log scale)"
            yLabel={metric === 'acc' ? 'exact-match accuracy' : 'P(correct byte)'}
          />
        )}
        <Legend
          items={ARM_ORDER.map((arm) => ({
            label: `${ARM_LABEL[arm]} — ${meta.arms[ARM_META[arm]].blurb}`,
            color: ARM_COLOR[arm],
            dash: ARM_DASH[arm],
          }))}
        />
      </Card>

      <div className="grid-3" style={{ marginTop: 16 }}>
        <Stat
          value={pct(sweepCell(icl, 'selfplay', task, k, 512) ??
            sweepCell(icl, 'selfplay', task, k, 256))}
          label={`self-play at the largest m this task was run at`}
        />
        <Stat
          value={pct(sweepCell(icl, 'uniform_prior', task, k, 512) ??
            sweepCell(icl, 'uniform_prior', task, k, 256))}
          label="universal prior, same cell"
        />
        <Stat
          value={pct(sweepCell(icl, 'pcfg', task, k, 512) ??
            sweepCell(icl, 'pcfg', task, k, 256))}
          label="PCFG, same cell"
        />
      </div>

      <Claim reference="fig4" also={['sec3.2']}>
        Self-play improves on all six tasks, though by very different margins:
        first and last saturate at 100%, sum reaches 87.5%, max and min land
        near 65–69%, and mean is the outlier at 12.5% — flooring a mean of
        several bytes is much harder than summing two. The universal-prior arm
        shows almost no effective in-context learning on this family at all
        (its best single cell across all 222 is 3.1%), so this is not a case of
        "any program-pretrained model learns to use context".
      </Claim>

      <Disclosure summary="Why the k axis is the interesting one">
        <p className="body">
          Accuracy falls off with k for every arm, and on max it falls off{' '}
          <em>fastest</em> for self-play: at m = 1 the self-play model answers{' '}
          {pct(sweepCell(icl, 'selfplay', 'max', 2, 1))} on max with two inputs
          but only {pct(sweepCell(icl, 'selfplay', 'max', 16, 1))} with sixteen.
          What is being learned is a <em>procedure</em> whose generalisation cost
          grows with the width of the input, not a lookup table keyed on short
          inputs — which is also why the grid is not rectangular. The harness
          skips any cell whose prompt would exceed the 4096-byte context, so at
          k = 16 the sweep stops at m = 128 and the missing cells are genuinely
          not measured rather than measured at zero. The empty top-right of the
          plot is a context limit, and this page does not draw through it.{' '}
          <PaperRef reference="fig4" inline />
        </p>
      </Disclosure>
    </>
  )
}

/** Line chart, one line per arm, log x (m runs 1..512). */
function ArmChart({
  series, ms, yMax, xLabel, yLabel,
}: {
  series: { arm: IclArmKey; pts: Pt[] | null }[]
  ms: number[]
  yMax: number
  xLabel: string
  yLabel: string
}) {
  const { payload, setPayload } = useTooltip()
  const hi = Math.min(1, Math.ceil(yMax * 10) / 10)
  const lo = ms[0] ?? 1
  const hiM = ms[ms.length - 1] ?? lo
  return (
    <>
      <ChartFrame
        height={340}
        xScale="log"
        xDomain={[lo, hiM]}
        yDomain={[0, hi]}
        xLabel={xLabel}
        yLabel={yLabel}
      >
        {({ x, y, innerWidth, innerHeight }) => (
          <g>
            {/* One label per swept m, in the tick row ChartFrame already
                reserves. ChartFrame also prints power-of-ten ticks there, so
                any power of two that would collide with one is dropped rather
                than overprinted -- the decade label reads as the same scale. */}
            <g className="chart__tick">
              {ms.map((m) => {
                const decades = decadeTicks([lo, hiM])
                // 34px of clearance in a ~640px plot: enough for a 3-digit
                // label plus a power-of-ten label on either side of it.
                if (decades.some((d) => Math.abs(x(d) - x(m)) < 34)) return null
                return (
                  <text key={`lab${m}`} x={x(m)} y={innerHeight + 18}
                        textAnchor="middle">
                    {m}
                  </text>
                )
              })}
            </g>
            <line
              x1={0} x2={innerWidth} y1={y(CHANCE)} y2={y(CHANCE)}
              stroke="var(--warn)" strokeDasharray="2 4" strokeWidth={1}
            />
            <text x={innerWidth} y={y(CHANCE) - 5} textAnchor="end"
                  fill="var(--warn)" fontSize={11}>
              chance, 1/256
            </text>
            {series.map((s) => (
              <Series
                key={s.arm}
                points={(s.pts ?? []).map((p) => [x(p.m), y(p.v)])}
                color={ARM_COLOR[s.arm]}
                dash={ARM_DASH[s.arm]}
                width={s.arm === 'selfplay' ? 2 : 1.5}
              />
            ))}
            {series.map((s) => (
              (s.pts ?? []).map((p) => (
                <circle
                  key={`${s.arm}${p.m}`}
                  className="chart__point"
                  cx={x(p.m)} cy={y(p.v)}
                  r={s.arm === 'selfplay' ? 3 : 2.5}
                  fill={ARM_COLOR[s.arm]}
                  onMouseMove={(e) => setPayload({
                    x: e.clientX,
                    y: e.clientY,
                    title: `m = ${p.m}`,
                    rows: [
                      [ARM_LABEL[s.arm], pct(p.v)],
                      ...(p.p !== null
                        ? [['P(correct byte)', pct(p.p, 2)] as [string, string]]
                        : []),
                    ],
                  })}
                  onMouseLeave={() => setPayload(null)}
                />
              ))
            ))}
          </g>
        )}
      </ChartFrame>
      <HoverReadout payload={payload} />
    </>
  )
}

// ---------------------------------------------------------------------------
// Associative recall and the arithmetic relations
// ---------------------------------------------------------------------------

function ExtrasExplorer({ icl, meta }: { icl: IclData; meta: Meta }) {
  const assocAvailable = useMemo(
    () => ARM_ORDER.some((arm) => assocSeries(icl, arm, ASSOC_V_MAX)),
    [icl],
  )
  const tasks = useMemo(() => {
    const found = new Set<string>()
    for (const arm of ARM_ORDER) {
      for (const row of armOf(icl, arm)?.files['icl_v3_results']?.rows ?? []) {
        found.add(row.task)
      }
    }
    return Array.from(found).sort()
  }, [icl])
  const [task, setTask] = useState('assoc')
  const [v, setV] = useState<number>(ASSOC_V_MAX)

  const series = useMemo(
    () => ARM_ORDER.map((arm) => ({
      arm,
      pts: task === 'assoc' ? assocSeries(icl, arm, v) : extraSeries(icl, arm, task),
    })),
    [icl, task, v],
  )
  const plotted = series.filter((s) => s.pts && s.pts.length > 1)
  const info = meta.iclExtraTasks[task] ?? meta.iclTasks[task]
  const bestAssoc = sweepCell(icl, 'pcfg', 'sum', 2, 512)

  return (
    <>
      <h2 className="section">The probe the paper leans on hardest</h2>
      <p className="body">
        Associative recall is the cleanest of these tasks: each sequence prints a
        dictionary of key–value byte pairs, and the query asks for a key the
        model has already seen. There is no rule to infer beyond "look it up",
        so success is contextual search and nothing else. It is also the one
        place the PCFG baseline is genuinely strong, which is worth seeing
        rather than hiding.
      </p>
      <div className="controls">
        <label className="field">
          <span className="field__label">Task</span>
          <select value={task} onChange={(e) => setTask(e.target.value)}>
            {tasks.map((t) => (
              <option key={t} value={t}>{meta.iclExtraTasks[t]?.label ?? t}</option>
            ))}
          </select>
        </label>
        {task === 'assoc' && (
          <label className="field">
            <span className="field__label">Dictionary size V</span>
            <select value={v} onChange={(e) => setV(Number(e.target.value))}>
              {ASSOC_V.map((vv) => <option key={vv} value={vv}>{vv}</option>)}
            </select>
          </label>
        )}
      </div>
      <Card
        title={info?.label ?? task}
        note={task === 'assoc'
          ? `One shared dictionary of ${v} byte pairs per sequence; the query key was printed in the context. 128 trials per cell.`
          : 'A fixed rule the model must apply to a fresh input. 128 trials per cell.'}
      >
        {plotted.length === 0 ? (
          <p className="body">Not measured for this arm set.</p>
        ) : (
          <ArmChart
            series={plotted}
            ms={Array.from(new Set(plotted.flatMap(
              (s) => s.pts?.map((p) => p.m) ?? [],
            ))).sort((a, b) => a - b)}
            yMax={Math.max(0.1, ...plotted.flatMap((s) => s.pts?.map((p) => p.v) ?? []))}
            xLabel="in-context demonstrations m (log scale)"
            yLabel="exact-match accuracy"
          />
        )}
        <Legend
          items={ARM_ORDER.map((arm) => ({
            label: ARM_LABEL[arm], color: ARM_COLOR[arm], dash: ARM_DASH[arm],
          }))}
        />
      </Card>

      <div className="grid-3" style={{ marginTop: 16 }}>
        <Stat
          value={pct(assocTop(icl, 'selfplay', v))}
          label={`self-play, V = ${v}, largest m`}
        />
        <Stat
          value={pct(assocTop(icl, 'pcfg', v))}
          label={`PCFG, V = ${v}, same cell`}
        />
        <Stat
          value={pct(extraTop(icl, 'selfplay', 'succ'))}
          label="self-play on successor, f(x) = x + 1, largest m"
        />
      </div>

      <Claim reference="sec3.2" also={['fig4']}>
        PCFG is the stronger learner on associative recall, and the gap widens
        with the dictionary size rather than staying fixed: at V = 4 both arms
        are near saturation, but at V = 64 PCFG reaches{' '}
        {pct(assocTop(icl, 'pcfg', 64))} against self-play's{' '}
        {pct(assocTop(icl, 'selfplay', 64))}. It then fails to transfer. On SUM
        the same PCFG model scores {pct(bestAssoc)} at the largest m, and on the
        successor task {pct(extraTop(icl, 'pcfg', 'succ'))} against self-play's{' '}
        {pct(extraTop(icl, 'selfplay', 'succ'))}. Self-play is not the best model
        everywhere; it is the one that learned something reusable.
      </Claim>

      <p className="card__note">
        The arithmetic relations in this file make the same point from the other
        side. On <code>compl</code> (y = 255 − x) and <code>double</code> (y = 2x
        mod 256) — where copying a context byte is useless — self-play climbs to{' '}
        {pct(extraTop(icl, 'selfplay', 'compl'))} and{' '}
        {pct(extraTop(icl, 'selfplay', 'double'))} at the largest m, the
        universal-prior arm never exceeds{' '}
        {pct(extraBest(icl, 'uniform_prior', 'compl'), 2)}, and PCFG never
        clears {pct(extraBest(icl, 'pcfg', 'compl'))}. Switch the task selector
        above to <code>succ</code> to see the saturated case: self-play hits
        100% where PCFG stays under{' '}
        {pct(extraBest(icl, 'pcfg', 'succ'))}.{' '}
        <PaperRef reference="fig4" inline />
      </p>

      {assocAvailable && (
        <p className="card__note">
          One more associative-recall cell is measured only as a sweep over how
          many <em>extra</em> dictionary prints follow the first: self-play goes{' '}
          {(() => {
            const rows = armOf(icl, 'selfplay')?.files['icl_assoc_dict_results']?.rows
            const a = rows?.[0]?.metrics['acc']
            const b = rows?.[rows.length - 1]?.metrics['acc']
            return a !== undefined && b !== undefined
              ? `${pct(a)} → ${pct(b)}` : '—'
          })()}
          against PCFG's{' '}
          {(() => {
            const rows = armOf(icl, 'pcfg')?.files['icl_assoc_dict_results']?.rows
            const a = rows?.[0]?.metrics['acc']
            const b = rows?.[rows.length - 1]?.metrics['acc']
            return a !== undefined && b !== undefined
              ? `${pct(a)} → ${pct(b)}` : '—'
          })()}
          , and the universal-prior arm is exactly 0.0% at every point. The
          released bundle drops the sweep coordinate from those rows, so we
          report the endpoints rather than draw a curve against a guessed axis.
        </p>
      )}
    </>
  )
}

// ---------------------------------------------------------------------------
// The m = 0 honesty note
// ---------------------------------------------------------------------------

function M0Note({ icl }: { icl: IclData }) {
  const spMax = m0(icl, 'selfplay', 'max', 'acc')
  const spMin = m0(icl, 'selfplay', 'min', 'acc')
  const upMax = m0(icl, 'uniform_prior', 'max', 'acc')
  const pcMax = m0(icl, 'pcfg', 'max', 'acc')
  const spStack = m0(icl, 'selfplay', 'stack', 'acc')
  const low = sumLowM(icl, 'selfplay')

  return (
    <>
      <h2 className="section">What m = 0 tells you — and what it does not</h2>
      <div className="grid-3">
        <Stat value={pct(spMax, 2)} label="self-play, max, zero examples" />
        <Stat value={pct(spMin, 2)} label="self-play, min, zero examples" />
        <Stat value={pct(upMax, 2)} label="universal prior, max, zero examples" />
      </div>
      <div className="claim claim--caveat" style={{ marginTop: 14 }}>
        <p className="claim__text">
          <strong>This is a copy prior, not task understanding.</strong> With no
          demonstrations at all the self-play model still scores{' '}
          {pct(spMax, 1)} on max and {pct(spMin, 1)} on min — roughly twelve
          times chance. The rule cannot have been inferred from an empty context, so
          the only thing that can be happening is that the model is copying a
          byte it has produced before, which on these inputs lands on the answer
          by accident. Any ICL claim that starts at m = 1 and does not subtract
          this floor is overstating its result.{' '}
          <PaperRef reference="fig4" inline />
        </p>
      </div>
      <p className="body">
        Two further cells are worth stating plainly rather than burying. The
        universal-prior arm scores exactly 0.0% here, so it has no copy prior at
        all — the gap between it and self-play at m = 0 is a real difference, not
        a shared offset. And PCFG scores {pct(pcMax, 1)} on max at m = 0, which
        looks like a strong result until you notice it comes with no context
        whatsoever: whatever it is doing, it is a fixed bias toward certain byte
        values, not task understanding. It does not survive contact with the
        task — at m = 1 with sixteen inputs PCFG is back at 1.6%.
      </p>
      {low && (
        <p className="card__note">
          The small-m cells are noisy enough to need their own run: with 4096
          trials instead of {armOf(icl, 'selfplay')?.trials ?? '—'}, self-play's
          SUM accuracy at m = 0, 1, 2, 3, 4 is{' '}
          {low.map((v) => pct(v, 2)).join(', ')}. The first two sit just
          <em>below</em> the {pct(CHANCE, 2)} chance line and only m = 4 is
          meaningfully above it, so SUM genuinely starts from nothing — unlike
          max and min, which carry an m = 0 floor of several percent.
        </p>
      )}
      <p className="card__note">
        The m = 0 stack cell is the odd one out: self-play scores{' '}
        {pct(spStack, 1)} with zero demonstrations. The released data does not
        say why, and it is well above the 1/256 a single-byte guess would earn,
        so we report the number and leave the mechanism unexplained.
      </p>
    </>
  )
}

// ---------------------------------------------------------------------------
// Figure 5
// ---------------------------------------------------------------------------

const CATEGORY_KEYS = [
  'correctAnswer', 'low4BitsCorrect', 'preferredBytes', 'contextByte', 'other',
] as const
type CategoryKey = (typeof CATEGORY_KEYS)[number]

function categoryValue(row: SumBehavior, key: string): number | null {
  const hit = CATEGORY_KEYS.find((c) => c === key)
  return hit === undefined ? null : row[hit]
}

function SumBehaviorSection({ icl }: { icl: IclData }) {
  const rows = [...icl.sumBehavior].sort((a, b) => a.m - b.m)
  if (rows.length === 0) {
    return (
      <>
        <h2 className="section">Figure 5 — what the model is emitting</h2>
        <p className="body">No behavioural rows in the bundle.</p>
      </>
    )
  }
  const first = rows[0] ?? null
  const at = (m: number): SumBehavior | null => rows.find((r) => r.m === m) ?? null
  /** Category share at a given m; falls back to the smallest m on record. */
  const shareAt = (m: number, k: CategoryKey): number | null => {
    const r = at(m) ?? first
    return r === null ? null : r[k]
  }
  const show = (m: number, k: CategoryKey, digits = 1): string =>
    pct(shareAt(m, k), digits)
  const m512 = at(512)
  const peak = rows.reduce((best, r) => (r.entropyMeanBits > best.entropyMeanBits ? r : best))
  const bandAt = (m: number): number => {
    const r = at(m)
    return r === null ? 0 : r.entropyQ75Bits - r.entropyQ25Bits
  }

  return (
    <>
      <h2 className="section">Figure 5 — the SUM task, behaviourally</h2>
      <p className="body">
        Figure 4 plots accuracy. Figure 5 asks the more useful question: what
        is the model actually emitting? Every trial's argmax prediction at the
        query position is classified into one of five buckets, and the entropy
        of the predictive distribution is recorded alongside. This is 1,024
        trials per context length on the self-play model.
      </p>

      <Card
        title="(a) What the prediction is, as a share of trials"
        note="Categories are applied in the order of precedence the harness used: correct answer, then the bytes the model favours, then low-nibble agreement, then any byte seen in the prompt, then everything else."
      >
        <CompositionChart rows={rows} categories={icl.categories} />
        <Legend
          items={icl.categories.map((c) => ({
            label: c.label, color: c.color,
          }))}
        />
      </Card>

      <Card
        title="(b) Predictive entropy of the next byte"
        note="Mean over trials, with the interquartile range shaded. A uniform distribution over 256 bytes is 8.0 bits."
      >
        <EntropyChart rows={rows} />
      </Card>

      <h3 className="section" style={{ marginTop: 28 }}>The trajectory</h3>
      <ol className="body">
        <li>
          <strong>Prior output.</strong> At m = 0 the model emits one of its
          favourite bytes {show(0, 'preferredBytes')} of the time and is exactly
          right {show(0, 'correctAnswer', 2)}. That is a marginal-frequency
          guess, not a computation: its exact accuracy at m = 0 is actually
          slightly below the 1/256 chance line, so a large favourite-byte share
          is not evidence of anything on its own.
        </li>
        <li>
          <strong>Copying from context.</strong> By m = 3 the context-byte bucket
          has risen from {show(0, 'contextByte')} to {show(3, 'contextByte')} and
          by m = 4 preferred bytes have collapsed to {show(4, 'preferredBytes')}.
          The model has learned to attend to the prompt, which is not yet the
          same as computing anything.
        </li>
        <li>
          <strong>Losing confidence.</strong> Entropy <em>rises</em> through this
          stage rather than falling, peaking at m = {peak.m} (
          {peak.entropyMeanBits.toFixed(2)} bits, up from{' '}
          {(first?.entropyMeanBits ?? 0).toFixed(2)} at m = 0). The reading is
          that the model commits, gets it wrong several times running, and is
          pushed off its confident prior onto a broad distribution. The
          confidence curve is not monotone, and a plot showing only accuracy
          would hide that completely.
        </li>
        <li>
          <strong>Partial arithmetic.</strong> The low-nibble bucket is applied
          after the exact one, so it counts trials that get the low four bits
          right <em>without</em> getting the whole byte right. At m = 4 that is{' '}
          {show(4, 'low4BitsCorrect')} of trials against {show(4, 'correctAnswer')}
          {' '}exact: the model is summing the low nibble and getting the high
          nibble wrong. It is a recognisable intermediate strategy, not a smooth
          fade-in of correctness, and it persists for a long time —{' '}
          {show(32, 'low4BitsCorrect')} of trials still have the low nibble alone
          correct at m = 32.
        </li>
        <li>
          <strong>Locking in.</strong> The exact bucket becomes the single
          largest category at m = 32, after which it dominates:{' '}
          {show(32, 'correctAnswer')} at m = 32, {show(64, 'correctAnswer')} at
          m = 64, {m512 ? pct(m512.correctAnswer) : '—'} at m = 512. Mean entropy
          falls to {m512 ? m512.entropyMeanBits.toFixed(2) : '—'} bits against{' '}
          {first ? first.entropyMeanBits.toFixed(2) : '—'} at m = 0.
        </li>
      </ol>

      <Claim reference="fig5" also={['sec3.2']}>
        The accuracy gain is not one model getting gradually less wrong. It is a
        model changing strategy — prior, then copying, then partial arithmetic,
        then the real computation — and each switch has a signature in panel
        (a) before it shows up as accuracy. The entropy spike around m ={' '}
        {peak.m} is the visible cost of the transition.
      </Claim>

      <p className="card__note">
        Panel (b) ends at {m512 ? m512.entropyMeanBits.toFixed(2) : '—'} bits,
        not at 0, so the model is far from a point mass at the end. One thing
        the interquartile band does <em>not</em> do is shrink: it is{' '}
        {bandAt(0).toFixed(2)} bits wide at m = 0 and{' '}
        {m512 ? bandAt(512).toFixed(2) : '—'} bits at m = 512, i.e. wider in
        absolute terms. That is not a contradiction — the distribution as a whole
        moves away from uniform, so a fixed spread subtends a smaller fraction of
        the remaining entropy. Read panel (b) as the level dropping, not as the
        trials converging on each other.
      </p>
    </>
  )
}

/**
 * Figure 5's two panels.
 *
 * These use a categorical strip frame rather than ChartFrame: m here is
 * {0,1,2,3,4,8,16,...,512}, so the cells are laid out at equal pitch and
 * labelled with the m they represent. ChartFrame's numeric ticks would be
 * index numbers pretending to be m, which would be wrong. Everything else --
 * margins, tick styling, axis labels -- matches ChartFrame's geometry, and the
 * hand-rolled SVG is the same choice D4 makes everywhere else.
 */
const STRIP = { width: 720, top: 18, right: 24, bottom: 46, left: 60 }

interface StripGeom {
  /** Slot centre for row i. */
  cx: (i: number) => number
  innerWidth: number
  innerHeight: number
}

function StripFrame({
  height, labels, yTicks, yToPx, yLabel, xLabel, children,
}: {
  height: number
  /** One per row, drawn under its slot. */
  labels: number[]
  yTicks: number[]
  /** Maps a y value to pixels from the top of the plot area. */
  yToPx: (v: number) => number
  yLabel: string
  xLabel: string
  children: (g: StripGeom) => ReactNode
}) {
  const innerWidth = STRIP.width - STRIP.left - STRIP.right
  const innerHeight = height - STRIP.top - STRIP.bottom
  const n = Math.max(labels.length, 1)
  const pitch = innerWidth / n
  const geom: StripGeom = {
    cx: (i) => pitch * (i + 0.5),
    innerWidth,
    innerHeight,
  }
  return (
    <svg
      className="chart"
      viewBox={`0 0 ${STRIP.width} ${height}`}
      role="img"
      aria-label={`${yLabel} by ${xLabel}`}
      preserveAspectRatio="xMidYMid meet"
    >
      <g transform={`translate(${STRIP.left},${STRIP.top})`}>
        <g className="chart__grid">
          {yTicks.map((t) => (
            <line key={t} x1={0} x2={innerWidth} y1={yToPx(t)} y2={yToPx(t)} />
          ))}
        </g>
        <g className="chart__tick">
          {yTicks.map((t) => (
            <text key={t} x={-8} y={yToPx(t)} dy="0.32em" textAnchor="end">
              {formatTick(t)}
            </text>
          ))}
        </g>
        <line x1={0} x2={innerWidth} y1={innerHeight} y2={innerHeight}
              stroke="var(--border-strong)" />
        {children(geom)}
        <g className="chart__tick">
          {labels.map((m, i) => (
            <text key={`${m}-${i}`} x={geom.cx(i)} y={innerHeight + 17}
                  textAnchor="middle">
              {m}
            </text>
          ))}
        </g>
        <text className="chart__axis-label" x={innerWidth / 2}
              y={innerHeight + 37} textAnchor="middle">
          {xLabel}
        </text>
        <text
          className="chart__axis-label"
          transform={`translate(${-STRIP.left + 12},${innerHeight / 2}) rotate(-90)`}
          textAnchor="middle"
        >
          {yLabel}
        </text>
      </g>
    </svg>
  )
}

/** Bar pitch depends on how many context lengths the file recorded. */
const BAR_W = (n: number): number =>
  ((STRIP.width - STRIP.left - STRIP.right) / n) * 0.62
const COMP_H = 300
const COMP_INNER_H = COMP_H - STRIP.top - STRIP.bottom

function CompositionChart({
  rows, categories,
}: {
  rows: SumBehavior[]
  categories: { key: string; label: string; color: string }[]
}) {
  return (
    <StripFrame
      height={COMP_H}
      labels={rows.map((r) => r.m)}
      yTicks={niceTicks([0, 1], 5)}
      yToPx={(v) => (1 - v) * COMP_INNER_H}
      yLabel="share of trials"
      xLabel="in-context demonstrations m"
    >
      {({ cx, innerHeight }) => (
        <g>
          {rows.map((row, i) => {
            let acc = 0
            const w = BAR_W(rows.length)
            return (
              <g key={row.m}>
                {categories.map((c) => {
                  const v = categoryValue(row, c.key)
                  if (v === null) return null
                  const top = (1 - (acc + v)) * innerHeight
                  const bottom = (1 - acc) * innerHeight
                  acc += v
                  return (
                    <rect
                      key={`${row.m}-${c.key}`}
                      x={cx(i) - w / 2}
                      y={top}
                      width={w}
                      height={Math.max(0, bottom - top)}
                      fill={c.color}
                      opacity={0.92}
                    >
                      <title>{`m = ${row.m} — ${c.label}: ${(v * 100).toFixed(1)}%`}</title>
                    </rect>
                  )
                })}
              </g>
            )
          })}
        </g>
      )}
    </StripFrame>
  )
}

function EntropyChart({ rows }: { rows: SumBehavior[] }) {
  // Anchor the top of the axis at the 8.0-bit uniform reference with headroom,
  // so the reference line is visibly inside the plot rather than on its edge.
  const lo = Math.min(...rows.map((r) => r.entropyQ25Bits))
  const hi = Math.max(...rows.map((r) => r.entropyQ75Bits))
  const pad = Math.max((hi - lo) * 0.08, 0.05)
  const dom: [number, number] = [lo - pad, hi + pad]
  const innerHeight = 260 - STRIP.top - STRIP.bottom
  const toPx = (v: number): number =>
    innerHeight - ((v - dom[0]) / (dom[1] - dom[0])) * innerHeight
  const back = [...rows].reverse()
  return (
    <StripFrame
      height={260}
      labels={rows.map((r) => r.m)}
      yTicks={niceTicks(dom, 5)}
      yToPx={toPx}
      yLabel="entropy (bits)"
      xLabel="in-context demonstrations m"
    >
      {({ cx }) => {
        const band =
          rows.map((r, i) => `${i === 0 ? 'M' : 'L'}${cx(i)},${toPx(r.entropyQ75Bits)}`).join(' ') +
          ' ' +
          back.map((r, i) => `L${cx(rows.length - 1 - i)},${toPx(r.entropyQ25Bits)}`).join(' ')
        return (
          <g>
            <line x1={0} x2={STRIP.width - STRIP.left - STRIP.right}
                  y1={toPx(8)} y2={toPx(8)} stroke="var(--warn)"
                  strokeDasharray="4 4" strokeWidth={1} />
            <text x={STRIP.width - STRIP.left - STRIP.right} y={toPx(8) - 5}
                  textAnchor="end" fill="var(--warn)" fontSize={11}>
              8.0 bits = uniform over 256 bytes
            </text>
            <path d={`${band} Z`} fill="var(--arm-selfplay)" opacity={0.18} />
            {rows.map((r, i) => (
              <circle key={`q${r.m}`} cx={cx(i)} cy={toPx(r.entropyQ25Bits)} r={1.5}
                      fill="var(--arm-selfplay)" opacity={0.55} />
            ))}
            {rows.map((r, i) => (
              <circle key={`Q${r.m}`} cx={cx(i)} cy={toPx(r.entropyQ75Bits)} r={1.5}
                      fill="var(--arm-selfplay)" opacity={0.55} />
            ))}
            <Series
              points={rows.map((r, i) => [cx(i), toPx(r.entropyMeanBits)])}
              color="var(--arm-selfplay)" width={2} showPoints pointRadius={3}
            />
          </g>
        )
      }}
    </StripFrame>
  )
}

// ---------------------------------------------------------------------------
// Caveat
// ---------------------------------------------------------------------------

function CaveatCard() {
  return (
    <>
      <h2 className="section">How to read every number on this page</h2>
      <Card
        title="Measurement conditions"
        note="From the model string recorded in the released result files."
      >
        <div className="kv">
          <span className="kv__k">model</span>
          <span className="kv__v">4-seed probability ensemble, d512h8L8 (24.3M non-emb), round 2816</span>
          <span className="kv__k">decoding</span>
          <span className="kv__v">greedy argmax, one forward pass</span>
          <span className="kv__k">trials</span>
          <span className="kv__v">64 per cell in the m sweep (Figure 4); 128 in the mechanism file; 1,024 in the behavioural file</span>
          <span className="kv__k">matched</span>
          <span className="kv__v">architecture, rung, training round, prompt format and scorer are identical across the three arms</span>
        </div>
        <div className="claim claim--caveat" style={{ marginTop: 14 }}>
          <p className="claim__text">
            <strong>64 trials means one point is 1/64 = 1.6% granular,</strong>{' '}
            and a single cell can swing by that much on its own. The small-m
            points are the noisiest: differences under about 5 points between
            arms are not meaningful at this trial count, which is why the m ≤ 4
            SUM cells were rerun with 4,096 trials. The four-seed ensemble
            averages the predictive distribution, not the accuracy, so the
            variance from seed-to-seed training is still inside these numbers.{' '}
            <PaperRef reference="fig4" inline />
          </p>
        </div>
        <p className="card__note">
          Nothing on this page is inferred from the paper's prose. Every value
          comes from <code>public/data/icl.json</code>, rebuilt by{' '}
          <code>tools/build_data.py</code> from the authors' released result
          files and gated by <code>tools/verify_data.py</code>.
        </p>
      </Card>
    </>
  )
}