/**
 * The pure data logic behind the scaling page: the compute-optimal frontier,
 * Table 2's per-modality exponents, and the three-arm comparison.
 *
 * Extracted so it can be tested without a DOM. Two rules govern everything
 * here, and both exist because breaking them produced wrong claims:
 *
 * 1. `scaling.ladder` (Table 2 / Figure 1) and `scaling.arms` (Figure 2) come
 *    from DIFFERENT files whose frontiers disagree on every row. They are never
 *    mixed in one series or one claim (docs/decisions.md D3).
 * 2. The arms are only ever compared at a compute budget all three reached.
 *    Comparing each arm's own best loss is the confound this page exists to
 *    avoid: self-play runs to ~10x the compute of the fixed-prior arm, so its
 *    frontier always ends lower without being better.
 *
 * scaling.test.ts pins the arithmetic behind every number the page quotes, and
 * in particular pins which corpora are *excluded* from the arm comparison --
 * an undisclosed exclusion once carried a load-bearing conclusion.
 */
import { predictFit } from '@/lib/scales'
import type {
  ArmKey, ArmScaling, Columnar, CorpusScaling, Meta, PowerFit, Scaling,
} from '@/data/types'

/**
 * `lossDomain`, `formatCompute` and `predictFit` already live in @/lib/scales
 * because Overview and Chart use them too. They are re-exported here so this
 * page's data logic has one import and one place to test against the bundle;
 * the definitions are NOT duplicated.
 */
export { formatCompute, lossDomain, predictFit } from '@/lib/scales'

/** The three pretraining regimes of Figure 2, in the order the page shows them. */
export const ARM_ORDER: readonly ArmKey[] = ['selfplay', 'uniform', 'pcfg']

/**
 * Never draw more raw circles than this; stride-subsample beyond it.
 *
 * Every ladder corpus carries 1,490 rows, so a 1,500 cap would never fire and
 * the stride path in subsample would be unreachable. 1,400 keeps the scatter
 * readable AND leaves the stride branch live.
 */
export const MAX_RAW_POINTS = 1400

/** Below this bits/byte gap the arms are not meaningfully separated. */
export const TIE_MARGIN = 0.05

/** How many log-spaced samples the dashed fit line is drawn with. */
export const FIT_SAMPLES = 48

/** One scored ladder point, unpacked from the columnar arrays. */
export interface RawPoint {
  c: number
  bpb: number
  n: number
  k: number
  round: number
}

/**
 * Render an exponent at the paper's printed precision of three decimals.
 * Ties round toward zero, because the authors printed 0.260 for the audio
 * 8-bit fit of 0.2605 (docs/findings.md F-note); `tools/verify_data.py`
 * accepts both renderings with a half-ulp window.
 */
export function paperExponent(v: number): string {
  const scaled = v * 1000
  const floor = Math.floor(scaled)
  return ((scaled - floor > 0.5 + 1e-9 ? floor + 1 : floor) / 1000).toFixed(3)
}

/** The loss-scale prefactor of L(C) = E + A*C^-b, at the paper's precision. */
export function amplitude(v: number): string {
  if (!Number.isFinite(v)) return '—'
  if (Math.abs(v) >= 1000) return v.toExponential(3)
  if (Math.abs(v) >= 10) return v.toFixed(2)
  return v.toFixed(3)
}

/** English ordinal suffix for the "every Nth point drawn" note. */
export function ordinal(n: number): string {
  if (n % 10 === 1 && n % 100 !== 11) return 'st'
  if (n % 10 === 2 && n % 100 !== 12) return 'nd'
  if (n % 10 === 3 && n % 100 !== 13) return 'rd'
  return 'th'
}

// ---------------------------------------------------------------------------
// The scored ladder (scaling.ladder)
// ---------------------------------------------------------------------------

/**
 * One corpus's rows out of the columnar ladder. The rows are sorted by
 * (corpus, compute), so the target corpus occupies one contiguous run and the
 * scan stops as soon as it has passed it -- one pass over ~39k rows, no
 * intermediate arrays of 39k elements.
 *
 * `total` counts every row carrying the corpus index, including any whose
 * numeric columns were missing; the page divides by it to state how many points
 * were drawn, so it must not silently shrink when a column is absent.
 */
export function sliceCorpus(ladder: ArmScaling, key: string): { rows: RawPoint[]; total: number } {
  const target = ladder.stringTables.corpora.indexOf(key)
  const rows: RawPoint[] = []
  if (target < 0) return { rows, total: 0 }
  const cols = ladder.columns
  let total = 0
  for (let i = 0; i < cols.ci.length; i += 1) {
    const ci = cols.ci[i]
    if (ci === undefined) continue
    if (ci > target) break
    if (ci !== target) continue
    total += 1
    const c = cols.C[i]
    const bpb = cols.bpb[i]
    const n = cols.N[i]
    const k = cols.K[i]
    const round = cols.round[i]
    if (c === undefined || bpb === undefined || n === undefined
      || k === undefined || round === undefined) continue
    rows.push({ c, bpb, n, k, round })
  }
  return { rows, total }
}

/**
 * Deterministic stride subsample, so the drawn point count is reproducible.
 * Returns the kept points plus the un-subsampled total, which the page prints
 * ("N of M scored ladder points drawn").
 */
export function subsample<T>(items: T[], cap: number): { kept: T[]; total: number } {
  const total = items.length
  if (total <= cap) return { kept: items, total }
  const stride = Math.ceil(total / cap)
  return { kept: items.filter((_, i) => i % stride === 0), total }
}

/** The stride the subsample actually used, for the "every Nth point" note. */
export function subsampleStride(total: number, cap: number = MAX_RAW_POINTS): number {
  return total <= cap ? 1 : Math.ceil(total / cap)
}

/** The corpus's ladder fit, or null when the frontier was too short to fit. */
export function ladderFit(cs: CorpusScaling): PowerFit | null {
  return cs.fit
}

/**
 * Whether the fitted floor may be read as "the bits per byte this arm would
 * still pay at infinite compute".
 *
 * Returns null when there is no fit at all (the page prints "no fit" instead),
 * 'broken-down' when the best measurement is below the fitted floor or the
 * floor is pinned at zero, and 'extrapolation' otherwise.
 *
 * This distinction is load-bearing: for the six corpora whose best measured
 * loss sits BELOW the floor, saying "the ladder does not approach it" would be
 * false, and for the six whose floor is exactly 0.000 the floor cannot be
 * called a measured irreducible loss at all.
 */
export type FloorReading = 'broken-down' | 'extrapolation' | null

export function floorReading(cs: CorpusScaling): FloorReading {
  if (!cs.fit) return null
  if (cs.bestLoss < cs.fit.floor || cs.fit.floor === 0) return 'broken-down'
  return 'extrapolation'
}

/** The corpora whose floor may NOT be quoted as an irreducible loss. */
export function brokenDownFloors(scaling: Scaling): string[] {
  return Object.entries(scaling.ladder.corpora)
    .filter(([, cs]) => floorReading(cs) === 'broken-down')
    .map(([key]) => key)
}

/**
 * The dashed fit line, sampled logarithmically between the first and last
 * frontier compute. Returns [] when there is no fit or the frontier cannot span
 * a positive range -- an empty array renders no line rather than a NaN one.
 */
export function fitCurve(cs: CorpusScaling, samples: number = FIT_SAMPLES): [number, number][] {
  if (!cs.fit || cs.frontier.length < 2) return []
  const lo = cs.frontier[0]?.[0] ?? 0
  const hi = cs.frontier[cs.frontier.length - 1]?.[0] ?? 0
  if (!(lo > 0) || !(hi > lo)) return []
  const out: [number, number][] = []
  for (let i = 0; i <= samples; i += 1) {
    const c = lo * Math.pow(hi / lo, i / samples)
    out.push([c, predictFit(cs.fit, c)])
  }
  return out
}

// ---------------------------------------------------------------------------
// Table 2
// ---------------------------------------------------------------------------

/**
 * One literature exponent for a Table 2 row: a published value, or one
 * converted from Chinchilla form at build time. `literatureChinchilla` already
 * holds b = αβ/(α+β) values, NOT (α, β) pairs -- do not convert them again.
 */
export interface LiteratureEntry {
  v: number
  derived: boolean
}

/** One Table 2 row with its literature values merged and sorted ascending. */
export interface ExponentRowView {
  key: string
  paperLabel: string
  group: string
  alpha: number | null
  literature: LiteratureEntry[]
  /** True when the authors tabulated "none found" rather than no values. */
  literatureDash: boolean
  refs: string[]
}

/** Table 2's rows, in bundle order, with literature values merged and sorted. */
export function exponentRows(scaling: Scaling): ExponentRowView[] {
  return scaling.exponents.map((row) => ({
    key: row.key,
    paperLabel: row.paperLabel,
    group: row.group,
    alpha: row.alpha,
    literature: [
      ...row.literatureValues.map((v) => ({ v, derived: false })),
      ...row.literatureChinchilla.map((v) => ({ v, derived: true })),
    ].sort((a, b) => a.v - b.v),
    literatureDash: row.literatureDash,
    refs: row.refs,
  }))
}

/** The literature cell for one row, or null when the row is not in Table 2. */
export function literatureEntries(
  scaling: Scaling, key: string,
): { entries: LiteratureEntry[]; dash: boolean } | null {
  const row = exponentRows(scaling).find((r) => r.key === key)
  return row === undefined ? null : { entries: row.literature, dash: row.literatureDash }
}

/**
 * Table 2 must be the ladder's fits, tabulated -- not a second, independently
 * fitted set. Returns every (row, field) where `scaling.exponents` disagrees
 * with `scaling.ladder.corpora[key].fit`, so a rebuild that refits the
 * exponents separately is caught here rather than being printed as Table 2.
 */
export interface ExponentMismatch {
  key: string
  field: string
  ladder: number | null
  table2: number | null
}

/** Fields of a fit that Table 2 is supposed to copy verbatim. */
export const FIT_FIELDS = ['alpha', 'amplitude', 'floor', 'rmse', 'nPoints'] as const

/** Every disagreement between Table 2 and the ladder fits it claims to tabulate. */
export function exponentMismatches(scaling: Scaling): ExponentMismatch[] {
  const out: ExponentMismatch[] = []
  for (const row of scaling.exponents) {
    const cs = scaling.ladder.corpora[row.key]
    for (const field of FIT_FIELDS) {
      const table2 = row[field] as number | null
      const ladder = cs?.fit ? (cs.fit[field] as number | null) : null
      if (ladder === table2) continue
      // Tolerate float noise from the builder, not a genuinely different fit.
      const same = ladder !== null && table2 !== null
        && Math.abs(ladder - table2) <= 1e-9 * Math.max(1, Math.abs(ladder))
      if (!same) out.push({ key: row.key, field, ladder, table2 })
    }
  }
  return out
}

// ---------------------------------------------------------------------------
// The compute axis: C = K * N * pool * CTX * (round + 1)
// ---------------------------------------------------------------------------

/**
 * Recover the programs-per-round pool actually used for one scored point, from
 * the columns alone. Returns null when C is not consistent with any integer
 * pool, which is how a mixed-up compute axis announces itself.
 *
 * This is the real guard on the compute axis: it reads the bundle rather than
 * the builder's intent. If one constant pool were used for all three arms
 * instead of the authors' per-rung rates, the shared-budget arm comparison
 * would silently move onto an axis the authors never used -- the defect these
 * checks exist to catch (see tools/verify_data.py).
 */
export function recoverPool(
  cols: Columnar['columns'], i: number, context: number, tol = 0.05,
): number | null {
  const c = cols.C[i]
  const k = cols.K[i]
  const n = cols.N[i]
  const round = cols.round[i]
  if (c === undefined || k === undefined || n === undefined || round === undefined) return null
  const denom = k * n * context * (round + 1)
  if (!(denom > 0)) return null
  const pool = c / denom
  const rounded = Math.round(pool)
  if (rounded <= 0 || Math.abs(pool - rounded) > tol) return null
  return rounded
}

/** The distinct pools a columnar arm actually used, ascending. */
export function poolsOf(arm: ArmScaling, context: number): number[] {
  const seen = new Set<number>()
  for (let i = 0; i < arm.columns.C.length; i += 1) {
    const pool = recoverPool(arm.columns, i, context)
    if (pool !== null) seen.add(pool)
  }
  return [...seen].sort((a, b) => a - b)
}

/**
 * The pool a specific point used, or null if the row does not decode. Exposed
 * so the per-rung pool can be pinned to a rung's own parameter count rather
 * than to the arm's pool set as a whole.
 */
export function poolAt(
  arm: ArmScaling, n: number, context: number,
): number[] {
  const out: number[] = []
  const cols = arm.columns
  for (let i = 0; i < cols.C.length; i += 1) {
    if (cols.N[i] !== n) continue
    const pool = recoverPool(cols, i, context)
    if (pool !== null) out.push(pool)
  }
  return [...new Set(out)].sort((a, b) => a - b)
}

// ---------------------------------------------------------------------------
// Figure 2: the three arms (scaling.arms only)
// ---------------------------------------------------------------------------

/**
 * Score the three arms at a compute budget they all reached, and say which is
 * lowest. Comparing `bestLoss` directly would be unfair: self-play runs to
 * ~10x the compute of the fixed-prior arm, so its frontier always ends lower.
 * Restricting every arm to the largest budget all three reached turns that
 * confound into the comparison.
 */
export interface ArmVerdict {
  key: string
  label: string
  group: string
  budget: number
  best: Record<ArmKey, number | null>
  winner: ArmKey | null
  /** bits/byte between the winner and the runner-up; null if <2 arms scored. */
  margin: number | null
}

/** True when a margin is inside TIE_MARGIN, i.e. not a meaningful win. */
export function isTie(margin: number | null): boolean {
  return margin !== null && margin < TIE_MARGIN
}

/** The group the bundle marks `excluded`: never scored, never compared. */
export function isExcluded(group: string): boolean {
  return group === 'excluded'
}

/**
 * One verdict per compared corpus.
 *
 * Only corpora whose group is not `excluded` are considered, and only those
 * where at least two arms have a frontier. The exclusions are load-bearing and
 * are disclosed on the page: glibc rand is the one corpus where the fixed
 * uniform prior reaches a lower loss than self-play, and it is excluded.
 */
export function compareArms(scaling: Scaling, meta: Meta): ArmVerdict[] {
  const out: ArmVerdict[] = []
  for (const corpus of meta.corpora) {
    if (isExcluded(corpus.group)) continue
    const frontiers = ARM_ORDER.map((arm) => ({
      arm,
      pts: scaling.arms[arm].corpora[corpus.key]?.frontier ?? [],
    })).filter((a) => a.pts.length > 0)
    if (frontiers.length < 2) continue
    const budget = Math.min(...frontiers.map((a) => a.pts[a.pts.length - 1]?.[0] ?? Infinity))
    const best = {} as Record<ArmKey, number | null>
    for (const { arm, pts } of frontiers) {
      const reachable = pts.filter((p) => p[0] <= budget * (1 + 1e-7))
      best[arm] = reachable.length
        ? Math.min(...reachable.map((p) => p[1]))
        : null
    }
    const scored = ARM_ORDER.filter((a) => best[a] !== null)
    const sorted = [...scored].sort((a, b) => (best[a] ?? 0) - (best[b] ?? 0))
    const top = sorted[0]
    const second = sorted[1]
    out.push({
      key: corpus.key,
      label: corpus.paperLabel,
      group: corpus.group,
      budget,
      best,
      winner: top ?? null,
      margin: top !== undefined && second !== undefined
        ? (best[second] ?? 0) - (best[top] ?? 0)
        : null,
    })
  }
  return out
}

/** Win counts per corpus group, plus how many of those wins were near-ties. */
export interface GroupTally {
  group: string
  counts: Record<ArmKey, number>
  total: number
  ties: number
}

export function tallyGroups(verdicts: ArmVerdict[]): GroupTally[] {
  const order: string[] = []
  const byGroup = new Map<string, GroupTally>()
  for (const v of verdicts) {
    let tally = byGroup.get(v.group)
    if (!tally) {
      tally = {
        group: v.group,
        counts: { selfplay: 0, uniform: 0, pcfg: 0 },
        total: 0,
        ties: 0,
      }
      byGroup.set(v.group, tally)
      order.push(v.group)
    }
    tally.total += 1
    if (v.winner) tally.counts[v.winner] += 1
    if (isTie(v.margin)) tally.ties += 1
  }
  return order.map((g) => byGroup.get(g)).filter((t): t is GroupTally => t !== undefined)
}

/**
 * Per-arm win totals over every verdict, and how many comparisons were ties.
 *
 * A tie is still counted for exactly one arm -- that is deliberate and
 * disclosed on the page: the table flags the row `(tie)` but the headline counts
 * do not drop it, so totals can exceed the number of clear wins.
 */
export function armTotals(verdicts: ArmVerdict[]): {
  totals: Record<ArmKey, number>
  ties: number
} {
  const totals = ARM_ORDER.reduce(
    (acc, arm) => {
      acc[arm] = verdicts.filter((v) => v.winner === arm).length
      return acc
    },
    {} as Record<ArmKey, number>,
  )
  return { totals, ties: verdicts.filter((v) => isTie(v.margin)).length }
}

/** The verdict line, written from the tally rather than asserted by hand. */
export function groupVerdict(t: GroupTally, meta: Meta): string {
  const entries = ARM_ORDER
    .map((arm) => ({ arm, n: t.counts[arm], name: meta.arms[arm].short }))
    .sort((a, b) => b.n - a.n)
  const lead = entries[0]
  if (!lead || lead.n === 0) return 'No arm reached the shared budget.'
  const noun = t.total === 1 ? 'corpus' : 'corpora'
  const share = `${lead.name} reaches the lowest loss on ${lead.n} of ${t.total} ${noun}`
  const rest = entries.slice(1).filter((e) => e.n > 0)
  const also = rest.length
    ? `; ${rest.map((e) => `${e.name} on ${e.n}`).join(', ')}`
    : ''
  const tie = t.ties > 0
    ? ` ${t.ties === 1 ? 'One of those is' : `${t.ties} of those are`} within `
      + `${TIE_MARGIN} bits/byte — treat ${t.ties === 1 ? 'it' : 'those'} as a tie.`
    : ''
  return `${share}${also}.${tie}`
}

/** The corpora grouped as the corpus `<select>`'s `<optgroup>`s render them. */
export function corpusGroups(meta: Meta): { group: string; items: Meta['corpora'] }[] {
  const seen: string[] = []
  for (const c of meta.corpora) if (!seen.includes(c.group)) seen.push(c.group)
  return seen.map((g) => ({
    group: g,
    items: meta.corpora.filter((c) => c.group === g),
  }))
}

/** The x domain of the ladder scatter: the scored points plus the frontier. */
export function computeDomain(points: RawPoint[], cs: CorpusScaling): [number, number] {
  const cs2 = points.map((p) => p.c).concat(cs.frontier.map((p) => p[0]))
  return cs2.length === 0 ? [1, 10] : [Math.min(...cs2), Math.max(...cs2)]
}