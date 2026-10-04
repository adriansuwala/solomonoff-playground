/**
 * The pure data logic behind the reward page, extracted so it can be tested
 * without a DOM.
 *
 * Every number this page prints about Table 5 is derived here rather than
 * typed into the prose, because the page's argument rests on claims like
 * "worse on 9 of 10 rows" and those claims are only worth anything if they are
 * derived from the same cells the table draws. reward.test.ts pins each of
 * them against the released bundle and against the authors' committed
 * reward_arms.tex.
 *
 * Two thresholds decide what counts as a difference, and both come from the
 * paper's own printing: TIE_EPS for "is this row worse", and the 2 dp rounding
 * inside rowBest. Neither is a tunable -- they exist because the authors round
 * every cell to 2 dp and share the bold between arms that agree there, so a
 * helper comparing at full precision would contradict the table printed beside
 * it.
 *
 * Every function is total: an unscored cell reads as null and is skipped, so a
 * missing cell can never contribute a NaN ratio or a divide-by-zero to a count.
 */
import type { Table5, Table5Row } from '@/data/types'

/**
 * Footnote markers on the ablation headers, matching the paper's Table 5
 * caption. `Table5.notes` carries the prose; only the mapping of which column
 * carries which marker has to be written down somewhere.
 */
export const FOOTNOTES: Record<string, string> = { last_step: 'b', loss_delta: 'a,b' }

/**
 * The uniform-over-256-bytes baseline: 8.0 bits/byte. This is what a predictor
 * ignorant of the corpus alphabet scores, NOT what our random-init control
 * scores -- that control reads 8.72 (see local_measurements.json), because a
 * randomly initialised transformer is not a uniform sampler. Every "above the
 * baseline" test is therefore against 8.0, which is the weaker bar.
 */
export const UNIFORM_256_BPB = 8

/**
 * Two losses are treated as a printed tie at the precision the authors use.
 * reward_arms.tex rounds every cell to 2 dp, and bolds `none`, `uniform` AND
 * `shuffle` together on the random-bytes row because they agree to that
 * precision. Counting the 0.0031 bits/byte gap as a strict loss made the page
 * claim "worse on 10 of 10 rows" where the paper's own table shows a tie.
 */
export const TIE_EPS = 0.005

/** bits/byte at the paper's printed precision: the rounding rowBest compares at. */
const PRINT_DP = 100

// ---------------------------------------------------------------------------
// Cell access. Every lookup goes through these, so a column key that does not
// exist reads as "unscored" instead of throwing or indexing a hole.
// ---------------------------------------------------------------------------

/** The bits/byte in one cell, or null if the column or the cell is absent. */
export function cellBpb(t: Table5, row: Table5Row, key: string): number | null {
  const i = t.columns.findIndex((c) => c.key === key)
  if (i < 0) return null
  const cell = row.cells[i]
  return cell ? cell.bpb : null
}

/** The seed count backing one cell, or null if the column or the cell is absent. */
export function cellSeeds(t: Table5, row: Table5Row, key: string): number | null {
  const i = t.columns.findIndex((c) => c.key === key)
  if (i < 0) return null
  return row.cells[i]?.kUsed ?? null
}

/** Rows where this arm actually has a loss, i.e. the ones a claim can cover. */
export function scoredRows(t: Table5, key: string): Table5Row[] {
  return t.rows.filter((r) => cellBpb(t, r, key) !== null)
}

/** How many rows this arm is scored on: the "X of Y" denominator. */
export function scoredRowCount(t: Table5, key: string): number {
  return scoredRows(t, key).length
}

export interface PairResult {
  /** Rows where `arm` ends up worse than `base`, with the ratio. */
  worse: [string, number][]
  /** Rows where it does not. */
  notWorse: [string, number][]
  /** Rows equal to `base` at the printed precision: neither better nor worse. */
  ties: string[]
  /**
   * Rows where both arms are scored, so both `base` and `arm` are positive.
   * This, not rows.length, is the denominator every "X of Y" on the page must
   * use: a row that was never scored cannot be evidence either way, and
   * counting it would let the page claim a sweep it did not measure.
   */
  scored: number
}

/**
 * Compare two arms across every row where both have a scored cell, with a
 * non-positive base skipped so the ratio cannot divide by zero.
 */
export function compareArms(t: Table5, arm: string, base: string): PairResult {
  const out: PairResult = { worse: [], notWorse: [], ties: [], scored: 0 }
  for (const row of t.rows) {
    const a = cellBpb(t, row, arm)
    const b = cellBpb(t, row, base)
    if (a === null || b === null || b <= 0) continue
    out.scored += 1
    const ratio = a / b
    if (Math.abs(a - b) <= TIE_EPS) out.ties.push(row.label)
    else if (a > b) out.worse.push([row.label, ratio])
    else out.notWorse.push([row.label, ratio])
  }
  return out
}

/** The row labels of a pair's worse / not-worse lists, for prose quotes. */
export function labels(xs: [string, number][]): string[] {
  return xs.map(([l]) => l)
}

/** The ratios of a pair's worse / not-worse lists, for the median claims. */
export function ratios(xs: [string, number][]): number[] {
  return xs.map(([, r]) => r)
}

/**
 * Median of a sample: the middle value, or the mean of the two middles. Returns
 * null on an empty sample, which is the case a "median X worse" claim would
 * otherwise read as 0.00x. The odd-length branch must not consult the lower
 * neighbour, or a single-element sample (mid = 0, lo = undefined) returns null
 * for a median that plainly exists.
 */
export function median(xs: number[]): number | null {
  if (xs.length === 0) return null
  const s = [...xs].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  if (s.length % 2 === 1) return s[mid] ?? null
  const lo = s[mid - 1]
  const hi = s[mid]
  return lo === undefined || hi === undefined ? null : (lo + hi) / 2
}

/** A bits/byte value at the table's precision; nullish renders as an em dash. */
export const fmt = (x: number | null, digits = 2): string =>
  x === null ? '—' : x.toFixed(digits)

/** "arithmetic 0.22 / 7.94" style quote of one arm's values on named rows. */
export function valuesOn(t: Table5, key: string, labels: string[]): string {
  return labels
    .map((l) => {
      const row = t.rows.find((r) => r.label === l)
      return row ? `${l} ${fmt(cellBpb(t, row, key))}` : null
    })
    .filter((s): s is string => s !== null)
    .join(', ')
}

/** The lowest kUsed seen in a column, for the ensemble caveat. 0 if unscored. */
export function minSeeds(t: Table5, key: string): number {
  let k = Infinity
  for (const row of t.rows) {
    const cell = cellSeeds(t, row, key)
    if (cell) k = Math.min(k, cell)
  }
  return Number.isFinite(k) ? k : 0
}

// ---------------------------------------------------------------------------
// Counts against the uniform-over-256-bytes reference
// ---------------------------------------------------------------------------

/**
 * Split the rows by whether an arm scores above a bits/byte threshold. The
 * divide is on measured cells only, so an unscored row lands in neither list
 * rather than being counted as "at or below" -- which would silently create
 * the exception the page then has to name.
 */
export function splitAbove(
  t: Table5, key: string, threshold: number,
): { above: string[]; atOrBelow: string[] } {
  const above: string[] = []
  const atOrBelow: string[] = []
  for (const row of t.rows) {
    const v = cellBpb(t, row, key)
    if (v === null) continue
    ;(v > threshold ? above : atOrBelow).push(row.label)
  }
  return { above, atOrBelow }
}

/** How many rows carry the canonical arm as the printed best. */
export function canonicalBestCount(t: Table5): number {
  return t.rows.filter((r) => r.bestIndex === 0).length
}

/** A row's label by key, with a fallback for prose that has to name a dataset. */
export function rowLabel(t: Table5, key: string, fallback: string): string {
  return t.rows.find((r) => r.key === key)?.label ?? fallback
}

/**
 * How many held-out sequences a corpus actually contributed. Not always the 256
 * the caption quotes -- mutopia_melody_16th contributes 17 -- and the signed
 * claim quotes that number rather than the caption's.
 */
export function heldOutSeqs(t: Table5, corpus: string, fallback: number): number {
  return t.provenance?.corpora?.[corpus]?.n_seq_used ?? fallback
}

// ---------------------------------------------------------------------------
// Heatmap: distance from the row's own best cell
// ---------------------------------------------------------------------------

/**
 * The best loss in a row, compared at the precision the authors' table is
 * printed to. reward_arms.tex rounds every cell to 2 dp and shares the bold
 * between arms that agree there, and build_data.py computes bestIndex the same
 * way. Comparing at full precision made this function disagree with bestIndex
 * on the random-bytes row, where uniform (8.0163) beats none (8.0204) by 0.004
 * -- both print as 8.02 and are bold together in the paper. The heatmap and the
 * table would then shade different cells as best on the same row.
 */
export function rowBest(row: Table5Row): number | null {
  let best: number | null = null
  for (const c of row.cells) {
    if (!c) continue
    if (best === null || Math.round(c.bpb * PRINT_DP) < Math.round(best * PRINT_DP)) {
      best = c.bpb
    }
  }
  return best
}

/** Opacity ∝ how much worse than the row's best, on a log ratio scale. */
export function heatAlpha(ratio: number, maxRatio: number): number {
  if (!(ratio > 1)) return 0
  const span = Math.log10(Math.max(maxRatio, 10))
  const t = Math.log10(ratio) / span
  return Math.min(1, Math.max(0, t)) * 0.72
}

/**
 * Every cell's ratio to its own row's best, over the rows that have a usable
 * best. `legendSpan` is floored at 10x because the paper's own worst row is
 * ~48x and a log scale from 1x to 48x puts most cells in the first fifth of
 * the ramp; the floor keeps the spread legible.
 */
export function heatScale(t: Table5): {
  ratios: number[]
  maxRatio: number
  legendSpan: number
} {
  const ratios: number[] = []
  for (const row of t.rows) {
    const best = rowBest(row)
    if (best === null || best <= 0) continue
    for (const c of row.cells) if (c) ratios.push(c.bpb / best)
  }
  const maxRatio = ratios.length ? Math.max(...ratios) : 10
  return { ratios, maxRatio, legendSpan: Math.max(maxRatio, 10) }
}

/** A cell's ratio to its row's best, or null where the ratio is undefined. */
export function cellRatio(t: Table5, row: Table5Row, key: string): number | null {
  const v = cellBpb(t, row, key)
  const best = rowBest(row)
  if (v === null || best === null || best <= 0) return null
  return v / best
}

// ---------------------------------------------------------------------------
// Arm explorer
// ---------------------------------------------------------------------------

export interface ArmBar {
  key: string
  label: string
  bpb: number
  kUsed: number | null
}

/**
 * The scored arms of one row, ascending in loss, which is the order the bar
 * chart plots them in. Unscored arms are dropped: a bar at 0.0 bits/byte would
 * read as a perfect run.
 */
export function armBars(t: Table5, row: Table5Row | undefined): ArmBar[] {
  return (row?.cells ?? [])
    .map((cell, i) => ({
      key: t.columns[i]?.key ?? String(i),
      label: t.columns[i]?.label ?? String(i),
      bpb: cell?.bpb ?? null,
      kUsed: cell?.kUsed ?? null,
    }))
    .filter((b): b is { key: string; label: string; bpb: number; kUsed: number | null } =>
      b.bpb !== null)
    .sort((a, b) => a.bpb - b.bpb)
}