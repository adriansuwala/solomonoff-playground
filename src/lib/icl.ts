/**
 * The pure data logic behind the in-context learning page, extracted so it can
 * be tested without a DOM. Every function is total: it returns null rather
 * than throwing or emitting NaN when a combination was not measured, and the
 * page renders that as "not measured" instead of plotting a guess.
 *
 * This is where the numbers the page quotes are derived, so it is where a
 * wrong claim would come from. icl.test.ts pins each quoted value against the
 * released bundle.
 */
import { pct } from '@/lib/format'
import type {
  IclArm, IclData, IclFile, IclRow, SumBehavior,
} from '@/data/types'

/** Keys as they appear in icl.json, which uses `uniform_prior`, not `uniform`. */
export const ARM_ORDER = ['selfplay', 'uniform_prior', 'pcfg'] as const
export type IclArmKey = (typeof ARM_ORDER)[number]

export const ARM_META: Record<IclArmKey, 'selfplay' | 'uniform' | 'pcfg'> = {
  selfplay: 'selfplay',
  uniform_prior: 'uniform',
  pcfg: 'pcfg',
}

/**
 * The three associative-recall dictionary sizes in icl_v3_results, keyed by the
 * harness's `V=` axis.
 */
export const ASSOC_V = [4, 16, 64] as const
/** The largest size in the sweep: where the three arms actually separate. */
export const ASSOC_V_MAX: number = ASSOC_V[2]

/**
 * File coordinates are now resolved upstream: tools/build_data.py parses both
 * the named harness keys (`m=8|k=2`, `m=128|V=64`) and the bare positional ones
 * (`max|8|2`) that run_icl.py writes, and tools/verify_data.py fails the build
 * if the main sweep has an unresolved cell. These helpers therefore read m and
 * k straight off the row and stay null-guarded, so an unplottable combination
 * renders as "not measured" rather than as a line through missing values.
 */
export function sweepRows(file: IclFile | undefined, task: string, k: number): IclRow[] {
  if (!file) return []
  return file.rows.filter((r) => r.task === task && r.k === k && r.m !== null)
}

// ---------------------------------------------------------------------------
// Series extraction (every metric lookup is guarded: names differ by harness)
// ---------------------------------------------------------------------------

export interface Pt { m: number; v: number; p: number | null }

export function armOf(icl: IclData, arm: IclArmKey): IclArm | null {
  return icl.arms[arm] ?? null
}

/** Accuracy against m for one arm, one task, one arity, one metric. */
export function sweepSeries(
  icl: IclData, arm: IclArmKey, fileName: string,
  task: string, k: number, metric: string,
): Pt[] | null {
  const file = armOf(icl, arm)?.files[fileName]
  if (!file) return null
  const out: Pt[] = []
  for (const row of sweepRows(file, task, k)) {
    if (row.m === null) continue
    const v = row.metrics[metric]
    if (v === undefined) continue
    out.push({ m: row.m, v, p: row.metrics['p_correct'] ?? null })
  }
  return out.length > 0 ? out.sort((a, b) => a.m - b.m) : null
}

/** One cell of the main sweep, for the numbers quoted in prose. */
export function sweepCell(
  icl: IclData, arm: IclArmKey, task: string, k: number, m: number,
): number | null {
  const series = sweepSeries(icl, arm, 'icl_results', task, k, 'acc')
  if (!series) return null
  return series.find((p) => p.m === m)?.v ?? null
}

export function extraSeries(
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

export function assocSeries(
  icl: IclData, arm: IclArmKey, v: number,
): Pt[] | null {
  const file = armOf(icl, arm)?.files['icl_v3_results']
  if (!file) return null
  const out: Pt[] = []
  for (const row of file.rows) {
    if (row.task !== 'assoc' || row.v !== v || row.m === null) continue
    const a = row.metrics['acc']
    if (a === undefined) continue
    out.push({ m: row.m, v: a, p: row.metrics['p_correct'] ?? null })
  }
  return out.length > 0 ? out.sort((a, b) => a.m - b.m) : null
}

/** First and last cell of the extra-dictionary-prints sweep, as a range. */
export function assocDictEnds(icl: IclData, arm: IclArmKey): string {
  const rows = (armOf(icl, arm)?.files['icl_assoc_dict_results']?.rows ?? [])
    .filter((r) => r.extra !== null)
    .sort((a, b) => (a.extra ?? 0) - (b.extra ?? 0))
  const a = rows[0]?.metrics['acc']
  const b = rows[rows.length - 1]?.metrics['acc']
  return a !== undefined && b !== undefined ? `${pct(a)} → ${pct(b)}` : '—'
}

/** Accuracy in the largest-m cell of a series, for the numbers quoted in prose. */
export function lastOf(pts: Pt[] | null): number | null {
  return pts !== null && pts.length > 0 ? (pts[pts.length - 1]?.v ?? null) : null
}
export function assocTop(icl: IclData, arm: IclArmKey, v: number): number | null {
  return lastOf(assocSeries(icl, arm, v))
}
export function extraTop(icl: IclData, arm: IclArmKey, task: string): number | null {
  return lastOf(extraSeries(icl, arm, task))
}
/** Best cell anywhere in a task's sweep, for claims phrased as "never above". */
export function extraBest(icl: IclData, arm: IclArmKey, task: string): number | null {
  const pts = extraSeries(icl, arm, task)
  return pts !== null && pts.length > 0
    ? Math.max(...pts.map((p) => p.v))
    : null
}

export function m0(icl: IclData, arm: IclArmKey, task: string, metric: string): number | null {
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
export function sumLowM(icl: IclData, arm: IclArmKey): number[] | null {
  const file = armOf(icl, arm)?.files['icl_sum_lowm_results']
  if (!file) return null
  const out = file.rows
    .filter((r) => r.task === 'sum' && r.m !== null && r.metrics['acc'] !== undefined)
    .sort((a, b) => (a.m ?? 0) - (b.m ?? 0))
    .map((r) => r.metrics['acc'] ?? Number.NaN)
  return out.length > 0 ? out : null
}

// ---------------------------------------------------------------------------
// Figure 5: behavioural composition and predictive entropy
// ---------------------------------------------------------------------------

/**
 * The five buckets the harness assigns each emitted byte to, in the order the
 * legend shows them: most-specific first. `correctAnswer` is the exact byte,
 * `low4BitsCorrect` the low nibble, and so on.
 */
export const CATEGORY_KEYS = [
  'correctAnswer', 'low4BitsCorrect', 'preferredBytes', 'contextByte', 'other',
] as const
export type CategoryKey = (typeof CATEGORY_KEYS)[number]

/** Uniform entropy over the 256-byte alphabet, in bits. */
export const BITS_FOR_256 = 8

/** Behavioural rows sorted by m, which is the order both panels plot in. */
export function sumBehaviorRows(icl: IclData): SumBehavior[] {
  return [...icl.sumBehavior].sort((a, b) => a.m - b.m)
}

/** The row at a given m, or null if that m was never run. */
export function sumRowAt(rows: SumBehavior[], m: number): SumBehavior | null {
  return rows.find((r) => r.m === m) ?? null
}

/**
 * Interquartile band width (Q75 - Q25) at a given m, in bits. Returns null
 * rather than 0 when m is absent: the page prints this width in prose to argue
 * that the band does not narrow, and a fabricated 0.0 would silently support
 * the opposite conclusion.
 */
export function iqrBits(rows: SumBehavior[], m: number): number | null {
  const r = sumRowAt(rows, m)
  return r === null ? null : r.entropyQ75Bits - r.entropyQ25Bits
}

/** The m at which mean predictive entropy peaks, with the peak value. */
export function entropyPeak(
  rows: SumBehavior[],
): { m: number; bits: number } | null {
  if (rows.length === 0) return null
  const best = rows.reduce((a, b) => (b.entropyMeanBits > a.entropyMeanBits ? b : a))
  return { m: best.m, bits: best.entropyMeanBits }
}

/**
 * The smallest m at which `key` is the largest of the five categories, or null
 * if it never is. This backs the claim that exact answers only take over as a
 * strategy once there is enough context, so it must be a real comparison
 * against every category rather than a fixed constant.
 */
export function firstDominant(
  rows: SumBehavior[], key: CategoryKey = 'correctAnswer',
): number | null {
  for (const r of rows) {
    const mine = r[key]
    if (CATEGORY_KEYS.some((c) => c !== key && r[c] > mine)) continue
    return r.m
  }
  return null
}

/** True when every category share at this m sums to 1 within tolerance. */
export function sharesSumToOne(row: SumBehavior, tol = 1e-6): boolean {
  const total = CATEGORY_KEYS.reduce((sum, c) => sum + row[c], 0)
  return Math.abs(total - 1) <= tol
}
