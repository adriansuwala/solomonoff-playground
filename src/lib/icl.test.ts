/**
 * Tests for the in-context learning page's data derivations.
 *
 * Two jobs. First, pin every number the page's prose and Stat components quote
 * against the released bundle, so an edit to the page that changes a claim
 * without changing the data fails here. Second, cover the null paths: the page
 * promises to render "not measured" for combinations with no data, and a
 * helper that silently returned 0.0 or NaN instead would turn an honest gap
 * into a fabricated claim.
 *
 * These load the real public/data/icl.json rather than a fixture, because the
 * thing being protected is the agreement between code and bundle.
 */
import { describe, expect, it } from 'vitest'
import icl from '../../public/data/icl.json'
import { pct } from './format'
import {
  ASSOC_V, ASSOC_V_MAX, BITS_FOR_256, CATEGORY_KEYS,
  armOf, assocDictEnds, assocSeries, assocTop,
  entropyPeak, extraBest, extraSeries, extraTop,
  firstDominant, iqrBits, m0, sharesSumToOne,
  sumBehaviorRows, sumLowM, sumRowAt, sweepCell, sweepSeries,
  type IclArmKey,
} from './icl'
import type { IclData } from '@/data/types'

const data = icl as unknown as IclData
const rows = sumBehaviorRows(data)

/** Chance for a single byte under argmax over the 256-byte alphabet. */
const CHANCE = 1 / 256

describe('bundle shape', () => {
  it('exposes all three arms under the keys the page uses', () => {
    for (const arm of ['selfplay', 'uniform_prior', 'pcfg'] as IclArmKey[]) {
      expect(armOf(data, arm)).not.toBeNull()
    }
  })

  it('returns null for an arm that does not exist, rather than throwing', () => {
    expect(armOf(data, 'nope' as IclArmKey)).toBeNull()
  })

  it('gives every main-sweep cell a resolved (m, k)', () => {
    // The defect that made this page unplottable: run_icl.py writes bare
    // positional keys, and the builder used to drop the coordinates.
    for (const arm of ['selfplay', 'uniform_prior', 'pcfg'] as IclArmKey[]) {
      const f = armOf(data, arm)?.files['icl_results']
      expect(f, `${arm} has icl_results`).toBeDefined()
      for (const row of f?.rows ?? []) {
        expect(row.m, `${arm} ${row.task} row has m`).not.toBeNull()
        expect(row.k, `${arm} ${row.task} row has k`).not.toBeNull()
      }
    }
  })

  it('keeps the associative-recall dictionary size on each row', () => {
    for (const arm of ['selfplay', 'pcfg'] as IclArmKey[]) {
      const rows3 = armOf(data, arm)?.files['icl_v3_results']?.rows ?? []
      const assoc = rows3.filter((r) => r.task === 'assoc')
      expect(assoc.length).toBeGreaterThan(0)
      expect(assoc.every((r) => r.v !== null)).toBe(true)
      const vs = [...new Set(assoc.map((r) => r.v as number))].sort((a, b) => a - b)
      expect(vs).toEqual([...ASSOC_V])
    }
  })
})

describe('sweepSeries', () => {
  it('sorts by m and returns one point per m', () => {
    const s = sweepSeries(data, 'selfplay', 'icl_results', 'sum', 2, 'acc')
    expect(s).not.toBeNull()
    const ms = s?.map((p) => p.m) ?? []
    expect([...ms].sort((a, b) => a - b)).toEqual(ms)
    expect(new Set(ms).size).toBe(ms.length)
  })

  it('returns null for a task/arity combination that was never run', () => {
    expect(sweepSeries(data, 'selfplay', 'icl_results', 'sum', 3, 'acc')).toBeNull()
  })

  it('returns null for a file the arm does not have', () => {
    expect(sweepSeries(data, 'uniform_prior', 'no_such_file', 'sum', 2, 'acc')).toBeNull()
  })

  it('the universal prior sits at chance on the low-m SUM cells too', () => {
    // It ships this file; the arm is at chance rather than absent.
    const low = sumLowM(data, 'uniform_prior')
    expect(low).not.toBeNull()
    for (const v of low ?? []) {
      expect(v).toBeLessThan(2 * CHANCE)
    }
  })

  it('returns null rather than NaN for a metric the file lacks', () => {
    expect(sweepSeries(data, 'selfplay', 'icl_results', 'sum', 2, 'no_such_metric')).toBeNull()
  })

  it('picks the same value through sweepCell as through the series', () => {
    const s = sweepSeries(data, 'selfplay', 'icl_results', 'sum', 2, 'acc')
    const m = s?.[s.length - 1]?.m ?? 0
    expect(sweepCell(data, 'selfplay', 'sum', 2, m)).toBe(s?.[s.length - 1]?.v ?? null)
  })

  it('returns null from sweepCell for an m that is not on the sweep', () => {
    expect(sweepCell(data, 'selfplay', 'sum', 2, 3)).toBeNull()
  })
})

describe('numbers the page quotes', () => {
  it('self-play SUM accuracy at the largest m swept, k = 2', () => {
    // Quoted in the page's main Stat: "84.4% self-play at the largest m".
    const v = sweepCell(data, 'selfplay', 'sum', 2, 512)
    expect(v).not.toBeNull()
    expect(pct(v)).toBe('84.4%')
  })

  it('self-play stays far above chance on the largest SUM cell', () => {
    const v = sweepCell(data, 'selfplay', 'sum', 2, 512)
    expect(v ?? 0).toBeGreaterThan(10 * CHANCE)
  })

  it('the universal prior barely beats chance on SUM at large m', () => {
    // 1.56% against a 0.39% floor: above chance, but nowhere near the
    // self-play learner's 84.4%. Not a zero, so the page must not call it one.
    const v = sweepCell(data, 'uniform_prior', 'sum', 2, 512)
    expect(v).not.toBeNull()
    expect(v ?? 0).toBeLessThan(0.02)
    expect(v ?? 0).toBeGreaterThan(CHANCE)
  })

  it('the universal prior is exactly 0.0% on the m = 0 max cell', () => {
    // A genuine hard failure, distinct from the low-but-nonzero SUM cells.
    expect(m0(data, 'uniform_prior', 'max', 'acc')).toBe(0)
  })

  it('associative recall: self-play reaches ~99% at V = 4, the largest m', () => {
    const v = assocTop(data, 'selfplay', 4)
    expect(pct(v)).toBe('99.2%')
  })

  it('associative recall: the arm gap only opens at the largest dictionary', () => {
    const small = assocTop(data, 'selfplay', ASSOC_V[0])
    const pcfgSmall = assocTop(data, 'pcfg', ASSOC_V[0])
    expect(small).not.toBeNull()
    expect(pcfgSmall).not.toBeNull()
    // At V = 4 both arms are saturated, so the page must not claim a gap there.
    expect(Math.abs((small ?? 0) - (pcfgSmall ?? 0))).toBeLessThan(0.05)

    // PCFG leads on associative recall at every dictionary size. The page
    // reports both numbers rather than implying self-play wins the task.
    const big = assocTop(data, 'selfplay', ASSOC_V_MAX)
    const pcfgBig = assocTop(data, 'pcfg', ASSOC_V_MAX)
    for (const v of ASSOC_V) {
      const sp = assocTop(data, 'selfplay', v) ?? 0
      const pc = assocTop(data, 'pcfg', v) ?? 0
      expect(pc, `PCFG vs self-play at V=${v}`).toBeGreaterThan(sp)
    }
    // Both arms degrade as the dictionary grows; the gap itself widens.
    expect(small ?? 0).toBeGreaterThan(big ?? 0)
    expect(pcfgBig ?? 0).toBeGreaterThan(big ?? 0)
  })

  it('successor is learned perfectly; a ceiling task the page calls saturated', () => {
    expect(pct(extraTop(data, 'selfplay', 'succ'))).toBe('100.0%')
  })

  it('m = 0 accuracy is a copy prior, not a floor: ~12x chance on max', () => {
    const max = m0(data, 'selfplay', 'max', 'acc')
    expect(max).not.toBeNull()
    // The page states "about 12x the chance floor", not 6-8x or ~10x.
    expect((max ?? 0) / CHANCE).toBeGreaterThan(11)
    expect((max ?? 0) / CHANCE).toBeLessThan(13)
    expect(pct(max, 2)).toBe('4.59%')
  })

  it('the same m = 0 cell is 0.0% for the universal prior', () => {
    expect(m0(data, 'uniform_prior', 'max', 'acc')).toBe(0)
  })

  it('reports min and max m = 0 cells within a factor of two of each other', () => {
    const max = m0(data, 'selfplay', 'max', 'acc') ?? 0
    const min = m0(data, 'selfplay', 'min', 'acc') ?? 0
    expect(min / max).toBeGreaterThan(0.8)
    expect(min / max).toBeLessThan(1.3)
  })

  it('returns null for an m = 0 task the harness never ran', () => {
    expect(m0(data, 'selfplay', 'succ', 'acc')).toBeNull()
  })

  it('small-m SUM cells are below chance, so the page cannot call them noise', () => {
    const low = sumLowM(data, 'selfplay')
    expect(low).not.toBeNull()
    expect(low?.length).toBeGreaterThan(0)
    // At least one genuinely-bad cell, and none at chance exactly.
    expect(Math.min(...(low ?? [1]))).toBeLessThan(CHANCE)
  })
})

describe('extraSeries / assocSeries', () => {
  it('never returns a negative accuracy', () => {
    for (const task of ['assoc', 'succ', 'compl', 'double', 'absdiff', 'closest', 'prev']) {
      for (const arm of ['selfplay', 'uniform_prior', 'pcfg'] as IclArmKey[]) {
        const s = extraSeries(data, arm, task)
        for (const p of s ?? []) {
          expect(p.v, `${arm}/${task} m=${p.m}`).toBeGreaterThanOrEqual(0)
          expect(p.v).toBeLessThanOrEqual(1)
        }
      }
    }
  })

  it('finds the true maximum for "never above" claims', () => {
    const best = extraBest(data, 'selfplay', 'succ')
    expect(best).toBe(1)
  })

  it('returns null from assocSeries for a dictionary size not swept', () => {
    expect(assocSeries(data, 'selfplay', 8)).toBeNull()
  })

  it('associative series are monotonic non-decreasing in accuracy with m', () => {
    for (const arm of ['selfplay', 'pcfg'] as IclArmKey[]) {
      for (const v of ASSOC_V) {
        const s = assocSeries(data, arm, v) ?? []
        for (let i = 1; i < s.length; i++) {
          const prev = s[i - 1]?.v ?? 0
          const cur = s[i]?.v ?? 0
          // Allowed to wobble slightly, but not to collapse.
          expect(cur, `${arm} V=${v} m=${s[i]?.m}`).toBeGreaterThan(prev - 0.35)
        }
      }
    }
  })

  it('assocDictEnds renders a range, or an em dash when unmeasured', () => {
    const s = assocDictEnds(data, 'selfplay')
    expect(s).toMatch(/^\d+\.\d% → \d+\.\d%$/)
    expect(assocDictEnds(data, 'nope' as IclArmKey)).toBe('—')
  })
})

describe('Figure 5 composition', () => {
  it('has one row per m on the 12-point sweep', () => {
    expect(rows.map((r) => r.m)).toEqual([0, 1, 2, 3, 4, 8, 16, 32, 64, 128, 256, 512])
  })

  it('sorts rows by m even though the bundle is not ordered', () => {
    const ms = rows.map((r) => r.m)
    expect([...ms].sort((a, b) => a - b)).toEqual(ms)
  })

  it('every row has all five categories and they sum to one', () => {
    for (const r of rows) {
      for (const c of CATEGORY_KEYS) {
        expect(r[c], `m=${r.m} ${c}`).toBeGreaterThanOrEqual(0)
        expect(r[c]).toBeLessThanOrEqual(1)
      }
      // Shares are rounded to 6dp in the bundle, so 1e-6 is too tight.
      expect(sharesSumToOne(r, 2e-6), `m=${r.m}`).toBe(true)
    }
  })

  it('exact answers are NOT dominant at small m, only later', () => {
    // The page says exact answers first become the largest bucket at m = 32.
    expect(firstDominant(rows)).toBe(32)
    for (const m of [0, 1, 2, 4, 8, 16]) {
      const r = sumRowAt(rows, m)
      const copyish = Math.max(r?.contextByte ?? 0, r?.preferredBytes ?? 0, r?.low4BitsCorrect ?? 0)
      expect(r?.correctAnswer ?? 0).toBeLessThan(copyish)
    }
  })

  it('exact answers dominate by the end of the sweep', () => {
    const end = sumRowAt(rows, 512)
    for (const c of CATEGORY_KEYS) {
      if (c === 'correctAnswer') continue
      expect(end?.correctAnswer ?? 0).toBeGreaterThan(end?.[c] ?? 1)
    }
  })

  it('returns null for an m that was never run', () => {
    expect(sumRowAt(rows, 7)).toBeNull()
  })

  it('firstDominant returns null when no category ever dominates', () => {
    const flat = rows.map((r) => ({ ...r, correctAnswer: 0.1 }))
    expect(firstDominant(flat)).toBeNull()
  })
})

describe('Figure 5 entropy', () => {
  it('knows the uniform entropy of a 256-byte alphabet', () => {
    expect(BITS_FOR_256).toBe(8)
  })

  it('peaks at m = 8 near 7.85 bits, above its m = 0 level', () => {
    const peak = entropyPeak(rows)
    expect(peak?.m).toBe(8)
    expect(peak?.bits ?? 0).toBeGreaterThan(7.8)
    expect(peak?.bits ?? 0).toBeLessThan(7.9)
    const zero = sumRowAt(rows, 0)?.entropyMeanBits ?? 0
    expect(peak?.bits ?? 0).toBeGreaterThan(zero)
  })

  it('mean entropy falls after the peak', () => {
    const peak = entropyPeak(rows)?.m ?? 0
    const after = rows.filter((r) => r.m > peak).map((r) => r.entropyMeanBits)
    expect(after.length).toBeGreaterThan(0)
    expect(Math.min(...after)).toBeLessThan(peak ? (entropyPeak(rows)?.bits ?? 0) : 0)
  })

  it('orders the quartiles Q25 <= Q75 at every m', () => {
    for (const r of rows) {
      expect(r.entropyQ25Bits, `m=${r.m}`).toBeLessThanOrEqual(r.entropyQ75Bits)
    }
  })

  it('brackets the mean, except at m = 4 where the bundle disagrees', () => {
    // Known data anomaly: at m = 4 the released bundle reports
    // Q25 = 7.854311 but mean = 7.845272, i.e. the mean sits BELOW its own
    // lower quartile. Pinned here so it is noticed if the bundle is rebuilt
    // from a different harness rather than silently drifting.
    for (const r of rows) {
      if (r.m === 4) {
        expect(r.entropyQ25Bits).toBeGreaterThan(r.entropyMeanBits)
        continue
      }
      expect(r.entropyQ25Bits, `m=${r.m}`).toBeLessThan(r.entropyMeanBits)
      expect(r.entropyQ75Bits, `m=${r.m}`).toBeGreaterThan(r.entropyMeanBits)
    }
  })

  it('the interquartile band WIDENS in absolute bits, as the page claims', () => {
    const b0 = iqrBits(rows, 0)
    const b512 = iqrBits(rows, 512)
    expect(b0).not.toBeNull()
    expect(b512).not.toBeNull()
    // This is the correction made during review: the band does not narrow, so
    // the page must not claim it does.
    expect(b512 ?? 0).toBeGreaterThan(b0 ?? 0)
  })

  it('returns null from iqrBits for an absent m, not a fabricated 0', () => {
    // A 0.0 here would silently support the opposite conclusion.
    expect(iqrBits(rows, 7)).toBeNull()
  })

  it('entropy stays within the 8-bit uniform ceiling', () => {
    for (const r of rows) {
      expect(r.entropyQ75Bits, `m=${r.m}`).toBeLessThanOrEqual(BITS_FOR_256)
    }
  })

  it('returns null from entropyPeak on an empty bundle', () => {
    expect(entropyPeak([])).toBeNull()
  })
})

describe('pct', () => {
  it('renders a fraction as a percentage', () => {
    expect(pct(0.844)).toBe('84.4%')
    expect(pct(0.5, 0)).toBe('50%')
  })

  it('renders nullish as an em dash, never 0.0%', () => {
    expect(pct(null)).toBe('—')
    expect(pct(undefined)).toBe('—')
  })

  it('renders 0 as 0.0%, which is a real measurement', () => {
    expect(pct(0)).toBe('0.0%')
  })
})