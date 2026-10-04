/**
 * Tests for the reward page's Table 5 derivations.
 *
 * Two jobs. First, pin every number the page's verdict quotes against the
 * released bundle and against the authors' committed reward_arms.tex, so an
 * edit that changes a claim without changing the data fails here. Second, cover
 * the null and tie paths: the page's argument is a set of "worse on X of Y
 * rows" claims, and a helper that counted an unscored row, divided by an
 * unscored base, or missed a printed tie would turn a weaker instrument into a
 * stronger-looking claim without anything visibly changing.
 *
 * These load the real public/data/table5.json rather than a fixture, because
 * the thing being protected is the agreement between the code, the bundle, and
 * the paper's committed table.
 */
import { describe, expect, it } from 'vitest'
import table5 from '../../public/data/table5.json'
import {
  FOOTNOTES, TIE_EPS, UNIFORM_256_BPB,
  armBars, canonicalBestCount, cellBpb, cellRatio, cellSeeds, compareArms,
  fmt, heatAlpha, heatScale, heldOutSeqs, labels, median, minSeeds, ratios,
  rowBest, rowLabel, scoredRowCount, scoredRows, splitAbove, valuesOn,
} from './reward'
import type { Table5, Table5Row } from '@/data/types'

const data = table5 as unknown as Table5
const COLUMN_KEYS = data.columns.map((c) => c.key)

/** Round to the precision reward_arms.tex prints every cell at. */
const printed = (v: number): string => v.toFixed(2)

describe('bundle shape', () => {
  it('carries the seven ablation columns the page names', () => {
    expect(COLUMN_KEYS).toEqual([
      'none', 'uniform', 'signed', 'shuffle', 'last_step', 'loss_delta', 'negate',
    ])
  })

  it('is scored on every row for every arm, so every denominator reads 10', () => {
    for (const key of COLUMN_KEYS) {
      expect(scoredRowCount(data, key), key).toBe(data.rows.length)
    }
    expect(data.rows.length).toBe(10)
  })

  it('maps the two footnote markers onto the arms the caption footnotes', () => {
    expect(FOOTNOTES.last_step).toBe('b')
    expect(FOOTNOTES.loss_delta).toBe('a,b')
    // The caption's own text, so the markers cannot drift off the prose.
    expect(data.notes[1]).toContain('last_step and loss_delta are bimodal')
  })

  it('marks the 3-seed ensemble as a short ensemble, not as 4', () => {
    expect(data.K).toBe(4)
    expect(minSeeds(data, 'loss_delta')).toBe(3)
    // The footnote a cell count is printed against: one seed stopped at 2587.
    for (const key of ['none', 'uniform', 'signed', 'shuffle', 'last_step', 'negate']) {
      expect(minSeeds(data, key), key).toBe(data.K)
    }
  })
})

describe('cell access', () => {
  it('reads a cell by column key', () => {
    const arith = data.rows.find((r) => r.key === 'arithmetic')
    expect(cellBpb(data, arith as Table5Row, 'none')).toBeCloseTo(0.2189, 4)
    expect(cellSeeds(data, arith as Table5Row, 'none')).toBe(4)
  })

  it('returns null for a column key the table does not have', () => {
    const row = data.rows[0] as Table5Row
    expect(cellBpb(data, row, 'nope')).toBeNull()
    expect(cellSeeds(data, row, 'nope')).toBeNull()
  })

  it('returns null for an unscored cell rather than NaN', () => {
    const first = data.rows[0] as Table5Row
    const holed: Table5 = {
      ...data,
      rows: [{ ...first, cells: [null, ...first.cells.slice(1)] }, ...data.rows.slice(1)],
    }
    expect(cellBpb(holed, holed.rows[0] as Table5Row, 'none')).toBeNull()
    expect(scoredRowCount(holed, 'none')).toBe(9)
    expect(scoredRows(holed, 'none')).toHaveLength(9)
  })

  it('cellRatio is null where the row has no best, never Infinity', () => {
    // On the dclm row the best is `signed`, not `none`, so the ratio of the
    // canonical arm is not 1 -- it must still be exactly bpb / rowBest.
    const row = data.rows[0] as Table5Row
    const signed = cellBpb(data, row, 'signed') as number
    const none = cellBpb(data, row, 'none') as number
    expect(rowBest(row)).toBe(signed)
    expect(cellRatio(data, row, 'signed')).toBeCloseTo(1, 10)
    expect(cellRatio(data, row, 'none')).toBeCloseTo(none / signed, 12)
    expect(cellRatio(data, row, 'none')).toBeGreaterThan(1)
    // A zero-loss row would divide to Infinity without the positive guard.
    const zeroBest: Table5Row = { ...row, cells: row.cells.map(() => ({ bpb: 0, kUsed: 1 })) }
    expect(rowBest(zeroBest)).toBe(0)
    expect(cellRatio(data, zeroBest, 'none')).toBeNull()
    expect(heatScale({ ...data, rows: [zeroBest] }).ratios).toEqual([])
    // A cell with no bpb, and a row with no best, both read null rather than
    // NaN or Infinity.
    expect(cellRatio(data, row, 'no_such_arm')).toBeNull()
    const empty: Table5Row = { ...row, cells: row.cells.map(() => null) }
    expect(cellRatio(data, empty, 'none')).toBeNull()
  })
})

describe('rowBest agrees with bestIndex', () => {
  it('picks the same cell the builder marked, on every row', () => {
    for (const row of data.rows) {
      const best = rowBest(row)
      const marked = row.cells[row.bestIndex ?? -1]
      expect(best, row.label).not.toBeNull()
      expect(marked, `${row.label} bestIndex cell exists`).not.toBeNull()
      expect(printed(best ?? Number.NaN), row.label).toBe(printed(marked?.bpb ?? Number.NaN))
    }
  })

  it('agrees on the random-bytes row, where full precision used to disagree', () => {
    // The row that motivated the 2 dp comparison: uniform (8.0163) is below
    // none (8.0204) at full precision, but reward_arms.tex prints both as 8.02
    // and bolds them together. Comparing at full precision shaded a different
    // cell as best than the table printed directly above it.
    const row = data.rows.find((r) => r.key === 'aitdcc_d_glibc_rand') as Table5Row
    const uniform = cellBpb(data, row, 'uniform') ?? 0
    const none = cellBpb(data, row, 'none') ?? 0
    expect(uniform).toBeLessThan(none)
    expect(row.bestIndex).toBe(0)
    // rowBest keeps the canonical arm, which is what the table bolds first.
    expect(rowBest(row)).toBeCloseTo(none, 4)
    expect(printed(rowBest(row) ?? 0)).toBe('8.02')
  })

  it('returns null for a row with no scored cell', () => {
    const row = data.rows[0] as Table5Row
    expect(rowBest({ ...row, cells: row.cells.map(() => null) })).toBeNull()
  })

  it('skips null cells instead of treating them as zero', () => {
    const row = data.rows.find((r) => r.key === 'arithmetic') as Table5Row
    const holed: Table5Row = { ...row, cells: [null, ...row.cells.slice(1)] }
    expect(rowBest(holed)).toBeCloseTo(0.5124, 4)
  })
})

describe('the tie bucket', () => {
  it('reads shuffle as worse on 9 of 10, with exactly one tie', () => {
    // The bug this pins: the 0.0031 bits/byte gap on random bytes (shuffle
    // 8.0235 vs none 8.0204) is below TIE_EPS, so the page must say 9 of 10
    // and name the tie -- not "worse on 10 of 10" where the paper's own table
    // prints a three-way bold.
    const shuffle = compareArms(data, 'shuffle', 'none')
    expect(shuffle.worse.length).toBe(9)
    expect(shuffle.ties).toEqual(['random bytes'])
    expect(shuffle.notWorse.length).toBe(0)
    expect(shuffle.scored).toBe(10)
  })

  it('puts uniform in the same tie bucket on the same row', () => {
    // uniform 8.0163 vs none 8.0204: a 0.0041 gap, also inside TIE_EPS. The
    // page's uniform claim is 9 of 10, not 10 of 10.
    const uniform = compareArms(data, 'uniform', 'none')
    expect(uniform.ties).toEqual(['random bytes'])
    expect(uniform.worse.length).toBe(9)
  })

  it('keeps a gap larger than TIE_EPS out of the tie bucket', () => {
    const lastStep = compareArms(data, 'last_step', 'none')
    expect(lastStep.ties).toEqual([])
    expect(lastStep.worse.length).toBe(10)
    // The tightest non-tie gap on that row still clears the threshold.
    const row = data.rows.find((r) => r.key === 'aitdcc_d_glibc_rand') as Table5Row
    const gap = (cellBpb(data, row, 'last_step') ?? 0) - (cellBpb(data, row, 'none') ?? 0)
    expect(gap).toBeGreaterThan(TIE_EPS)
  })

  it('treats exactly TIE_EPS as a tie, and just above as a loss', () => {
    // Boundary of the printed precision, pinned so a changed constant is caught.
    const t2: Table5 = { ...data, K: 1, rows: synthetic(1 + TIE_EPS, 1) }
    expect(compareArms(t2, 'none', 'uniform').ties.length).toBe(1)
    const t3: Table5 = { ...data, K: 1, rows: synthetic(1 + TIE_EPS * 1.01, 1) }
    expect(compareArms(t3, 'none', 'uniform').worse.length).toBe(1)
    expect(compareArms(t3, 'none', 'uniform').ties.length).toBe(0)
  })
})

describe('compareArms denominators', () => {
  it('counts only rows where both arms are scored', () => {
    const partial: Table5 = {
      ...data,
      rows: data.rows.slice(0, 3),
    }
    const pair = compareArms(partial, 'signed', 'none')
    expect(pair.scored).toBe(3)
    // ...not rows.length of some other table, and never more than the rows.
    expect(pair.scored).toBeLessThanOrEqual(partial.rows.length)
  })

  it('drops a row where either arm is unscored, from numerator and denominator', () => {
    const row = data.rows[0] as Table5Row
    const holed = { ...row, cells: row.cells.map(() => null) }
    const t: Table5 = { ...data, rows: [holed, ...data.rows.slice(1)] }
    const pair = compareArms(t, 'signed', 'none')
    expect(pair.scored).toBe(data.rows.length - 1)
    expect(labels(pair.worse)).not.toContain(holed.label)
    expect(pair.ties).not.toContain(holed.label)
    // every bucket plus the denominator accounts for exactly the scored rows
    expect(pair.worse.length + pair.notWorse.length + pair.ties.length).toBe(pair.scored)
  })

  it('never divides by a zero or negative base', () => {
    const zeroed: Table5 = { ...data, rows: synthetic(0, 0) }
    const pair = compareArms(zeroed, 'none', 'uniform')
    expect(pair.scored).toBe(0)
    for (const [, r] of pair.worse) expect(Number.isFinite(r)).toBe(true)
    expect(ratios(pair.worse)).toEqual([])
    // A zero arm value is a real measurement and still counted when the base
    // is positive, rather than filtered out.
    const zeroArm: Table5 = { ...data, rows: synthetic(0, 1) }
    expect(compareArms(zeroArm, 'none', 'uniform').scored).toBe(1)
    expect(compareArms(zeroArm, 'none', 'uniform').notWorse.length).toBe(1)
    expect(ratios(compareArms(zeroArm, 'none', 'uniform').notWorse)).toEqual([0])
  })

  it('buckets every scored row exactly once, for all six comparisons', () => {
    for (const [arm, base] of [
      ['negate', 'none'], ['shuffle', 'none'], ['last_step', 'none'],
      ['loss_delta', 'last_step'], ['signed', 'none'], ['uniform', 'none'],
    ] as [string, string][]) {
      const pair = compareArms(data, arm, base)
      const counted = pair.worse.length + pair.notWorse.length + pair.ties.length
      expect(counted, `${arm} vs ${base}`).toBe(pair.scored)
      expect(pair.scored, `${arm} vs ${base}`).toBe(10)
    }
  })
})

describe('numbers the verdict quotes', () => {
  it('negate is above the uniform-over-256-bytes baseline on 9 of 10', () => {
    // Only audio 8-bit PCM lands below 8.0, and the page names it.
    const { above, atOrBelow } = splitAbove(data, 'negate', UNIFORM_256_BPB)
    expect(above.length).toBe(9)
    expect(atOrBelow).toEqual(['audio 8-bit PCM'])
    expect(scoredRowCount(data, 'negate')).toBe(10)
    // The named exception is a real measurement, and a bad one, not a gap.
    const row = data.rows.find((r) => r.label === 'audio 8-bit PCM') as Table5Row
    expect(cellBpb(data, row, 'negate')).toBeCloseTo(7.6745, 4)
    expect(cellBpb(data, row, 'negate')).toBeLessThan(UNIFORM_256_BPB)
  })

  it('even that exception sits far above the canonical arm', () => {
    expect(valuesOn(data, 'negate', ['audio 8-bit PCM'])).toBe('audio 8-bit PCM 7.67')
    expect(valuesOn(data, 'none', ['audio 8-bit PCM'])).toBe('audio 8-bit PCM 2.27')
  })

  it('medians of the worse-row ratios, recomputed from the bundle', () => {
    // Independently derived here rather than copied: the median of the ratios
    // over the rows the helper says are worse.
    const check = (arm: string, base: string, want: string) => {
      const pair = compareArms(data, arm, base)
      const sorted = ratios(pair.worse).sort((a, b) => a - b)
      const mid = Math.floor(sorted.length / 2)
      const expectMid = sorted.length % 2 === 1
        ? (sorted[mid] as number)
        : (((sorted[mid - 1] as number) + (sorted[mid] as number)) / 2)
      expect(sorted.length, `${arm} worse rows`).toBe(pair.worse.length)
      expect(fmt(median(pair.worse.map(([, r]) => r))), arm).toBe(want)
      expect(fmt(expectMid)).toBe(want)
    }
    check('negate', 'none', '2.83')
    check('shuffle', 'none', '1.20')
    check('last_step', 'none', '1.25')
    check('loss_delta', 'last_step', '1.13')
    check('uniform', 'none', '1.73')
  })

  it('the loss_delta median is over 10 ratios even though it runs 3 seeds', () => {
    const pair = compareArms(data, 'loss_delta', 'last_step')
    expect(ratios(pair.worse).length).toBe(10)
    expect(minSeeds(data, 'loss_delta')).toBe(3)
    // The 3-seed ensemble never drops out of the row count: the caveat is
    // about noise, not about an unscored cell.
    expect(scoredRowCount(data, 'loss_delta')).toBe(10)
  })

  it('signed is better on exactly 2 of 10, and the page keeps saying so', () => {
    // The honest disclosure: the abs is the weakest of the six. Pinned so a
    // change to TIE_EPS or to the comparison cannot quietly round it away.
    const signed = compareArms(data, 'signed', 'none')
    expect(signed.notWorse.length).toBe(2)
    expect(signed.worse.length).toBe(8)
    expect(labels(signed.notWorse)).toEqual(['text (dclm)', 'C source'])
    expect(valuesOn(data, 'signed', labels(signed.notWorse)))
      .toBe('text (dclm) 5.06, C source 4.10')
    expect(valuesOn(data, 'none', labels(signed.notWorse)))
      .toBe('text (dclm) 5.34, C source 4.16')
  })

  it('on those two rows the paper prints signed as the row best', () => {
    for (const label of ['text (dclm)', 'C source']) {
      const row = data.rows.find((r) => r.label === label) as Table5Row
      expect(row.bestIndex, label).toBe(2)
      expect(data.columns[row.bestIndex ?? 0]?.key, label).toBe('signed')
    }
  })

  it('the melody margin is thin: signed under 0.01 bits/byte from canonical', () => {
    const row = data.rows.find((r) => r.key === 'mutopia_melody_16th') as Table5Row
    const gap = (cellBpb(data, row, 'signed') ?? 0) - (cellBpb(data, row, 'none') ?? 0)
    expect(gap).toBeGreaterThan(0)
    expect(gap).toBeLessThan(0.01)
    // Not a tie: it is a strict loss, and counts as one.
    expect(compareArms(data, 'signed', 'none').worse.length).toBe(8)
    expect(labels(compareArms(data, 'signed', 'none').worse)).toContain('melody (Mutopia)')
  })

  it('melody rests on 17 held-out sequences, not the 256 the caption claims', () => {
    expect(heldOutSeqs(data, 'mutopia_melody_16th', 17)).toBe(17)
    expect(heldOutSeqs(data, 'no_such_corpus', 17)).toBe(17)
    // The rest do contribute the caption's 256.
    expect(heldOutSeqs(data, 'aitdcc_d_glibc_rand', 256)).toBe(256)
  })

  it('arithmetic is where the narrow frontier hurts: 36x the canonical loss', () => {
    expect(valuesOn(data, 'shuffle', ['arithmetic'])).toBe('arithmetic 0.73')
    expect(valuesOn(data, 'none', ['arithmetic'])).toBe('arithmetic 0.22')
    expect(valuesOn(data, 'uniform', ['arithmetic'])).toBe('arithmetic 7.94')
    expect(valuesOn(data, 'uniform', ['melody (Mutopia)'])).toBe('melody (Mutopia) 7.09')
  })

  it('uniform lands within 0.01 of canonical on random bytes, as the page says', () => {
    const row = rowByKey('aitdcc_d_glibc_rand')
    const gap = Math.abs(
      (cellBpb(data, row, 'uniform') ?? 0) - (cellBpb(data, row, 'none') ?? 0),
    )
    expect(gap).toBeLessThan(0.01)
    expect(valuesOn(data, 'uniform', ['random bytes'])).toBe('random bytes 8.02')
    expect(rowLabel(data, 'aitdcc_d_glibc_rand', 'random bytes')).toBe('random bytes')
  })

  it('the canonical arm is the printed best on 8 of 10, not 10', () => {
    // "A consistent winner, not a uniform one": the two rows signed takes are
    // the exception, and the caveat must not round up to all 10. The 10 is the
    // canonical arm's own scored-row count, not bestIndex's.
    expect(canonicalBestCount(data)).toBe(8)
    expect(scoredRowCount(data, 'none')).toBe(10)
    for (const row of data.rows) {
      if (row.bestIndex === 0) continue
      expect(data.columns[row.bestIndex ?? 0]?.key, row.label).toBe('signed')
    }
  })
})

describe('median', () => {
  it('averages the two middle values on an even sample', () => {
    expect(median([1, 2, 3, 4])).toBe(2.5)
    expect(median([4, 1, 3, 2])).toBe(2.5)
  })

  it('takes the middle value on an odd sample', () => {
    expect(median([3, 1, 2])).toBe(2)
    expect(median([10])).toBe(10)
  })

  it('does not mutate its input', () => {
    const xs = [3, 1, 2]
    median(xs)
    expect(xs).toEqual([3, 1, 2])
  })

  it('returns null on an empty sample rather than NaN', () => {
    expect(median([])).toBeNull()
    expect(fmt(median([]))).toBe('—')
  })
})

describe('valuesOn / fmt', () => {
  it('quotes only rows the table actually has', () => {
    expect(valuesOn(data, 'none', ['Metamath'])).toBe('Metamath 3.38')
    expect(valuesOn(data, 'none', ['no such dataset'])).toBe('')
    expect(valuesOn(data, 'no_such_arm', ['Metamath'])).toBe('Metamath —')
  })

  it('formats a bits/byte value at 2 dp and nullish as an em dash', () => {
    expect(fmt(0.2189)).toBe('0.22')
    expect(fmt(null)).toBe('—')
    expect(fmt(3.3796, 4)).toBe('3.3796')
  })
})

describe('splitAbove / scoredRows', () => {
  it('puts an unscored row in neither bucket', () => {
    const holed: Table5Row = { ...(data.rows[0] as Table5Row), cells: [] }
    const t: Table5 = { ...data, rows: [holed, ...data.rows.slice(1)] }
    const split = splitAbove(t, 'none', UNIFORM_256_BPB)
    expect(split.above.length + split.atOrBelow.length).toBe(9)
    expect(split.above).not.toContain(holed.label)
    expect(split.atOrBelow).not.toContain(holed.label)
  })

  it('scoredRows returns the rows themselves, not just the count', () => {
    expect(scoredRows(data, 'none').length).toBe(data.rows.length)
    expect(scoredRows(data, 'none')[0]?.key).toBe(data.rows[0]?.key)
  })
})

describe('heatScale / heatAlpha', () => {
  it('spans the table, with the worst cell ~48x its own row best', () => {
    const { ratios: rs, maxRatio, legendSpan } = heatScale(data)
    expect(rs.length).toBe(data.rows.length * data.columns.length)
    expect(maxRatio).toBeGreaterThan(48)
    expect(maxRatio).toBeLessThan(49)
    expect(legendSpan).toBeCloseTo(maxRatio, 10)
  })

  it('is 0 for the best cell of a row, and rises with the log ratio', () => {
    const maxRatio = heatScale(data).maxRatio
    expect(heatAlpha(1, maxRatio)).toBe(0)
    expect(heatAlpha(0.9995, maxRatio)).toBe(0)
    // Below 1 (the random-bytes uniform cell) is a tie, not a negative shade.
    expect(heatAlpha(0.5, maxRatio)).toBe(0)
    const mid = heatAlpha(10, maxRatio)
    expect(mid).toBeGreaterThan(0)
    expect(mid).toBeLessThan(1)
    expect(heatAlpha(1000, maxRatio)).toBe(0.72)
  })

  it('is logarithmic, so equal steps in the log ratio are equal steps of shade', () => {
    const maxRatio = heatScale(data).maxRatio
    const span = Math.log10(maxRatio)
    // Halfway in log ratio must be exactly half the opacity band. A linear
    // ramp would put it far below half and crowd every cell into the first
    // fifth of the ramp, which is what the 48x worst row forces.
    for (const f of [0.25, 0.5, 0.75]) {
      expect(heatAlpha(10 ** (f * span), maxRatio)).toBeCloseTo(f * 0.72, 10)
    }
    // ...and the band is filled exactly at the top of the scale.
    expect(heatAlpha(maxRatio, maxRatio)).toBeCloseTo(0.72, 10)
    // A doubling of the ratio is the same increment of shade everywhere.
    const d1 = heatAlpha(4, maxRatio) - heatAlpha(2, maxRatio)
    const d2 = heatAlpha(8, maxRatio) - heatAlpha(4, maxRatio)
    expect(d1).toBeCloseTo(d2, 10)
    expect(d1).toBeGreaterThan(0)
  })

  it('gives the random-bytes row its exact near-tie shades, not all zero', () => {
    // At 3 dp of printed opacity these are all 0.000, but they are not the
    // same number: shuffle (1.0004x) must shade a hair more than the
    // sub-1.0 uniform cell, which must read as an outright tie at 0. Any
    // widening of the `ratio > 1` cutoff would flatten both.
    const maxRatio = heatScale(data).maxRatio
    const row = rowByKey('aitdcc_d_glibc_rand')
    const none = heatAlpha(cellRatio(data, row, 'none') as number, maxRatio)
    const uniform = heatAlpha(cellRatio(data, row, 'uniform') as number, maxRatio)
    const shuffle = heatAlpha(cellRatio(data, row, 'shuffle') as number, maxRatio)
    expect(none).toBe(0)
    expect(uniform).toBe(0)
    expect(shuffle).toBeGreaterThan(0)
    // shuffle vs the row best is the same 0.0031 gap the tie bucket treats as
    // printed-equal, shaded as a real (if invisible) step.
    const shuffleBpb = cellBpb(data, row, 'shuffle') as number
    const bestBpb = rowBest(row) as number
    expect(shuffleBpb - bestBpb).toBeCloseTo(0.0031, 4)
    expect(shuffleBpb - bestBpb).toBeLessThanOrEqual(TIE_EPS)
    expect(shuffle).toBeCloseTo(
      Math.log10(shuffleBpb / bestBpb) / Math.log10(maxRatio) * 0.72, 12,
    )
    expect(none).toBeLessThan(shuffle)
  })

  it('floors the legend span at 10x when every row is within 10x of its best', () => {
    // Without the floor a tight table would render a legend of 1.0x-1.4x and
    // the ramp would be mostly empty. maxRatio itself must stay untouched.
    const tight: Table5 = {
      ...data,
      rows: data.rows.map((r) => ({
        ...r,
        cells: r.cells.map((c) => (c ? { ...c, bpb: 1 + (c.bpb % 0.3) } : c)),
      })),
    }
    const { maxRatio, legendSpan } = heatScale(tight)
    expect(maxRatio).toBeLessThan(10)
    expect(legendSpan).toBe(10)
  })

  it('stays inside the opacity band for every real cell', () => {
    const { ratios: rs, maxRatio } = heatScale(data)
    for (const r of rs) {
      const a = heatAlpha(r, maxRatio)
      expect(Number.isNaN(a)).toBe(false)
      expect(a).toBeGreaterThanOrEqual(0)
      expect(a).toBeLessThanOrEqual(0.72)
    }
  })

  it('leaves every row with one fully zero-shaded cell (its own best)', () => {
    const { maxRatio } = heatScale(data)
    for (const row of data.rows) {
      const alphas = data.columns
        .map((c) => cellRatio(data, row, c.key))
        .map((r) => (r === null ? 0 : heatAlpha(r, maxRatio)))
      expect(Math.min(...alphas), row.label).toBe(0)
    }
  })

  it('falls back to a 10x span on a table with no usable row best', () => {
    const empty: Table5 = { ...data, rows: data.rows.map((r) => ({ ...r, cells: [] })) }
    const { ratios: rs, maxRatio, legendSpan } = heatScale(empty)
    expect(rs).toEqual([])
    expect(maxRatio).toBe(10)
    expect(legendSpan).toBe(10)
  })
})

describe('armBars', () => {
  it('sorts a row ascending in loss and labels the canonical arm', () => {
    const bars = armBars(data, rowByKey('arithmetic'))
    expect(bars.length).toBe(data.columns.length)
    const bs = bars.map((b) => b.bpb)
    expect([...bs].sort((a, b) => a - b)).toEqual(bs)
    expect(bars[0]?.key).toBe('none')
    expect(fmt(bars[0]?.bpb ?? null)).toBe('0.22')
    // loss_delta runs 3 seeds and must carry that through to the label.
    const ld = bars.find((b) => b.key === 'loss_delta')
    expect(ld?.kUsed).toBe(3)
  })

  it('drops an unscored arm instead of charting it at 0.00', () => {
    const row = rowByKey('arithmetic')
    const holed: Table5Row = { ...row, cells: [null, ...row.cells.slice(1)] }
    const bars = armBars(data, holed)
    expect(bars.length).toBe(data.columns.length - 1)
    expect(bars.some((b) => b.bpb === 0)).toBe(false)
  })

  it('returns an empty list for a missing row', () => {
    expect(armBars(data, undefined)).toEqual([])
  })
})

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function rowByKey(key: string): Table5Row {
  const row = data.rows.find((r) => r.key === key)
  if (!row) throw new Error(`no row ${key}`)
  return row
}

/**
 * A two-column, one-row table carrying the given `none` and `uniform` losses,
 * for the boundary and divide-by-zero paths where the real table's precision
 * is not the thing under test.
 */
function synthetic(none: number, uniform: number): Table5Row[] {
  return [{
    key: 'synthetic',
    label: 'synthetic',
    bestIndex: null,
    cells: [
      { bpb: none, kUsed: 1 },
      { bpb: uniform, kUsed: 1 },
    ],
  }]
}
