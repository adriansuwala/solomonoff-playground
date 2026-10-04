/**
 * Tests for the scaling page's data derivations.
 *
 * Two jobs. First, pin the arithmetic behind every claim the page makes --
 * especially the shared-budget arm comparison, the floor conditional and the
 * per-rung compute pools, each of which has already been wrong once. Second,
 * pin the *exclusions*: which corpora are compared and which are not, because
 * an undisclosed exclusion once carried a load-bearing conclusion.
 *
 * These load the real public/data/scaling.json and meta.json rather than a
 * fixture, because the thing being protected is the agreement between code,
 * bundle and the prose on the page.
 */
import { describe, expect, it } from 'vitest'
import scalingJson from '../../public/data/scaling.json'
import metaJson from '../../public/data/meta.json'
import {
  ARM_ORDER, FIT_SAMPLES, MAX_RAW_POINTS, TIE_MARGIN,
  type GroupTally,
  amplitude, armTotals, brokenDownFloors, compareArms, corpusGroups,
  exponentMismatches, exponentRows, fitCurve, floorReading, groupVerdict,
  isExcluded, isTie, ladderFit, literatureEntries, lossDomain, ordinal,
  paperExponent, poolAt, poolsOf, recoverPool, sliceCorpus, subsample,
  subsampleStride, tallyGroups,
} from './scaling'
import { formatCompute, predictFit } from './scales'
import type {
  ArmKey, ArmScaling, CorpusScaling, Meta, PowerFit, Scaling,
} from '@/data/types'

const scaling = scalingJson as unknown as Scaling
const meta = metaJson as unknown as Meta
const CTX = meta.compute.context

/**
 * The pool the authors' own per-rung files record, transcribed from
 * vendor/spp/figures/fig2_transfer_across_modalities/data/*_programs_per_round.json
 * and cross-checked by tools/verify_data.py. Self-play is a constant 1536; the
 * two fixed arms were run at per-rung rates.
 */
const AUTHOR_POOLS: Record<string, Record<string, number>> = {
  selfplay: {
    d64h1L1: 1536, d128h2L2: 1536, d128h2L4: 1536,
    d256h4L4: 1536, d256h4L8: 1536, d512h8L8: 1536,
  },
  // The authors' file for the fixed-prior arm is named "prior"; the bundle
  // calls the arm "uniform".
  prior: {
    d64h1L1: 1024, d128h2L2: 2048, d128h2L4: 1024,
    d256h4L4: 2048, d256h4L8: 1024,
  },
  pcfg: {
    d64h1L1: 512, d128h2L2: 2048, d128h2L4: 2048,
    d256h4L4: 1024, d256h4L8: 1024,
  },
}

/** The N each rung contributes, from the Figure 2 CSVs' own N column. */
const RUNG_N: Record<string, number> = {
  d64h1L1: 65728, d128h2L2: 492160, d128h2L4: 984192,
  d256h4L4: 3016960, d256h4L8: 6033664, d512h8L8: 24253184,
}

/** The six corpora the bundle marks `excluded`, i.e. never compared. */
const EXCLUDED = [
  'arithmetic', 'aitdcc_a_protein', 'kolmogorov_dna',
  'aitdcc_d_glibc_rand', 'aitdcc_e_atlas_float32', 'aitdcc_g_astronomy',
]

/**
 * One ladder corpus, asserted present. The bundle types its maps as
 * Record<string, T>, so a plain index is `T | undefined` under
 * noUncheckedIndexedAccess; going through here means a missing corpus fails as
 * a clear "not in the bundle" rather than as a null-dereference further down.
 */
function ladderCs(key: string): CorpusScaling {
  const cs = scaling.ladder.corpora[key]
  expect(cs, `${key} is in the ladder bundle`).toBeDefined()
  return cs as CorpusScaling
}

/** One arm's columnar bundle, asserted present. */
function armBundle(key: ArmKey): ArmScaling {
  const a = scaling.arms[key]
  expect(a, `${key} arm is in the bundle`).toBeDefined()
  return a as ArmScaling
}

// ---------------------------------------------------------------------------

describe('bundle shape', () => {
  it('carries 26 ladder corpora, six of them marked excluded', () => {
    expect(meta.corpora.length).toBe(26)
    const excluded = meta.corpora.filter((c) => c.group === 'excluded').map((c) => c.key)
    expect(excluded.sort()).toEqual([...EXCLUDED].sort())
  })

  it('groups every ladder corpus, with no corpus in the bundle but not the meta', () => {
    const metaKeys = meta.corpora.map((c) => c.key).sort()
    const ladderKeys = scaling.ladder.stringTables.corpora.slice().sort()
    expect(ladderKeys).toEqual(metaKeys)
  })

  it('encodes the compute formula the page prints', () => {
    expect(meta.compute.formula).toBe('C = K * N * POOL * CTX * (round + 1)')
    expect(CTX).toBe(4096)
    expect(meta.compute.pool).toBe(1536)
  })

  it('keeps the ladder and the three arm bundles separate (docs/decisions.md D3)', () => {
    // Table 2 comes from scaling.ladder, Figure 2 from scaling.arms, and their
    // frontiers disagree. If a rebuild ever merged them, this is what breaks.
    const ladder = ladderCs('dclm')
    const arms = armBundle('selfplay').corpora['dclm'] as CorpusScaling
    expect(ladder).toBeDefined()
    expect(arms).toBeDefined()
    expect(ladder?.frontier).not.toEqual(arms?.frontier)
  })

  it('corpusGroups preserves first-seen group order and covers every corpus', () => {
    const groups = corpusGroups(meta)
    expect(groups.map((g) => g.group)).toEqual([
      'Text', 'Images', 'Audio / speech', 'Math / formal',
      'Biological sequences', 'Code', 'excluded',
    ])
    expect(groups.reduce((n, g) => n + g.items.length, 0)).toBe(meta.corpora.length)
  })
})

// ---------------------------------------------------------------------------

describe('sliceCorpus', () => {
  it('slices one corpus out of the 38,740-row ladder by a single pass', () => {
    const { rows, total } = sliceCorpus(scaling.ladder, 'dclm')
    expect(total).toBe(1490)
    expect(rows.length).toBe(1490)
  })

  it('gives every ladder corpus the same 1,490 rows', () => {
    // The count the MAX_RAW_POINTS comment rests on: the stride branch is only
    // reachable because 1,490 > 1,400.
    for (const key of scaling.ladder.stringTables.corpora) {
      expect(sliceCorpus(scaling.ladder, key).total, key).toBe(1490)
    }
  })

  it('returns an empty slice for a corpus that is not in the ladder', () => {
    expect(sliceCorpus(scaling.ladder, 'no_such_corpus')).toEqual({ rows: [], total: 0 })
  })

  it('carries the five columns the hover readout prints', () => {
    const { rows } = sliceCorpus(scaling.ladder, 'dclm')
    for (const p of rows) {
      expect(p.c).toBeGreaterThan(0)
      expect(p.bpb).toBeGreaterThan(0)
      expect(p.n).toBeGreaterThan(0)
      expect(p.k).toBeGreaterThan(0)
      expect(p.round).toBeGreaterThanOrEqual(0)
    }
  })

  it('never returns a row whose compute column disagrees with its own factors', () => {
    // C = K*N*pool*CTX*(round+1) with the self-play pool of 1536.
    const { rows } = sliceCorpus(scaling.ladder, 'dclm')
    for (const p of rows) {
      expect(p.c).toBeCloseTo(p.k * p.n * 1536 * CTX * (p.round + 1), 0)
    }
  })
})

// ---------------------------------------------------------------------------

describe('subsample', () => {
  it('leaves a corpus under the cap untouched', () => {
    const items = [1, 2, 3, 4, 5]
    const { kept, total } = subsample(items, MAX_RAW_POINTS)
    expect(kept).toEqual(items)
    expect(total).toBe(5)
    expect(subsampleStride(5)).toBe(1)
  })

  it('keeps at most MAX_RAW_POINTS of the real 1,490-row corpus', () => {
    const { rows } = sliceCorpus(scaling.ladder, 'dclm')
    const { kept, total } = subsample(rows, MAX_RAW_POINTS)
    expect(total).toBe(1490)
    expect(kept.length).toBeLessThanOrEqual(MAX_RAW_POINTS)
    // The stride branch is live, not theoretical: 1,490 > 1,400.
    expect(kept.length).toBeLessThan(rows.length)
    expect(subsampleStride(1490)).toBe(2)
  })

  it('preserves the first point, so the axis still starts where scoring starts', () => {
    const { rows } = sliceCorpus(scaling.ladder, 'dclm')
    const { kept } = subsample(rows, MAX_RAW_POINTS)
    expect(kept[0]).toEqual(rows[0])
  })

  it('drops the LAST point, because stride 2 skips the odd final index', () => {
    // 1,490 rows at stride 2 keeps indices 0,2,...,1488 -- index 1489, the
    // highest-compute point, is skipped even though 1,490 divides by 2,
    // because the kept range is [0, n-1) not [0, n]. This costs exactly one
    // circle out of 745 drawn and nothing else: the frontier is unaffected
    // and the axis is set by the raw cloud. Pinned so a change to the
    // subsample cannot silently alter the drawn range.
    const { rows } = sliceCorpus(scaling.ladder, 'dclm')
    const { kept } = subsample(rows, MAX_RAW_POINTS)
    expect(rows.length % subsampleStride(rows.length)).toBe(0)
    expect(kept.length).toBe(rows.length / subsampleStride(rows.length))
    expect(kept[kept.length - 1]).toEqual(rows[rows.length - 2])
    expect(kept[kept.length - 1]).not.toEqual(rows[rows.length - 1])
  })

  it('loses at most one point off the top of the x axis', () => {
    // The dropped circle is the highest-compute raw point, and it is 3.2%
    // above the last kept one on a log axis. Negligible, and asserted so it
    // stays negligible rather than becoming a real truncation.
    const cs = ladderCs('dclm')
    const { rows } = sliceCorpus(scaling.ladder, 'dclm')
    const { kept } = subsample(rows, MAX_RAW_POINTS)
    const top = Math.max(...rows.map((p) => p.c))
    const keptTop = Math.max(...kept.map((p) => p.c))
    expect(top / keptTop).toBeLessThan(1.05)
    // The frontier is untouched by the subsample: it is a separate series.
    expect(Math.max(...cs.frontier.map((p) => p[0])))
      .toBe(cs.maxCompute)
    expect(cs.maxCompute).toBeLessThan(top)
  })

  it('is deterministic: the same corpus always draws the same circles', () => {
    const { rows } = sliceCorpus(scaling.ladder, 'dclm')
    expect(subsample(rows, MAX_RAW_POINTS).kept).toEqual(
      subsample(rows, MAX_RAW_POINTS).kept,
    )
  })

  it('every kept point is a real point from the corpus, in compute order', () => {
    const { rows } = sliceCorpus(scaling.ladder, 'dclm')
    const { kept } = subsample(rows, MAX_RAW_POINTS)
    const cs = kept.map((p) => p.c)
    expect([...cs].sort((a, b) => a - b)).toEqual(cs)
    for (const p of kept) expect(rows).toContain(p)
  })
})

// ---------------------------------------------------------------------------

describe('the floor conditional', () => {
  it('flags the six corpora whose best measured loss is BELOW the fitted floor', () => {
    // For these the page must NOT claim "the ladder does not approach it" --
    // the measurement is already past the floor the fit predicts.
    const below = Object.entries(scaling.ladder.corpora)
      .filter(([, cs]) => cs.fit !== null && cs.bestLoss < cs.fit.floor)
      .map(([k]) => k).sort()
    expect(below).toEqual([
      'aitdcc_d_glibc_rand', 'dna', 'esc50_pcm8_11khz', 'kolmogorov_dna',
      'speech_commands_pcm8', 'speech_commands_pcm8_8khz',
    ])
  })

  it('flags the corpora whose floor is pinned at exactly zero', () => {
    // E = 0.000 is a fit that failed to find a floor, not a measured
    // irreducible loss, so it must not be quoted as one.
    const zero = Object.entries(scaling.ladder.corpora)
      .filter(([, cs]) => cs.fit !== null && cs.fit.floor === 0)
      .map(([k]) => k).sort()
    expect(zero).toEqual([
      'aitdcc_b_c_source', 'arithmetic', 'kolmogorov_text',
      'llm_compression_arxiv_math', 'llm_compression_python', 'metamath',
    ])
  })

  it('never reports "the ladder does not approach it" for any of them', () => {
    const broken = brokenDownFloors(scaling)
    expect(broken.length).toBe(12)
    for (const key of broken) {
      expect(floorReading(ladderCs(key)), key)
        .toBe('broken-down')
    }
  })

  it('reads a genuine floor above the best loss as an extrapolation', () => {
    const dclm = ladderCs('dclm')
    expect(ladderFit(dclm)).not.toBeNull()
    expect(dclm.bestLoss).toBeGreaterThan(dclm.fit?.floor ?? 0)
    expect(floorReading(dclm)).toBe('extrapolation')
  })

  it('most corpora DO have an intact floor, so the caveat is not universal', () => {
    const intact = Object.entries(scaling.ladder.corpora)
      .filter(([, cs]) => floorReading(cs) === 'extrapolation').length
    expect(intact).toBe(14)
  })

  it('returns null rather than an opinion when there is no fit at all', () => {
    const noFit = { frontier: [], nFrontier: 0, bestLoss: 0, maxCompute: 0, fit: null }
    expect(floorReading(noFit)).toBeNull()
  })
})

// ---------------------------------------------------------------------------

describe('fitCurve', () => {
  it('samples 49 log-spaced points (FIT_SAMPLES + 1) across the frontier', () => {
    const curve = fitCurve(ladderCs('dclm'))
    expect(curve.length).toBe(FIT_SAMPLES + 1)
    const cs = curve.map((p) => p[0])
    expect([...cs].sort((a, b) => a - b)).toEqual(cs)
  })

  it('starts and ends exactly on the frontier compute range', () => {
    const cs0 = ladderCs('dclm')
    const curve = fitCurve(cs0)
    expect(curve[0]?.[0]).toBeCloseTo(cs0.frontier[0]?.[0] ?? 0, 0)
    expect(curve[curve.length - 1]?.[0])
      .toBeCloseTo(cs0.frontier[cs0.frontier.length - 1]?.[0] ?? 0, 0)
  })

  it('every sampled point is exactly the fitted law at that compute', () => {
    const cs0 = ladderCs('dclm')
    const fit = cs0.fit
    expect(fit).not.toBeNull()
    for (const [c, l] of fitCurve(cs0)) {
      expect(l).toBeCloseTo(predictFit(fit as PowerFit, c), 12)
    }
  })

  it('decreases monotonically with compute, because alpha is positive', () => {
    const losses = fitCurve(ladderCs('dclm')).map((p) => p[1])
    for (let i = 1; i < losses.length; i += 1) {
      expect(losses[i]).toBeLessThanOrEqual(losses[i - 1] ?? Infinity)
    }
  })

  it('returns no curve rather than a NaN one for a short or zero frontier', () => {
    expect(fitCurve({
      frontier: [[1, 1]], nFrontier: 1, bestLoss: 1, maxCompute: 1,
      fit: { alpha: 0.1, amplitude: 1, floor: 0, rmse: 0, nPoints: 1 },
    })).toEqual([])
    expect(fitCurve({
      frontier: [[0, 1], [0, 2]], nFrontier: 2, bestLoss: 1, maxCompute: 0,
      fit: { alpha: 0.1, amplitude: 1, floor: 0, rmse: 0, nPoints: 2 },
    })).toEqual([])
    expect(fitCurve({
      frontier: [], nFrontier: 0, bestLoss: 0, maxCompute: 0, fit: null,
    })).toEqual([])
  })

  it('the curve converges toward the fitted floor, and never below it', () => {
    const cs0 = ladderCs('dclm')
    const floor = cs0.fit?.floor ?? 0
    const curve = fitCurve(cs0)
    const last = curve[curve.length - 1]?.[1] ?? 0
    expect(last).toBeGreaterThan(floor)
    expect(last).toBeLessThan(floor + (cs0.fit?.amplitude ?? 0))
  })
})

// ---------------------------------------------------------------------------

describe('per-rung compute pools', () => {
  it('recovers each rung\'s exact per-rung pool for every arm', () => {
    // The real bug this guards: one constant pool for all arms moved the
    // shared-budget comparison onto an axis the authors never used. The check
    // reads the bundle's own columns, not the builder's intent -- the same
    // thing tools/verify_data.py does.
    for (const [arm, fileKey] of [['uniform', 'prior'], ['pcfg', 'pcfg']] as const) {
      for (const [rung, n] of Object.entries(RUNG_N)) {
        const pools = poolAt(armBundle(arm), n, CTX)
        const expected = AUTHOR_POOLS[fileKey]?.[rung]
        if (expected === undefined) {
          // The fixed arms were never run at the largest rung.
          expect(pools.length, `${arm}/${rung}`).toBeLessThanOrEqual(1)
          continue
        }
        expect(pools, `${arm} at ${rung}`).toEqual([expected])
      }
    }
  })

  it('pins each rung\'s pool to a SINGLE value: no rung mixes two rates', () => {
    for (const arm of ARM_ORDER) {
      for (const [rung, n] of Object.entries(RUNG_N)) {
        expect(poolAt(armBundle(arm), n, CTX).length, `${arm}/${rung}`)
          .toBeLessThanOrEqual(1)
      }
    }
  })

  it('the three arms disagree at the smallest rung, so it is not one relabelled pool', () => {
    const n = RUNG_N['d64h1L1'] ?? 0
    expect(poolAt(armBundle('selfplay'), n, CTX)).toEqual([1536])
    expect(poolAt(armBundle('uniform'), n, CTX)).toEqual([1024])
    expect(poolAt(armBundle('pcfg'), n, CTX)).toEqual([512])
  })

  it('recovers one integer pool from every scored point', () => {
    for (const arm of ARM_ORDER) {
      const cols = armBundle(arm).columns
      for (let i = 0; i < cols.C.length; i += 1) {
        expect(recoverPool(cols, i, CTX), `${arm}[${i}]`).not.toBeNull()
      }
    }
  })

  it('self-play uses ONE constant pool, 1536, at every rung', () => {
    expect(poolsOf(armBundle('selfplay'), CTX)).toEqual([1536])
  })

  it('the uniform arm uses the authors\' per-rung pools {1024, 2048}', () => {
    expect(poolsOf(armBundle('uniform'), CTX)).toEqual([1024, 2048])
    expect([...new Set(Object.values(AUTHOR_POOLS.prior ?? {}))].sort((a, b) => a - b))
      .toEqual([1024, 2048])
  })

  it('the PCFG arm uses the authors\' per-rung pools {512, 1024, 2048}', () => {
    expect(poolsOf(armBundle('pcfg'), CTX)).toEqual([512, 1024, 2048])
    expect([...new Set(Object.values(AUTHOR_POOLS.pcfg ?? {}))].sort((a, b) => a - b))
      .toEqual([512, 1024, 2048])
  })

  it('the fixed arms are genuinely per-rung, not one rate relabelled', () => {
    expect(poolsOf(armBundle('uniform'), CTX).length).toBeGreaterThan(1)
    expect(poolsOf(armBundle('pcfg'), CTX).length).toBeGreaterThan(1)
  })

  it('the pool set over all three arms is exactly the ladder pool plus 512/1024/2048', () => {
    const all = new Set(ARM_ORDER.flatMap((a) => poolsOf(scaling.arms[a], CTX)))
    expect([...all].sort((a, b) => a - b)).toEqual([512, 1024, 1536, 2048])
  })

  it('the ladder itself uses the constant 1536 pool at every one of its 38,740 rows', () => {
    expect(poolsOf(scaling.ladder as ArmScaling, CTX)).toEqual([1536])
  })

  it('returns null, not a guess, when C does not decode to an integer pool', () => {
    const cols = armBundle('selfplay').columns
    const bad = { ...cols, C: cols.C.slice() }
    // Multiply C by a factor that is not a ratio of two integer pools, so the
    // decoded value falls between integers. This is what a compute axis built
    // from the wrong context length looks like.
    bad.C[0] = (bad.C[0] ?? 0) * 1.37
    expect(recoverPool(bad, 0, CTX)).toBeNull()
  })

  it('returns null for a row whose columns are missing, not NaN', () => {
    const cols = { ...armBundle('selfplay').columns, C: [] }
    expect(recoverPool(cols, 0, CTX)).toBeNull()
  })
})

// ---------------------------------------------------------------------------

describe('Table 2 against the ladder fits', () => {
  it('has no field where Table 2 disagrees with the ladder fit it tabulates', () => {
    // Table 2 IS the ladder's fits, tabulated. A second, independent fit here
    // would be printed as the paper's table while being something else.
    expect(exponentMismatches(scaling)).toEqual([])
  })

  it('tabulates 10 of the 26 ladder corpora', () => {
    const rows = exponentRows(scaling)
    expect(rows.length).toBe(10)
    for (const r of rows) {
      expect(scaling.ladder.corpora[r.key], r.key).toBeDefined()
      expect(r.alpha, `${r.key} has a fitted exponent`).not.toBeNull()
    }
  })

  it('reports the DNA exponent the page\'s caveat is written against', () => {
    const dna = exponentRows(scaling).find((r) => r.key === 'dna')
    expect(paperExponent(dna?.alpha ?? 0)).toBe('0.435')
    // Seven times the highest published value (0.01-0.06).
    const highest = Math.max(...(dna?.literature ?? []).map((e) => e.v))
    expect((dna?.alpha ?? 0) / highest).toBeGreaterThan(6)
  })

  it('renders every exponent at the paper\'s three-decimal precision', () => {
    for (const r of exponentRows(scaling)) {
      expect(paperExponent(r.alpha ?? 0), r.key).toMatch(/^\d\.\d{3}$/)
      for (const e of r.literature) {
        expect(paperExponent(e.v), `${r.key} literature`).toMatch(/^\d\.\d{3}$/)
      }
    }
  })

  it('merges published and Chinchilla-converted values into one sorted list', () => {
    const dclm = literatureEntries(scaling, 'dclm')
    expect(dclm).not.toBeNull()
    const vs = (dclm?.entries ?? []).map((e) => e.v)
    expect([...vs].sort((a, b) => a - b)).toEqual(vs)
    // The converted value is marked, so the page can print a dagger on it.
    expect((dclm?.entries ?? []).some((e) => e.derived)).toBe(true)
  })

  it('does not re-convert the Chinchilla values: they are already b values', () => {
    const row = scaling.exponents.find((e) => e.key === 'dclm')
    const chin = row?.literatureChinchilla?.[0] ?? 0
    // The bundle stores b = a*b/(a+b) ~ 0.099, not an (alpha, beta) pair.
    expect(chin).toBeGreaterThan(0)
    expect(chin).toBeLessThan(0.5)
  })

  it('distinguishes "none found" from a row that merely has no values', () => {
    const rows = exponentRows(scaling)
    const dashes = rows.filter((r) => r.literatureDash)
    expect(dashes.length).toBeGreaterThan(0)
    for (const r of dashes) {
      // The page renders the pill only when there are no values at all.
      expect(r.literature.length, r.key).toBe(0)
    }
  })

  it('returns null for a corpus that is not in Table 2', () => {
    expect(literatureEntries(scaling, 'dclm_ranked')).toBeNull()
  })

  it('the text exponent is 2.6x its LOWEST published value, as the caveat says', () => {
    // The page's caveat calls this "its nearest published value". Against the
    // nearest (0.099) the ratio is 1.24x; against the lowest (0.048) it is
    // 2.57x, which is the number printed. Pinned to the LOWEST so the printed
    // figure cannot drift.
    const dclm = exponentRows(scaling).find((r) => r.key === 'dclm')
    const lit = (dclm?.literature ?? []).map((e) => e.v)
    expect(lit.length).toBe(2)
    expect((dclm?.alpha ?? 0) / Math.min(...lit)).toBeCloseTo(2.566, 2)
    expect((dclm?.alpha ?? 0) / Math.max(...lit)).toBeCloseTo(1.244, 2)
  })

  it('exactly two image/audio rows exceed 2.1x their lowest literature value', () => {
    // dclm (Text), cifar10_rgb_planar (Images) and audio_8bit (Audio) do; DNA is
    // counted separately by the caveat. The page says "two image/audio rows",
    // which holds once Text and DNA are excluded from that phrasing.
    const over = scaling.exponents
      .filter((e) => e.group !== 'Biological sequences' && e.group !== 'Text')
      .flatMap((e) => {
        const lit = [...e.literatureValues, ...e.literatureChinchilla]
        return lit.length && (e.alpha ?? 0) / Math.min(...lit) > 2.1 ? [e.key] : []
      })
    expect(over.sort()).toEqual(['audio_8bit', 'cifar10_rgb_planar'])
  })

  it('four Table 2 rows have no literature value to compare against', () => {
    const rows = exponentRows(scaling)
    expect(rows.filter((r) => r.literature.length === 0).length).toBe(4)
    // Of those four, exactly one is tabulated "none found"; the other three
    // are rendered as a bare em dash. Both are honest, but they are different
    // statements, so the count of each is pinned.
    expect(rows.filter((r) => r.literature.length === 0 && r.literatureDash).length).toBe(1)
    expect(rows.filter((r) => r.literature.length === 0 && !r.literatureDash).length).toBe(3)
  })
})

// ---------------------------------------------------------------------------

describe('compareArms: who is compared', () => {
  const verdicts = compareArms(scaling, meta)

  it('scores exactly 20 of the 26 ladder corpora', () => {
    expect(verdicts.length).toBe(20)
    expect(verdicts.length + EXCLUDED.length).toBe(meta.corpora.length)
  })

  it('compares no corpus the bundle marks excluded', () => {
    for (const v of verdicts) {
      expect(isExcluded(v.group), v.key).toBe(false)
      expect(EXCLUDED, v.key).not.toContain(v.key)
    }
  })

  it('gives every compared corpus a verdict from all three arms', () => {
    // "scored by all three arms" -- the phrase the page's headline uses.
    for (const v of verdicts) {
      for (const arm of ARM_ORDER) {
        expect(v.best[arm], `${v.key}/${arm}`).not.toBeNull()
      }
      expect(v.winner).not.toBeNull()
      expect(v.margin).not.toBeNull()
    }
  })

  it('every compared corpus has a real runner-up, so no margin is null in practice', () => {
    // budget = min over the arms of each frontier's last compute, so an arm
    // with a frontier always has at least one point at or below the budget.
    // The null-margin path is therefore defensive: it can only fire for a
    // corpus whose frontiers are non-empty but all unreachable, which no
    // bundle produces. Pinned so a future rule change to the budget shows up
    // here rather than as an em dash in the margin column.
    for (const v of verdicts) expect(v.margin, v.key).not.toBeNull()
  })

  it('a corpus with only one armed frontier is dropped, not scored on one arm', () => {
    const oneArm = {
      ...scaling,
      arms: Object.fromEntries(ARM_ORDER.map((a) => [a, {
        ...scaling.arms[a],
        corpora: a === 'selfplay'
          ? scaling.arms[a].corpora
          : Object.fromEntries(Object.entries(scaling.arms[a].corpora)
            .filter(([k]) => k !== 'dclm')),
      }])) as Scaling['arms'],
    } as Scaling
    expect(compareArms(oneArm, meta).some((v) => v.key === 'dclm')).toBe(false)
    expect(compareArms(oneArm, meta).length).toBe(19)
  })

  it('marks glibc rand excluded, and it is the corpus that breaks the prior claim', () => {
    // The load-bearing exclusion. On glibc rand the fixed uniform prior reaches
    // a LOWER loss than self-play, so "the prior never wins" depends on this
    // corpus being out of the comparison -- which the page discloses.
    const g = 'aitdcc_d_glibc_rand'
    expect(isExcluded(meta.corpora.find((c) => c.key === g)?.group ?? '')).toBe(true)
    expect(verdicts.some((v) => v.key === g)).toBe(false)

    const budget = Math.min(...ARM_ORDER.map(
      (a) => scaling.arms[a].corpora[g]?.frontier.slice(-1)[0]?.[0] ?? Infinity,
    ))
    const at = (arm: string) => {
      const pts = scaling.arms[arm as 'selfplay']?.corpora[g]?.frontier ?? []
      const reach = pts.filter((p) => p[0] <= budget * (1 + 1e-7))
      return reach.length ? Math.min(...reach.map((p) => p[1])) : null
    }
    const sp = at('selfplay') ?? 0
    const un = at('uniform') ?? 0
    expect(un, 'uniform prior beats self-play on glibc rand').toBeLessThan(sp)
    // The page quotes these two numbers, rounded to three decimals.
    expect(un.toFixed(3)).toBe('8.026')
    expect(sp.toFixed(3)).toBe('8.052')
    // ...and it concedes the gap is inside the tie margin, so the exclusion
    // would not have flipped the verdict either way.
    expect(sp - un).toBeLessThan(TIE_MARGIN)
  })

  it('the fixed uniform prior wins NONE of the 20 compared corpora', () => {
    // The page's own "load-bearing control". If this ever changes, the
    // exclusion above stops being a footnote and becomes the headline.
    const { totals } = armTotals(verdicts)
    expect(totals.uniform).toBe(0)
    expect(verdicts.filter((v) => v.winner === 'uniform')).toEqual([])
  })

  it('quotes 14 self-play wins and 6 PCFG wins', () => {
    const { totals } = armTotals(verdicts)
    expect(totals).toEqual({ selfplay: 14, uniform: 0, pcfg: 6 })
    expect(totals.selfplay + totals.pcfg + totals.uniform).toBe(verdicts.length)
  })
})

// ---------------------------------------------------------------------------

describe('compareArms: the shared-budget rule', () => {
  const verdicts = compareArms(scaling, meta)

  it('scores every arm at the SMALLEST frontier-end compute, not its own', () => {
    // The confound this rule removes: self-play runs to ~10x the compute of
    // the fixed-prior arm, so its own best loss always looks better.
    for (const v of verdicts) {
      for (const arm of ARM_ORDER) {
        const pts = armBundle(arm).corpora[v.key]?.frontier ?? []
        const own = pts[pts.length - 1]?.[0] ?? Infinity
        expect(v.budget, `${v.key}`).toBeLessThanOrEqual(own)
        expect(v.budget).toBeGreaterThan(0)
      }
    }
  })

  it('never credits an arm with a loss measured above the shared budget', () => {
    for (const v of verdicts) {
      for (const arm of ARM_ORDER) {
        const pts = armBundle(arm).corpora[v.key]?.frontier ?? []
        const allowed = pts.filter((p) => p[0] <= v.budget * (1 + 1e-7)).map((p) => p[1])
        expect(allowed.length, `${v.key}/${arm}`).toBeGreaterThan(0)
        expect(v.best[arm]).toBe(Math.min(...allowed))
      }
    }
  })

  it('self-play really does reach far more compute than the fixed arms', () => {
    // Confirms the rule is doing work: without it the comparison is unfair.
    const span = (arm: 'selfplay' | 'uniform' | 'pcfg') => Object.values(
      armBundle(arm).corpora,
    ).reduce((m, cs) => Math.max(m, cs.maxCompute), 0)
    expect(span('selfplay')).toBeGreaterThan(2 * span('uniform'))
    expect(span('selfplay')).toBeGreaterThan(2 * span('pcfg'))
  })

  it('the winner is the arm with the lowest loss at the shared budget', () => {
    for (const v of verdicts) {
      const ranked = ARM_ORDER
        .filter((a) => v.best[a] !== null)
        .sort((a, b) => (v.best[a] ?? 0) - (v.best[b] ?? 0))
      expect(v.winner).toBe(ranked[0])
      const second = ranked[1]
      expect(second, `${v.key} has a runner-up`).toBeDefined()
      expect(v.margin).toBeCloseTo(
        (v.best[second as ArmKey] ?? 0) - (v.best[ranked[0] as ArmKey] ?? 0), 12,
      )
    }
  })

  it('loses are non-negative: the winner is never also the runner-up', () => {
    for (const v of verdicts) {
      expect(v.margin ?? 0).toBeGreaterThanOrEqual(0)
    }
  })
})

// ---------------------------------------------------------------------------

describe('ties', () => {
  const verdicts = compareArms(scaling, meta)

  it('flags metamath as the one tie, and it is inside TIE_MARGIN', () => {
    const ties = verdicts.filter((v) => isTie(v.margin))
    expect(ties.map((v) => v.key)).toEqual(['metamath'])
    const mm = ties[0]
    expect(mm?.winner).toBe('selfplay')
    expect(mm?.margin ?? 0).toBeCloseTo(0.0124, 4)
    expect(isTie(mm?.margin ?? null)).toBe(true)
  })

  it('still counts the tie for exactly one arm in the totals', () => {
    // The page discloses this: the row is flagged "(tie)" but the headline
    // counts do not drop it, so totals can exceed the number of clear wins.
    const mm = verdicts.find((v) => v.key === 'metamath')
    expect(mm?.winner).not.toBeNull()
    const { totals } = armTotals(verdicts)
    const clearSelfplay = verdicts.filter(
      (v) => v.winner === 'selfplay' && !isTie(v.margin),
    ).length
    expect(totals.selfplay).toBe(clearSelfplay + 1)
  })

  it('the tie arithmetic matches the prose: totals exceed clear wins by one', () => {
    const { totals, ties } = armTotals(verdicts)
    expect(ties).toBe(1)
    const clearWins = verdicts.filter((v) => v.winner !== null && !isTie(v.margin)).length
    expect(totals.selfplay + totals.uniform + totals.pcfg)
      .toBe(clearWins + ties)
  })

  it('treats a margin exactly at the margin as a win, not a tie', () => {
    // The comparison is strict `<`, matching the page's "anything under
    // 0.05 bits/byte is reported as a tie".
    expect(isTie(0.05)).toBe(false)
    expect(isTie(0.0499999)).toBe(true)
    expect(isTie(null)).toBe(false)
  })
})

// ---------------------------------------------------------------------------

describe('tallyGroups', () => {
  const verdicts = compareArms(scaling, meta)
  const tallies = tallyGroups(verdicts)

  it('covers every compared corpus exactly once', () => {
    expect(tallies.reduce((n, t) => n + t.total, 0)).toBe(verdicts.length)
    expect(tallies.every((t) => t.total === verdicts.filter((v) => v.group === t.group).length))
      .toBe(true)
  })

  it('reproduces the per-group counts the page prints', () => {
    const by = Object.fromEntries(tallies.map((t) => [t.group, t]))
    expect(by['Text']?.counts).toEqual({ selfplay: 1, uniform: 0, pcfg: 3 })
    expect(by['Images']?.counts).toEqual({ selfplay: 2, uniform: 0, pcfg: 0 })
    expect(by['Audio / speech']?.counts).toEqual({ selfplay: 9, uniform: 0, pcfg: 0 })
    expect(by['Math / formal']?.counts).toEqual({ selfplay: 1, uniform: 0, pcfg: 1 })
    expect(by['Code']?.counts).toEqual({ selfplay: 0, uniform: 0, pcfg: 2 })
    expect(by['Biological sequences']?.counts)
      .toEqual({ selfplay: 1, uniform: 0, pcfg: 0 })
  })

  it('counts the Math / formal tie without dropping a corpus', () => {
    const math = tallies.find((t) => t.group === 'Math / formal') as GroupTally
    expect(math.total).toBe(2)
    expect(math.ties).toBe(1)
    // A tie is still counted for one arm, so the per-arm counts still add up
    // to the number of corpora in the group.
    expect(math.counts.selfplay + math.counts.pcfg + math.counts.uniform)
      .toBe(math.total)
  })

  it('never emits an excluded group', () => {
    for (const t of tallies) expect(isExcluded(t.group)).toBe(false)
    expect(tallies.map((t) => t.group)).not.toContain('excluded')
  })

  it('preserves first-seen order so the rendered cards do not shuffle', () => {
    expect(tallies.map((t) => t.group)).toEqual(
      verdicts.map((v) => v.group).filter((g, i, a) => a.indexOf(g) === i),
    )
  })
})

// ---------------------------------------------------------------------------

describe('groupVerdict', () => {
  const verdicts = compareArms(scaling, meta)

  it('writes each group line from the tally, matching the page verbatim', () => {
    const lines = tallyGroups(verdicts).map((t) => groupVerdict(t, meta))
    expect(lines).toEqual([
      'PCFG reaches the lowest loss on 3 of 4 corpora; Self-play on 1.',
      'Self-play reaches the lowest loss on 2 of 2 corpora.',
      'Self-play reaches the lowest loss on 9 of 9 corpora.',
      'Self-play reaches the lowest loss on 1 of 2 corpora; PCFG on 1. '
      + 'One of those is within 0.05 bits/byte — treat it as a tie.',
      'Self-play reaches the lowest loss on 1 of 1 corpus.',
      'PCFG reaches the lowest loss on 2 of 2 corpora.',
    ])
  })

  it('uses the singular for a one-corpus group', () => {
    const bio = tallyGroups(verdicts).find((t) => t.group === 'Biological sequences')
    expect(groupVerdict(bio as GroupTally, meta)).toContain('1 of 1 corpus.')
    expect(groupVerdict(bio as GroupTally, meta)).not.toContain('corpora')
  })

  it('omits the tie clause entirely when there are no ties', () => {
    const audio = tallyGroups(verdicts).find((t) => t.group === 'Audio / speech')
    expect(groupVerdict(audio as GroupTally, meta)).not.toContain('tie')
  })

  it('pluralises the tie clause when more than one is a tie', () => {
    const t = {
      group: 'X', total: 3, ties: 2, counts: { selfplay: 1, uniform: 0, pcfg: 2 },
    }
    expect(groupVerdict(t, meta))
      .toContain('2 of those are within 0.05 bits/byte — treat those as a tie.')
  })

  it('says so plainly when no arm reached the budget at all', () => {
    const empty = {
      group: 'X', total: 2, ties: 0, counts: { selfplay: 0, uniform: 0, pcfg: 0 },
    }
    expect(groupVerdict(empty, meta)).toBe('No arm reached the shared budget.')
  })

  it('names arms by their short label, never by an internal key', () => {
    for (const line of tallyGroups(verdicts).map((t) => groupVerdict(t, meta))) {
      expect(line).not.toMatch(/selfplay|uniform|pcfg/)
    }
  })
})

// ---------------------------------------------------------------------------

describe('the numbers the page displays', () => {
  it('formats the ladder compute axis the way the hover readout does', () => {
    // The ladder spans 4.1e11 to 1.3e18, so the axis exercises four of the
    // five unit bands, and drops to zero decimals above 100 of a unit.
    expect(formatCompute(413524819968)).toBe('414G')
    expect(formatCompute(4.13524819968e11)).toBe('414G')
    expect(formatCompute(1.2908631560835564e+18)).toBe('1.3E')
    expect(formatCompute(1e15)).toBe('1.0P')
    expect(formatCompute(7.501930813697556e+18)).toBe('7.5E')
    expect(formatCompute(0)).toBe('—')
    expect(formatCompute(-1)).toBe('—')
    expect(formatCompute(Number.NaN)).toBe('—')
  })

  it('shows no decimal above 100 of a unit, and one decimal below it', () => {
    expect(formatCompute(150e9)).toBe('150G')
    expect(formatCompute(15e9)).toBe('15.0G')
  })

  it('falls back to a bare digit string below a million', () => {
    // formatCompute's smallest unit band is 1e6 (M); below that it prints the
    // number with no suffix at all. No ladder point lands here, but the branch
    // exists and must not invent a unit.
    expect(formatCompute(1536 * 4096)).toBe('6.3M')
    expect(formatCompute(1e6)).toBe('1.0M')
  })

  it('keeps the loss domain positive, and does not clamp at the 8-bit ceiling', () => {
    // glibc rand's frontier exceeds 8 bits/byte (its best loss is 8.006 and
    // its worst is 8.464): the arm is worse than a uniform 256-symbol
    // predictor, so a domain clamped to 8 would silently crop real data.
    for (const [key, cs] of Object.entries(scaling.ladder.corpora)) {
      const [lo, hi] = lossDomain(cs.frontier.map((p) => p[1]))
      expect(lo, key).toBeGreaterThanOrEqual(0)
      expect(hi, key).toBeGreaterThan(lo)
      const worst = Math.max(...cs.frontier.map((p) => p[1]))
      expect(hi, key).toBeGreaterThanOrEqual(worst)
    }
    expect(Math.max(...ladderCs('aitdcc_d_glibc_rand')
      .frontier.map((p) => p[1]))).toBeGreaterThan(8)
  })

  it('renders the amplitude prefactor at a precision that suits its magnitude', () => {
    expect(amplitude(277.298364652189)).toBe('277.30')
    expect(amplitude(1234.5)).toBe('1.235e+3')
    expect(amplitude(2.5)).toBe('2.500')
    expect(amplitude(Number.NaN)).toBe('—')
  })

  it('prints exponents at three decimals, rounding exact ties toward zero', () => {
    // docs/findings.md: the authors printed 0.260 for a fit of 0.2605, so an
    // exact .0005 must round DOWN, not to the nearest away from zero.
    expect(paperExponent(0.2605)).toBe('0.260')
    expect(paperExponent(0.1235)).toBe('0.123')
    expect(paperExponent(0.0005)).toBe('0.000')
    // Just past the tie still rounds up, so the rule is a tie rule and not a
    // blanket truncation.
    expect(paperExponent(0.1236)).toBe('0.124')
    expect(paperExponent(0.12317)).toBe('0.123')
    expect(paperExponent(0.43451)).toBe('0.435')
  })

  it('ordinal suffixes the drawn-point note correctly', () => {
    expect(ordinal(1)).toBe('st')
    expect(ordinal(2)).toBe('nd')
    expect(ordinal(3)).toBe('rd')
    expect(ordinal(4)).toBe('th')
    expect(ordinal(11)).toBe('th')
    expect(ordinal(21)).toBe('st')
  })
})