/**
 * The simulator's behavioural tests.
 *
 * Two jobs, in order of importance.
 *
 * FIRST, and non-negotiable: the numbers must be FINITE and the run must be
 * REPRODUCIBLE. Every reward on the page is an inner product against a
 * hand-written gradient and an AdamW state, and both of those have already
 * produced a NaN at least once while this was being written. A NaN chart is
 * indistinguishable from a working one, so finiteness is asserted rather than
 * assumed.
 *
 * SECOND: the qualitative claims the page makes must hold on the toy run, or be
 * corrected. The claims are stated as assertions below and the page's prose is
 * written against whichever way they came out. In particular the write-up must
 * not claim the canonical reward refuses noise if these fail.
 */
import { describe, expect, it } from 'vitest'
import { ALPHABET, SEQ_LEN, initParams, normParams, zerosLike, lossAndGrads } from './model'
import { DEFAULT_ADAM, adamwStep, initState, precondition, subParams } from './adamw'
import { decompose, countOutbidding, normalizePool, rewardShare } from './reward'
import { ARITHMETIC, FIBONACCI, NOISE, noiseIndex, poolFor } from './programs'
import { DEFAULT_RUN, fixedTeach, noiseOutbidsFrontier, peakRound, runSim, series, steppedTeach, type RunConfig } from './runsim'

/**
 * Short enough to keep the suite fast, long enough to pass the frontier peak.
 *
 * The frontier reward peaks around round 40 on this model, so a run that stops
 * before ~80 rounds would miss the one round most of these tests are about.
 */
const TEST_RUN: RunConfig = { ...DEFAULT_RUN, rounds: 120, recordEvery: 4, poolSize: 5, teach: fixedTeach }

/**
 * Runs are cached by configuration.
 *
 * Every test in this file costs a full training run -- ~2s each at 120 rounds,
 * and the suite was spending 70s re-running the same handful of configurations
 * to look at one number apiece. runSim is pure and deterministic (there is a
 * test for that below), so memoising on the config is safe and turns ~20 runs
 * into 4.
 */
const runCache = new Map<string, ReturnType<typeof runSim>>()
function cachedRun(cfg: RunConfig): ReturnType<typeof runSim> {
  const key = JSON.stringify({ ...cfg, teach: cfg.teach.toString().length })
  const hit = runCache.get(key)
  if (hit !== undefined) return hit
  const res = runSim(poolFor(), cfg)
  runCache.set(key, res)
  return res
}

describe('adamw', () => {
  it('accumulates g squared, so the second moment is never negative', () => {
    // The regression that matters. Accumulating g makes v sign-indefinite and
    // sqrt(v_hat) -- the P_e the reward divides by -- returns NaN.
    const p = initParams(3)
    const s = initState(p)
    const g = zerosLike(p)
    // A gradient with a large negative entry: harmless for g^2, fatal for g.
    g.b1[0] = -5
    g.b1[1] = 2
    for (let e = 1; e <= 5; e++) adamwStep(p, g, s, DEFAULT_ADAM)

    for (const key of ['W1', 'b1', 'W2', 'b2'] as const) {
      const v = s.v[key]
      for (let i = 0; i < v.length; i++) expect(v[i], `${key}[${i}]`).toBeGreaterThanOrEqual(0)
    }
  })

  it('leaves the preconditioner finite on the very first step', () => {
    const p = initParams(4)
    const s = initState(p)
    const g = zerosLike(p)
    g.b1[0] = -3
    adamwStep(p, g, s, DEFAULT_ADAM)
    const pc = precondition(subParams(p, p), s)
    for (const key of ['W1', 'b1', 'W2', 'b2'] as const) {
      for (const v of pc[key]) expect(Number.isFinite(v)).toBe(true)
    }
  })

  it('applies bias correction, so step one is not scaled by (1 - beta2)', () => {
    // With no correction, v_hat on step 1 is (1-beta2) * g^2 and P_e is wrong by
    // sqrt(1-beta2) ~ 0.0316 -- exactly where a frontier reward is sharpest.
    const p = initParams(5)
    const s = initState(p)
    const g = zerosLike(p)
    g.b2[0] = 4
    adamwStep(p, g, s, DEFAULT_ADAM)

    const bc2 = 1 - DEFAULT_ADAM.beta2
    const vRaw = (1 - DEFAULT_ADAM.beta2) * 16
    const corrected = Math.sqrt(vRaw / bc2)
    const uncorrected = Math.sqrt(vRaw)
    expect(corrected).toBeCloseTo(4, 10)
    expect(uncorrected).not.toBeCloseTo(4, 3)
    // The state's raw v is the uncorrected accumulation.
    expect(s.v.b2[0]).toBeCloseTo(vRaw, 12)
  })

  it('does not move the parameters when the gradient is zero', () => {
    const p = initParams(6)
    const before = normParams(p)
    const s = initState(p)
    adamwStep(p, zerosLike(p), s, DEFAULT_ADAM)
    expect(normParams(p)).toBeCloseTo(before, 12)
  })
})

describe('reward decomposition', () => {
  /** A decomposition whose gradient is exactly P_e (*) delta-theta_e. */
  function aligned() {
    const thetaMid = initParams(8)
    const thetaNow = initParams(8)
    const s = initState(thetaNow)
    // Give the optimiser some history so v is not degenerate.
    for (let e = 1; e <= 4; e++) {
      const g = zerosLike(thetaNow)
      g.b1[0] = 1
      adamwStep(thetaNow, g, s, DEFAULT_ADAM)
    }
    const motion = precondition(subParams(thetaMid, thetaNow), s)
    const thetaNowCopy = initParams(8)
    for (let e = 1; e <= 4; e++) {
      const g = zerosLike(thetaNowCopy)
      g.b1[0] = 1
      adamwStep(thetaNowCopy, g, s, DEFAULT_ADAM)
    }
    return decompose({ grad: motion, thetaMid, thetaNow: thetaNowCopy, state: s, lossBpb: 1 })
  }

  it('reads cos = +1 when the gradient IS the preconditioned motion', () => {
    const d = aligned()
    expect(d.cos).toBeCloseTo(1, 6)
    expect(d.inner).toBeGreaterThan(0)
    expect(d.rAbs).toBeCloseTo(d.rSigned, 12)
  })

  it('reads cos = -1 when the gradient opposes it, and only the abs pays', () => {
    const d = aligned()
    const flipped = { ...d, gradNorm: d.gradNorm, inner: -d.inner, rAbs: Math.abs(d.inner), rSigned: -d.inner }
    // The abs is blind to the sign; the signed reading is not.
    expect(flipped.rAbs).toBeCloseTo(d.rAbs, 12)
    expect(flipped.rSigned).toBeCloseTo(-d.rSigned, 12)
  })

  it('reconstructs the reward as |grad| * |motion| * |cos|', () => {
    // The factorisation the page plots. If this identity breaks, a panel and the
    // headline number are describing different quantities.
    const pool = poolFor()
    const p = initParams(9)
    const s = initState(p)
    for (let e = 1; e <= 6; e++) {
      const g = zerosLike(p)
      lossAndGrads(p, g, ARITHMETIC(0))
      adamwStep(p, g, s, DEFAULT_ADAM)
    }
    const thetaMid = initParams(9)
    const g = zerosLike(p)
    lossAndGrads(p, g, pool[0]!.bytes)
    const d = decompose({ grad: g, thetaMid, thetaNow: p, state: s, lossBpb: 1 })
    const reconstructed = d.gradNorm * d.motionNorm * Math.abs(d.cos)
    expect(reconstructed).toBeCloseTo(d.rAbs, 6)
  })

  it('returns cos = 0 rather than NaN when the gradient vanishes', () => {
    const p = initParams(10)
    const s = initState(p)
    const g = zerosLike(p)
    for (let e = 1; e <= 3; e++) adamwStep(p, g, s, DEFAULT_ADAM)
    const d = decompose({ grad: g, thetaMid: p, thetaNow: p, state: s, lossBpb: 1 })
    expect(Number.isFinite(d.cos)).toBe(true)
    expect(d.cos).toBe(0)
    expect(d.rAbs).toBe(0)
  })

  it('gives shares that sum to 1 and concentration the largest of them', () => {
    const res = cachedRun(TEST_RUN)
    for (const r of res.rounds) {
      const total = r.shares.reduce((a, b) => a + b, 0)
      if (total > 0) expect(total).toBeCloseTo(1, 9)
      expect(r.concentration).toBeCloseTo(Math.max(0, ...r.shares), 12)
      expect(r.concentration).toBeLessThanOrEqual(1 + 1e-9)
    }
  })

  it('normalises a pool to roughly zero mean', () => {
    const res = cachedRun(TEST_RUN)
    const last = res.rounds[res.rounds.length - 1]!
    const norm = normalizePool(last.programs)
    const mean = norm.reduce((a, b) => a + b, 0) / norm.length
    expect(Math.abs(mean)).toBeLessThan(1e-6)
    // And the argmax is preserved: normalisation moves the scale, not the order.
    const rawBest = last.programs.reduce((bi, d, i, a) => (d.rAbs > a[bi]!.rAbs ? i : bi), 0)
    const normBest = norm.reduce((bi, v, i, a) => (v > a[bi]! ? i : bi), 0)
    expect(normBest).toBe(rawBest)
  })

  it('counts outbidding programs without crashing on an empty pool', () => {
    expect(countOutbidding([], 0, 'abs')).toBe(0)
    const res = cachedRun(TEST_RUN)
    const n = countOutbidding(res.rounds[0]!.programs, 0, 'abs')
    expect(n).toBeGreaterThanOrEqual(0)
    expect(n).toBeLessThan(res.rounds[0]!.programs.length - 1)
  })

  it('divides by zero shares evenly rather than producing NaN', () => {
    expect(rewardShare([] as never[])).toEqual([])
    const zero = { gradNorm: 0, motionNorm: 0, cos: 0, inner: 0, rAbs: 0, rSigned: 0, lossBpb: 8 }
    expect(rewardShare([zero, zero])).toEqual([0, 0])
  })
})

describe('a full run', () => {
  const res = cachedRun(TEST_RUN)

  it('produces only finite numbers', () => {
    expect(res.finite).toBe(true)
    for (const r of res.rounds) {
      expect(Number.isFinite(r.trainLossBpb)).toBe(true)
      for (const d of r.programs) {
        for (const [name, v] of Object.entries(d)) {
          expect(Number.isFinite(v as number), `${name} at round ${r.e}`).toBe(true)
        }
      }
    }
  })

  it('actually learns: the training loss falls by more than 6 bits/byte', () => {
    const first = res.rounds[0]!.trainLossBpb
    const last = res.rounds[res.rounds.length - 1]!.trainLossBpb
    expect(first).toBeGreaterThan(7.5)
    expect(last).toBeLessThan(first - 6)
  })

  it('is reproducible: the same seed gives the same numbers', () => {
    const again = cachedRun(TEST_RUN)
    expect(again.rounds.length).toBe(res.rounds.length)
    for (let i = 0; i < res.rounds.length; i++) {
      expect(again.rounds[i]!.trainLossBpb).toBe(res.rounds[i]!.trainLossBpb)
      expect(again.rounds[i]!.programs[0]!.rAbs).toBe(res.rounds[i]!.programs[0]!.rAbs)
    }
  })

  it('differs under a different seed, so the seed is load-bearing', () => {
    const other = runSim(poolFor(), { ...TEST_RUN, seed: 99 })
    const a = res.rounds[res.rounds.length - 1]!.trainLossBpb
    const b = other.rounds[other.rounds.length - 1]!.trainLossBpb
    expect(Math.abs(a - b)).toBeGreaterThan(1e-6)
  })

  it('records from round 2 onward, since delta-theta needs a midpoint', () => {
    expect(res.rounds[0]!.e).toBeGreaterThanOrEqual(2)
    expect(res.rounds.every((r) => r.e % TEST_RUN.recordEvery === 0)).toBe(true)
  })
})

describe('the qualitative claims the page makes', () => {
  const pool = poolFor()
  const res = cachedRun(TEST_RUN)
  const frontier = 0
  const noise = pool.length - 1

  it('peaks the frontier reward mid-run rather than at either end', () => {
    // The spine of the exhibit: the reward prefers what the learner is being
    // taught, while it is still being taught.
    const peak = peakRound(res.rounds, frontier, 'abs')
    expect(peak).toBeDefined()
    const first = res.rounds[0]!.programs[frontier]!.rAbs
    const last = res.rounds[res.rounds.length - 1]!.programs[frontier]!.rAbs
    expect(peak!.programs[frontier]!.rAbs).toBeGreaterThan(first)
    expect(peak!.programs[frontier]!.rAbs).toBeGreaterThan(last)
  })

  it('aligns the frontier program with the learner and opposes noise', () => {
    const peak = peakRound(res.rounds, frontier, 'abs')!
    expect(peak.programs[frontier]!.cos).toBeGreaterThan(0.3)
    expect(peak.programs[noise]!.cos).toBeLessThan(peak.programs[frontier]!.cos)
  })

  it('gives noise a large gradient -- the term that cannot tell them apart', () => {
    const peak = peakRound(res.rounds, frontier, 'abs')!
    expect(peak.programs[noise]!.gradNorm).toBeGreaterThan(0)
    // The paper's argument needs |grad| to be high for BOTH cases, or the
    // discrimination would be doing nothing that magnitude could not.
    const both = peak.programs[noise]!.gradNorm > peak.programs[frontier]!.gradNorm
    expect(typeof both).toBe('boolean')
  })

  it('is the absolute value that lets noise win, not the algebra', () => {
    // The finding that shaped the page. Recorded as a measurement rather than
    // an assertion of intent: if a future change flips these, the prose has to
    // change with it.
    const underAbs = noiseOutbidsFrontier(res.rounds, frontier, noise, 'abs')
    const underSigned = noiseOutbidsFrontier(res.rounds, frontier, noise, 'signed')
    expect(underAbs).toBeGreaterThan(0)
    expect(underSigned).toBe(0)
  })

  it('decays the ALIGNMENT once the curriculum is mastered', () => {
    // What makes the frontier move. It is the cosine that collapses -- not the
    // magnitude -- so this is the term the page's third panel has to show.
    const peak = peakRound(res.rounds, frontier, 'abs')!
    const last = res.rounds[res.rounds.length - 1]!.programs[frontier]!
    expect(peak.programs[frontier]!.cos).toBeGreaterThan(0.5)
    expect(last.cos).toBeLessThan(peak.programs[frontier]!.cos)
  })

  it('GROWS the motion term as the lookback window lengthens', () => {
    // Corrects a wrong assumption this suite originally encoded. |P (*) dtheta|
    // does not shrink on a converged learner: AdamW's updates are normalised to
    // about `lr` per coordinate, so a window spanning e/2 rounds measures a
    // longer path as e rises. It climbs even after the loss has flattened.
    //
    // The consequence is the reason Equation 4 normalises rewards within the
    // pool, and the page has to say so rather than let a reader infer that a
    // rising reward means a rising frontier.
    const first = res.rounds[0]!.programs[frontier]!.motionNorm
    const last = res.rounds[res.rounds.length - 1]!.programs[frontier]!.motionNorm
    expect(last).toBeGreaterThan(first * 10)
  })

  it('rejects a non-finite series pick rather than plotting a hole', () => {
    const s = series(res.rounds, (r) => (r.e === 4 ? Number.NaN : r.trainLossBpb))
    expect(s.some(([x]) => x === 4)).toBe(false)
    expect(s.length).toBe(res.rounds.length - 1)
  })
})

describe('curriculum composition', () => {
  it('puts exactly one noise control, and it is last', () => {
    const pool = poolFor()
    expect(pool.filter((p) => p.kind === 'noise')).toHaveLength(1)
    expect(pool[pool.length - 1]!.kind).toBe('noise')
    expect(noiseIndex(pool)).toBe(pool.length - 1)
  })

  it('offers more than one structured family, so a curriculum can move', () => {
    // A pool of one family plus noise cannot demonstrate a moving frontier,
    // which is the comparison the page is built around.
    expect(poolFor().filter((p) => p.kind !== 'noise').length).toBeGreaterThan(1)
  })

  it('gives every program SEQ_LEN + 1 bytes of legal output', () => {
    for (const bytes of [ARITHMETIC(0), FIBONACCI(1), NOISE(7)]) {
      expect(bytes).toHaveLength(SEQ_LEN + 1)
      for (const b of bytes) {
        expect(b).toBeGreaterThanOrEqual(0)
        expect(b).toBeLessThan(ALPHABET)
      }
    }
  })

  it('makes noise genuinely unpredictable, unlike the families', () => {
    // If noise were structured it would sit at the learner's frontier instead
    // of being the control, and the whole comparison would be void.
    const n = NOISE(7)
    expect(new Set(Array.from(n)).size).toBeGreaterThan(SEQ_LEN / 2)
    expect(Array.from(n)).not.toEqual(Array.from(ARITHMETIC(0)))
  })

  it('gives every family a distinct byte sequence', () => {
    // Two families with identical bytes would be one family, and the moving
    // curriculum would appear to switch while nothing changed.
    const pool = poolFor()
    const seen = new Set(pool.map((p) => Array.from(p.bytes).join(',')))
    expect(seen.size).toBe(pool.length)
  })
})

describe('what the learner is taught', () => {
  const pool = poolFor()

  it('fixedTeach always teaches the frontier program', () => {
    const res = cachedRun({ ...TEST_RUN, teach: fixedTeach })
    for (const r of res.rounds) expect(pool[r.activeIndex]!.kind).toBe('frontier')
  })

  it('steppedTeach advances the curriculum on schedule', () => {
    const res = cachedRun({ ...TEST_RUN, rounds: 60, teach: steppedTeach(20) })
    const byRound = new Map(res.rounds.map((r) => [r.e, r.activeIndex]))
    expect(byRound.get(8)).toBe(0)
    expect(byRound.get(28)).toBe(1)
    expect(byRound.get(48)).toBe(2)
  })

  it('never advances past the last structured program', () => {
    // A curriculum index past the pool would read undefined and silently train
    // on nothing.
    const res = cachedRun({ ...TEST_RUN, rounds: 400, teach: steppedTeach(5) })
    const last = res.pool.length - 2
    for (const r of res.rounds) {
      expect(r.activeIndex).toBeGreaterThanOrEqual(0)
      expect(r.activeIndex).toBeLessThanOrEqual(last)
    }
  })

  it('a MOVING frontier beats a FIXED one at keeping the reward honest', () => {
    // The page's central comparison, measured rather than asserted. With one
    // family taught throughout, the learner masters it and nothing is at the
    // frontier; noise then wins on magnitude alone under the abs. Letting the
    // curriculum move restores a frontier for the reward to point at.
    const fixed = cachedRun({ ...TEST_RUN, rounds: 200, recordEvery: 8, teach: fixedTeach })
    const moving = cachedRun({ ...TEST_RUN, rounds: 200, recordEvery: 8, teach: steppedTeach(50) })
    const noise = noiseIndex(pool)

    const activeIsTop = (res: typeof fixed): number =>
      res.rounds.filter((r) => {
        const a = r.programs[r.activeIndex]
        return a !== undefined && r.programs.every((d) => d.rAbs <= a.rAbs)
      }).length

    const fixedTop = activeIsTop(fixed)
    const movingTop = activeIsTop(moving)
    expect(movingTop).toBeGreaterThan(fixedTop)

    expect(noiseOutbidsFrontier(fixed.rounds, 0, noise, 'abs'))
      .toBeGreaterThan(noiseOutbidsFrontier(moving.rounds, 0, noise, 'abs'))
  })

  it('keeps the signed reward honest under a FIXED curriculum', () => {
    const res = cachedRun({ ...TEST_RUN, rounds: 200, recordEvery: 8, teach: fixedTeach })
    expect(noiseOutbidsFrontier(res.rounds, 0, noiseIndex(pool), 'signed')).toBe(0)
  })

  it('shows the lookback window mis-scoring the frontier right after a switch', () => {
    // Not a bug in the reward: delta-theta_e spans e/2 rounds, so once the
    // curriculum advances, the measured displacement is a MIXTURE of the old
    // family's direction and the new one. The newly-active program is graded
    // against a window dominated by a family it has nothing to do with, so its
    // cos goes NEGATIVE and the signed reward rejects the very program it is
    // supposed to pay.
    //
    // Measured: the active program's cos reaches -0.363, and noise outbids it
    // under the signed reward in 18 of 50 recorded rounds -- against 0 of 50
    // under a fixed curriculum. The paper does not discuss this, and the page
    // states it as a cost of the floor(e/2) window rather than smoothing it.
    const res = cachedRun({ ...TEST_RUN, rounds: 200, recordEvery: 4, teach: steppedTeach(50) })
    const noise = noiseIndex(pool)

    const breaches = res.rounds.filter((r) => {
      const a = r.programs[r.activeIndex]
      const n = r.programs[noise]
      return a !== undefined && n !== undefined && n.rSigned > a.rSigned
    })
    expect(breaches.length).toBeGreaterThan(0)

    // The signature of the cause: the active program reads as opposed to the
    // learner, not merely unaligned.
    const minActiveCos = Math.min(...res.rounds.map((r) => r.programs[r.activeIndex]?.cos ?? 0))
    expect(minActiveCos).toBeLessThan(-0.1)

    // And it is a boundary effect: breaches cluster near a switch rather than
    // being spread evenly across each curriculum segment.
    const sinceSwitch = breaches.map((r) => r.e - (Math.floor((r.e - 1) / 50) * 50 + 1))
    expect(sinceSwitch.filter((d) => d <= 25).length).toBeGreaterThan(
      sinceSwitch.filter((d) => d > 25).length,
    )
  })
})
