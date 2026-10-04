/**
 * The training loop: teach a curriculum, and score every program under
 * Equation 2 at every recorded round.
 *
 * This is the simulator. Everything the page plots comes out of `runSim`, and
 * every number in it is derived here rather than typed into the prose -- the
 * same rule the rest of the site follows (D2).
 *
 * Design notes that are load-bearing rather than stylistic:
 *
 * CHECKPOINTS. Equation 2's delta-theta_e compares theta_e against
 * theta_{floor(e/2)}, so the loop keeps every round's parameters and reaches
 * back into the middle of the run. That is why this cannot be a single forward
 * pass: the reward at round e is not computable until round e exists.
 *
 * DETERMINISM. Everything is seeded. A reader who reloads sees identical
 * numbers, which is the precondition for being able to check any of them.
 *
 * COST. A full run is ~120 rounds x (pool size) gradient evaluations. At
 * HIDDEN=64 / SEQ_LEN=32 that is a few hundred ms per program-pool sweep, so a
 * full run is seconds rather than milliseconds. Phase 4 replaces this with an
 * incremental worker; the pure function here stays the reference both must
 * agree with, which is the same arrangement bf.ts uses for its stepper.
 */
import { cloneParams, forward, initParams, lossAndGrads, zerosLike, type Params } from './model'
import { DEFAULT_ADAM, adamwStep, initState, type AdamConfig } from './adamw'
import { decompose, rewardShare, type Decomposition, type RewardMode } from './reward'
import type { SimProgram } from './programs'

export interface RunConfig {
  /** Training rounds. The paper's rungs run to ~2600; this toy needs ~120. */
  rounds: number
  /** Pool size, i.e. programs scored per round. The paper's pool is 1536. */
  poolSize: number
  /** Parameter init seed. */
  seed: number
  adam: AdamConfig
  /** Score every Nth round rather than every round, to bound the trace size. */
  recordEvery: number
  /**
   * Which program the learner is TAUGHT at round e, by index into the pool.
   *
   * Load-bearing, and the reason is easy to get wrong. What the learner is
   * taught sets the direction of delta-theta_e, and cos measures alignment
   * against exactly that direction -- so changing what is scored changes
   * nothing, while changing what is taught changes everything. An earlier
   * version of this simulator tried to separate the two arms by widening the
   * scored pool and the arms came out numerically identical, which is the
   * correct result for a comparison that does not touch the independent
   * variable.
   *
   * The default teaches the frontier program and nothing else.
   */
  teach: (pool: SimProgram[], e: number) => number
}

export const DEFAULT_RUN: RunConfig = {
  rounds: 120,
  poolSize: 8,
  seed: 1,
  adam: DEFAULT_ADAM,
  recordEvery: 4,
  teach: (pool) => pool.findIndex((p) => p.kind === 'frontier'),
}

/**
 * Advance to the next STRUCTURED program every `span` rounds.
 *
 * Clamped to the last program that is not the noise control. Clamping to
 * `pool.length - 1` instead would put the learner on the control at the end of
 * a long run -- training on uniform random bytes, which teaches nothing and
 * quietly invalidates every downstream comparison. That is what the "never
 * advances past the last structured program" test exists to catch.
 */
export function steppedTeach(span: number) {
  return (pool: SimProgram[], e: number): number => {
    const structured = pool.filter((p) => p.kind !== 'noise').length
    return Math.max(0, Math.min(structured - 1, Math.floor((e - 1) / span)))
  }
}

/** The fixed curriculum: teach the frontier program for the whole run. */
export function fixedTeach(pool: SimProgram[]): number {
  const i = pool.findIndex((p) => p.kind === 'frontier')
  // Never -1: a missing frontier would fall through to index 0 by accident and
  // train on whatever happened to be first.
  return i >= 0 ? i : 0
}

export interface RoundRecord {
  /** 1-based round index. */
  e: number
  /** The learner's loss on its training family, bits per byte. */
  trainLossBpb: number
  /** Per-program decomposition, in pool order. */
  programs: Decomposition[]
  /** Each program's share of the round's total abs reward. */
  shares: number[]
  /** Largest single share: one number for "has the generator collapsed". */
  concentration: number
  /** Pool index of the program being taught at this round. */
  activeIndex: number
}

export interface SimResult {
  /** Recorded rounds, ascending in `e`. */
  rounds: RoundRecord[]
  /** Pool actually used, so the page can label each program. */
  pool: SimProgram[]
  /** Final learner loss, for the summary line. */
  finalLossBpb: number
  /** Every parameter value is finite. A false here means a chart is a lie. */
  finite: boolean
}

/**
 * Run a curriculum and score it.
 *
 * The learner is taught `cfg.teach(pool, e)` and scored on all of `pool`.
 */
export function runSim(
  pool: SimProgram[],
  cfg: RunConfig = DEFAULT_RUN,
): SimResult {
  const scored = pool.slice(0, cfg.poolSize)
  const params = initParams(cfg.seed)
  const state = initState(params)
  const grads = zerosLike(params)

  // Every round's parameters, for the theta_{floor(e/2)} lookup.
  const history: Params[] = [cloneParams(params)]

  const rounds: RoundRecord[] = []
  let finite = true

  for (let e = 1; e <= cfg.rounds; e++) {
    const activeIndex = cfg.teach(scored, e)
    const active = scored[activeIndex] ?? scored[0]
    if (active === undefined) break

    // One gradient step on the program this round teaches, then advance.
    grads.W1.fill(0); grads.b1.fill(0); grads.W2.fill(0); grads.b2.fill(0)
    const trainLossBpb = lossAndGrads(params, grads, active.bytes)
    adamwStep(params, grads, state, cfg.adam)
    history.push(cloneParams(params))

    // delta-theta_e compares against the checkpoint from the middle of the run
    // so far. Round 1 has no midpoint yet, so scoring starts at round 2.
    if (e < 2 || e % cfg.recordEvery !== 0) continue

    const thetaMid = history[Math.floor(e / 2)] as Params

    const programs = scored.map((prog) => {
      const g = zerosLike(params)
      const loss = forward(params, prog.bytes).lossBpb
      lossAndGrads(params, g, prog.bytes)
      return decompose({
        grad: g,
        thetaMid,
        thetaNow: params,
        state,
        lossBpb: loss,
        cfg: cfg.adam,
      })
    })

    const shares = rewardShare(programs, 'abs')
    rounds.push({
      e,
      trainLossBpb,
      programs,
      shares,
      concentration: shares.length > 0 ? Math.max(...shares) : 0,
      activeIndex,
    })
  }

  // One finiteness sweep over everything the page will draw. A NaN anywhere in
  // here means some chart is plotting a hole, so it is a first-class result
  // rather than something to discover in the browser.
  const last = rounds[rounds.length - 1]
  if (last) {
    for (const d of last.programs) {
      for (const v of [d.gradNorm, d.motionNorm, d.cos, d.inner, d.rAbs, d.rSigned, d.lossBpb]) {
        if (!Number.isFinite(v)) finite = false
      }
    }
  }

  return {
    rounds,
    pool: scored,
    finalLossBpb: rounds[rounds.length - 1]?.trainLossBpb ?? Number.NaN,
    finite,
  }
}

/**
 * The round whose frontier reward is largest.
 *
 * The peak is the most interesting single round on the page: it is where the
 * reward most clearly prefers what the learner is learning over what it has
 * finished, and it is the round the page opens on.
 *
 * Scored under `mode`, because the peak's location depends on the reading --
 * and if it does not, that is itself worth the page saying.
 */
export function peakRound(rounds: RoundRecord[], frontier: number, mode: RewardMode): RoundRecord | undefined {
  // The incumbent's value has to be read under the SAME mode as the candidate.
  // Comparing a signed candidate against an absolute incumbent would make the
  // peak a function of the toggle rather than of the data.
  const value = (r: RoundRecord): number => {
    const d = r.programs[frontier]
    if (d === undefined) return Number.NEGATIVE_INFINITY
    return mode === 'abs' ? d.rAbs : d.rSigned
  }
  let best: RoundRecord | undefined
  for (const r of rounds) {
    const v = value(r)
    if (!Number.isFinite(v)) continue
    if (best === undefined || v > value(best)) best = r
  }
  return best
}

/**
 * How many recorded rounds noise outbids the frontier under a mode.
 *
 * The headline number for the gaming demo, and the one that inverts when the
 * absolute value is dropped.
 */
export function noiseOutbidsFrontier(
  rounds: RoundRecord[], frontier: number, noise: number, mode: RewardMode,
): number {
  let n = 0
  for (const r of rounds) {
    const f = r.programs[frontier]
    const z = r.programs[noise]
    if (f === undefined || z === undefined) continue
    const fv = mode === 'abs' ? f.rAbs : f.rSigned
    const zv = mode === 'abs' ? z.rAbs : z.rSigned
    if (zv > fv) n += 1
  }
  return n
}

/**
 * The recorded rounds as (round, value) pairs, for a chart series.
 *
 * Rejects non-finite values rather than passing them through: Chart.tsx plots
 * what it is given, and a NaN coordinate draws a gap that reads as missing data
 * rather than as a bug.
 */
export function series(
  rounds: RoundRecord[], pick: (r: RoundRecord) => number | null,
): [number, number][] {
  const out: [number, number][] = []
  for (const r of rounds) {
    const v = pick(r)
    if (v !== null && Number.isFinite(v)) out.push([r.e, v])
  }
  return out
}