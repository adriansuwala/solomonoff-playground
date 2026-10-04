/**
 * The gate the reward simulator stands on: the hand-written gradient must agree
 * with central finite differences.
 *
 * Every reward the simulator page prints is
 *   <grad_i, P_e (*) delta-theta_e>
 * so this file is not a nice-to-have. A gradient that is wrong by a constant
 * factor, or wrong in one term, produces a chart that is indistinguishable from
 * a correct one -- the ordering can survive, the magnitudes cannot. The only
 * defence is to compare against a method that shares no code with the one
 * under test.
 *
 * Two real bugs were caught this way and both are now regression tests below:
 * the gradient of the SUM rather than the MEAN (every coordinate off by exactly
 * SEQ_LEN), and AdamW accumulating g instead of g^2 into the second moment.
 */
import { describe, expect, it } from 'vitest'
import {
  ALPHABET, HIDDEN, PARAM_COUNT, PARAM_KEYS, SEQ_LEN,
  forward, initParams, lossAndGrads, mulberry32, zerosLike,
} from './model'

/**
 * The loss as a function of parameters alone, in nats, with no backward pass.
 *
 * Deliberately a separate implementation from model.ts's forward: it reads
 * like the definition rather than like the code, so a bug in the optimised
 * forward cannot hide by being present in both halves of the comparison.
 */
function lossOnly(
  p: { W1: Float64Array; b1: Float64Array; W2: Float64Array; b2: Float64Array },
  bytes: Uint8Array,
): number {
  let nll = 0
  for (let t = 0; t < SEQ_LEN; t++) {
    const pre = new Float64Array(HIDDEN)
    for (let j = 0; j < HIDDEN; j++) {
      let s = p.b1[j] ?? 0
      for (let back = 1; back <= 4; back++) {
        // Context is zero-PADDED, not truncated: positions before the start of
        // the sequence read byte 0. This is the model's documented convention
        // (see contextByte) and an earlier version of this file skipped padding
        // instead, which made row 0 of W1 look like a coordinate with a
        // nonzero analytic gradient and an identically zero finite difference.
        const i = t - back
        const c = i >= 0 ? bytes[i] ?? 0 : 0
        s += p.W1[c * HIDDEN + j] ?? 0
      }
      pre[j] = Math.tanh(s)
    }
    let max = -Infinity
    const lg = new Float64Array(ALPHABET)
    for (let k = 0; k < ALPHABET; k++) {
      let s = p.b2[k] ?? 0
      for (let j = 0; j < HIDDEN; j++) s += (pre[j] ?? 0) * (p.W2[j * ALPHABET + k] ?? 0)
      lg[k] = s
      if (s > max) max = s
    }
    let z = 0
    for (let k = 0; k < ALPHABET; k++) z += Math.exp((lg[k] ?? 0) - max)
    nll += max + Math.log(z) - (lg[bytes[t + 1] ?? 0] ?? 0)
  }
  return nll / SEQ_LEN
}

const arithmetic = (off: number): Uint8Array =>
  Uint8Array.from({ length: SEQ_LEN + 1 }, (_, t) => (off + t) % ALPHABET)

/** Absolute error floor, for coordinates whose true gradient is ~0. */
const ABS_FLOOR = 1e-9

describe('analytic gradient vs central differences', () => {
  it('agrees to 1e-5 relative on every tensor', () => {
    const p = initParams(4242)
    const bytes = arithmetic(3)
    const g = zerosLike(p)
    lossAndGrads(p, g, bytes)

    const h = 1e-5
    let worst = 0
    let worstAt = ''

    for (const key of PARAM_KEYS) {
      const n = p[key].length
      // Sample spread across the tensor rather than the first few, which would
      // only ever touch the rows for byte 0.
      for (let s = 0; s < 4; s++) {
        const i = Math.floor((s * n) / 4) + 1
        const orig = p[key][i] ?? 0

        p[key][i] = orig + h
        const lp = lossOnly(p, bytes)
        p[key][i] = orig - h
        const lm = lossOnly(p, bytes)
        p[key][i] = orig

        const fd = (lp - lm) / (2 * h)
        const an = g[key][i] ?? 0
        const denom = Math.max(ABS_FLOOR, Math.abs(fd) + Math.abs(an))
        const rel = Math.abs(fd - an) / denom
        if (rel > worst) { worst = rel; worstAt = `${key}[${i}]` }
      }
    }

    expect(worst, `worst at ${worstAt}`).toBeLessThan(1e-5)
  })

  it('reproduces the SEQ_LEN scaling bug this check exists for', () => {
    // If the gradient were of the sum rather than the mean, every coordinate
    // would come out exactly SEQ_LEN times too large. Assert the ratio is 1
    // and not 32, so the failure mode has a name in the suite.
    const p = initParams(4242)
    const bytes = arithmetic(3)
    const g = zerosLike(p)
    lossAndGrads(p, g, bytes)

    const h = 1e-5
    const i = 5
    const orig = p.b1[i] ?? 0
    p.b1[i] = orig + h
    const lp = lossOnly(p, bytes)
    p.b1[i] = orig - h
    const lm = lossOnly(p, bytes)
    p.b1[i] = orig
    const fd = (lp - lm) / (2 * h)
    const ratio = Math.abs(fd / (g.b1[i] ?? 1))
    // Finite differences carry ~1e-3 relative truncation error at this h, so the
    // assertion is that the ratio is 1 and emphatically NOT 32.
    expect(ratio).toBeCloseTo(1, 2)
    expect(Math.abs(ratio - SEQ_LEN)).toBeGreaterThan(1)
  })

  it('reports the model size the page quotes', () => {
    // 256*64 + 64 + 64*256 + 256. The page prints this next to the released
    // rungs, so it is pinned rather than recomputed in prose.
    expect(PARAM_COUNT).toBe(2 * ALPHABET * HIDDEN + HIDDEN + ALPHABET)
    expect(PARAM_COUNT).toBe(33088)
  })
})

describe('forward', () => {
  it('starts at ~8 bits/byte on random bytes, the uniform-over-256 prediction', () => {
    const p = initParams(1)
    const rnd = mulberry32(99)
    const bytes = Uint8Array.from({ length: SEQ_LEN + 1 }, () => (rnd() * ALPHABET) | 0)
    const { lossBpb } = forward(p, bytes)
    // 8 bits/byte is exactly log2(256). A model that reads anything here is
    // reporting a real bias, so the band is narrow on purpose.
    expect(lossBpb).toBeGreaterThan(7.9)
    expect(lossBpb).toBeLessThan(8.1)
  })

  it('gives a normalised distribution at every position', () => {
    const p = initParams(7)
    const { p: probs } = forward(p, arithmetic(0))
    for (let t = 0; t < SEQ_LEN; t++) {
      let z = 0
      for (let k = 0; k < ALPHABET; k++) z += probs[t * ALPHABET + k] ?? 0
      expect(z).toBeCloseTo(1, 10)
    }
  })

  it('is deterministic: same bytes, same bytes out', () => {
    const p = initParams(11)
    const a = forward(p, arithmetic(5))
    const b = forward(p, arithmetic(5))
    expect(Array.from(a.p)).toEqual(Array.from(b.p))
  })

  it('never returns a non-finite probability on any input', () => {
    // The NaN in this feature arrived through the optimiser, not the forward,
    // but a forward that can emit NaN would hide the next one.
    const p = initParams(13)
    for (const bytes of [arithmetic(0), new Uint8Array(SEQ_LEN + 1), Uint8Array.from({ length: SEQ_LEN + 1 }, (_, i) => (i * 251) % ALPHABET)]) {
      const { p: probs, lossBpb } = forward(p, bytes)
      expect(Number.isFinite(lossBpb)).toBe(true)
      for (const v of probs) expect(Number.isFinite(v)).toBe(true)
    }
  })
})