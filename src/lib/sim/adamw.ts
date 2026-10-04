/**
 * AdamW, kept separate from the model because the reward needs its state.
 *
 * Equation 2 divides by P_e = diag(sqrt(v_hat_e) + eps), so v_hat is not an
 * optimiser internal here -- it is an input to the thing the page is about.
 * That makes the one detail below load-bearing rather than incidental.
 *
 * `v` is a SECOND MOMENT. It accumulates g * g. Accumulating g instead makes v
 * sign-indefinite, so sqrt(v_hat) returns NaN on any negative entry and the
 * entire reward goes NaN one round later. That bug happened here, and it is
 * pinned by adamw.test.ts so it cannot come back as a "simplification".
 *
 * Decoupled weight decay is in the paper's name (AdamW) but its coefficient is
 * not in any released artefact, so it defaults to 0 and the page says so.
 */
import { PARAM_KEYS, type Params, type Grads } from './model'

export interface AdamConfig {
  lr: number
  beta1: number
  beta2: number
  /** eps inside the sqrt of P_e, and the AdamW denominator guard. */
  eps: number
  /** Decoupled weight decay. 0 because the paper's coefficient is unreleased. */
  weightDecay?: number
}

/**
 * The paper's own optimiser defaults are unreleased, so these are ours and the
 * page labels them as such. lr is the only one tuned here, against a 120-round
 * toy run that reaches ~0.2 bits/byte on the arithmetic family without
 * diverging.
 */
export const DEFAULT_ADAM: AdamConfig = {
  lr: 1e-3,
  beta1: 0.9,
  beta2: 0.999,
  eps: 1e-8,
  weightDecay: 0,
}

export interface AdamState {
  m: Params
  v: Params
  /** Step count, for bias correction. */
  t: number
}

function zeroLike(p: Params): Params {
  return {
    W1: new Float64Array(p.W1.length),
    b1: new Float64Array(p.b1.length),
    W2: new Float64Array(p.W2.length),
    b2: new Float64Array(p.b2.length),
  }
}

export function initState(p: Params): AdamState {
  return { m: zeroLike(p), v: zeroLike(p), t: 0 }
}

/**
 * One AdamW step, in place. Returns the step index after the update.
 *
 * Bias correction is applied to both moments before use, so v_hat_e on the
 * first step is the gradient squared rather than (1 - beta2) times it. Skipping
 * it makes P_e wrong by a factor sqrt(1 - beta2) ~ 0.03 in the first rounds,
 * which is exactly where a frontier program's reward is most sensitive.
 */
export function adamwStep(
  p: Params, g: Grads, s: AdamState, cfg: AdamConfig = DEFAULT_ADAM,
): number {
  s.t += 1
  const bc1 = 1 - Math.pow(cfg.beta1, s.t)
  const bc2 = 1 - Math.pow(cfg.beta2, s.t)
  const wd = cfg.weightDecay ?? 0

  for (const k of PARAM_KEYS) {
    const pk = p[k]
    const gk = g[k]
    const mk = s.m[k]
    const vk = s.v[k]
    for (let i = 0; i < pk.length; i++) {
      const gi = gk[i] ?? 0
      const mi = (mk[i] ?? 0) + (1 - cfg.beta1) * gi
      // g * g, not g. See the note at the top of this file.
      const vi = (vk[i] ?? 0) + (1 - cfg.beta2) * gi * gi
      mk[i] = mi
      vk[i] = vi
      const mHat = mi / bc1
      const vHat = vi / bc2
      let w = (pk[i] ?? 0) - cfg.lr * mHat / (Math.sqrt(vHat) + cfg.eps)
      if (wd !== 0) w -= cfg.lr * wd * (pk[i] ?? 0)
      pk[i] = w
    }
  }
  return s.t
}

/**
 * P_e applied to a vector, as a diagonal operator.
 *
 * Returns a NEW parameter-shaped block rather than mutating: the caller is
 * usually forming P_e (*) delta-theta_e and wants to keep delta-theta itself,
 * and an in-place version would be one refactor away from destroying it.
 */
export function precondition(
  v: Params, s: AdamState, cfg: AdamConfig = DEFAULT_ADAM,
): Params {
  const bc2 = 1 - Math.pow(cfg.beta2, s.t)
  const out = {} as Params
  for (const k of PARAM_KEYS) {
    const src = v[k]
    const dst = new Float64Array(src.length)
    const vk = s.v[k]
    for (let i = 0; i < src.length; i++) {
      dst[i] = (Math.sqrt((vk[i] ?? 0) / bc2) + cfg.eps) * (src[i] ?? 0)
    }
    out[k] = dst
  }
  return out
}

/** Elementwise difference a - b. */
export function subParams(a: Params, b: Params): Params {
  const out = {} as Params
  for (const k of PARAM_KEYS) {
    const x = a[k]
    const y = b[k]
    const z = new Float64Array(x.length)
    for (let i = 0; i < x.length; i++) z[i] = (x[i] ?? 0) - (y[i] ?? 0)
    out[k] = z
  }
  return out
}