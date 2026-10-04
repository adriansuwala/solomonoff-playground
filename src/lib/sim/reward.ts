/**
 * Equation 2, and the decomposition the simulator page is built to show.
 *
 * The reward is
 *   r_i = |<grad_i, P_e (*) delta-theta_e>|
 * with delta-theta_e = theta_{floor(e/2)} - theta_e and P_e = diag(sqrt(v_hat_e)
 * + eps). Written as a product of magnitudes and a cosine, that is
 *
 *   r_i = |grad_i| * |P_e (*) delta-theta_e| * cos(grad_i, P_e (*) delta-theta_e)
 *
 * and this is the form the page plots. The factorisation is not decoration --
 * it is the whole argument, because the three factors do three different jobs:
 *
 *   |grad|            high for a frontier program AND high for noise. It is
 *                     the term that cannot tell them apart, so the page shows it
 *                     precisely so the reader can see it failing to.
 *   cos               the discriminating term. Frontier programs sit near +1
 *                     (the learner's recent motion agrees with this gradient);
 *                     noise sits at or below 0.
 *   |P (*) dtheta|    the length of the measured path. This one does NOT decay
 *                     on a converged learner, which is worth knowing: AdamW's
 *                     updates are normalised to roughly `lr` per coordinate, so
 *                     a window spanning e/2 rounds measures a LONGER path as e
 *                     grows. It rises even after the loss has flattened. See
 *                     reward.test.ts, which pins the direction.
 *
 * Sign convention, verified numerically rather than assumed. AdamW moves theta
 * along -g, so theta_e - theta_{floor(e/2)} is roughly -lr * (mean
 * preconditioned gradient); delta-theta_e is defined as the NEGATIVE of that,
 * i.e. it points the way the optimiser has been going. The inner product is
 * therefore positive exactly when a program's gradient AGREES with the
 * learner's motion, which is what makes the signed variant meaningful and the
 * abs variant lossy. See reward.test.ts, which pins the sign on a hand-built
 * case where the answer is known by construction.
 */
import { normParams, dotParams, type Grads, type Params } from './model'
import { precondition, subParams, type AdamState, type AdamConfig } from './adamw'

/**
 * How the reward treats the sign of the inner product.
 *
 * This is the page's central toggle and it is not cosmetic: measured on the
 * toy run, dropping the abs takes noise from outbidding the frontier by ~324x to
 * never outbidding it at all. The paper's canonical arm is 'abs'.
 */
export type RewardMode = 'abs' | 'signed'

export interface Decomposition {
  /** |grad_i|: the learner's gradient magnitude on this program's bytes. */
  gradNorm: number
  /** |P_e (*) delta-theta_e|: the learner's preconditioned recent motion. */
  motionNorm: number
  /** cos(grad_i, P_e (*) delta-theta_e): +1 aligned, -1 opposed, 0 orthogonal. */
  cos: number
  /** The signed inner product, before any abs. */
  inner: number
  /** The reward under the canonical (abs) reading. */
  rAbs: number
  /** The reward under the signed reading. */
  rSigned: number
  /** The learner's loss on this program, in bits per byte. */
  lossBpb: number
}

/**
 * Everything the reward is built from, gathered in one place.
 *
 * `deltaTheta` is passed already signed: delta-theta_e = theta_{floor(e/2)} -
 * theta_e. Computing it here from two checkpoints would be one line, but the
 * page's whole point is that the reader can see which vector is which, so the
 * direction is made explicit at the call site.
 */
export interface RewardInputs {
  /** grad_i = the learner's gradient on program i's bytes at theta_now. */
  grad: Grads
  /** theta_{floor(e/2)}. */
  thetaMid: Params
  /** theta_e, i.e. theta_now. */
  thetaNow: Params
  /** The AdamW state at round e, for P_e. */
  state: AdamState
  /** The learner's loss on this program, in bits per byte. */
  lossBpb: number
  cfg?: AdamConfig
}

/**
 * Decompose one program's reward.
 *
 * Total by construction: no division can produce NaN, because a zero gradient
 * norm or a zero motion norm short-circuits `cos` to 0 rather than dividing.
 * A NaN here would propagate into every chart on the page.
 */
export function decompose(x: RewardInputs): Decomposition {
  const cfg = x.cfg
  const deltaTheta = subParams(x.thetaMid, x.thetaNow)
  const motion = precondition(deltaTheta, x.state, cfg)

  const gradNorm = normParams(x.grad)
  const motionNorm = normParams(motion)
  const inner = dotParams(x.grad, motion)

  const cos = gradNorm > 0 && motionNorm > 0 ? inner / (gradNorm * motionNorm) : 0

  return {
    gradNorm,
    motionNorm,
    cos,
    inner,
    rAbs: Math.abs(inner),
    rSigned: inner,
    lossBpb: x.lossBpb,
  }
}

/** The reward a decomposition pays out under a given mode. */
export function rewardOf(d: Decomposition, mode: RewardMode): number {
  return mode === 'abs' ? d.rAbs : d.rSigned
}

/**
 * The reward's sign convention, stated as a number rather than prose.
 *
 * +1 when a program's gradient points the same way as the learner's recent
 * preconditioned displacement, which -- given delta-theta_e = theta_{floor(e/2)}
 * - theta_e and a descent optimiser -- is the direction the learner has been
 * improving in. Zero when orthogonal. The page prints this next to each
 * program so "aligned" is a measured quantity and not an adjective.
 */
export const ALIGNED = 1
export const ORTHOGONAL = 0
export const OPPOSED = -1

/**
 * Normalise rewards within a pool, the way Equation 4's GRPO advantage does.
 *
 * This is not cosmetic. Because |P (*) dtheta| grows with the length of the
 * lookback window, the raw reward scale drifts upward across a run even when
 * nothing interesting is happening -- measured on the toy run, motionNorm rises
 * from 2.0e-3 at round 8 to 2.4e-1 at round 200, a factor of 124. A chart
 * comparing rounds without dividing that out is mostly plotting the window
 * getting longer. Mean and standard deviation over the pool, guarding the
 * degenerate cases.
 */
export function normalizePool(ds: Decomposition[]): number[] {
  if (ds.length === 0) return []
  const mean = ds.reduce((s, d) => s + rewardOf(d, 'abs'), 0) / ds.length
  const variance = ds.reduce((s, d) => {
    const x = rewardOf(d, 'abs') - mean
    return s + x * x
  }, 0) / ds.length
  const sd = Math.sqrt(variance)
  return ds.map((d) => (rewardOf(d, 'abs') - mean) / (sd + 1e-12))
}

/**
 * How much of a round's total reward one program took.
 *
 * The share is the quantity that shows degeneracy: as one program approaches
 * all of the reward, the generator is collapsing, whatever the reward function
 * says. Reported so the page can show the share next to the frontier share.
 */
export function rewardShare(ds: Decomposition[], mode: RewardMode = 'abs'): number[] {
  const total = ds.reduce((s, d) => s + Math.max(0, rewardOf(d, mode)), 0)
  if (!(total > 0)) return ds.map(() => 0)
  return ds.map((d) => Math.max(0, rewardOf(d, mode)) / total)
}

/**
 * The largest share any single program took in a round.
 *
 * One number for "has the generator collapsed". A sampler that is following the
 * frontier spreads reward across several programs; one that has degenerated
 * takes nearly all of it in one.
 */
export function concentration(shares: number[]): number {
  if (shares.length === 0) return 0
  return Math.max(...shares)
}

/**
 * The count of programs that outbid the frontier program under a given mode.
 *
 * The page's headline comparison. Under 'abs' noise wins this repeatedly on a
 * single-family curriculum; under 'signed' it never does, which is the clearest
 * statement of what the absolute value costs.
 */
export function countOutbidding(ds: Decomposition[], frontier: number, mode: RewardMode): number {
  const ref = ds[frontier]
  if (ref === undefined) return 0
  const refReward = rewardOf(ref, mode)
  return ds.filter((d, i) => i !== frontier && rewardOf(d, mode) > refReward).length
}