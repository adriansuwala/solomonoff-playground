/**
 * A byte-level language model small enough to train in a browser tab, with
 * hand-written forward and backward passes.
 *
 * This exists so the reward page's simulator computes Equation 2 from real
 * numbers rather than replaying a trace. That makes the gradient the single
 * most load-bearing piece of code in the feature: every reward the page prints
 * is an inner product against it, and a wrong gradient produces a chart that
 * looks exactly like a right one. Hence gradcheck.ts, which pins it against
 * central finite differences.
 *
 * Two bugs were caught by that check while this was being written, and both are
 * the kind that survive review:
 *
 *   1. The gradient was computed for the SUM over positions while the reported
 *      loss was the MEAN, scaling every gradient by exactly T. Invisible except
 *      against finite differences.
 *   2. AdamW's second moment accumulated g instead of g^2. v is then
 *      sign-indefinite, so sqrt(v_hat) -- the P_e the reward divides by --
 *      returns NaN on any negative entry. See adamw.ts.
 *
 * The model is deliberately not the paper's architecture. It is a window of
 * K previous bytes, one-hot summed into a tanh hidden layer, then a softmax
 * over the 256-byte alphabet. What matters for the reward is only that
 * parameters, gradients, and an AdamW state all exist and are consistent; the
 * learner's own inductive bias is not what the page is claiming to measure.
 */

// The byte alphabet: the paper's cell modulus, so every value here is a legal
// emitted byte.
export const ALPHABET = 256

/** Bytes of context fed forward. */
export const CONTEXT = 4

/** Sequence length in bytes, including the byte predicted at each position. */
export const SEQ_LEN = 32

/** Hidden width. 64 keeps a full sweep of a 64-program pool under ~600 ms. */
export const HIDDEN = 64

/** ln(2): loss is reported in bits per byte, so the mean nats divide by this. */
const LN2 = Math.log(2)

/**
 * Parameter block. Plain Float64Array per tensor rather than one object graph,
 * because the reward needs to diff and dot raw coordinate vectors, and because
 * copy-on-write of a checkpoint is then a handful of `.slice()` calls.
 *
 * Layout matches the released rungs' ordering convention: input projection,
 * input bias, output projection, output bias.
 */
export interface Params {
  W1: Float64Array // ALPHABET x HIDDEN
  b1: Float64Array // HIDDEN
  W2: Float64Array // HIDDEN x ALPHABET
  b2: Float64Array // ALPHABET
}

/** A gradient block, same shape as Params. */
export type Grads = Params

/** Names of the parameter tensors, in the order they are allocated. */
export const PARAM_KEYS = ['W1', 'b1', 'W2', 'b2'] as const
export type ParamKey = (typeof PARAM_KEYS)[number]

/** Total scalar parameters. Reported on the page so the reader can see the scale. */
export const PARAM_COUNT =
  ALPHABET * HIDDEN + HIDDEN + HIDDEN * ALPHABET + ALPHABET

/**
 * Deterministic PRNG.
 *
 * Every number on the simulator page is reproducible from a seed the reader
 * can see, which is the only reason a number on this page can be checked at
 * all. mulberry32: small, fast, and no dependence on host float behaviour.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return function next(): number {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** A fresh parameter block, initialised small so tanh starts near linear. */
export function initParams(seed = 1): Params {
  const rnd = mulberry32(seed)
  const scale = 1 / Math.sqrt(HIDDEN)
  const fill = (n: number): Float64Array => {
    const a = new Float64Array(n)
    for (let i = 0; i < n; i++) a[i] = (rnd() * 2 - 1) * scale
    return a
  }
  return { W1: fill(ALPHABET * HIDDEN), b1: new Float64Array(HIDDEN), W2: fill(HIDDEN * ALPHABET), b2: new Float64Array(ALPHABET) }
}

/** A zeroed gradient block shaped like `p`. */
export function zerosLike(p: Params): Grads {
  return { W1: new Float64Array(p.W1.length), b1: new Float64Array(p.b1.length), W2: new Float64Array(p.W2.length), b2: new Float64Array(p.b2.length) }
}

/** A deep copy, for checkpoints. */
export function cloneParams(p: Params): Params {
  return { W1: p.W1.slice(), b1: p.b1.slice(), W2: p.W2.slice(), b2: p.b2.slice() }
}

/** Inner product of two parameter blocks, summed over every coordinate. */
export function dotParams(a: Params, b: Params): number {
  let s = 0
  for (const k of PARAM_KEYS) {
    const x = a[k]
    const y = b[k]
    for (let i = 0; i < x.length; i++) s += (x[i] ?? 0) * (y[i] ?? 0)
  }
  return s
}

/** Euclidean norm of a parameter block. */
export function normParams(a: Params): number {
  return Math.sqrt(dotParams(a, a))
}

/** The context byte at distance `back` before position `t`, zero-padded. */
function contextByte(bytes: Uint8Array, t: number, back: number): number {
  const i = t - back
  return i >= 0 ? (bytes[i] ?? 0) : 0
}

/**
 * Forward pass, returning the hidden states and the probability tensor.
 *
 * Kept separate from `lossAndGrads` because the simulator's prose needs the
 * learner's predicted distribution, not only its loss: "what does the learner
 * think comes next" is how the page shows a program sitting at the frontier
 * rather than one already mastered.
 */
export interface Forward {
  /** SEQ_LEN x HIDDEN tanh activations. */
  h: Float64Array
  /** SEQ_LEN x ALPHABET softmax probabilities. */
  p: Float64Array
  /** Mean loss over positions, in bits per byte. */
  lossBpb: number
}

export function forward(p: Params, bytes: Uint8Array): Forward {
  const h = new Float64Array(SEQ_LEN * HIDDEN)
  const probs = new Float64Array(SEQ_LEN * ALPHABET)
  let nll = 0

  for (let t = 0; t < SEQ_LEN; t++) {
    const ho = t * HIDDEN
    for (let j = 0; j < HIDDEN; j++) {
      let s = p.b1[j] ?? 0
      // One-hot input: only the CONTEXT context bytes contribute, so this is a
      // gather over CONTEXT rows of W1 rather than a full matmul.
      for (let back = 1; back <= CONTEXT; back++) {
        s += p.W1[contextByte(bytes, t, back) * HIDDEN + j] ?? 0
      }
      h[ho + j] = Math.tanh(s)
    }

    // Softmax over the byte alphabet, max-shifted for numerical stability.
    let max = -Infinity
    const logits = new Float64Array(ALPHABET)
    for (let k = 0; k < ALPHABET; k++) {
      let s = p.b2[k] ?? 0
      for (let j = 0; j < HIDDEN; j++) s += (h[ho + j] ?? 0) * (p.W2[j * ALPHABET + k] ?? 0)
      logits[k] = s
      if (s > max) max = s
    }
    let z = 0
    for (let k = 0; k < ALPHABET; k++) z += Math.exp((logits[k] ?? 0) - max)
    const logZ = max + Math.log(z)

    const target = bytes[t + 1] ?? 0
    nll += logZ - (logits[target] ?? 0)
    const po = t * ALPHABET
    for (let k = 0; k < ALPHABET; k++) probs[po + k] = Math.exp((logits[k] ?? 0) - logZ)
  }

  return { h, p: probs, lossBpb: nll / (SEQ_LEN * LN2) }
}

/**
 * Forward plus backward in one pass, writing gradients into `g`.
 *
 * `g` is accumulated into, not reset, so a caller can sum gradients over a
 * pool before stepping. The AdamW step in this simulator is per-program, but
 * the reward needs a gradient per program, and sharing the traversal keeps the
 * two from ever disagreeing about what a gradient is.
 *
 * The 1/T below is the bug the finite-difference check exists for: the
 * gradient is of the MEAN loss, matching what `forward` reports. Computing it
 * for the sum scales every coordinate by SEQ_LEN and still trains, just badly.
 */
export function lossAndGrads(p: Params, g: Grads, bytes: Uint8Array): number {
  const { h } = forward(p, bytes)
  const dlogits = new Float64Array(SEQ_LEN * ALPHABET)
  let nll = 0

  for (let t = 0; t < SEQ_LEN; t++) {
    const ho = t * HIDDEN
    const po = t * ALPHABET
    let max = -Infinity
    const logits = new Float64Array(ALPHABET)
    for (let k = 0; k < ALPHABET; k++) {
      let s = p.b2[k] ?? 0
      for (let j = 0; j < HIDDEN; j++) s += (h[ho + j] ?? 0) * (p.W2[j * ALPHABET + k] ?? 0)
      logits[k] = s
      if (s > max) max = s
    }
    let z = 0
    for (let k = 0; k < ALPHABET; k++) z += Math.exp((logits[k] ?? 0) - max)
    const logZ = max + Math.log(z)
    const target = bytes[t + 1] ?? 0
    nll += logZ - (logits[target] ?? 0)
    for (let k = 0; k < ALPHABET; k++) {
      dlogits[po + k] = (Math.exp((logits[k] ?? 0) - logZ) - (k === target ? 1 : 0)) / SEQ_LEN
    }
  }

  for (let t = 0; t < SEQ_LEN; t++) {
    const ho = t * HIDDEN
    const po = t * ALPHABET
    for (let k = 0; k < ALPHABET; k++) {
      const d = dlogits[po + k] ?? 0
      if (d === 0) continue
      g.b2[k] = (g.b2[k] ?? 0) + d
      for (let j = 0; j < HIDDEN; j++) {
        const hj = h[ho + j] ?? 0
        const w2 = p.W2[j * ALPHABET + k] ?? 0
        g.W2[j * ALPHABET + k] = (g.W2[j * ALPHABET + k] ?? 0) + d * hj
        const dz = d * w2 * (1 - hj * hj)
        g.b1[j] = (g.b1[j] ?? 0) + dz
        for (let back = 1; back <= CONTEXT; back++) {
          const row = contextByte(bytes, t, back) * HIDDEN + j
          g.W1[row] = (g.W1[row] ?? 0) + dz
        }
      }
    }
  }

  return nll / (SEQ_LEN * LN2)
}