/**
 * The programs the generator is simulated over.
 *
 * These are NOT real Brainfuck programs. They are emitted byte sequences with
 * the structural properties the reward is supposed to distinguish, and the
 * page says so: a reader who assumes these are UTM output would be wrong, and
 * the honest framing is that the simulator exercises the reward's algebra on a
 * controlled input set, not the paper's program space.
 *
 * The reason for byte sequences rather than actual bf.ts programs is the
 * experiment: to ask whether the reward prefers the learner's frontier over
 * noise, the frontier has to be a KNOWN position in a KNOWN curriculum. A real
 * random program draw has no such handle. Feeding the simulator a family it is
 * currently learning, a family it has finished, and a noise control makes the
 * three cases a property of the setup rather than of luck.
 *
 * The families mirror the ones bf.ts already runs, so the page's structural
 * vocabulary is the paper's: arithmetic and Fibonacci mod 256, plus uniform
 * noise as the control.
 */
import { ALPHABET, SEQ_LEN, mulberry32 } from './model'

export type ProgramKind = 'frontier' | 'mastered' | 'noise'

export interface SimProgram {
  key: string
  label: string
  kind: ProgramKind
  /** What this program is for, in one clause, for the tooltip. */
  blurb: string
  bytes: Uint8Array
}

/** Arithmetic mod 256: the paper's Table 1 worked example. */
function arithmeticBytes(offset: number): Uint8Array {
  return Uint8Array.from({ length: SEQ_LEN + 1 }, (_, t) => (offset + t) % ALPHABET)
}

/** Fibonacci mod 256, seeded from the first drawn byte. */
function fibonacciBytes(offset: number): Uint8Array {
  let a = offset % ALPHABET
  let b = (offset + 1) % ALPHABET
  const out = new Uint8Array(SEQ_LEN + 1)
  for (let t = 0; t <= SEQ_LEN; t++) {
    out[t] = t % 2 === 0 ? a : b
    const next = (a + b) % ALPHABET
    a = b
    b = next
  }
  return out
}

/** Quadratic mod 256. */
function quadraticBytes(offset: number): Uint8Array {
  return Uint8Array.from({ length: SEQ_LEN + 1 }, (_, t) => (offset + t * t) % ALPHABET)
}

/** Geometric mod 256: doubles each step, wrapping at the modulus. */
function geometricBytes(offset: number): Uint8Array {
  let v = offset % ALPHABET
  const out = new Uint8Array(SEQ_LEN + 1)
  for (let t = 0; t <= SEQ_LEN; t++) {
    out[t] = v
    v = (v * 2) % ALPHABET
  }
  return out
}

/**
 * Uniform random bytes, and the control that matters.
 *
 * A seeded LCG rather than Math.random, because the page's numbers must be
 * reproducible: a reader who reloads and sees different rewards cannot check
 * anything.
 */
function noiseBytes(seed: number): Uint8Array {
  const rnd = mulberry32(seed)
  return Uint8Array.from({ length: SEQ_LEN + 1 }, () => (rnd() * ALPHABET) | 0)
}

/**
 * The pool, for a curriculum that teaches `learned`.
 *
 * The composition is the experiment. `learned` names the families the learner
 * has already been trained on, so it becomes the mastered case; the frontier is
 * the family currently being taught; noise is always present as the control the
 * paper's argument turns on.
 */
/**
 * The scored pool: every structured family the curriculum can move through,
 * plus the noise control.
 *
 * Order matters and is not arbitrary. `noise` is LAST so its index is
 * `pool.length - 1`, which is what the page's noise comparisons index. The
 * structured families come first in curriculum order, so a moving curriculum
 * that indexes forward walks them in the order a reader would guess.
 */
export function poolFor(noiseSeed = 20260904): SimProgram[] {
  return [
    {
      key: 'arith',
      label: 'Arithmetic, mod 256',
      kind: 'frontier',
      blurb: 'The first family taught. Counting, in the order the paper\u2019s worked example emits.',
      bytes: arithmeticBytes(0),
    },
    {
      key: 'fib',
      label: 'Fibonacci, mod 256',
      kind: 'frontier',
      blurb: 'Two coupled accumulators. Structure the learner has to hold a state for to predict.',
      bytes: fibonacciBytes(1),
    },
    {
      key: 'quad',
      label: 'Quadratic, mod 256',
      kind: 'frontier',
      blurb: 'Squares. Locally as regular as arithmetic, but the second difference is what carries it.',
      bytes: quadraticBytes(3),
    },
    {
      key: 'geom',
      label: 'Geometric, mod 256',
      kind: 'frontier',
      blurb: 'Doubles each step and wraps at 256. Regular until the modulus breaks it.',
      bytes: geometricBytes(2),
    },
    {
      key: 'noise',
      label: 'Uniform random bytes',
      kind: 'noise',
      blurb: 'The control. Large gradient, no alignment with where the learner is going \u2014 this is what the reward is supposed to refuse to pay for.',
      bytes: noiseBytes(noiseSeed),
    },
  ]
}

/** Index of the noise control in a pool built by `poolFor`. */
export function noiseIndex(pool: SimProgram[]): number {
  return pool.findIndex((p) => p.kind === 'noise')
}

/** The pool for the degenerate single-family curriculum: arithmetic only. */
export function singleFamilyPool(noiseSeed = 20260904): SimProgram[] {
  return poolFor(noiseSeed)
}

export const ARITHMETIC = arithmeticBytes
export const FIBONACCI = fibonacciBytes
export const QUADRATIC = quadraticBytes
export const GEOMETRIC = geometricBytes
export const NOISE = noiseBytes