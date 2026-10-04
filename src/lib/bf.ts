/**
 * The Brainfuck machine, as an explicit state machine.
 *
 * The page used to run a whole program in a single while-loop, which is the
 * right shape for the question "what does this program emit" and the wrong
 * shape for "show me this program running". So the one instruction is pulled
 * out as stepMachine and both callers sit on top of it: the run-to-completion
 * loop and the explorer's stepper. They cannot disagree, because there is only
 * one implementation of what a `>` does.
 *
 * The two callers do differ in what they promise. runBrainfuck answers with a
 * finished run -- T bytes, zero-padded, plus the halted/stepLimited/unmatched
 * flags the page's prose reasons about -- and its behaviour is pinned by
 * src/lib/bf.test.ts. The stepper deals in raw MachineState and lets the caller
 * decide what "done" means and how much of the tape to draw.
 */
import type { BfMacro } from '@/data/types'

/** Circular tape. The paper's cell modulus is 256; the tape is our choice. */
export const TAPE = 256
/** Guard against a program that loops without emitting. */
export const MAX_STEPS = 300_000
/** Characters rendered in the tape strip. */
export const WINDOW = 32

export interface BfRun {
  /** Exactly `T` bytes: the emitted prefix, zero-padded. */
  out: number[]
  /**
   * Bytes the program actually emitted, before zero-padding. `out.length` is
   * always T, so it cannot answer "how much did this program produce"; this
   * can. Equals `out.length` for a program that fills the window.
   */
  emitted: number
  tape: number[]
  head: number
  steps: number
  halted: boolean
  /** True when the step guard stopped the program before T bytes came out. */
  stepLimited: boolean
  expanded: string
  unmatched: number
}

/**
 * Everything one instruction needs, and nothing else. The tape is the only
 * large field, which is why copying it is worth avoiding (see stepMachine).
 */
export interface MachineState {
  tape: number[]
  /** Cell index under the head. Always in [0, TAPE). */
  head: number
  /** Index of the next instruction in `BfProgram.code`. */
  pc: number
  /** Emitted bytes so far, unpadded. */
  out: number[]
  /** Instructions executed. The step guard counts these. */
  steps: number
  /** LCG state for `,`. Kept in the state so a stepper cannot desynchronise it. */
  rng: number
}

/**
 * A program compiled once and then stepped as many times as the reader likes.
 *
 * The source map exists for one reason: the ProgramView shows the macro
 * program the reader typed, while the machine executes the expansion. Without
 * a map, a cursor at pc=11 in a program that expanded to 40 characters has
 * nowhere to point.
 */
export interface BfProgram {
  /** Macro tokens expanded to plain Brainfuck. */
  code: string
  /** `[` <-> `]` index pairs into `code`. Unmatched brackets are absent. */
  pairs: Map<number, number>
  /** Brackets that matched nothing, in either direction. They are no-ops. */
  unmatched: number
  /** For each character of `code`, the index of the source char it came from. */
  sourceMap: number[]
}

/**
 * Expand the macro alphabet into plain Brainfuck. Table 4, verbatim.
 *
 * Kept as its own function because the page quotes the expansion length, and
 * because a caller that only wants the text should not have to think about the
 * source map it does not need.
 */
export function expandMacros(src: string, macros: BfMacro[]): string {
  return expandWithMap(src, macros).code
}

/**
 * Expand, and record provenance as we go. Char-by-char rather than by counting
 * expansion lengths afterwards so the two can never fall out of step.
 */
function expandWithMap(
  src: string, macros: BfMacro[],
): { code: string; sourceMap: number[] } {
  const table = new Map(macros.map((m) => [m.token, m.expansion]))
  let code = ''
  const sourceMap: number[] = []
  let i = 0
  for (const ch of src) {
    const expansion = table.get(ch) ?? ch
    code += expansion
    for (let k = 0; k < expansion.length; k += 1) sourceMap.push(i)
    i += 1
  }
  return { code, sourceMap }
}

/** Bracket pairs. Unmatched brackets are left unmapped and become no-ops. */
function matchBrackets(code: string): { pairs: Map<number, number>; unmatched: number } {
  const open: number[] = []
  const pairs = new Map<number, number>()
  for (let i = 0; i < code.length; i += 1) {
    const c = code[i]
    if (c === '[') open.push(i)
    else if (c === ']') {
      const j = open.pop()
      if (j !== undefined) {
        pairs.set(j, i)
        pairs.set(i, j)
      }
    }
  }
  // `open` holds brackets that opened but never closed; `closePairs` counts
  // closers that found an empty stack. Both become no-ops at runtime.
  return { pairs, unmatched: open.length + closePairs(code) }
}

/** Number of `]` that close nothing, i.e. that never had an open `[`. */
function closePairs(code: string): number {
  let depth = 0
  let n = 0
  for (const c of code) {
    if (c === '[') depth += 1
    else if (c === ']') {
      if (depth > 0) depth -= 1
      else n += 1
    }
  }
  return n
}

/**
 * A seeded LCG for `,`. Deterministic so the reader can reproduce a byte; the
 * paper only says the byte is uniform, not how it is drawn.
 */
function nextRandom(state: number): [number, number] {
  const s = (state * 1664525 + 1013904223) >>> 0
  return [s, (s >>> 16) & 255]
}

/** Macro expansion plus the bracket table, computed once per program edit. */
export function compileProgram(src: string, macros: BfMacro[]): BfProgram {
  const { code, sourceMap } = expandWithMap(src, macros)
  const { pairs, unmatched } = matchBrackets(code)
  return { code, pairs, unmatched, sourceMap }
}

/** The machine before the first instruction. A seed of 0 is coerced to 1. */
export function initialState(seed: number): MachineState {
  return {
    tape: new Array<number>(TAPE).fill(0),
    head: 0,
    pc: 0,
    out: [],
    steps: 0,
    rng: (seed >>> 0) || 1,
  }
}

/**
 * Execute exactly one instruction and return the next state.
 *
 * Copy-on-write for the tape: the three instructions that can change a cell
 * (`+`, `-` and `,`) allocate a fresh array, and the five that cannot share the
 * old one. A full run is allowed 300k instructions, and copying 256 numbers on
 * each of them would dominate the run for no benefit -- but a stepper that
 * aliased its tape would quietly rewrite the state the reader is looking at,
 * which is the bug this avoids.
 *
 * Anything that is not one of the eight primitives is a no-op that still costs
 * a step, because the released programs are newline-separated rows and a step
 * count that skipped them would not mean what the page says it means.
 */
export function stepMachine(prog: BfProgram, state: MachineState): MachineState {
  const cell = state.tape[state.head] ?? 0
  let tape = state.tape
  let out = state.out
  let head = state.head
  let pc = state.pc
  let rng = state.rng

  const write = (v: number): void => {
    tape = state.tape.slice()
    tape[state.head] = v
  }

  const c = prog.code[state.pc]
  if (c === '>') { head = (head + 1) % TAPE; pc += 1 }
  else if (c === '<') { head = (head - 1 + TAPE) % TAPE; pc += 1 }
  else if (c === '+') { write((cell + 1) % 256); pc += 1 }
  else if (c === '-') { write((cell + 255) % 256); pc += 1 }
  else if (c === '.') { out = [...state.out, cell]; pc += 1 }
  else if (c === ',') {
    const r = nextRandom(rng)
    rng = r[0]
    write(r[1])
    pc += 1
  } else if (c === '[') { pc = cell === 0 ? ((prog.pairs.get(pc) ?? pc) + 1) : pc + 1 } else if (c === ']') { const open = prog.pairs.get(pc); pc = open !== undefined && cell !== 0 ? open + 1 : pc + 1 } else { pc += 1 }

  return { tape, head, pc, out, steps: state.steps + 1, rng }
}

/**
 * Done means: ran off the end of the program, filled the output window, or ran
 * out of step budget. The first two are the program's own doing; the third is
 * this page's doing, which is why the finished run reports them separately.
 */
export function isDone(prog: BfProgram, state: MachineState, t: number): boolean {
  return state.pc >= prog.code.length || state.out.length >= t || state.steps >= MAX_STEPS
}

/** Step until `isDone`, or until `max` instructions have been spent. */
export function drive(
  prog: BfProgram, state: MachineState, t: number, max = MAX_STEPS,
): MachineState {
  let s = state
  let spent = 0
  while (!isDone(prog, s, t) && spent < max) {
    s = stepMachine(prog, s)
    spent += 1
  }
  return s
}

/**
 * The source character the cursor sits on, for the ProgramView to highlight.
 * Null when the machine has run off the end, where there is no instruction to
 * point at -- pointing at the last character would read as "still executing".
 */
export function sourceIndexOfPc(prog: BfProgram, pc: number): number | null {
  if (pc < 0 || pc >= prog.code.length) return null
  return prog.sourceMap[pc] ?? null
}

/** Exactly T bytes: the emitted prefix, zero-padded. */
export function padOutput(out: number[], t: number): number[] {
  const padded = out.slice(0, t)
  while (padded.length < t) padded.push(0)
  return padded
}

/** Shape a raw state into the finished-run record the page renders. */
export function finalize(prog: BfProgram, state: MachineState, t: number): BfRun {
  return {
    out: padOutput(state.out, t),
    emitted: Math.min(state.out.length, t),
    tape: state.tape,
    head: state.head,
    steps: state.steps,
    halted: state.pc >= prog.code.length,
    stepLimited: state.out.length < t && state.steps >= MAX_STEPS,
    expanded: prog.code,
    unmatched: prog.unmatched,
  }
}

/**
 * Run a program to completion. Signature and behaviour are the page's original
 * contract and bf.test.ts pins them against a transcription of the pre-refactor
 * loop; the stepper is the new thing, not a change to this.
 */
export function runBrainfuck(
  src: string, t: number, seed: number, macros: BfMacro[],
): BfRun {
  const prog = compileProgram(src, macros)
  return finalize(prog, drive(prog, initialState(seed), t), t)
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

export interface Preset {
  label: string
  program: string
  seed: number
  /** What the paper says this family emits. */
  paperTerms: string
  reference: 'sec2.1' | 'table1'
  note: string
}

/**
 * The paper's own example programs. Seeds are chosen so that `,` draws the byte
 * the paper's stated term list starts from; they are ours, not the paper's.
 *
 * These live beside the machine rather than in the page because they are the
 * differential test's fixture: bf.test.ts runs each of them through the stepper
 * and through runBrainfuck and requires the two to agree byte for byte. A
 * preset that stopped being executable would be a claim on the page with
 * nothing behind it.
 */
export const PRESETS: Preset[] = [
  {
    label: 'Our worked example',
    program: '+++[>+.<-]',
    seed: 1,
    paperTerms: '1, 2, 3, 0, 0, 0, …',
    reference: 'sec2.1',
    note:
      'Ten characters. The head sweeps right three times, and the cell it left '
      + 'behind is what gets emitted, so the emitted bytes count up to three '
      + 'and the program halts in 22 steps.',
  },
  {
    label: 'Arithmetic, mod 256',
    program: '+[.++]',
    seed: 1,
    paperTerms: '1, 3, 5, 7, 9, …',
    reference: 'table1',
    note:
      'The loop never halts: each pass emits, then adds two to the cell it is '
      + 'reading. Output length is bounded only by the context window T.',
  },
  {
    label: 'Geometric, mod 256',
    program: '+[.L>]',
    seed: 1,
    paperTerms: '1, 3, 9, 27, 81, …',
    reference: 'table1',
    note:
      'Six characters, using the L macro. Each pass clears the cell, triples it '
      + 'into its right neighbour and moves the head there. At 243 the sequence '
      + 'wraps mod 256 and stops looking like a geometric progression — which is '
      + 'exactly the arithmetical structure the paper scores.',
  },
  {
    label: 'Fibonacci, mod 256',
    program: ',[[.C>.C>]',
    seed: 36,
    paperTerms: '1, 1, 2, 3, 5, 8, …',
    reference: 'table1',
    note:
      'Two accumulator cells, written together by the C macro. The first byte '
      + 'drawn is whatever `,` returns, so the whole sequence scales with it.',
  },
  {
    label: 'Quadratic — read the caveat',
    program: ',.[<C>>VX<RX++]',
    seed: 248,
    paperTerms: '9, 25, 59, 111, …',
    reference: 'table1',
    note:
      'This one does not reproduce under the macro table as shipped. See the '
      + 'caveat below the program-space table.',
  },
]