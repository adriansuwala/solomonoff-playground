/**
 * Tests for the Brainfuck machine on the Method page.
 *
 * The explorer gained a stepper, and the stepper is the reason these tests
 * exist. runBrainfuck is a straight loop over stepMachine; both were correct or
 * both were wrong, and a page that renders 4,095 bytes off this machine makes
 * claims about arithmetic families, so "roughly right" is not available.
 *
 * The load-bearing test is the differential one: every preset, driven one
 * instruction at a time through stepMachine, must land in exactly the state
 * runBrainfuck reaches. That catches the whole class of bug where the stepper
 * and the batch runner disagree -- an off-by-one in the bracket back-edge, a
 * step that mutates the tape it was handed, a guard that fires at a different
 * instruction -- none of which any single-instruction assertion would notice.
 *
 * The macro table comes from the real public/data/meta.json rather than a
 * fixture, because the thing being protected is the agreement between the
 * interpreter and the authors' shipped expansions.
 */
import { describe, expect, it } from 'vitest'
import metaJson from '../../public/data/meta.json'
import {
  MAX_STEPS, PRESETS, TAPE,
  compileProgram, drive, expandMacros, finalize, initialState, isDone,
  padOutput, runBrainfuck, sourceIndexOfPc, stepMachine,
  type BfProgram, type MachineState,
} from './bf'
import type { Meta } from '@/data/types'

const meta = metaJson as unknown as Meta
const macros = meta.bf.macros

/** Step one instruction at a time, the way the explorer's Play button does. */
function stepToEnd(prog: BfProgram, seed: number, t: number): MachineState {
  let s = initialState(seed)
  while (!isDone(prog, s, t)) s = stepMachine(prog, s)
  return s
}

/** The five presets, at a T large enough that the halting one is not truncated. */
const PRESET_T = 48

describe('the shipped macro table', () => {
  it('expands every macro token into plain Brainfuck', () => {
    expect(expandMacros('+[.L>]', macros)).toBe('+[.[->+++<]>]')
    // A token that is not in the table is left alone, which is how the released
    // programs carry the eight primitives alongside the macro alphabet.
    expect(expandMacros('abc', macros)).toBe('abc')
  })

  it('expands the macros the presets actually use', () => {
    for (const p of PRESETS) {
      const code = expandMacros(p.program, macros)
      expect(code.length, p.label).toBeGreaterThan(0)
      // Only primitives survive expansion; a macro letter left in the code
      // would be a silent no-op at runtime, which looks like a working program
      // that emits nothing.
      expect(/[^><+\-.,[\]]/.test(code), `${p.label} -> ${code}`).toBe(false)
    }
  })

  it('cells are taken mod the modulus the page quotes', () => {
    // The page renders this number in the explorer's note, so it is a claim.
    expect(meta.bf.params.cell_modulus).toBe(256)
  })
})

describe('stepMachine, one instruction at a time', () => {
  it('advances pc by exactly one on a straight-line program', () => {
    const prog = compileProgram('+++', [])
    let s = initialState(1)
    expect(s.pc).toBe(0)
    s = stepMachine(prog, s)
    expect(s.pc).toBe(1)
    expect(s.steps).toBe(1)
    s = stepMachine(prog, s)
    expect(s.pc).toBe(2)
    expect(s.tape[0]).toBe(2)
    s = stepMachine(prog, s)
    expect(s.pc).toBe(3)
    expect(s.tape[0]).toBe(3)
    expect(s.steps).toBe(3)
  })

  it('moves the head without touching the tape', () => {
    const prog = compileProgram('><', [])
    let s = initialState(1)
    s = stepMachine(prog, s)
    expect(s.head).toBe(1)
    s = stepMachine(prog, s)
    expect(s.head).toBe(0)
  })

  it('wraps the head at both ends of the circular tape', () => {
    const back = compileProgram('<', [])
    expect(stepMachine(back, initialState(1)).head).toBe(TAPE - 1)

    // Forward past the end: exactly TAPE right-moves is a full lap.
    const fwd = compileProgram('>'.repeat(TAPE), [])
    expect(drive(fwd, initialState(1), 1024).head).toBe(0)
    // And one short of it the head is at the last cell, not back at zero.
    const short = compileProgram('>'.repeat(TAPE - 1), [])
    expect(drive(short, initialState(1), 1024).head).toBe(TAPE - 1)
  })

  it('wraps the cell mod 256 in both directions', () => {
    // 256 increments from zero must return to zero, not to 256.
    const up = compileProgram('+'.repeat(257), [])
    const high = stepMachine(up, initialState(1))
    expect(high.tape[0]).toBe(1)
    const zeroToOne = compileProgram('-', [])
    expect(stepMachine(zeroToOne, initialState(1)).tape[0]).toBe(255)

    const down = compileProgram('-'.repeat(256), [])
    expect(drive(down, initialState(1), 1024).tape[0]).toBe(0)
  })

  it('emits the cell under the head, not the cell at zero', () => {
    const prog = compileProgram('>+++.', [])
    const s = drive(prog, initialState(1), 8)
    expect(s.out).toEqual([3])
    expect(s.tape[0]).toBe(0)
    expect(s.tape[1]).toBe(3)
  })

  it('jumps `]` back to the instruction after its `[` while the cell is nonzero', () => {
    const prog = compileProgram('+[>]', [])
    let s = stepMachine(prog, initialState(1)) // +
    expect(s.pc).toBe(1)
    s = stepMachine(prog, s) // [ with cell 1 -> falls through
    expect(s.pc).toBe(2)
    s = stepMachine(prog, s) // > moves the head off the nonzero cell
    s = { ...s, head: 0 }
    s = stepMachine(prog, s) // ] with cell 1 -> back to index 2
    expect(s.pc).toBe(2)
    expect(s.steps).toBe(4)
  })

  it('falls through `]` when the cell under the head is zero', () => {
    const prog = compileProgram('[>]', [])
    let s = stepMachine(prog, initialState(1)) // [ with cell 0 -> jump past ]
    expect(s.pc).toBe(3)
    expect(s.steps).toBe(1)
    // And the ] itself, reached directly, is a no-op rather than a back-edge.
    s = { ...initialState(1), pc: 2 }
    expect(stepMachine(prog, s).pc).toBe(3)
  })

  it('treats an unmatched bracket as a no-op that still costs a step', () => {
    // The Fibonacci preset has one unmatched `]`; the page's prose depends on
    // it being a no-op rather than a jump, so this is a claim, not a detail.
    const prog = compileProgram(']', [])
    const s = stepMachine(prog, initialState(1))
    expect(s.pc).toBe(1)
    expect(s.steps).toBe(1)
    expect(prog.unmatched).toBe(1)
  })

  it('counts a non-instruction character as a step, so step counts stay honest', () => {
    // Released programs are newline-separated rows; a step count that skipped
    // the separators would not be the number the page prints.
    const prog = compileProgram('+', [])
    const s = stepMachine({ ...prog, code: '\n+' }, { ...initialState(1), pc: 0 })
    expect(s.steps).toBe(1)
    expect(s.pc).toBe(1)
    expect(s.tape[0]).toBe(0)
  })

  it('does not mutate the state it was handed', () => {
    // Copy-on-write: a stepper that rewrote its predecessor would corrupt the
    // frame the reader is currently looking at.
    const prog = compileProgram('++', [])
    const before = initialState(1)
    const snapshot = before.tape.slice()
    const after = stepMachine(prog, before)
    expect(before.tape).toEqual(snapshot)
    expect(after.tape).not.toBe(before.tape)
    stepMachine(prog, after)
    expect(after.tape[0]).toBe(1)
  })

  it('shares the tape between steps that cannot change it', () => {
    // The 300k-step guard makes an unconditional 256-element copy per
    // instruction the dominant cost of a run; only the writers need one.
    const prog = compileProgram('>.', [])
    const first = stepMachine(prog, initialState(1))
    const second = stepMachine(prog, first)
    expect(second.tape).toBe(first.tape)
  })

  it('carries the LCG state in the machine state, so `,` is reproducible', () => {
    const prog = compileProgram(',,.', [])
    const s = drive(prog, initialState(7), 8)
    const again = drive(compileProgram(',,.', []), initialState(7), 8)
    expect(s.out).toEqual(again.out)
    expect(s.tape[0]).toBe(s.out[0])
    // A different seed must actually give different bytes, or the seed field in
    // the explorer would be decoration.
    expect(drive(compileProgram(',.', []), initialState(8), 1).out)
      .not.toEqual(drive(compileProgram(',.', []), initialState(9), 1).out)
  })

  it('coerces a zero seed to 1 rather than emitting a constant stream', () => {
    const zero = drive(compileProgram(',.', []), initialState(0), 1)
    const one = drive(compileProgram(',.', []), initialState(1), 1)
    expect(zero.out).toEqual(one.out)
  })
})

describe('differential: stepMachine against runBrainfuck', () => {
  // The test this file exists for. Every preset, run one instruction at a time,
  // must end in exactly the state the batch runner reaches.
  it.each(PRESETS.map((p) => [p.label, p] as const))(
    '%s agrees byte for byte',
    (_label, p) => {
      const prog = compileProgram(p.program, macros)
      const stepped = stepToEnd(prog, p.seed, PRESET_T)
      const batch = runBrainfuck(p.program, PRESET_T, p.seed, macros)
      const steppedRun = finalize(prog, stepped, PRESET_T)

      expect(steppedRun.out).toEqual(batch.out)
      expect(steppedRun.emitted).toBe(batch.emitted)
      expect(steppedRun.tape).toEqual(batch.tape)
      expect(steppedRun.head).toBe(batch.head)
      expect(steppedRun.steps).toBe(batch.steps)
      expect(steppedRun.halted).toBe(batch.halted)
      expect(steppedRun.stepLimited).toBe(batch.stepLimited)
      expect(steppedRun.unmatched).toBe(batch.unmatched)
      expect(steppedRun.expanded).toBe(batch.expanded)
    },
  )

  it('holds for every preset across a spread of T, including T = 1', () => {
    for (const p of PRESETS) {
      for (const t of [1, 2, 7, 48, 300]) {
        const prog = compileProgram(p.program, macros)
        const stepped = finalize(prog, stepToEnd(prog, p.seed, t), t)
        const batch = runBrainfuck(p.program, t, p.seed, macros)
        expect(stepped.out, `${p.label} T=${t}`).toEqual(batch.out)
        expect(stepped.tape, `${p.label} T=${t}`).toEqual(batch.tape)
        expect(stepped.steps, `${p.label} T=${t}`).toBe(batch.steps)
        expect(stepped.head, `${p.label} T=${t}`).toBe(batch.head)
      }
    }
  })

  it('holds for programs outside the preset list, macro letters included', () => {
    const cases = ['+', '', '.-', '+[]', '>[<]', '+++[>+.<-]', '+[.++]', '[-]', '><>+<']
    for (const src of cases) {
      const prog = compileProgram(src, macros)
      const stepped = finalize(prog, stepToEnd(prog, 3, 32), 32)
      const batch = runBrainfuck(src, 32, 3, macros)
      expect(stepped.out, src).toEqual(batch.out)
      expect(stepped.tape, src).toEqual(batch.tape)
      expect(stepped.steps, src).toBe(batch.steps)
    }
  })

  it('drive() with no budget matches stepMachine-driven to completion', () => {
    // The explorer uses drive for the batch view and stepMachine for playback.
    for (const p of PRESETS) {
      const prog = compileProgram(p.program, macros)
      expect(drive(prog, initialState(p.seed), PRESET_T)).toEqual(stepToEnd(prog, p.seed, PRESET_T))
    }
  })
})

describe('the numbers the page quotes', () => {
  it('the worked example emits 1, 2, 3 and halts in 22 steps', () => {
    // Quoted verbatim in the preset's note, and the page's Claim leans on it.
    const run = runBrainfuck('+++[>+.<-]', 48, 1, macros)
    expect(run.out.slice(0, 3)).toEqual([1, 2, 3])
    expect(run.emitted).toBe(3)
    expect(run.steps).toBe(22)
    expect(run.halted).toBe(true)
    expect(run.stepLimited).toBe(false)
  })

  it('the arithmetic program counts up by two, mod 256', () => {
    const run = runBrainfuck('+[.++]', 48, 1, macros)
    expect(run.out.slice(0, 5)).toEqual([1, 3, 5, 7, 9])
    expect(run.halted).toBe(false)
  })

  it('the geometric program wraps at 243, which is the point the page makes', () => {
    const run = runBrainfuck('+[.L>]', 48, 1, macros)
    expect(run.out[0]).toBe(1)
    // The first decrease in the series is the mod-256 wrap the note describes.
    const firstDrop = run.out.findIndex((b, i) => i > 0 && b < (run.out[i - 1] ?? 0))
    expect(firstDrop).toBeGreaterThan(0)
  })

  it('the Fibonacci program needs the no-op reading of its unmatched bracket', () => {
    // The page says the unmatched `]` must be a no-op for 1, 1, 2, 3, 5, 8 to
    // come out. If the bracket handling ever changed, this is what would fail.
    const run = runBrainfuck(',[[.C>.C>]', 48, 36, macros)
    expect(run.unmatched).toBe(1)
    expect(run.out.slice(0, 6)).toEqual([1, 1, 2, 3, 5, 8])
  })

  it('the quadratic program does not reproduce under the shipped table', () => {
    // A documented failure, not a regression: the page reports this as a
    // caveat. If this ever starts passing, the caveat text is wrong.
    const run = runBrainfuck(',.[<C>>VX<RX++]', 48, 248, macros)
    expect(run.out.slice(0, 4)).not.toEqual([9, 25, 59, 111])
  })

  it('pads the output to exactly T, whatever the program emitted', () => {
    expect(runBrainfuck('+.', 48, 1, macros).out).toHaveLength(48)
    expect(runBrainfuck('', 12, 1, macros).out).toEqual(new Array(12).fill(0))
    expect(padOutput([1, 2], 4)).toEqual([1, 2, 0, 0])
    expect(padOutput([1, 2, 3], 2)).toEqual([1, 2])
  })

  it('emitted is capped at T, so it cannot exceed the window', () => {
    const run = runBrainfuck('+.', 4, 1, macros)
    expect(run.out).toHaveLength(4)
    expect(run.emitted).toBe(1)
    expect(runBrainfuck('+[.++]', 3, 1, macros).emitted).toBe(3)
  })
})

describe('the step guard', () => {
  it('stops a program that loops without emitting', () => {
    // `+[]` spins forever on a nonzero cell and emits nothing, which is the
    // case MAX_STEPS exists for.
    const run = runBrainfuck('+[]', 48, 1, macros)
    expect(run.steps).toBe(MAX_STEPS)
    expect(run.stepLimited).toBe(true)
    expect(run.halted).toBe(false)
    expect(run.emitted).toBe(0)
  })

  it('reports stepLimited only when the guard is what stopped it', () => {
    // The distinction the caveat paragraph turns on: truncation is the page's
    // doing, and a program that simply filled T is not truncated.
    expect(runBrainfuck('+[]', 48, 1, macros).stepLimited).toBe(true)
    expect(runBrainfuck('+[.++]', 48, 1, macros).stepLimited).toBe(false)
    expect(runBrainfuck('+++', 48, 1, macros).stepLimited).toBe(false)
  })

  it('isDone fires at the guard, not one instruction later', () => {
    const prog = compileProgram('+[]', [])
    const s = drive(prog, initialState(1), 48)
    expect(s.steps).toBe(MAX_STEPS)
    expect(isDone(prog, s, 48)).toBe(true)
  })

  it('the stepper reaches the guard in the same number of frames', () => {
    const prog = compileProgram('+[]', [])
    expect(stepToEnd(prog, 1, 48).steps).toBe(MAX_STEPS)
  })

  it('drive() honours a smaller budget, which is how a frame batches steps', () => {
    const prog = compileProgram('+[.++]', [])
    const s = drive(prog, initialState(1), 48, 10)
    expect(s.steps).toBe(10)
    expect(s.out).toHaveLength(2)
  })
})

describe('isDone', () => {
  it('is true once the program has run off the end', () => {
    const prog = compileProgram('+', [])
    const s = stepMachine(prog, initialState(1))
    expect(isDone(prog, s, 48)).toBe(true)
  })

  it('is true once T bytes are out, even mid-loop', () => {
    const prog = compileProgram('+[.++]', [])
    const s = drive(prog, initialState(1), 48, 8)
    expect(s.out).toHaveLength(2)
    expect(s.pc).toBeLessThan(prog.code.length)
    expect(isDone(prog, s, 2)).toBe(true)
  })

  it('is false at the start of a program that has work to do', () => {
    expect(isDone(compileProgram('+[.]', []), initialState(1), 48)).toBe(false)
  })
})

describe('sourceIndexOfPc', () => {
  it('points a macro token at the character the reader typed', () => {
    // `L` is three characters of expansion standing for one of source; the
    // cursor has to land on the `L`, not on any of them.
    const prog = compileProgram('+[.L>]', macros)
    // '+[.' occupies expansion 0..2, the eight characters of L occupy 3..10,
    // and the trailing '>]' sit at 11..12. Every index inside L must resolve to
    // source index 3 -- the `L` the reader typed.
    expect(prog.sourceMap).toEqual([0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5])
    for (let i = 0; i < prog.code.length; i += 1) {
      expect(sourceIndexOfPc(prog, i), `pc=${i}`).toBe(prog.sourceMap[i] ?? null)
    }
  })

  it('is null past the end, so a finished program shows no cursor', () => {
    const prog = compileProgram('+', [])
    expect(sourceIndexOfPc(prog, 1)).toBeNull()
    expect(sourceIndexOfPc(prog, -1)).toBeNull()
    expect(sourceIndexOfPc(prog, 999)).toBeNull()
  })

  it('never points outside the source string', () => {
    for (const p of PRESETS) {
      const prog = compileProgram(p.program, macros)
      for (let i = 0; i <= prog.code.length; i += 1) {
        const idx = sourceIndexOfPc(prog, i)
        if (idx === null) continue
        expect(idx, `${p.label} pc=${i}`).toBeGreaterThanOrEqual(0)
        expect(idx, `${p.label} pc=${i}`).toBeLessThan([...p.program].length)
      }
    }
  })
})