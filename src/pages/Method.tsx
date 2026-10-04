/**
 * Method: how a round of self-play pretraining actually works.
 *
 * The centrepiece is a Brainfuck interpreter written here in TypeScript, run
 * against the paper's own program-space definition (Appendix E / Table 4). The
 * point is not to reimplement the paper's training loop -- it is to let the
 * reader watch the premise hold: a program of 9 characters emits an unbounded
 * structured sequence, and that is the only kind of data the model ever sees.
 *
 * Macro expansions come from meta.bf.macros (the authors' table, verbatim).
 * Where the shipped expansion contradicts its own stated effect -- it does, for
 * `X` -- the interpreter uses the shipped expansion and the page says so.
 * See the caveat under the program-space table.
 */
import { useMemo, useState } from 'react'
import { loadMeta } from '@/data/loaders'
import type { BfMacro, Meta } from '@/data/types'
import {
  Async, Card, Claim, Disclosure, Legend, PaperRef, Stat,
} from '@/components/UI'
import { ChartFrame, Series } from '@/components/Chart'

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

/** Circular tape. The paper's cell modulus is 256; the tape is our choice. */
const TAPE = 256
/** Guard against a program that loops without emitting. */
const MAX_STEPS = 300_000
/** Characters rendered in the tape strip. */
const WINDOW = 32

export interface BfRun {
  /** Exactly `T` bytes: the emitted prefix, zero-padded. */
  out: number[]
  tape: number[]
  head: number
  steps: number
  halted: boolean
  /** True when the step guard stopped the program before T bytes came out. */
  stepLimited: boolean
  expanded: string
  unmatched: number
}

/** Expand the macro alphabet into plain Brainfuck. Table 4, verbatim. */
export function expandMacros(src: string, macros: BfMacro[]): string {
  const table = new Map(macros.map((m) => [m.token, m.expansion]))
  let out = ''
  for (const ch of src) out += table.get(ch) ?? ch
  return out
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

export function runBrainfuck(
  src: string, t: number, seed: number, macros: BfMacro[],
): BfRun {
  const expanded = expandMacros(src, macros)
  const { pairs, unmatched } = matchBrackets(expanded)
  const code = expanded
  const tape = new Array<number>(TAPE).fill(0)
  const out: number[] = []
  let head = 0
  let pc = 0
  let steps = 0
  let rng = (seed >>> 0) || 1

  while (pc < code.length && out.length < t && steps < MAX_STEPS) {
    const c = code[pc]
    if (c === '>') { head = (head + 1) % TAPE; pc += 1 } else if (c === '<') { head = (head - 1 + TAPE) % TAPE; pc += 1 } else if (c === '+') { tape[head] = ((tape[head] ?? 0) + 1) % 256; pc += 1 } else if (c === '-') { tape[head] = ((tape[head] ?? 0) + 255) % 256; pc += 1 } else if (c === '.') { out.push(tape[head] ?? 0); pc += 1 } else if (c === ',') { const r = nextRandom(rng); rng = r[0]; tape[head] = r[1]; pc += 1 } else if (c === '[') { pc = tape[head] === 0 ? ((pairs.get(pc) ?? pc) + 1) : pc + 1 } else if (c === ']') { const open = pairs.get(pc); pc = open !== undefined && tape[head] !== 0 ? open + 1 : pc + 1 } else { pc += 1 }
    steps += 1
  }

  const padded = out.slice(0, t)
  while (padded.length < t) padded.push(0)

  return {
    out: padded,
    tape,
    head,
    steps,
    halted: pc >= code.length,
    stepLimited: out.length < t && steps >= MAX_STEPS,
    expanded: code,
    unmatched,
  }
}



// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

interface Preset {
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
 */
const PRESETS: Preset[] = [
  {
    label: 'Paper’s §2.1 example',
    program: '+++[>+.<-]',
    seed: 1,
    paperTerms: '1, 2, 3, 0, 0, 0, …',
    reference: 'sec2.1',
    note:
      'Nine characters. The head sweeps right three times, and the cell it left '
      + 'behind is what gets emitted, so the output counts back down to zero and '
      + 'the program halts in 22 steps.',
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
      'Two accumulator cells, kept one apart by the C macro. The first byte '
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

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export function Method() {
  return (
    <Async load={loadMeta}>
      {(meta) => <MethodBody meta={meta} />}
    </Async>
  )
}

/**
 * Bytes a program actually emits. The context is 4096, but one of those bytes
 * is the `O` output prefix, so the tape holds 4095 corpus bytes -- which is the
 * `tape_len` every one of the 20,045 records in the release carries, and what
 * scoring/README.md describes. Quoting the context as the emitted length was
 * off by one on a page whose whole point is byte-level precision.
 */
function tapeLen(meta: Meta): number {
  return meta.compute.context - 1
}

function MethodBody({ meta }: { meta: Meta }) {
  const TAPE_LEN = tapeLen(meta)
  const { primitives, macros, params } = meta.bf

  return (
    <>
      <h1 className="page__title">Method</h1>
      <p className="page__lede">
        One round of self-play pretraining is three things: a generator writes a
        program, a universal Turing machine executes it into bytes, and a
        learner is trained on those bytes. The generator is then paid according
        to how much the learner actually moved on them. This page walks the loop,
        the program space, and the objectives — and gives you a working
        interpreter for the machine at the centre of it.
      </p>

      <h2 className="section">The round</h2>
      <p className="body">
        Both networks start from random initialisation. In each round the
        generator emits a pool of {meta.compute.pool.toLocaleString()} programs;
        each program is executed on the machine to produce a length-{TAPE_LEN.toLocaleString()}
        byte sequence; the learner takes gradient steps on those sequences; and
        the generator takes a reinforcement-learning step whose reward is a
        function of the learner’s progress on them. Nothing in that loop touches
        natural data. The tokens consumed per round are what the scaling axis on
        the Scaling page is built from, <code>{meta.compute.formula}</code>.
      </p>

      <ol className="body" style={{ maxWidth: '68ch' }}>
        <li>
          <strong>Sample.</strong> The generator writes{' '}
          {meta.compute.pool.toLocaleString()} programs from its learned
          distribution over the program alphabet.
        </li>
        <li>
          <strong>Execute.</strong> Each program runs on the universal machine
          for up to {TAPE_LEN.toLocaleString()} emitted bytes.
        </li>
        <li>
          <strong>Train the learner.</strong> One gradient step per program on
          its emitted bytes (Equation 1).
        </li>
        <li>
          <strong>Pay the generator.</strong> Each program is scored by how far
          the learner moved on it (Equation 2), and the generator takes an RL
          step on that score (Equation 3).
        </li>
      </ol>
      <Claim reference="sec2" also={['sec2.2']}>
        The learner's loss and the generator's reward are separate objectives,
        and the reward is the only place the two models talk to each other.
      </Claim>

      <h2 className="section">The program space</h2>
      <p className="body">
        The program space is Brainfuck: eight primitives over a circular tape of
        byte-valued cells. That choice is the load-bearing one, because a
        Turing-complete space contains every computable sequence, so the ceiling
        on what pretraining can absorb is not the space — it is the learner.
        {' '}
        <PaperRef reference="sec2.1" also={['appE']} inline />
      </p>

      <Card
        title={`The ${primitives.length} primitives`}
        note={`Cells are taken mod ${params.cell_modulus}. The full alphabet the generator samples over is ${params.program_alphabet} — the eight primitives plus the ${macros.length} macro tokens below.`}
      >
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr><th>token</th><th>effect</th></tr>
            </thead>
            <tbody>
              {primitives.map((p) => (
                <tr key={p.token}>
                  <td className="num">{p.token}</td>
                  <td style={{ textAlign: 'left' }}>{p.effect}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="card__note">
          Programs in the data carry an <code>{params.prefix_program}</code>{' '}
          prefix before the program and an <code>{params.prefix_output}</code>{' '}
          prefix before its output; neither is an instruction. Unmatched brackets
          are no-ops, so a random program is almost always still executable.
          {' '}
          <PaperRef reference="appE" also={['table4']} inline />
        </p>
      </Card>

      <Card
        title={`The ${macros.length} macros, with the authors' expansions`}
        note="Table 4. These exist to make arithmetic and scanning expressible in few characters, which is what lets a short program carry a long structured output."
      >
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr><th>token</th><th>expansion</th><th>effect</th></tr>
            </thead>
            <tbody>
              {macros.map((m) => (
                <tr key={m.token}>
                  <td className="num" style={{ color: '#f0883e' }}>{m.token}</td>
                  <td className="num" style={{ textAlign: 'left' }}>{m.expansion}</td>
                  <td style={{ textAlign: 'left' }}>{m.effect}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <Claim caveat reference="table4" also={['appE']}>
        <strong>One macro in the shipped table does not match its own
        description.</strong> <code>X</code> is listed with effect “set cell to
        16”, but its expansion field carries <code>[-]</code> — byte-identical to{' '}
        <code>Z</code> (“clear current cell”). The interpreter below uses the
        expansion as shipped, not the effect as described, because the expansion
        is the authors’ table and the effect is a one-line gloss. The consequence
        is visible: the quadratic program does not reproduce. If{' '}
        <code>X</code> is expanded to <code>[-]</code> followed by sixteen{' '}
        <code>+</code>, <code>,.[&lt;C&gt;&gt;VX&lt;RX++]</code> emits 9, 25, 59,
        111, … — the paper’s terms. We have not changed the shipped data; this is
        reported rather than patched.
      </Claim>

      <h2 className="section">The machine, running in your browser</h2>
      <p className="body">
        Below is the interpreter, written against the semantics above: cells mod{' '}
        {params.cell_modulus}, circular tape, <code>,</code> reads a uniform random
        byte, <code>.</code> emits the cell under the head, unmatched brackets are
        no-ops, and the output is the first T emitted bytes zero-padded. Macro
        tokens are expanded first. The random byte is drawn from a seeded LCG, so
        a given seed always gives the same bytes — the paper specifies uniform,
        not the generator, so the exact stream is ours.
      </p>

      <BfExplorer macros={macros} cellModulus={params.cell_modulus} />

      <Claim reference="sec2.1" also={['appC']}>
        This is the paper’s whole premise, and it is worth being blunt about what
        it does and does not show. A 6-character program emits an unbounded
        geometric progression; a ten-character one emits 1, 2, 3 and halts.
        Finding a short program that does this is not easy by chance: a{' '}
        {meta.mathPrior.totalSamples.toLocaleString()}-program uniform sweep of
        the instruction alphabet found arithmetic families and nothing else,
        which is what makes the self-play generator&rsquo;s hits worth
        reporting. The program space being universal is what
        makes the ceiling high; the RL sampler is what makes the floor reachable.
        The uniform-prior arm in Figure 2 is the evidence that the second half
        matters as much as the first.
      </Claim>

      <h2 className="section">Why a short program can carry a long sequence</h2>
      <p className="body">
        Because the machine is not reading the output — it is being read. A loop
        is a counter, and the counter can be the emitted byte. The program is
        O(1) in characters; the output is O(T). With T ={' '}
        {TAPE_LEN.toLocaleString()}, a handful of characters defines
        {TAPE_LEN.toLocaleString()} bytes of structure, and the
        learner’s job is to discover which few characters those were. That
        inference problem is the pretraining task: compress the mapping from
        program to output, which is a general compression problem with no
        natural data anywhere in it.
      </p>
      <p className="body">
        The interesting failure modes are visible in the explorer above rather
        than hidden. Programs that emit nothing teach nothing. Programs that emit
        random bytes are unlearnable and produce a gradient that leads nowhere.
        Programs sitting at the edge of what the learner can currently absorb are
        the only ones the reward pays for — which is the design the Reward page
        ablates.
      </p>

      <h2 className="section">The objectives</h2>
      <p className="body">
        Four terms, in the order they act. Rendered as text rather than typeset;
        the notation is ours, the structure is theirs.
      </p>

      <Card
        title="Equation 1 — the learner’s loss"
        note="Ordinary next-byte cross-entropy on the emitted sequence. No natural data enters here, at any round."
      >
        <div className="program">
          L(θ) = −(1/T) · Σ[ t = 1..T ] log p_θ( y_t | y_&lt;t )
        </div>
        <p className="body" style={{ marginTop: 10 }}>
          A single gradient step on one program’s bytes. Repeated across the
          pool, this is the whole of the learner’s training.{' '}
          <PaperRef reference="eq1" also={['sec2.2']} inline />
        </p>
      </Card>

      <Card
        title="Equation 2 — the reward"
        note="The magnitude of the preconditioned gradient inner product against the learner’s recent displacement."
      >
        <div className="program">
          r_i = | ⟨ ∇_θ L(y_i; θ_now) , P_e ⊙ δθ_e ⟩ |
          <br />
          δθ_e = θ_⌊e/2⌋ − θ_e  ,  P_e = diag( sqrt(v̂_e) + ε )
        </div>
        <p className="body" style={{ marginTop: 10 }}>
          Read it as: how big a step would this program’s gradient have produced,
          taken in the direction the learner has actually been moving. Mastered
          program → zero gradient → no reward. Unlearnable program → gradient
          pointing nowhere useful → no reward. The preconditioner{' '}
          <code>P</code> weights each coordinate by its own running scale, so a
          rarely-updated parameter can still produce a usable-sized step.{' '}
          <PaperRef reference="eq2" also={['sec2.2', 'appF']} inline />
        </p>
      </Card>

      <Card
        title="Equation 3 — the generator’s RL objective"
        note="KL-regularised policy gradient against a Solomonoff prior, so the generator is pulled toward short programs rather than drifting to whatever is currently paying."
      >
        <div className="program">
          J(φ) = E_x[ (r(x)/β) · log( g_φ(x) / g_0(x) ) ] − β · KL( g_φ ‖ g_0 )
          <br />
          g_0(x) = |A|^( −len(x) )   with |A| = alphabet size
        </div>
        <p className="body" style={{ marginTop: 10 }}>
          <code>g_0</code> is the length-prior over the program alphabet — the
          fixed uniform-prior arm is exactly <code>g_0</code> with the generator
          switched off, which is why Figure 2’s comparison is clean. The KL term
          is what keeps the sampler near the prior instead of collapsing onto a
          single high-reward program.{' '}
          <PaperRef reference="eq3" also={['sec2.2']} inline />
        </p>
      </Card>

      <Card
        title="Equation 4 — the GRPO advantage"
        note="Rewards are normalised within the pool, so the step size does not drift as the reward scale changes across rounds."
      >
        <div className="program">
          Â_i = ( r_i − mean_j r_j ) / ( std_j r_j + ε )
        </div>
        <p className="body" style={{ marginTop: 10 }}>
          Group-relative: the comparison is always against the other programs in
          the same pool, never against an absolute scale.{' '}
          <PaperRef reference="eq4" also={['sec2.2']} inline />
        </p>
      </Card>

      <Card
        title="Equation 5 — expert iteration"
        note="A reward-weighted SFT term on the learner side, pulling the generator toward its own high-reward programs."
      >
        <div className="program">
          L_SFT(φ) = − E_x~g_φ [ w(x) · log g_φ(x) ],  w(x) ∝ max( 0, r(x) − τ )
        </div>
        <p className="body" style={{ marginTop: 10 }}>
          This is the term that makes the loop self-reinforcing rather than purely
          exploratory: programs that paid once get imitated. We render the
          threshold form; the paper’s notation differs in detail.{' '}
          <PaperRef reference="eq5" also={['sec2.2']} inline />
        </p>
      </Card>

      <h2 className="section">The program pool</h2>
      <p className="body">
        Programs do not come only from fresh sampling. Each pool mixes three
        sources, and the pool itself is an archive rather than a single round’s
        batch.
      </p>
      <Card
        title="Where the pool’s programs come from"
        note="Appendix G. The archive is the reason the curriculum has a memory."
      >
        <p className="body">
          Every round, the {meta.compute.pool} programs in the pool are not all
          freshly sampled. Fresh draws from the current generator are mixed with{' '}
          <strong>mutations of programs that scored well in earlier rounds</strong>,
          and with <strong>programs carried forward</strong> from previous
          rounds. A program that once taught the learner something therefore
          keeps appearing in later rounds instead of having to be rediscovered
          by chance.
        </p>
        <p className="body">
          That reuse is the mechanism behind the curriculum page: the pool is a
          memory of what the learner has already been shown, not a fresh sample
          each round.
        </p>
      </Card>
      <Claim reference="appG" also={['sec2.2']}>
        Carrying programs forward across rounds is what stops the generator from
        re-learning the same handful of programs. Whether that retention is what
        produces the scaling exponent, or merely accompanies it, is not
        something the paper isolates — there is no archive-free ablation arm.
      </Claim>
      <p className="card__note">
        Appendix G is titled “Pool construction”, and the pool size above is
        stated exactly. The released artefacts do not specify the archive’s
        niche index, retention count, or any decay schedule, so this page does
        not state them.
      </p>

      <Disclosure summary="What this page does not cover">
        <p className="body">
          The reward ablations and their causal evidence are on the Reward page.
          The curriculum measurement — epiplexity of the pool against what the
          learner has actually absorbed — is on the Curriculum page. The scaling
          exponents the compute formula above feeds are on the Scaling page. And
          the twenty thousand real discovered programs are on the Discovered Math
          page; the presets above are the paper’s illustrative examples, not
          samples the generator actually emitted.
        </p>
      </Disclosure>
    </>
  )
}

// ---------------------------------------------------------------------------
// Explorer
// ---------------------------------------------------------------------------

function BfExplorer({ macros, cellModulus }: { macros: BfMacro[]; cellModulus: number }) {
  const [program, setProgram] = useState(PRESETS[0]?.program ?? '+++[>+.<-]')
  const [seed, setSeed] = useState(PRESETS[0]?.seed ?? 1)
  const [t, setT] = useState(48)

  const run = useMemo(
    () => runBrainfuck(program, t, seed, macros),
    [program, seed, t, macros],
  )
  const active = PRESETS.find((p) => p.program === program) ?? null

  // Tape window centred a little left of the head so the accumulators a program
  // has already walked past stay visible.
  const start = (run.head - 6 + TAPE * 2) % TAPE
  const cells = Array.from({ length: WINDOW }, (_, k) => (start + k) % TAPE)

  return (
    <Card
      title="Brainfuck interpreter"
      note={`Cells mod ${cellModulus}, circular tape of ${TAPE}, macro tokens expanded from Table 4 before execution.`}
      className="card--inset"
    >
      <div className="row" style={{ marginBottom: 10 }}>
        {PRESETS.map((p) => (
          <button
            key={p.program}
            aria-pressed={program === p.program}
            onClick={() => { setProgram(p.program); setSeed(p.seed) }}
            title={p.paperTerms}
          >
            {p.label}
          </button>
        ))}
      </div>

      <div className="field">
        <span className="field__label">program</span>
        <input
          type="text"
          value={program}
          spellCheck={false}
          onChange={(e) => setProgram(e.target.value)}
          style={{ width: '100%', fontFamily: 'var(--mono)' }}
        />
      </div>

      <div className="controls">
        <label className="field">
          <span className="field__label">T (bytes to emit)</span>
          <input
            type="number" min={1} max={1024} value={t}
            onChange={(e) => setT(Math.max(1, Math.min(1024, Number(e.target.value) || 1)))}
          />
        </label>
        <label className="field">
          <span className="field__label">seed</span>
          <input
            type="number" min={0} value={seed}
            onChange={(e) => setSeed(Math.max(0, Number(e.target.value) || 0))}
          />
        </label>
        <div className="field">
          <span className="field__label">expansion</span>
          <code style={{ fontSize: 12, color: 'var(--text-faint)' }}>
            {run.expanded.length} chars
          </code>
        </div>
      </div>

      <h3 className="subhead">Emitted bytes</h3>
      <ProgramView src={program} macros={macros} />
      <div className="program" style={{ marginTop: 6 }}>
        {run.out.map((b, i) => (
          <span key={i} style={{ color: b === 0 ? 'var(--text-faint)' : 'var(--text)' }}>
            {b}
            {i < run.out.length - 1 ? ' ' : ''}
          </span>
        ))}
      </div>
      <p className="card__note">
        {active
          ? <>Paper states: <code>{active.paperTerms}</code> · {active.note}</>
          : 'Your program. The paper’s own examples are the preset buttons above.'}
      </p>

      <h3 className="subhead">Byte value by position</h3>
      <ByteChart out={run.out} />
      <Legend
        items={[
          { label: 'emitted byte', color: 'var(--arm-selfplay)' },
          { label: 'mod-256 wrap (value decreases)', color: 'var(--warn)', dash: '4 3' },
        ]}
      />

      <h3 className="subhead">Tape</h3>
      <p className="card__note">
        {WINDOW} cells starting at index {start}; the head is highlighted. Cells
        left of the start are off-screen — the tape is circular and wraps at index{' '}
        {TAPE - 1}.
      </p>
      <div className="tape" style={{ marginTop: 6 }}>
        {cells.map((i) => (
          <div
            key={i}
            className={[
              'tape__cell',
              i === run.head ? 'tape__cell--head' : '',
              (run.tape[i] ?? 0) === 0 ? 'tape__cell--zero' : '',
            ].filter(Boolean).join(' ')}
            title={`cell ${i} = ${run.tape[i] ?? 0}`}
          >
            {run.tape[i] ?? 0}
          </div>
        ))}
      </div>

      <div className="grid-3" style={{ marginTop: 16 }}>
        <Stat value={run.out.length} label="bytes emitted (zero-padded to T)" />
        <Stat value={run.steps.toLocaleString()} label="instructions executed" />
        <Stat
          value={run.head}
          label={`head at cell ${run.head}${run.halted ? ' — program halted' : ''}`}
        />
      </div>

      {run.stepLimited && (
        <div className="claim claim--caveat" style={{ marginTop: 16 }}>
          <p className="claim__text">
            <strong>Step limit reached.</strong> This program had not emitted T
            bytes after {MAX_STEPS.toLocaleString()} instructions, so it is
            looping without producing output. The paper’s own data would score
            this sequence at the uniform-256 baseline of 8.0 bits/byte: nothing
            to learn.
          </p>
        </div>
      )}
      {run.unmatched > 0 && (
        <div className="claim claim--caveat" style={{ marginTop: 16 }}>
          <p className="claim__text">
            <strong>{run.unmatched} unmatched bracket{run.unmatched === 1 ? '' : 's'}.</strong>{' '}
            Treated as a no-op, which is what the paper specifies — a random
            program is nearly always still executable.
          </p>
        </div>
      )}
      {active && (
        <p className="card__note" style={{ marginTop: 12 }}>
          Source: <PaperRef reference={active.reference} also={['appE', 'table4']} inline />
        </p>
      )}
    </Card>
  )
}

/** Syntax-highlighted program, marking macro tokens the way Table 4 does. */
function ProgramView({ src, macros }: { src: string; macros: BfMacro[] }) {
  const macroTokens = new Set(macros.map((m) => m.token))
  return (
    <div className="program">
      {[...src].map((c, i) => (
        <span
          key={i}
          className={
            macroTokens.has(c) ? 'program__macro' : 'program__op'
          }
        >
          {c}
        </span>
      ))}
    </div>
  )
}

/** Emitted byte against its position in the output. */
function ByteChart({ out }: { out: number[] }) {
  const last = Math.max(out.length - 1, 1)
  return (
    <ChartFrame
      height={220}
      xScale="linear"
      xDomain={[0, last + 1]}
      yDomain={[0, 256]}
      xLabel="position in the output"
      yLabel="byte value"
    >
      {({ x, y }) => (
        <g>
          <Series
            points={out.map((b, i) => [x(i), y(b)])}
            color="var(--arm-selfplay)"
            showPoints
            pointRadius={2.5}
          />
        </g>
      )}
    </ChartFrame>
  )
}