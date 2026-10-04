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
 *
 * The machine itself lives in @/lib/bf, not here. The explorer's stepper and
 * the run-to-completion path are two callers of one `stepMachine`, and a test
 * cannot import a page module without dragging React and the KaTeX stylesheet
 * into a node-environment test run. This page re-exports the interpreter's
 * surface, so anything that reached for it here still finds it.
 */
import { useEffect, useMemo, useState } from 'react'
import { loadMeta } from '@/data/loaders'
import type { BfMacro, Meta } from '@/data/types'
import {
  Async, Card, Claim, Disclosure, Legend, PaperRef, Stat,
} from '@/components/UI'
import { ChartFrame, Series } from '@/components/Chart'
// Aliased: a component named `Math` shadows the global in every type position,
// which TS then rejects as a JSX element type.
import { Math as MathBlock } from '@/components/Math'
import {
  MAX_STEPS, PRESETS, TAPE, WINDOW, compileProgram, drive, finalize,
  initialState, isDone, sourceIndexOfPc, stepMachine, type MachineState,
} from '@/lib/bf'

// Re-exported so the interpreter keeps answering to the page it grew out of.
export {
  PRESETS, compileProgram, drive, expandMacros, finalize, initialState, isDone,
  runBrainfuck, sourceIndexOfPc, stepMachine, TAPE, WINDOW,
  type BfProgram, type BfRun, type MachineState, type Preset,
} from '@/lib/bf'

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
        note={`Cells are taken mod ${params.cell_modulus}. The primitive alphabet is ${params.program_alphabet} — the eight primitives plus F, which terminates a row. The ${macros.length} macro tokens below expand into those same primitives, so a program written with macros uses 19 distinct characters in total, which is exactly what the ${params.program_alphabet.length + macros.length} distinct characters across all released programs amount to.`}
      >
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr><th className="left">token</th><th className="left">effect</th></tr>
            </thead>
            <tbody>
              {primitives.map((p) => (
                <tr key={p.token}>
                  <td className="num">{p.token}</td>
                  <td className="left">{p.effect}</td>
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
              <tr><th className="left">token</th><th className="left">expansion</th><th className="left">effect</th></tr>
            </thead>
            <tbody>
              {macros.map((m) => (
                <tr key={m.token}>
                  <td className="num" style={{ color: '#f0883e' }}>{m.token}</td>
                  <td className="num left">{m.expansion}</td>
                  <td className="left">{m.effect}</td>
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
        Five terms, in the order they act. Typeset with KaTeX; the notation is
        ours, and equations 3 to 5 are reconstructions rather than
        transcriptions.
      </p>

      <Card
        title="Equation 1 — the learner’s loss"
        note="Ordinary next-byte cross-entropy on the emitted sequence. No natural data enters here, at any round."
      >
        <MathBlock
          display
          tex={String.raw`L(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta\!\left(y_t \mid y_{<t}\right)`}
        />
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
        <MathBlock
          display
          tex={String.raw`r_i = \left|\left\langle \nabla_\theta L(y_i;\theta_{\text{now}}),\, P_e \odot \delta\theta_e \right\rangle\right|`}
        />
        <MathBlock
          display
          tex={String.raw`\delta\theta_e = \theta_{\lfloor e/2 \rfloor} - \theta_e, \qquad P_e = \operatorname{diag}\!\left(\sqrt{\hat{v}_e} + \varepsilon\right)`}
        />
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
        note="Reconstruction, not a transcription. The released artefacts do not contain this formula; see the note below."
      >
        <MathBlock
          display
          tex={String.raw`J(\phi) = \mathbb{E}_x\!\left[\frac{r(x)}{\beta}\log\frac{g_\phi(x)}{g_0(x)}\right] - \beta\, \mathrm{KL}\!\left(g_\phi \| g_0\right)`}
        />
        <MathBlock
          display
          tex={String.raw`g_0(x) = |A|^{-\mathrm{len}(x)}, \qquad |A| = \text{alphabet size}`}
        />
        <p className="body" style={{ marginTop: 10 }}>
          <code>g_0</code> is the length-prior over the program alphabet — the
          fixed uniform-prior arm is exactly <code>g_0</code> with the generator
          switched off, which is why Figure 2’s comparison is clean. The KL term
          is what keeps the sampler near the prior instead of collapsing onto a
          single high-reward program.{' '}
          <PaperRef reference="eq3" also={['sec2.2']} inline />
        </p>
        <p className="card__note">
          This card is our reconstruction of the objective, not the paper&rsquo;s
          printed equation: no released artefact contains the 1/β reward scaling
          or the −β·KL coefficient, so treat the constants as illustrative. The
          structure — reward-weighted log-ratio against the prior, plus a KL
          penalty — is what §2.2 describes.
        </p>
      </Card>

      <Card
        title="Equation 4 — the GRPO advantage"
        note="Rewards are normalised within the pool, so the step size does not drift as the reward scale changes across rounds."
      >
        <MathBlock
          display
          tex={String.raw`\hat{A}_i = \frac{r_i - \mathrm{mean}_j r_j}{\mathrm{std}_j r_j + \varepsilon}`}
        />
        <p className="body" style={{ marginTop: 10 }}>
          Group-relative: the comparison is always against the other programs in
          the same pool, never against an absolute scale.{' '}
          <PaperRef reference="eq4" also={['sec2.2']} inline />
        </p>
        <p className="card__note">
          Also a reconstruction. The mean-and-standard-deviation normaliser is
          the standard group-relative form, but the released artefacts do not
          fix the ε or state whether the standard deviation is taken over the
          pool or over seeds.
        </p>
      </Card>

      <Card
        title="Equation 5 — expert iteration"
        note="Reconstruction, not a transcription. See the note below."
      >
        <MathBlock
          display
          tex={String.raw`\mathcal{L}_{\mathrm{SFT}}(\phi) = -\,\mathbb{E}_{x \sim g_\phi}\!\left[w(x)\log g_\phi(x)\right], \qquad w(x) \propto \max\left(0,\ r(x) - \tau\right)`}
        />
        <p className="body" style={{ marginTop: 10 }}>
          This is the term that makes the loop self-reinforcing rather than purely
          exploratory: programs that paid once get imitated.{' '}
          <PaperRef reference="eq5" also={['sec2.2']} inline />
        </p>
        <p className="card__note">
          Note the parameterisation:{' '}
          <MathBlock tex={String.raw`\mathcal{L}_{\mathrm{SFT}}(\phi)`} /> is a
          loss over the{' '}
          <em>generator</em> <MathBlock tex={String.raw`g_\phi`} />, not over the
          learner, even though it is
          supervised fine-tuning — the generator is being trained to imitate its
          own high-reward programs. The threshold form is our rendering; no
          released artefact fixes <MathBlock tex={String.raw`\tau`} /> or the exact
          weight.
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
          page. The presets above are ours — worked examples chosen to exercise
          the interpreter, not samples the generator emitted; the ones the
          generator actually produced are on that page.
        </p>
      </Disclosure>
    </>
  )
}

// ---------------------------------------------------------------------------
// Explorer
// ---------------------------------------------------------------------------

/**
 * Instructions retired per animation frame. One is the only rate at which a
 * ten-character program is legible, and 1024 is the rate at which a 4,095-byte
 * one finishes before the reader has decided they wanted to see it. The middle
 * settings exist because the interesting programs sit between those.
 */
const SPEEDS = [1, 8, 64, 1024]

function BfExplorer({ macros, cellModulus }: { macros: BfMacro[]; cellModulus: number }) {
  const [program, setProgram] = useState(PRESETS[0]?.program ?? '+++[>+.<-]')
  const [seed, setSeed] = useState(PRESETS[0]?.seed ?? 1)
  const [t, setT] = useState(48)
  const [speed, setSpeed] = useState(8)

  /**
   * `live === null` means "not animating", and the explorer then renders the
   * finished run -- which is what this page did before there was a stepper, and
   * is the right default: someone arriving here wants the emitted sequence, not
   * a machine to watch. Rewinding sets a state, and from then on the tape,
   * the cursor and the step counter follow it instead of the end state.
   */
  const [live, setLive] = useState<MachineState | null>(null)
  const [playing, setPlaying] = useState(false)

  const prog = useMemo(() => compileProgram(program, macros), [program, macros])
  const end = useMemo(
    () => finalize(prog, drive(prog, initialState(seed), t), t),
    [prog, seed, t],
  )

  const view = live === null ? end : finalize(prog, live, t)
  const active = PRESETS.find((p) => p.program === program) ?? null
  const cursor = live === null ? null : sourceIndexOfPc(prog, live.pc)

  // Editing the program while a replay is on screen rewinds the new program to
  // its first instruction rather than leaving a cursor pointing into code that
  // no longer exists. With no replay engaged there is nothing to rewind.
  useEffect(() => {
    setLive((prev) => (prev === null ? null : initialState(seed)))
  }, [prog, t, seed])

  // The frame loop. Everything it reads is in the dependency list, so the
  // cleanup runs on every change -- which is also what makes it safe under
  // StrictMode, where React mounts, tears down and mounts again.
  useEffect(() => {
    if (!playing) return
    let raf = requestAnimationFrame(function tick() {
      // Functional update: the frame callback must not close over the state
      // from the render that scheduled it, or the replay runs at one step.
      setLive((prev) => drive(prog, prev ?? initialState(seed), t, speed))
      raf = requestAnimationFrame(tick)
    })
    return () => cancelAnimationFrame(raf)
  }, [playing, prog, t, seed, speed])

  // Stop on arrival. Kept out of the frame callback: setState inside another
  // setState's updater is a side effect in the render phase, and React is
  // within its rights to call that updater twice.
  useEffect(() => {
    if (playing && live !== null && isDone(prog, live, t)) setPlaying(false)
  }, [playing, live, prog, t])

  const rewind = (): void => {
    setPlaying(false)
    setLive(initialState(seed))
  }
  const stepOnce = (): void => {
    setPlaying(false)
    setLive((prev) => {
      const s = prev ?? initialState(seed)
      return isDone(prog, s, t) ? s : stepMachine(prog, s)
    })
  }
  const toggle = (): void => {
    // Pressing play from the finished state means "show me the run", so it
    // starts from the first instruction rather than resuming at the end.
    if (live === null || isDone(prog, live, t)) setLive(initialState(seed))
    setPlaying((p) => !p)
  }

  // Tape window centred a little left of the head so the accumulators a program
  // has already walked past stay visible.
  const start = (view.head - 6 + TAPE * 2) % TAPE
  const cells = Array.from({ length: WINDOW }, (_, k) => (start + k) % TAPE)
  const animating = live !== null

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
        <label className="field">
          <span className="field__label">speed</span>
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
            {SPEEDS.map((s) => (
              <option key={s} value={s}>{s} step{s === 1 ? '' : 's'}/frame</option>
            ))}
          </select>
        </label>
        <div className="field">
          <span className="field__label">expansion</span>
          <code style={{ fontSize: 12, color: 'var(--text-faint)' }}>
            {end.expanded.length} chars
          </code>
        </div>
      </div>

      <div className="row row--tight" style={{ marginBottom: 10 }}>
        <button onClick={toggle} aria-pressed={playing} title="Run the machine forward from the current instruction">
          {playing ? 'Pause' : 'Play'}
        </button>
        <button onClick={stepOnce} disabled={live !== null && isDone(prog, live, t)} title="Execute one instruction">
          Step
        </button>
        <button onClick={rewind} title="Back to the instruction before the first">Rewind</button>
        <span className="card__note" style={{ margin: 0 }}>
          {animating
            ? `${playing ? 'running' : 'paused'} · instruction ${cursor === null ? '—' : cursor + 1} of ${[...program].length} · ${live.steps.toLocaleString()} steps executed`
            : 'showing the finished run'}
        </span>
      </div>

      <h3 className="subhead">Emitted bytes</h3>
      <ProgramView src={program} macros={macros} cursor={cursor} />
      <div className="program" style={{ marginTop: 6 }}>
        {view.out.map((b, i) => (
          <span key={i} style={{ color: b === 0 ? 'var(--text-faint)' : 'var(--text)' }}>
            {b}
            {i < view.out.length - 1 ? ' ' : ''}
          </span>
        ))}
      </div>
      <p className="card__note">
        {active
          ? <>Paper states: <code>{active.paperTerms}</code> · {active.note}</>
          : 'Your program. The paper’s own examples are the preset buttons above.'}
      </p>

      <h3 className="subhead">Byte value by position</h3>
      <ByteChart out={view.out} />
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
              i === view.head ? 'tape__cell--head' : '',
              (view.tape[i] ?? 0) === 0 ? 'tape__cell--zero' : '',
            ].filter(Boolean).join(' ')}
            title={`cell ${i} = ${view.tape[i] ?? 0}`}
          >
            {view.tape[i] ?? 0}
          </div>
        ))}
      </div>

      <div className="grid-3" style={{ marginTop: 16 }}>
        <Stat value={view.emitted} label={`bytes emitted of ${view.out.length} shown`} />
        <Stat value={view.steps.toLocaleString()} label="instructions executed" />
        <Stat
          value={view.head}
          label={`head at cell ${view.head}${view.halted ? ' — program halted' : ''}`}
        />
      </div>

      {view.stepLimited && (
        <div className="claim claim--caveat" style={{ marginTop: 16 }}>
          <p className="claim__text">
            <strong>Step limit reached.</strong> This program had not emitted T
            bytes after {MAX_STEPS.toLocaleString()} instructions. What you see is
            therefore <em>truncated</em>, not a failed program: the sequence may
            well have kept going. Lower T below {MAX_STEPS.toLocaleString()} (or
            raise the step limit) to see the whole thing — the truncation is this
            page&rsquo;s doing, not a property of the program.
          </p>
        </div>
      )}
      {view.unmatched > 0 && (
        <div className="claim claim--caveat" style={{ marginTop: 16 }}>
          <p className="claim__text">
            <strong>{view.unmatched} unmatched bracket{view.unmatched === 1 ? '' : 's'}.</strong>{' '}
            Treated as a no-op here — the reading the paper&rsquo;s own Table 1
            programs require, since its Fibonacci example has one unmatched
            bracket and only the no-op reading makes it emit 1, 1, 2, 3, 5, 8. A
            random program is nearly always still executable.
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

/**
 * Syntax-highlighted program, marking macro tokens the way Table 4 does, and
 * the instruction about to execute while a replay is running.
 *
 * `cursor` is an index into `src`, not into the expansion, which is what
 * sourceIndexOfPc spends its time computing: the highlight has to land on the
 * character the reader typed, and a macro token is one character standing for
 * several.
 */
function ProgramView({
  src, macros, cursor = null,
}: { src: string; macros: BfMacro[]; cursor?: number | null }) {
  const macroTokens = new Set(macros.map((m) => m.token))
  return (
    <div className="program">
      {[...src].map((c, i) => (
        <span
          key={i}
          className={[
            macroTokens.has(c) ? 'program__macro' : 'program__op',
            i === cursor ? 'program__pc' : '',
          ].filter(Boolean).join(' ')}
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