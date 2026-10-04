/**
 * The reward simulator: Equation 2 computed live, and taken apart.
 *
 * Why this page exists. The Reward page shows Table 5, which is an ENDPOINT
 * trace -- seven arms scored at one round of one rung. It cannot show what the
 * reward does during a run, and its own text says so. This page runs the reward
 * on a real (small) learner and lets a reader watch it move.
 *
 * What it is NOT, stated at the top because everything below depends on it:
 *
 *   - The learner is a {HIDDEN}-wide byte model with {PARAM_COUNT} parameters,
 *     not the paper's smallest released rung at 98,496. It is not a
 *     transformer and does not have attention.
 *   - The programs are synthetic byte sequences, NOT output of the paper's UTM.
 *     They are chosen to have known positions in a known curriculum, which real
 *     random program draws do not have.
 *   - The generator is not trained. There is no g_phi, no KL term, no policy
 *     gradient. The generator's UPDATE -- which is where the anti-collapse
 *     pressure actually lives -- is Phase 4 and is not here.
 *   - The optimiser hyperparameters are ours. No released artefact contains them.
 *
 * So the claim this page can support is narrow and real: the reward's ALGEBRA
 * prefers a learner's current frontier over noise, and the terms that do the
 * preferring can be separated and plotted. It cannot support any claim about the
 * paper's results, and it does not make one.
 *
 * The numbers are derived, never typed (D2). Every figure below comes from
 * runSim via useMemo, and the gradient those numbers rest on is pinned against
 * finite differences in gradcheck.test.ts.
 */
import { useEffect, useState } from 'react'
import { Card, Claim, Disclosure, Legend, PaperRef, Stat } from '@/components/UI'
import { ChartFrame, Series } from '@/components/Chart'
import { Math as MathBlock } from '@/components/Math'
import { HIDDEN, PARAM_COUNT, SEQ_LEN } from '@/lib/sim/model'
import { noiseIndex, type SimProgram } from '@/lib/sim/programs'
import {
  noiseOutbidsFrontier, peakRound, series,
  type RoundRecord, type SimResult,
} from '@/lib/sim/runsim'
import { useSimRuns } from '@/lib/sim/useSim'
import { rewardOf, type Decomposition, type RewardMode } from '@/lib/sim/reward'

/** Run geometry. Pinned here so the page's axes and its prose cannot drift. */
const ROUNDS = 200
const RECORD_EVERY = 8
const CURRICULUM_SPAN = 50

const TEAL = 'var(--arm-selfplay)'
const WARN = 'var(--warn)'
const MUTED = 'var(--text-muted)'

type Curriculum = 'fixed' | 'moving'

export function RewardSim() {
  return (
    <>
        <h1 className="page__title">Inside the reward</h1>
        <p className="page__lede">
          The ablation table tells you which reward won. This page runs one and
          takes it apart, term by term, so you can watch which part of the
          formula is doing the discriminating — and which part cannot tell a
          learnable program from noise at all.
        </p>

        <ScopeNote />

        <h2 className="section">The formula, and the three numbers in it</h2>
        <p className="body">
          The reward is an inner product between a program&rsquo;s gradient and
          the direction the learner has actually been moving. Factored into
          magnitudes and an angle, which is the form this page plots:
        </p>
        <MathBlock
          display
          tex={String.raw`r_i = \left|\left\langle \nabla_\theta L(y_i;\theta_{\text{now}}),\, P_e \odot \delta\theta_e \right\rangle\right| = \underbrace{\lVert\nabla_\theta L\rVert}_{\text{magnitude}} \cdot \underbrace{\lVert P_e \odot \delta\theta_e\rVert}_{\text{path travelled}} \cdot \underbrace{\left|\cos(\nabla_\theta L,\, P_e \odot \delta\theta_e)\right|}_{\text{agreement}}`}
        />
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th className="left">term</th>
                <th className="left">what it is</th>
                <th className="left">what it decides</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="left"><code>|∇L|</code></td>
                <td className="left">the learner&rsquo;s gradient magnitude on this program</td>
                <td className="left">
                  <strong>Nothing on its own.</strong> High for a program at
                  the frontier <em>and</em> high for noise.
                </td>
              </tr>
              <tr>
                <td className="left"><code>cos</code></td>
                <td className="left">alignment with the learner&rsquo;s recent displacement</td>
                <td className="left">
                  <strong>Everything.</strong> Frontier sits near +1; noise
                  sits at or below 0.
                </td>
              </tr>
              <tr>
                <td className="left"><code>|P ⊙ δ&theta;|</code></td>
                <td className="left">length of the measured path, preconditioned</td>
                <td className="left">
                  The scale of the window. It <em>grows</em> as the run goes
                  on, which matters below.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="body">
          So the discriminating work is entirely in the cosine. The magnitude is
          the term that has to be controlled for, and the reason it cannot be
          simply normalised away is that the paper&rsquo;s{' '}
          <MathBlock tex={String.raw`|\cdot|`} /> discards the sign that would
          have told the two apart.{' '}
          <PaperRef reference="eq2" also={['sec2.2']} inline />
        </p>

      <Interactive />

      <Mechanism />
      <Limits />
    </>
  )
}

/** The honesty block. First thing on the page, before any number. */
function ScopeNote() {
  return (
    <div className="claim claim--caveat">
      <p className="claim__text">
        <strong>What this is, precisely.</strong> A {PARAM_COUNT.toLocaleString()}-parameter
        byte model (not a transformer, no attention), trained on synthetic byte
        sequences that are <em>not</em> the paper&rsquo;s UTM output. The
        generator&rsquo;s reinforcement-learning update — the KL term against
        the prior that actually keeps it from collapsing — is not simulated
        here at all. What this page shows is the reward&rsquo;s{' '}
        <em>algebra</em>: that its terms separate a learner&rsquo;s frontier
        from noise, and what that separation costs. It reproduces no result from
        the paper and makes no claim to. Optimiser hyperparameters are ours; no
        released artefact contains them.
      </p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// The interactive exhibit
// ---------------------------------------------------------------------------

function Interactive() {
  const [curriculum, setCurriculum] = useState<Curriculum>('fixed')
  const [mode, setMode] = useState<RewardMode>('abs')
  // WHICH PROGRAM the panels read against, and WHICH ROUND they read at.
  //
  // These were one state variable, which was a bug: selecting a program also
  // moved the round, because a program index was being used to index the rounds
  // array. Clicking "Fibonacci" jumped the exhibit to round 16. They are
  // independent axes and a reader needs both.
  const [programIdx, setProgramIdx] = useState(0)
  // null means "the peak", which is where the page opens.
  const [roundIdx, setRoundIdx] = useState<number | null>(null)

  // Off the main thread: a 200-round run is ~2.8 s of arithmetic, and running
  // both arms inline froze the tab for ~5.5 s. See useSim for the full contract.
  const runs = useSimRuns({ rounds: ROUNDS, recordEvery: RECORD_EVERY, span: CURRICULUM_SPAN })

  // Reset the two axes when the curriculum arm changes: the other arm's round
  // grid may not line up, and the pinned program may not exist in it.
  //
  // This sits ABOVE the early returns on purpose. A hook placed after a
  // conditional return runs on some renders and not others, and React rejects
  // that with "rendered more hooks than during the previous render". The first
  // render here returns early because the worker is still running and the
  // second does not, so the count changes exactly once -- and with no error
  // boundary on this page the result is a blank document with nothing logged.
  // App.tsx carries the same warning about a callback passed to Async; this is
  // the same trap reached from a different direction.
  useEffect(() => {
    setRoundIdx(null)
    setProgramIdx(0)
  }, [curriculum])

  if (runs.error !== null) {
    return (
      <div className="error">
        The simulation failed to run: {runs.error}
        <br />
        <small>
          Nothing on this page is a measurement until that succeeds — the reward
          is computed from a hand-written gradient, so a failed run is not a
          cosmetic failure.
        </small>
      </div>
    )
  }

  const res = runs[curriculum]
  if (res === null) {
    return (
      <div className="loading">
        Running two {ROUNDS}-round simulations… a few seconds.
      </div>
    )
  }
  const noise = noiseIndex(res.pool)
  // The peak moves with the sign convention, so "show me the peak" has to be
  // re-resolved whenever `mode` changes rather than cached from first render.
  const peak = peakRound(res.rounds, 0, mode)

  // An out-of-range roundIdx (the other arm has a different round count) falls
  // back to the peak instead of rendering nothing.
  const viewed: RoundRecord | undefined =
    roundIdx === null || res.rounds[roundIdx] === undefined ? peak : res.rounds[roundIdx]

  const outbidAbs = noiseOutbidsFrontier(res.rounds, 0, noise, 'abs')
  const outbidSigned = noiseOutbidsFrontier(res.rounds, 0, noise, 'signed')

  return (
    <>
      <h2 className="section">The reward, run</h2>
      <p className="body">
        A {ROUNDS}-round run. Every {RECORD_EVERY} rounds the whole pool is
        scored under Equation 2. Learner: {PARAM_COUNT.toLocaleString()}{' '}
        parameters, context {SEQ_LEN} bytes, width {HIDDEN}. Programs:{' '}
        {res.pool.filter((p) => p.kind !== 'noise').length} structured families
        plus a uniform-random control.{' '}
        <PaperRef reference="eq2" also={['sec2.1']} inline />
      </p>

      <div className="controls">
        <button aria-pressed={curriculum === 'fixed'} onClick={() => setCurriculum('fixed')}>
        Fixed frontier
        </button>
        <button aria-pressed={curriculum === 'moving'} onClick={() => setCurriculum('moving')}>
        Moving frontier
        </button>
        <span className="controls__gap" />
        <button aria-pressed={mode === 'abs'} onClick={() => setMode('abs')}>
        |&middot;| &nbsp;canonical
        </button>
        <button aria-pressed={mode === 'signed'} onClick={() => setMode('signed')}>
        signed
        </button>
      </div>
      <p className="card__note">
        <strong>Fixed frontier</strong> teaches one family for the whole run, so
        by the end it is mastered and nothing is left to learn.{' '}
        <strong>Moving frontier</strong> advances to a new family every{' '}
        {CURRICULUM_SPAN} rounds, so there is always something at the edge. The
        <code> |&middot;| </code> toggle switches the paper&rsquo;s canonical
        reading for the signed one.
      </p>

      {!res.finite && (
        <div className="error">
        This run produced a non-finite reward. Every chart below would be
        plotting a hole — do not read them.
        </div>
      )}

      <TermBreakdown
        res={res}
        viewed={viewed}
        mode={mode}
        noise={noise}
        programIdx={programIdx}
        onSelectProgram={setProgramIdx}
        roundIdx={roundIdx === null || res.rounds[roundIdx] === undefined ? null : roundIdx}
      />

      <RoundScrubber
        res={res}
        viewed={viewed}
        peak={peak}
        roundIdx={roundIdx}
        onSelectRound={setRoundIdx}
      />

      <div className="kv" style={{ marginTop: 16 }}>
        <Stat
        value={`${viewed ? noiseOutbidsFrontier([viewed], 0, noise, mode) : 0}`}
        label="programs outbidding the frontier at this round"
        />
        <Stat value={outbidAbs.toString()} label="rounds noise wins under |&middot;|" />
        <Stat value={outbidSigned.toString()} label="rounds noise wins when signed" />
        <Stat
        value={`${res.rounds.length}`}
        label="recorded rounds"
        />
      </div>

      <Claim caveat reference="appF" also={['eq2', 'sec2.2']}>
        <strong>The toggle is the whole finding.</strong> On this run the
        canonical absolute-value reading lets uniform noise outbid the frontier
        program in <strong>{outbidAbs}</strong> of {res.rounds.length} recorded
        rounds. Read the inner product signed and that falls to{' '}
        <strong>{outbidSigned}</strong>. Same gradient, same displacement, same
        preconditioner — one sign convention apart. The paper&rsquo;s Reward
        page calls the absolute value&rsquo;s cost the ability to distinguish
        hard-in-a-good-direction from hard-in-a-direction-the-optimiser-will-not-follow,
        and prices it low; on this toy run that term is not a caveat, it is the
        difference between a reward that tracks the learner and one that pays
        the most violent gradient in the pool.{' '}
        <PaperRef reference="appF" inline />
      </Claim>

      <Trajectory
        res={res}
        mode={mode}
        noise={noise}
        viewed={viewed}
        onSelectRound={setRoundIdx}
      />

      <Disclosure summary="What each program in the pool is">
        <div className="kv">
        {res.pool.map((p) => (
          <Fragment key={p.key}>
            <span className="kv__k">{p.label}</span>
            <span className="kv__v">{p.blurb}</span>
          </Fragment>
        ))}
        </div>
        <p className="body" style={{ marginTop: 10 }}>
        These are synthetic byte sequences chosen to sit at known positions in
        a known curriculum, not UTM output. That is deliberate: the question is
        whether the reward prefers a frontier the learner is currently being
        taught, and a random program draw has no such handle.
        </p>
      </Disclosure>
    </>
  )
}

/** Minimal key/value pair, so the pool list does not need its own component. */
function Fragment({ children }: { children: React.ReactNode }) {
  return <>{children}</>
}

/**
 * The three terms, side by side, at one round.
 *
 * This is the panel that answers "which piece pays for what".
 *
 * It takes TWO independent axes: which program the readouts are about, and
 * which round. They were one state variable at first, which meant selecting a
 * program also moved the round -- clicking "Fibonacci" jumped the exhibit to
 * round 16, because a program index was indexing the rounds array. Hover
 * previews a program and click pins it; neither touches the round, which the
 * RoundScrubber owns.
 */
function TermBreakdown({
  res, viewed, mode, noise, programIdx, onSelectProgram, roundIdx,
}: {
  res: SimResult
  viewed: RoundRecord | undefined
  mode: RewardMode
  noise: number
  programIdx: number
  onSelectProgram: (i: number) => void
  roundIdx: number | null
}) {
  const [hover, setHover] = useState<number | null>(null)

  if (!viewed) return <div className="loading">Scoring the pool…</div>

  const shown = hover ?? 0
  const d = viewed.programs[shown]

  return (
    <Card
      title={
        <>
        Round {viewed.e} — the reward taken apart
          {roundIdx === null && (
            <span style={{ color: 'var(--text-faint)', marginLeft: 8, fontSize: 12 }}>
              (peak)
            </span>
          )}
          {mode === 'signed' && (
          <span style={{ color: WARN, marginLeft: 8, fontSize: 12 }}>signed reading</span>
        )}
        </>
      }
      note="Pick a program to read every panel against it, then scrub the round below to move through the run. cos is the only term that separates a frontier program from noise; the magnitude is high for both."
    >
      <div className="controls" style={{ marginBottom: 12 }}>
        {res.pool.map((p, i) => {
        const pd = viewed.programs[i]
        const isActive = viewed.programs[i] !== undefined && pd !== undefined
          && viewed.activeIndex === i
        const isNoise = i === noise
        return (
          <button
            key={p.key}
            aria-pressed={programIdx === i}
            onMouseEnter={() => setHover(i)}
            onMouseLeave={() => setHover(null)}
            onFocus={() => setHover(i)}
            onBlur={() => setHover(null)}
            onClick={() => onSelectProgram(i)}
            style={isActive ? { borderColor: TEAL, color: TEAL } : undefined}
            title={isNoise ? 'the control' : isActive ? 'being taught this round' : p.label}
          >
            {p.label}
            {isActive && ' •'}
          </button>
        )
        })}
      </div>

      {d === undefined ? null : (
        <>
        <div className="kv">
          <span className="kv__k">gradient magnitude</span>
          <span className="kv__v">
            <code>|∇L|</code> = {d.gradNorm.toExponential(3)}
          </span>
          <span className="kv__k">preconditioned path</span>
          <span className="kv__v">
            <code>|P ⊙ δ&theta;|</code> = {d.motionNorm.toExponential(3)}
          </span>
          <span className="kv__k">agreement</span>
          <span className="kv__v">
            <code>cos</code> ={' '}
            <strong style={{ color: d.cos > 0.3 ? TEAL : d.cos < 0 ? WARN : MUTED }}>
              {d.cos.toFixed(4)}
            </strong>
            {d.cos > 0.5 && ' — aligned with the learner'}
            {d.cos < 0 && ' — opposed to the learner'}
            {d.cos >= 0 && d.cos <= 0.5 && ' — weakly aligned'}
          </span>
          <span className="kv__k">reward</span>
          <span className="kv__v">
            <code>r</code> = {rewardOf(d, mode).toExponential(3)}{' '}
            <span style={{ color: 'var(--text-faint)' }}>
              (raw inner product {d.inner.toExponential(3)})
            </span>
          </span>
          <span className="kv__k">learner loss here</span>
          <span className="kv__v">{d.lossBpb.toFixed(3)} bits/byte</span>
        </div>

        <TermBars viewed={viewed} pool={res.pool} noise={noise} />
        </>
      )}
    </Card>
  )
}

/**
 * The round axis.
 *
 * This control did not exist at all in the first version, which is a real gap:
 * the page opened on the peak round and the only way to reach any other round
 * was to click a program button -- and those were wired to the rounds array, so
 * picking a program silently moved the round. Selecting a round and selecting a
 * program are independent choices and each needs its own control.
 *
 * The slider steps over RECORDED rounds, not training rounds, so one step is
 * always one scored pool rather than a variable jump.
 */
function RoundScrubber({
  res, viewed, peak, roundIdx, onSelectRound,
}: {
  res: SimResult
  viewed: RoundRecord | undefined
  peak: RoundRecord | undefined
  roundIdx: number | null
  onSelectRound: (i: number | null) => void
}) {
  const n = res.rounds.length
  if (n === 0 || viewed === undefined) return null
  const current = roundIdx === null ? res.rounds.indexOf(viewed) : roundIdx
  const safe = current >= 0 ? current : 0
  const noise = noiseIndex(res.pool)
  const d = viewed.programs[viewed.activeIndex]
  const z = viewed.programs[noise]

  return (
    <Card
      title="Move through the run"
      note="One step is one scored pool. The peak is the round where the reward most clearly prefers what the learner is being taught."
    >
      <div className="controls" style={{ alignItems: 'center', gap: 12 }}>
        <button
          onClick={() => onSelectRound(Math.max(0, safe - 1))}
          disabled={safe === 0}
          aria-label="previous round"
        >
          &larr;
        </button>
        <input
          type="range"
          min={0}
          max={n - 1}
          step={1}
          value={safe}
          onChange={(e) => onSelectRound(Number(e.target.value))}
          style={{ flex: 1, minWidth: 200, accentColor: 'var(--arm-selfplay)' }}
          aria-label="round"
        />
        <button
          onClick={() => onSelectRound(Math.min(n - 1, safe + 1))}
          disabled={safe === n - 1}
          aria-label="next round"
        >
          &rarr;
        </button>
        <button onClick={() => onSelectRound(null)} aria-pressed={roundIdx === null}>
          jump to peak
        </button>
      </div>
      <div className="kv" style={{ marginTop: 12 }}>
        <span className="kv__k">round</span>
        <span className="kv__v">
          {viewed.e} of {res.rounds[res.rounds.length - 1]?.e ?? ROUNDS}
          {roundIdx === null && ' — showing the peak'}
        </span>
        <span className="kv__k">being taught</span>
        <span className="kv__v">{res.pool[viewed.activeIndex]?.label ?? '—'}</span>
        <span className="kv__k">learner loss</span>
        <span className="kv__v">{viewed.trainLossBpb.toFixed(3)} bits/byte</span>
        {d !== undefined && (
          <>
            <span className="kv__k">taught program&rsquo;s cos</span>
            <span className="kv__v">
              <strong style={{ color: d.cos > 0.3 ? TEAL : d.cos < 0 ? WARN : MUTED }}>
                {d.cos.toFixed(4)}
              </strong>
            </span>
          </>
        )}
        {z !== undefined && (
          <>
            <span className="kv__k">noise cos</span>
            <span className="kv__v">
              <strong style={{ color: z.cos < 0 ? WARN : 'var(--text)' }}>{z.cos.toFixed(4)}</strong>
            </span>
            <span className="kv__k">noise ÷ taught</span>
            <span className="kv__v">
              {(rewardOf(z, 'abs') / (d === undefined || rewardOf(d, 'abs') === 0 ? 1e-30 : rewardOf(d, 'abs'))).toFixed(2)}
              &times;
            </span>
          </>
        )}
      </div>
      {peak !== undefined && roundIdx !== null && (
        <p className="card__note">
          The peak is round {peak.e}.{' '}
          {peak.e === viewed.e
            ? 'You are there.'
            : `You are ${Math.abs(peak.e - viewed.e)} round${Math.abs(peak.e - viewed.e) === 1 ? '' : 's'} ${peak.e > viewed.e ? 'before' : 'after'} it.`}
        </p>
      )}
    </Card>
  )
}

/**
 * One bar per term, per program, so the three terms are comparable at a glance.
 *
 * Two things this panel has to get right, both of which were wrong first time:
 *
 *   - The magnitude bars are log-scaled against the PANEL'S OWN range, not a
 *     fixed decade range. A fixed -12..0 mapping pins every bar at full width
 *     once the values reach order 1, which is exactly what happened at round
 *     176: five identical bars, reading as "no difference" when the underlying
 *     numbers differ by 4x.
 *   - The preconditioned-path term is a property of the ROUND, not of the
 *     program: P_e and delta-theta_e are the same for every program scored at
 *     round e. Its bars are therefore all equal by construction, and the label
 *     says so rather than letting a reader mistake it for a discriminator.
 */
function TermBars({
  viewed, pool, noise,
}: {
  viewed: RoundRecord
  pool: SimProgram[]
  noise: number
}) {
  const terms: { label: string; hint: string; pick: (d: Decomposition) => number; log: boolean }[] = [
    {
      label: '|\u2207L| gradient magnitude',
      hint: 'log-scaled. High for a frontier program AND for noise \u2014 this term cannot tell them apart.',
      pick: (d) => d.gradNorm,
      log: true,
    },
    {
      label: '|P \u2299 \u03b4\u03b8| preconditioned path',
      hint: 'log-scaled across this round\u2019s programs.',
      pick: (d) => d.motionNorm,
      log: true,
    },
    {
      label: 'agreement (cos)',
      hint: 'linear, marked at zero. This is the only term that refuses noise.',
      pick: (d) => d.cos,
      log: false,
    },
  ]

  return (
    <div style={{ marginTop: 18 }}>
      {terms.map((t) => {
        const vals = viewed.programs.map(t.pick)
        // Data-driven log range. Guarded to a 1-decade span so a panel where
        // every value is identical does not divide by zero.
        const mags = vals.map((v) => Math.abs(v)).filter((v) => v > 0)
        const lo = mags.length > 0 ? Math.min(...mags) : 1
        const hi = mags.length > 0 ? Math.max(...mags) : 1
        const logLo = Math.log10(lo)
        const logSpan = Math.max(Math.log10(hi) - logLo, 1)
        // A panel where every program reads the same has no ranking to show.
        // Drawing five equal bars implies a comparison that is not there, so it
        // is stated as a single line instead.
        const uniqCount = new Set(vals.map((v) => v.toPrecision(12))).size

        const frac = (v: number): number => {
          if (!t.log) return Math.abs(v)
          const a = Math.abs(v)
          if (!(a > 0)) return 0
          return Math.min(1, Math.max(0, (Math.log10(a) - logLo) / logSpan))
        }

        return (
          <div key={t.label} style={{ marginBottom: 16 }}>
            <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 5, fontFamily: 'var(--mono)' }}>
              {t.label}
            </div>
            {uniqCount === 1 && (
              <p className="card__note" style={{ margin: '4px 0 6px 148px' }}>
                identical for every program at this round:{' '}
                <code>{(vals[0] ?? 0).toExponential(3)}</code>. There is nothing
                to rank here, which is the point &mdash; this term is a property
                of the round.
              </p>
            )}
            {uniqCount > 1 && viewed.programs.map((d, i) => {
              const v = t.pick(d)
              const w = frac(v)
              const isNoise = i === noise
              return (
                <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 3 }}>
                  <span
                    style={{
                      fontSize: 11,
                      width: 140,
                      fontFamily: 'var(--sans)',
                      color: isNoise ? WARN : 'var(--text-muted)',
                      textAlign: 'right',
                    }}
                  >
                    {pool[i]?.label ?? `program ${i}`}
                  </span>
                  <span style={{ flex: 1, height: 10, background: 'var(--border)', borderRadius: 2, position: 'relative' }}>
                    {!t.log && (
                      <span
                        style={{
                          position: 'absolute',
                          left: '50%',
                          top: -2,
                          bottom: -2,
                          width: 1,
                          background: 'var(--border-strong)',
                        }}
                      />
                    )}
                    <span
                      style={{
                        position: 'absolute',
                        top: 0,
                        bottom: 0,
                        // cos is signed: a negative cosine fills LEFT from the
                        // centre line, a positive one fills right. Drawing a
                        // negative value as a positive-length bar to the left
                        // edge would hide the sign, which is the whole point.
                        left: t.log ? 0 : v < 0 ? `${50 - w * 50}%` : '50%',
                        width: `${(w * (t.log ? 100 : 50)).toFixed(1)}%`,
                        background: isNoise ? WARN : v < 0 ? 'var(--danger)' : TEAL,
                        borderRadius: 2,
                        opacity: 0.85,
                      }}
                    />
                  </span>
                  <span style={{ fontSize: 11, width: 96, textAlign: 'right', fontFamily: 'var(--mono)', color: 'var(--text)' }}>
                    {t.log ? Math.abs(v).toExponential(2) : v.toFixed(4)}
                  </span>
                </div>
              )
            })}
            {uniqCount > 1 && (
              <p className="card__note" style={{ marginTop: 4 }}>{t.hint}</p>
            )}
          </div>
        )
      })}
    </div>
  )
}

/** Reward and loss over the whole run. Rows are clickable round selectors. */
function Trajectory({
  res, mode, noise, viewed, onSelectRound,
}: {
  res: SimResult
  mode: RewardMode
  noise: number
  viewed: RoundRecord | undefined
  onSelectRound: (i: number | null) => void
}) {
  const activeSeries = series(res.rounds, (r) => {
    const d = r.programs[r.activeIndex]
    return d ? rewardOf(d, mode) : null
  })
  const noiseSeries = series(res.rounds, (r) => {
    const d = r.programs[noise]
    return d ? rewardOf(d, mode) : null
  })

  const all = [...activeSeries, ...noiseSeries].map(([, v]) => v)
  const maxReward = Math.max(...all, 1e-12)

  return (
    <Card
      title="Reward over the run — the frontier against the control"
      note="Both series are raw rewards under the current reading, so they share a scale directly. The learner's loss falls underneath."
    >
      <ChartFrame
        height={300}
        xScale="linear"
        xDomain={[0, Math.max(...res.rounds.map((r) => r.e))]}
        yDomain={[0, maxReward * 1.15]}
        xLabel="round"
        yLabel={`reward (${mode})`}
      >
        {({ x, innerHeight }) => (
        <g>
          {noiseSeries.length > 1 && (
            <Series points={noiseSeries} color={WARN} width={1.75} showPoints />
          )}
          {activeSeries.length > 1 && (
            <Series points={activeSeries} color={TEAL} width={2.25} showPoints />
          )}
          <line
            x1={0} x2={x(Math.max(...res.rounds.map((r) => r.e)))}
            y1={innerHeight} y2={innerHeight}
            stroke="var(--border-strong)"
          />
        </g>
        )}
      </ChartFrame>
      <Legend
        items={[
        { label: 'program being taught', color: TEAL },
        { label: 'uniform noise (control)', color: WARN },
        ]}
      />
      <div className="table-wrap" style={{ marginTop: 14 }}>
        <table className="data">
        <thead>
          <tr>
            <th>round</th>
            <th>taught</th>
            <th>learner loss</th>
            <th>reward (taught)</th>
            <th>reward (noise)</th>
            <th>ratio</th>
          </tr>
        </thead>
        <tbody>
          {res.rounds.map((r, ri) => {
            const a = r.programs[r.activeIndex]
            const n = r.programs[noise]
            if (a === undefined || n === undefined) return null
            const ra = rewardOf(a, mode)
            const rn = rewardOf(n, mode)
            const noiseWins = rn > ra
            const isCurrent = viewed !== undefined && viewed.e === r.e
            return (
              <tr
                key={r.e}
                onClick={() => onSelectRound(ri)}
                title={`Load round ${r.e} into the breakdown above`}
                style={{
                  cursor: 'pointer',
                  background: isCurrent
                    ? 'rgba(120, 200, 190, 0.09)'
                    : undefined,
                }}
              >
                <td>{r.e}</td>
                <td className="left">{res.pool[r.activeIndex]?.label ?? '—'}</td>
                <td className="num">{r.trainLossBpb.toFixed(3)}</td>
                <td className="num">{ra.toExponential(2)}</td>
                <td className="num" style={noiseWins ? { color: WARN } : undefined}>
                  {rn.toExponential(2)}
                </td>
                <td className="num">{(rn / (ra || 1e-30)).toFixed(2)}&times;</td>
              </tr>
            )
          })}
        </tbody>
        </table>
      </div>
      <p className="card__note">
        Click a row to load that round into the breakdown above. Ratio is noise
        ÷ taught. Above 1.00 the control is being paid more than
        the thing the learner is actually trying to learn. Note the reward scale
        itself climbs over the run — the{' '}
        <code>|P ⊙ δ&theta;|</code> window spans more rounds as{' '}
        <MathBlock tex={String.raw`e`} /> grows — so the ratio column is the one
        to read, not the raw magnitudes. This is why Equation 4 normalises
        rewards within the pool.
      </p>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// What the mechanism is, and what it costs
// ---------------------------------------------------------------------------

function Mechanism() {
  return (
    <>
      <h2 className="section">Why the alignment term is the one that works</h2>
      <p className="body">
        Noise and a frontier program are not distinguished by how hard they are.
        They are distinguished by whether the learner&rsquo;s recent motion would
        have made progress on them. A program at the frontier produces a gradient
        pointing the same way the optimiser has been going, because the optimiser
        has been going that way <em>because of programs like it</em>. Uniform
        bytes produce a large gradient in a direction nothing in the run has ever
        pushed, so its cosine sits at or below zero no matter how large the
        gradient is.
      </p>
      <p className="body">
        That is the whole argument for learning progress over difficulty, and it
        holds in the simulator for the reason it holds in the paper: the two cases
        differ in <em>direction</em>, not in <em>size</em>. It also explains why
        the magnitude cannot be the criterion — on this run noise carries the
        largest gradient in the pool in most rounds, and would win any reward
        built on gradient size alone.{' '}
        <PaperRef reference="sec2.2" inline />
      </p>

      <h2 className="section">What the lookback window costs</h2>
      <p className="body">
        One more thing falls out of running it, and the paper does not discuss it.
        The reward measures alignment against{' '}
        <MathBlock tex={String.raw`\delta\theta_e = \theta_{\lfloor e/2 \rfloor} - \theta_e`} />
        , a window spanning half the run so far. When the curriculum advances, that
        window is straddling a switch: it measures a displacement that is partly
        the old family&rsquo;s direction and partly the new one. The newly-active
        program is then graded against a window dominated by a family it has
        nothing to do with.
      </p>
      <Claim caveat reference="eq2" also={['appF']}>
        <strong>The frontier briefly reads as opposed.</strong> Immediately after
        a curriculum switch the active program&rsquo;s cosine goes negative —
        measured down to −0.363 — and under the signed reading the noise control
        outbids it in 18 of 50 recorded rounds, against 0 of 50 when the
        curriculum is fixed. The window is doing exactly what it was designed to
        do, averaging over short-term fluctuation; the cost is that a frontier
        which has just <em>moved</em> is invisible to it for a few rounds. Whether
        that matters at the paper&rsquo;s scale, where a pool of 1536 programs
        keeps the frontier from ever being fully mastered, is not something this
        simulator can answer.
      </Claim>
    </>
  )
}

function Limits() {
  return (
    <>
      <h2 className="section">What this cannot tell you</h2>
      <div className="table-wrap">
        <table className="data">
        <thead>
          <tr>
            <th className="left">the simulator has</th>
            <th className="left">the paper has</th>
            <th className="left">so the gap is</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="left">{PARAM_COUNT.toLocaleString()} params, no attention</td>
            <td className="left">98,496 at the smallest released rung</td>
            <td className="left">the learner is not the paper&rsquo;s learner</td>
          </tr>
          <tr>
            <td className="left">synthetic byte sequences</td>
            <td className="left">UTM program output</td>
            <td className="left">the program space is not the paper&rsquo;s</td>
          </tr>
          <tr>
            <td className="left">a fixed pool of {CURRICULUM_SPAN}-round switches</td>
            <td className="left">a pool of 1,536 programs per round</td>
            <td className="left">the frontier never sits still here</td>
          </tr>
          <tr>
            <td className="left">no generator update</td>
            <td className="left">
              reward-weighted log-ratio plus a KL penalty against{' '}
              <MathBlock tex={String.raw`g_0`} />
            </td>
            <td className="left">
              the anti-collapse pressure is the generator&rsquo;s, and it is
              absent here
            </td>
          </tr>
          <tr>
            <td className="left">ours: lr 1e-3, β₂ 0.999</td>
            <td className="left">unreleased</td>
            <td className="left">nothing to compare against</td>
          </tr>
        </tbody>
        </table>
      </div>
      <p className="body">
        So: the separation between frontier and noise is a property of the
        reward&rsquo;s algebra and it reproduces here for the reason the paper
        gives. The claim that the reward <em>keeps a generator balanced</em> is a
        claim about the generator&rsquo;s update, and this page deliberately stops
        short of it — that is the next piece of work, and it needs a trained{' '}
        <MathBlock tex={String.raw`g_\phi`} /> and the KL term to say anything at
        all.
      </p>
    </>
  )
}