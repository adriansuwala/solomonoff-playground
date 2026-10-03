/**
 * Overview: what the paper claims and how to read this app.
 *
 * Doubles as the reference implementation for page structure: every number is
 * followed by a <PaperRef>, every chart goes through ChartFrame, and the
 * "our own measurement" block is deliberately visually separated from the
 * paper's numbers (D5).
 */
import { loadLocalMeasurements, loadScaling } from '@/data/loaders'
import {
  Async, Card, Claim, Disclosure, Legend, PaperRef, Stat,
} from '@/components/UI'
import { ChartFrame, Series } from '@/components/Chart'
import { lossDomain } from '@/lib/scales'

const ARM_COLORS = {
  selfplay: 'var(--arm-selfplay)',
  uniform: 'var(--arm-uniform)',
  pcfg: 'var(--arm-pcfg)',
} as const

export function Overview() {
  return (
    <>
      <h1 className="page__title">Self-Play Pretraining with Zero Data</h1>
      <p className="page__lede">
        Two transformers start from random initialisation and never see a byte of
        natural data. One writes programs for a universal Turing machine; the
        other learns to predict the bytes those programs emit. The question this
        app answers is not "does it work" but <em>what exactly the evidence
        shows</em> — and where the paper's own numbers stop.
      </p>

      <h2 className="section">The one idea</h2>
      <p className="body">
        The generator is not rewarded for producing{' '}
        <em>hard-to-predict</em> data, which is trivially gameable by spraying
        random bytes into an otherwise regular sequence. It is rewarded for
        producing data whose gradient points the same way the learner has{' '}
        <em>actually been moving</em>. Programs the learner has mastered produce
        no gradient and earn nothing. Programs containing unlearnable noise
        produce a gradient that leads nowhere and also earn nothing. Only programs
        sitting at the edge of what the learner can absorb get paid.
      </p>

      <Disclosure summary="The reward, formally (Equation 2)">
        <div className="kv">
          <span className="kv__k">reward</span>
          <span className="kv__v">r_i = |⟨∇θ L(y_i; θ_now), P_e ⊙ δθ_e⟩|</span>
          <span className="kv__k">δθ_e</span>
          <span className="kv__v">θ_⌊e/2⌋ − θ_e</span>
          <span className="kv__k">P_e</span>
          <span className="kv__v">diag(sqrt(v̂_e) + ε), the AdamW step operator</span>
        </div>
        <p className="body" style={{ marginTop: 10 }}>
          The preconditioner matters: the paper reports that using it was
          important, following Thrush et al. (2026).{' '}
          <PaperRef reference="eq2" also={['sec2.2']} inline />
        </p>
      </Disclosure>

      <h2 className="section">What the evidence actually shows</h2>
      <Claim reference="sec3.1">
        Prediction on held-out natural data — text, images, audio, speech,
        melody, DNA, code — improves as a power law in <em>self-play compute</em>,
        with no gradient step ever taken on any of those datasets.
      </Claim>
      <Claim reference="sec3.1" also={['fig2']}>
        The curriculum is what does the work. Hold the program space fixed and
        replace the learned sampler with the uniform Solomonoff-style prior, and
        scaling gets substantially slower. Access to a universal space of
        programs is not sufficient on its own.
      </Claim>
      <Claim reference="appC" also={['table1']}>
        The generator discovers recognisable mathematical sequences. Fibonacci,
        geometric, quadratic and cubic families all appear by round 512, against
        an expected first appearance beyond round 53,000 under uniform sampling
        from 1.64×10⁸ programs.
      </Claim>
      <Claim reference="sec3.2" also={['fig4']}>
        The learner does in-context learning on held-out tasks with no gradient
        updates at all — associative recall (contextual search), reverse string
        (dynamic indexing), stack (simulating a context-free grammar), and
        arithmetic relations.
      </Claim>

      <h2 className="section">The arms compared throughout</h2>
      <Card
        title="Three pretraining regimes over the same program space"
        note="Figures 2 and 7 compare these. The Universal-prior arm is the load-bearing control."
      >
        <Legend
          items={[
            { label: 'Self-play (learned sampler, RL on learning progress)', color: ARM_COLORS.selfplay },
            { label: 'Universal prior (fixed Solomonoff-style sampler)', color: ARM_COLORS.uniform, dash: '6 4' },
            { label: 'PCFG (hand-designed grammars)', color: ARM_COLORS.pcfg, dash: '2 3' },
          ]}
        />
        <p className="body" style={{ marginTop: 12 }}>
          PCFG is <em>not</em> a weak baseline: it beats self-play on text and
          code, where its inductive bias is well matched. Self-play wins on
          images, melody, audio and speech. The honest summary is that self-play
          transfers more <em>broadly</em>, not that it dominates everywhere.{' '}
          <PaperRef reference="sec3.1" inline />
        </p>
      </Card>

      <h2 className="section">We re-measured the core claim on CPU</h2>
      <Async load={loadLocalMeasurements}>
        {(m) => (
          <>
            <p className="body">
              The authors released checkpoints but not their training code, so we
              cannot retrain. We can still check the central claim directly: score
              a released checkpoint on real natural bytes it never trained on.
            </p>
            <Card
              title={`Released ${m.provenance.rung} checkpoint, ${m.provenance.params.toLocaleString()} parameters, scored by us`}
              note={`${m.provenance.nSequences} sequences of ${m.provenance.corpus} at context ${m.provenance.contextLength}. ${m.provenance.host.cpu}, ${m.provenance.host.torchThreads} CPU threads, no GPU.`}
            >
              <LocalMeasurementChart points={m.points.map((p) => p.bitsPerByte)} />
              <div className="grid-3" style={{ marginTop: 16 }}>
                {m.points.map((p) => (
                  <Stat
                    key={p.round}
                    value={
                      <>
                        {p.bitsPerByte.toFixed(2)}
                        <span style={{ fontSize: 13, color: 'var(--text-faint)' }}>
                          {' '}bits/byte
                        </span>
                      </>
                    }
                    label={
                      p.isRandomInit
                        ? 'round 0 — random init (the control)'
                        : `round ${p.round.toLocaleString()}`
                    }
                  />
                ))}
              </div>
              <div className="claim claim--caveat" style={{ marginTop: 16 }}>
                <p className="claim__text">
                  <strong>Read the control first.</strong> Round 0 is the random
                  initialisation and scores <em>above</em> the uniform-256
                  baseline of 8.0 bits/byte, which is exactly what an untrained
                  network should do. If it had read below 8.0, the measurement
                  would be broken. The drop to ≈6.15 is the paper's claim,
                  reproduced on two CPU cores.{' '}
                  <PaperRef reference="sec3.1" inline />
                </p>
              </div>
              <p className="card__note" style={{ marginTop: 10 }}>
                These are <strong>our</strong> measurements at the smallest
                released rung, not the paper's headline scaling exponents. The
                authors' own numbers for this checkpoint are on the Scaling page.
              </p>
            </Card>
          </>
        )}
      </Async>

      <h2 className="section">Where to go next</h2>
      <div className="grid-2">
        <Card title="If you want the mechanism" note="How it works → Reward ablations">
          The reward is the paper's central design choice, and the ablation table
          is the causal evidence for it. Everything else is downstream of that
          decision.
        </Card>
        <Card title="If you want the headline result" note="Scaling laws">
          The compute-optimal frontier and per-modality exponents, recomputed
          live from the authors' scored ladder, with our own CPU measurement as a
          cross-check.
        </Card>
        <Card title="If you want the surprising bit" note="Discovered structure">
          Twenty thousand real programs the generator wrote, each with the
          sequence it emitted and the round it first appeared.
        </Card>
        <Card title="If you want to check us" note="Every page">
          Every number is recomputed from released data by{' '}
          <code>tools/build_data.py</code> and gated by 134 assertions in{' '}
          <code>tools/verify_data.py</code>. No figure is transcribed by hand.
        </Card>
      </div>
    </>
  )
}

/** Our CPU measurements: bits/byte per self-play round, with the uniform line. */
function LocalMeasurementChart({ points }: { points: number[] }) {
  const lastIndex = Math.max(points.length - 1, 1)
  const domain: [number, number] = [0.6, 9]
  return (
    <ChartFrame
      height={280}
      xScale="linear"
      xDomain={[0, lastIndex + 1]}
      yDomain={domain}
      xLabel="self-play round (index, not to scale)"
      yLabel="bits/byte on real DCLM text"
    >
      {({ x, y, innerWidth }) => (
        <g>
          <line
            x1={0} x2={innerWidth}
            y1={y(8)} y2={y(8)}
            stroke="var(--warn)" strokeDasharray="4 4" strokeWidth={1}
          />
          <text x={innerWidth - 4} y={y(8) - 5} textAnchor="end"
                fill="var(--warn)" fontSize={11}>
            uniform over 256 bytes = 8.0
          </text>
          {points.map((bpb, i) => (
            <g key={i}>
              <circle cx={x(i)} cy={y(bpb)} r={i === 0 ? 5 : 4}
                      fill={i === 0 ? 'var(--warn)' : 'var(--arm-selfplay)'} />
              <text x={x(i)} y={y(bpb) - 10} textAnchor="middle"
                    fill="var(--text-muted)" fontSize={11}>
                {bpb.toFixed(2)}
              </text>
            </g>
          ))}
          {points.length > 1 && (
            <Series points={points.map((bpb, i) => [i, bpb])}
                    color="var(--arm-selfplay)" showPoints={false} />
          )}
        </g>
      )}
    </ChartFrame>
  )
}

/** Small preview of the three arms, reused by the Scaling page's summary. */
export function ArmPreview() {
  return (
    <Async load={loadScaling}>
      {(s) => {
        const dclm = 'dclm'
        const series = (['selfplay', 'uniform', 'pcfg'] as const)
          .map((arm) => ({
            arm,
            pts: s.arms[arm].corpora[dclm]?.frontier ?? [],
          }))
          .filter((x) => x.pts.length > 1)
        const all = series.flatMap((x) => x.pts.map((p) => p[1]))
        return (
          <ChartFrame
            height={300}
            xDomain={[
              Math.min(...series.flatMap((x) => x.pts.map((p) => p[0]))),
              Math.max(...series.flatMap((x) => x.pts.map((p) => p[0]))),
            ]}
            yDomain={lossDomain(all)}
            xLabel="effective compute"
            yLabel="bits/byte (DCLM text)"
          >
            {({ x, y }) => (
              <g>
                {series.map((s2) => (
                  <Series
                    key={s2.arm}
                    points={s2.pts.map((p) => [x(p[0]), y(p[1])])}
                    color={ARM_COLORS[s2.arm]}
                    dash={s2.arm === 'selfplay' ? null : '5 4'}
                  />
                ))}
              </g>
            )}
          </ChartFrame>
        )
      }}
    </Async>
  )
}