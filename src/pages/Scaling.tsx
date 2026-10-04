/**
 * Scaling: the compute-optimal frontier, the per-modality exponents, and the
 * three-arm comparison.
 *
 * D3 is the load-bearing constraint on this page. Table 2 and Figure 2 come
 * from DIFFERENT files -- `scaling.ladder` (frontier_traj_perk.json) versus
 * `scaling.arms` (the fig2 CSVs) -- and their frontiers disagree on every row.
 * The ladder section is therefore labelled as Table 2's source and the arms
 * section as Figure 2's, and the two are never mixed in one chart or one claim.
 */
import { useMemo, useState, type MouseEvent as ReactMouseEvent } from 'react'
import { loadMeta, loadScaling } from '@/data/loaders'
import {
  Async, Card, Claim, Disclosure, Legend, PaperRef, Stat,
} from '@/components/UI'
import {
  ChartFrame, HoverReadout, Series, useTooltip, type TooltipPayload,
} from '@/components/Chart'
import {
  ARM_ORDER, MAX_RAW_POINTS, TIE_MARGIN, armTotals, compareArms, corpusGroups,
  fitCurve, floorReading, formatCompute, groupVerdict, isTie, ladderFit,
  literatureEntries, lossDomain, ordinal, paperExponent, amplitude,
  sliceCorpus, subsample, subsampleStride, tallyGroups, type RawPoint,
} from '@/lib/scaling'
import type {
  ArmScaling, CorpusScaling, Meta, Scaling,
} from '@/data/types'

/** Both bundles the page needs, fetched once and cached by the loaders. */
function loadPage(): Promise<{ meta: Meta; scaling: Scaling }> {
  return Promise.all([loadMeta(), loadScaling()]).then(([meta, scaling]) => ({
    meta,
    scaling,
  }))
}

// ---------------------------------------------------------------------------

export function Scaling() {
  return (
    <>
      <h1 className="page__title">Scaling laws</h1>
      <p className="page__lede">
        The headline claim is a shape, not a number: zero-shot loss on held-out
        natural data falls as a power law in <em>self-play compute</em>, with no
        gradient step ever taken on that data. This page lets you pick a corpus,
        look at every scored point behind its frontier, read the fitted
        exponent, and check the claim that the curriculum — not merely access to
        a universal program space — is what produces the scaling.
      </p>
      <Claim reference="sec3.1" also={['fig1', 'table2']}>
        Prediction loss on text, images, audio, speech, melody, DNA and code
        improves as a power law in the compute spent on self-play pretraining,
        and the fitted exponents are broadly comparable to — if a bit higher
        than — those from pretraining directly on the same modalities.
      </Claim>

      <Async load={loadPage}>
        {(page) => <ScalingExplorer meta={page.meta} scaling={page.scaling} />}
      </Async>
    </>
  )
}

function ScalingExplorer({ meta, scaling }: { meta: Meta; scaling: Scaling }) {
  const [key, setKey] = useState('dclm')
  const corpus = meta.corpora.find((c) => c.key === key) ?? meta.corpora[0]
  const selectedKey = corpus?.key ?? 'dclm'
  const cs = scaling.ladder.corpora[selectedKey]

  const groups = useMemo(() => corpusGroups(meta), [meta])

  if (!corpus || !cs) {
    return <div className="error">The ladder bundle has no frontier for {selectedKey}.</div>
  }

  return (
    <>
      <h2 className="section">The scored self-play ladder</h2>
      <p className="body">
        Every point below is one released model: a rung of the parameter ladder
        at a self-play round, scored on bytes it was never trained on, with an
        ensemble of K independently seeded copies averaged into one
        bits/byte number. The faint cloud is all of them; the solid line is the
        compute-optimal frontier — the lower envelope, i.e. the best loss
        reachable at or below each compute budget; the dashed line is the fitted
        law L(C) = E + A·C<sup>−b</sup>.{' '}
        <PaperRef reference="sec3.1" also={['fig1']} inline />
      </p>

      <div className="controls">
        <div className="field">
          <label className="field__label" htmlFor="scaling-corpus">Corpus</label>
          <select
            id="scaling-corpus"
            value={selectedKey}
            onChange={(e) => setKey(e.target.value)}
          >
            {groups.map((g) => (
              <optgroup key={g.group} label={g.group}>
                {g.items.map((c) => (
                  <option key={c.key} value={c.key}>
                    {c.paperLabel}
                    {c.inTable2 ? '' : ' — not in Table 2'}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
        <div className="field">
          <span className="field__label">Group</span>
          <span style={{ fontSize: 13 }}>{corpus.group}</span>
        </div>
        <div className="field">
          <span className="field__label">In the paper&rsquo;s Table 2</span>
          <span style={{ fontSize: 13 }}>
            {corpus.inTable2 ? 'yes' : 'no — scored here, not tabulated'}
          </span>
        </div>
      </div>

      <Card
        title={`${corpus.paperLabel}: frontier and fit`}
        note="Source: scaling.ladder — the authors' scored ladder (frontier_traj_perk.json). This is Table 2's source, not Figure 2's."
      >
        <FrontierChart
          label={corpus.paperLabel}
          cs={cs}
          rows={scaling.ladder}
          corpusKey={selectedKey}
        />
      </Card>

      <FitCard cs={cs} label={corpus.paperLabel} meta={meta} />

      <h2 className="section">Table 2: per-modality compute exponents</h2>
      <p className="body">
        The same fits, tabulated as the paper tabulates them, against published
        exponents for pretraining directly on the same modality. Values marked
        † were converted from published Chinchilla-form (α, β) fits via
        b = αβ/(α+β) under the compute-optimal allocation; the conversion is
        applied once, at build time.{' '}
        <PaperRef reference="table2" also={['sec3.1']} inline />
      </p>
      <ExponentTable scaling={scaling} />
      <div className="claim claim--caveat">
        <p className="claim__text">
          <strong>DNA is the row to distrust.</strong> Our exponent of 0.435 is
          well above every published DNA value
          (0.01–0.06) — seven times the highest — and the paper prints it without
          hedging. A steeper exponent here means the fit is still bending
          downward over the range it was given, not that DNA is seven times more
          learnable from programs than from genomes. The other rows with
          literature values are not all close: our text exponent is 2.6× its
          nearest published value and two image/audio rows exceed 2.1×, while
          four rows have no published value to compare against.{' '}
          <PaperRef reference="table2" inline />
        </p>
      </div>

      <h2 className="section">Figure 2: the three arms, separately</h2>
      <p className="body">
        Everything below comes from <strong>a different file</strong>: the
        per-K trajectory CSVs behind Figure 2, which carry all three pretraining
        regimes in one shape. Their frontiers do not coincide with the ladder's,
        so these exponents are not Table 2's and are not shown as such.{' '}
        <PaperRef reference="fig2" also={['fig7', 'sec3.1']} inline />
      </p>
      <Card
        title={`${corpus.paperLabel}: all three arms`}
        note="Source: scaling.arms — the Figure 2 CSVs. Never mixed with the ladder above (docs/decisions.md D3)."
      >
        <ArmChart scaling={scaling} meta={meta} corpusKey={selectedKey} />
      </Card>

      <ArmVerdicts scaling={scaling} meta={meta} />

      <Disclosure summary="What the compute axis actually counts">
        <div className="kv">
          <span className="kv__k">C</span>
          <span className="kv__v">K · N · pool · {meta.compute.context} · (round + 1)</span>
          <span className="kv__k">K</span>
          <span className="kv__v">ensemble size — number of independently seeded copies averaged</span>
          <span className="kv__k">N</span>
          <span className="kv__v">parameter count of the rung</span>
          <span className="kv__k">pool</span>
          <span className="kv__v">
            programs sampled per round: a constant {meta.compute.pool} for
            self-play, and a per-rung count for each fixed arm, which is what
            the authors used when they drew these curves
          </span>
          <span className="kv__k">{meta.compute.context}</span>
          <span className="kv__v">context length in bytes, i.e. tokens per program</span>
          <span className="kv__k">round + 1</span>
          <span className="kv__v">number of learner rounds trained</span>
        </div>
        <p className="body" style={{ marginTop: 10 }}>
          C is a FLOP-equivalent count, not a GPU-hour measurement: parameters ×
          tokens × constant. The ensemble factor is there because each of the K
          seeds is a separately trained model over the same token stream, so
          producing the averaged prediction genuinely costs K times one
          model&rsquo;s forward and backward pass. That is also why the ladder
          reaches K = 14 on some points: ensembling is a compute-buying move
          the ladder is allowed to make, and the axis charges it honestly.{' '}
          <PaperRef reference="sec3.1" also={['fig1', 'appB']} inline />
        </p>
        <p className="body">
          A consequence worth stating plainly: because C multiplies K, the same
          rung and round appear at many different x positions, and the frontier
          is a Pareto envelope over (round, K) within each rung and then across
          rungs. That is why the frontier is not a function of round alone.{' '}
          <PaperRef reference="fig1" inline />
        </p>
      </Disclosure>

      <h2 className="section">How far to push this</h2>
      <div className="claim claim--caveat">
        <p className="claim__text">
          <strong>These are small models.</strong> The ladder tops out at{' '}
          {((meta.rungs[meta.rungs.length - 1]?.paper_params ?? 0) / 1e6).toFixed(1)}M
          parameters at a context of {meta.compute.context.toLocaleString()}{' '}
          bytes, so every exponent here is fitted over roughly four orders of
          magnitude of compute and nothing beyond it. Power laws fitted over a
          finite range are statements about that range; the paper is explicit
          that the exponents are broadly comparable to, if a bit higher than,
          natural-data pretraining — a statement about shape, not a promise that
          the curve holds at 10²⁵ FLOP.{' '}
          <PaperRef reference="sec6" also={['sec4', 'table2']} inline />
        </p>
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// Frontier chart (scaling.ladder only)
// ---------------------------------------------------------------------------

function FrontierChart({
  label, cs, rows, corpusKey,
}: {
  label: string
  cs: CorpusScaling
  rows: ArmScaling
  corpusKey: string
}) {
  const { payload, setPayload } = useTooltip()
  const { kept, total } = useMemo(
    () => subsample(sliceCorpus(rows, corpusKey).rows, MAX_RAW_POINTS),
    [rows, corpusKey],
  )

  const frontier = cs.frontier
  const fitPoints = useMemo(() => fitCurve(cs), [cs])

  const allC = kept.map((p) => p.c).concat(frontier.map((p) => p[0]))
  const allL = kept.map((p) => p.bpb).concat(frontier.map((p) => p[1]))
  if (allC.length === 0) return <div className="loading">No scored points.</div>
  const xDomain: [number, number] = [Math.min(...allC), Math.max(...allC)]
  const yDomain = lossDomain(allL.concat(fitPoints.map((p) => p[1])))

  return (
    <>
      <ChartFrame
        height={420}
        xDomain={xDomain}
        yDomain={yDomain}
        xLabel="effective compute C = K · N · pool · 4096 · (round+1)"
        yLabel={`bits/byte on ${label}`}
      >
        {({ x, y, innerWidth, innerHeight }) => {
          const projected = kept.map((p) => [x(p.c), y(p.bpb)] as const)
          const hover = (e: ReactMouseEvent<SVGRectElement>) => {
            const svg = e.currentTarget.ownerSVGElement
            const ctm = svg?.getScreenCTM()
            if (!svg || !ctm) return
            const loc = new DOMPoint(e.clientX, e.clientY).matrixTransform(ctm.inverse())
            let bestIdx = -1
            let bestD = 18 * 18
            projected.forEach(([px, py], i) => {
              const d = (px - loc.x) ** 2 + (py - loc.y) ** 2
              if (d < bestD) { bestD = d; bestIdx = i }
            })
            const p = bestIdx >= 0 ? kept[bestIdx] : undefined
            setPayload(p ? pointPayload(p, label, e) : null)
          }
          return (
            <g>
              {kept.map((_, i) => (
                <circle
                  key={i}
                  cx={projected[i]?.[0] ?? 0}
                  cy={projected[i]?.[1] ?? 0}
                  r={1.6}
                  fill="var(--text-faint)"
                  opacity={0.45}
                />
              ))}
              <Series
                points={frontier.map((p) => [x(p[0]), y(p[1])])}
                color="var(--arm-selfplay)"
                width={2}
              />
              <Series
                points={frontier.map((p) => [x(p[0]), y(p[1])])}
                color="var(--arm-selfplay)"
                showPoints
                pointRadius={2.6}
              />
              <Series
                points={fitPoints.map(([c, l]) => [x(c), y(l)])}
                color="var(--arm-uniform)"
                dash="6 4"
                width={1.6}
              />
              <rect
                x={0} y={0}
                width={innerWidth} height={innerHeight}
                fill="transparent"
                onMouseMove={hover}
                onMouseLeave={() => setPayload(null)}
              />
            </g>
          )
        }}
      </ChartFrame>
      <Legend
        items={[
          { label: 'every scored ladder point', color: 'var(--text-faint)' },
          { label: 'compute-optimal frontier', color: 'var(--arm-selfplay)' },
          { label: 'fit L(C) = E + A·C⁻ᵇ', color: 'var(--arm-uniform)', dash: '6 4' },
        ]}
      />
      <p className="card__note" style={{ marginTop: 8 }}>
        {(() => {
          const stride = subsampleStride(total)
          return kept.length === total
            ? `All ${total.toLocaleString()} scored ladder points for this corpus are drawn.`
            : `${kept.length.toLocaleString()} of ${total.toLocaleString()} scored ladder points drawn (every ${stride}${ordinal(stride)} in compute order), capped at ${MAX_RAW_POINTS} circles.`
        })()}
        {' '}Hover a point for its compute, loss, parameter count, ensemble size
        and round.
      </p>
      <HoverReadout payload={payload} />
    </>
  )
}

function pointPayload(
  p: RawPoint,
  label: string,
  e: ReactMouseEvent<SVGRectElement>,
): TooltipPayload {
  return {
    x: e.clientX,
    y: e.clientY,
    title: `${label} — C = ${formatCompute(p.c)}`,
    rows: [
      ['bits/byte', p.bpb.toFixed(3)],
      ['compute C', `${formatCompute(p.c)} (${p.c.toExponential(2)})`],
      ['params N', p.n.toLocaleString()],
      ['ensemble K', String(p.k)],
      ['round', String(p.round)],
    ],
  }
}

// ---------------------------------------------------------------------------
// The fit, stated plainly
// ---------------------------------------------------------------------------

function FitCard({
  cs, label, meta,
}: { cs: CorpusScaling; label: string; meta: Meta }) {
  const fit = ladderFit(cs)
  const reading = floorReading(cs)
  return (
    <Card
      title={`The fit for ${label}`}
      note={`${cs.nFrontier} points on the frontier, ${meta.compute.pool} programs per self-play round, ${meta.compute.context}-byte context.`}
    >
      {fit ? (
        <>
          <div className="grid-3">
            <Stat value={paperExponent(fit.alpha)} label="b — the compute exponent" />
            <Stat value={amplitude(fit.amplitude)} label="A — loss-scale prefactor" />
            <Stat value={fit.floor.toFixed(3)} label="E — fitted floor (bits/byte)" />
            <Stat value={fit.rmse.toFixed(3)} label="RMSE (bits/byte)" />
            <Stat value={String(fit.nPoints)} label="points in the fit" />
            <Stat
              value={`${cs.bestLoss.toFixed(3)}`}
              label={`best frontier loss at C = ${formatCompute(cs.maxCompute)}`}
            />
          </div>
          <p className="body" style={{ marginTop: 14 }}>
            The fitted law is L(C) = E + A·C<sup>−b</sup>.{' '}
            <strong>E is the asymptotic irreducible loss</strong> — the bits per
            byte this arm would still pay at infinite self-play compute, with no
            natural data at any point. For {label} the fit puts it at{' '}
            {fit.floor.toFixed(3)} bits/byte, against a best measured loss of{' '}
            {cs.bestLoss.toFixed(3)} at C = {formatCompute(cs.maxCompute)}.{' '}
            {(reading === 'broken-down') ? (
              <>
                {' '}Note that the best measurement is <em>below</em> the fitted
                floor, or the floor itself pinned at zero: on this corpus the
                extrapolation has broken down, and E should be read as &ldquo;less
                than anything measured&rdquo; rather than as a number.
              </>
            ) : (
              <>
                {' '}The ladder does not approach it, so E remains an
                extrapolation well beyond the measured range.
              </>
            )}{' '}
            <PaperRef reference="table2" also={['sec3.1']} inline />
          </p>
          <p className="card__note">
            RMSE is in bits/byte on the raw loss, not in log space — the same
            units as the axis, so it reads directly against the frontier's
            scatter.
          </p>
        </>
      ) : (
        <p className="body">
          No fit for this corpus: its frontier is too short or spans too few
          decades for the paper&rsquo;s fitting procedure, which requires at
          least six points.{' '}
          <PaperRef reference="table2" inline />
        </p>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Table 2
// ---------------------------------------------------------------------------

/**
 * Literature exponents, already converted where the conversion was needed.
 * `literatureChinchilla` holds b = αβ/(α+β) values, NOT (α, β) pairs — do not
 * convert them again.
 */
function literatureCell(scaling: Scaling, key: string) {
  const cell = literatureEntries(scaling, key)
  if (!cell) return null
  const { entries, dash } = cell
  if (entries.length === 0) {
    return dash ? (
      <span className="pill">— none found</span>
    ) : (
      <span style={{ color: 'var(--text-faint)' }}>—</span>
    )
  }
  return (
    <>
      {entries.map((e, i) => (
        <span key={i}>
          {i > 0 && ' – '}
          {paperExponent(e.v)}
          {e.derived && <sup title="converted from Chinchilla form">†</sup>}
        </span>
      ))}
    </>
  )
}

function ExponentTable({ scaling }: { scaling: Scaling }) {
  const rows = scaling.exponents
  return (
    <div className="table-wrap">
      <table className="data">
        <thead>
          <tr>
            <th>Modality / dataset</th>
            <th>Group</th>
            <th>our b</th>
            <th>literature b</th>
            <th className="left">reference</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => {
            const outlier = r.key === 'dna'
            return (
              <tr key={r.key}>
                <td>
                  {r.paperLabel}{' '}
                  {outlier && <span className="pill">outlier</span>}
                </td>
                <td>{r.group}</td>
                <td className="num" style={outlier ? { color: 'var(--warn)' } : undefined}>
                  {r.alpha === null ? '—' : paperExponent(r.alpha)}
                </td>
                <td className="num">{literatureCell(scaling, r.key)}</td>
                <td className="left" style={{ color: 'var(--text-faint)' }}>
                  {r.refs.length ? r.refs.join('; ') : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Figure 2: three arms (scaling.arms only)
// ---------------------------------------------------------------------------

function ArmChart({
  scaling, meta, corpusKey,
}: { scaling: Scaling; meta: Meta; corpusKey: string }) {
  const series = ARM_ORDER.map((arm) => ({
    arm,
    info: meta.arms[arm],
    pts: scaling.arms[arm].corpora[corpusKey]?.frontier ?? [],
  })).filter((s) => s.pts.length > 1)
  if (series.length === 0) return <div className="loading">No frontiers for this corpus.</div>
  const cs = series.flatMap((s) => s.pts)
  const xDomain: [number, number] = [
    Math.min(...cs.map((p) => p[0])),
    Math.max(...cs.map((p) => p[0])),
  ]
  return (
    <>
      <ChartFrame
        height={380}
        xDomain={xDomain}
        yDomain={lossDomain(cs.map((p) => p[1]))}
        xLabel="effective compute C"
        yLabel="bits/byte"
      >
        {({ x, y }) => (
          <g>
            {series.map((s) => (
              <Series
                key={s.arm}
                points={s.pts.map((p) => [x(p[0]), y(p[1])])}
                color={s.info.color}
                dash={s.info.dash}
                width={2}
              />
            ))}
          </g>
        )}
      </ChartFrame>
      <Legend
        items={series.map((s) => ({
          label: `${s.info.short} — ${s.info.blurb}`,
          color: s.info.color,
          dash: s.info.dash,
        }))}
      />
    </>
  )
}

function ArmVerdicts({ scaling, meta }: { scaling: Scaling; meta: Meta }) {
  const verdicts = useMemo(() => compareArms(scaling, meta), [scaling, meta])
  const tallies = useMemo(() => tallyGroups(verdicts), [verdicts])
  const { totals, ties: tieCount } = armTotals(verdicts)

  return (
    <>
      <Card
        title="Who wins where, at a compute budget all three arms reached"
        note={`Each arm is scored at the largest effective compute its own frontier reached AND all the others reached, so no arm is credited for compute the others never trained to. Source: scaling.arms.`}
      >
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>Corpus</th>
                <th>group</th>
                {ARM_ORDER.map((a) => (
                  <th key={a}>{meta.arms[a].short}</th>
                ))}
                <th>lowest</th>
                <th>margin</th>
              </tr>
            </thead>
            <tbody>
              {verdicts.map((v) => (
                <tr key={v.key}>
                  <td>{v.label}</td>
                  <td>{v.group}</td>
                  {ARM_ORDER.map((a) => (
                    <td key={a} className="num">
                      {v.best[a] === null ? '—' : (v.best[a] ?? 0).toFixed(3)}
                    </td>
                  ))}
                  <td style={{ color: v.winner ? meta.arms[v.winner].color : undefined }}>
                    {v.winner ? meta.arms[v.winner].short : '—'}
                  </td>
                  <td className="num">
                    {v.margin === null
                      ? '—'
                      : isTie(v.margin)
                        ? `${v.margin.toFixed(3)} (tie)`
                        : v.margin.toFixed(3)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="card__note" style={{ marginTop: 8 }}>
          All losses in bits/byte at the shared budget. Margin is the gap to the
          runner-up; anything under {TIE_MARGIN} bits/byte is reported as a tie
          rather than a win.
        </p>
      </Card>

      <h3 className="subhead">Per-group verdict</h3>
      <div className="grid-2">
        {tallies.map((t) => (
          <Card key={t.group} title={t.group}>
            <p className="body">{groupVerdict(t, meta)}</p>
          </Card>
        ))}
      </div>
      <p className="card__note">
        This count covers the {verdicts.length} corpora above and omits six
        binary and synthetic sets, which the bundle marks{' '}
        <code>excluded</code>: arithmetic, protein, KoLMogorov DNA, glibc rand,
        ATLAS float32 and astronomy. Six of the twenty-six ladder corpora are
        therefore not in this comparison. One of them matters: on glibc rand the
        fixed prior edges ahead of self-play (8.016 against 8.006 bits/byte),
        though by less than the {TIE_MARGIN} tie margin, so &ldquo;the prior never
        wins&rdquo; would not survive including it either way.
      </p>
      <p className="body" style={{ marginTop: 12 }}>
        Counted over all {verdicts.length} corpora scored by all three arms:
        self-play has the lowest loss on {totals.selfplay}, PCFG on{' '}
        {totals.pcfg}, and the fixed universal prior on {totals.uniform}. The
        honest reading is breadth, not dominance — PCFG is the better-matched
        inductive bias where a grammar was hand-designed for the modality. Over
        these {verdicts.length} the fixed prior never wins, which is the
        load-bearing control: same
        program space, no curriculum, materially slower scaling.{' '}
        <PaperRef reference="sec3.1" also={['fig2', 'fig7']} inline />
      </p>
      {tieCount > 0 && (
        <p className="card__note">
          {tieCount} of these {verdicts.length} comparisons are within{' '}
          {TIE_MARGIN} bits/byte — close enough that the winner is not
          meaningful. Those rows are flagged <code>(tie)</code> in the table
          above but still counted for one arm in the totals and per-group
          lines, so the headline counts can exceed the number of clear wins.
        </p>
      )}
    </>
  )
}