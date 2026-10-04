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
import { formatCompute, lossDomain, predictFit } from '@/lib/scales'
import type {
  ArmKey, ArmScaling, CorpusScaling, Meta, Scaling,
} from '@/data/types'

const ARM_ORDER: readonly ArmKey[] = ['selfplay', 'uniform', 'pcfg']

/** Never draw more raw circles than this; stride-subsample beyond it. */
/**
 * Every ladder corpus carries 1,490 rows, so a 1,500 cap would never fire and
 * the stride path in subsample would be unreachable. 1,400 keeps the scatter
 * readable AND leaves the stride branch live.
 */
const MAX_RAW_POINTS = 1400

/** Below this bits/byte gap the arms are not meaningfully separated. */
const TIE_MARGIN = 0.05

/** One scored ladder point, unpacked from the columnar arrays. */
interface RawPoint {
  c: number
  bpb: number
  n: number
  k: number
  round: number
}

/** Both bundles the page needs, fetched once and cached by the loaders. */
function loadPage(): Promise<{ meta: Meta; scaling: Scaling }> {
  return Promise.all([loadMeta(), loadScaling()]).then(([meta, scaling]) => ({
    meta,
    scaling,
  }))
}

/**
 * Render an exponent at the paper's printed precision of three decimals.
 * Ties round toward zero, because the authors printed 0.260 for the audio
 * 8-bit fit of 0.2605 (docs/findings.md F-note); `tools/verify_data.py`
 * accepts both renderings with a half-ulp window.
 */
function paperExponent(v: number): string {
  const scaled = v * 1000
  const floor = Math.floor(scaled)
  return ((scaled - floor > 0.5 + 1e-9 ? floor + 1 : floor) / 1000).toFixed(3)
}

/** The loss-scale prefactor of L(C) = E + A*C^-b, at the paper's precision. */
function amplitude(v: number): string {
  if (!Number.isFinite(v)) return '—'
  if (Math.abs(v) >= 1000) return v.toExponential(3)
  if (Math.abs(v) >= 10) return v.toFixed(2)
  return v.toFixed(3)
}

/**
 * One corpus's rows out of the columnar ladder. The rows are sorted by
 * (corpus, compute), so the target corpus occupies one contiguous run and the
 * scan stops as soon as it has passed it -- one pass over ~39k rows, no
 * intermediate arrays of 39k elements.
 */
function sliceCorpus(ladder: ArmScaling, key: string): { rows: RawPoint[]; total: number } {
  const target = ladder.stringTables.corpora.indexOf(key)
  const rows: RawPoint[] = []
  if (target < 0) return { rows, total: 0 }
  const cols = ladder.columns
  let total = 0
  for (let i = 0; i < cols.ci.length; i += 1) {
    const ci = cols.ci[i]
    if (ci === undefined) continue
    if (ci > target) break
    if (ci !== target) continue
    total += 1
    const c = cols.C[i]
    const bpb = cols.bpb[i]
    const n = cols.N[i]
    const k = cols.K[i]
    const round = cols.round[i]
    if (c === undefined || bpb === undefined || n === undefined
      || k === undefined || round === undefined) continue
    rows.push({ c, bpb, n, k, round })
  }
  return { rows, total }
}

/** Deterministic stride subsample, so the drawn point count is reproducible. */
function subsample<T>(items: T[], cap: number): { kept: T[]; total: number } {
  const total = items.length
  if (total <= cap) return { kept: items, total }
  const stride = Math.ceil(total / cap)
  return { kept: items.filter((_, i) => i % stride === 0), total }
}

/**
 * Score the three arms at a compute budget they all reached, and say which is
 * lowest. Comparing `bestLoss` directly would be unfair: self-play runs to
 * ~10x the compute of the fixed-prior arm, so its frontier always ends lower.
 * Restricting every arm to the largest budget all three reached turns that
 * confound into the comparison.
 */
interface ArmVerdict {
  key: string
  label: string
  group: string
  budget: number
  best: Record<ArmKey, number | null>
  winner: ArmKey | null
  /** bits/byte between the winner and the runner-up; null if <2 arms scored. */
  margin: number | null
}

function compareArms(scaling: Scaling, meta: Meta): ArmVerdict[] {
  const out: ArmVerdict[] = []
  for (const corpus of meta.corpora) {
    if (corpus.group === 'excluded') continue
    const frontiers = ARM_ORDER.map((arm) => ({
      arm,
      pts: scaling.arms[arm].corpora[corpus.key]?.frontier ?? [],
    })).filter((a) => a.pts.length > 0)
    if (frontiers.length < 2) continue
    const budget = Math.min(...frontiers.map((a) => a.pts[a.pts.length - 1]?.[0] ?? Infinity))
    const best = {} as Record<ArmKey, number | null>
    for (const { arm, pts } of frontiers) {
      const reachable = pts.filter((p) => p[0] <= budget * (1 + 1e-7))
      best[arm] = reachable.length
        ? Math.min(...reachable.map((p) => p[1]))
        : null
    }
    const scored = ARM_ORDER.filter((a) => best[a] !== null)
    const sorted = [...scored].sort((a, b) => (best[a] ?? 0) - (best[b] ?? 0))
    const top = sorted[0]
    const second = sorted[1]
    out.push({
      key: corpus.key,
      label: corpus.paperLabel,
      group: corpus.group,
      budget,
      best,
      winner: top ?? null,
      margin: top !== undefined && second !== undefined
        ? (best[second] ?? 0) - (best[top] ?? 0)
        : null,
    })
  }
  return out
}

/** Win counts per corpus group, plus how many of those wins were near-ties. */
interface GroupTally {
  group: string
  counts: Record<ArmKey, number>
  total: number
  ties: number
}

function tallyGroups(verdicts: ArmVerdict[]): GroupTally[] {
  const order: string[] = []
  const byGroup = new Map<string, GroupTally>()
  for (const v of verdicts) {
    let tally = byGroup.get(v.group)
    if (!tally) {
      tally = {
        group: v.group,
        counts: { selfplay: 0, uniform: 0, pcfg: 0 },
        total: 0,
        ties: 0,
      }
      byGroup.set(v.group, tally)
      order.push(v.group)
    }
    tally.total += 1
    if (v.winner) tally.counts[v.winner] += 1
    if (v.margin !== null && v.margin < TIE_MARGIN) tally.ties += 1
  }
  return order.map((g) => byGroup.get(g)).filter((t): t is GroupTally => t !== undefined)
}

/** The verdict line, written from the tally rather than asserted by hand. */
function groupVerdict(t: GroupTally, meta: Meta): string {
  const entries = ARM_ORDER
    .map((arm) => ({ arm, n: t.counts[arm], name: meta.arms[arm].short }))
    .sort((a, b) => b.n - a.n)
  const lead = entries[0]
  if (!lead || lead.n === 0) return 'No arm reached the shared budget.'
  const noun = t.total === 1 ? 'corpus' : 'corpora'
  const share = `${lead.name} reaches the lowest loss on ${lead.n} of ${t.total} ${noun}`
  const rest = entries.slice(1).filter((e) => e.n > 0)
  const also = rest.length
    ? `; ${rest.map((e) => `${e.name} on ${e.n}`).join(', ')}`
    : ''
  const tie = t.ties > 0
    ? ` ${t.ties === 1 ? 'One of those is' : `${t.ties} of those are`} within `
      + `${TIE_MARGIN} bits/byte — treat ${t.ties === 1 ? 'it' : 'those'} as a tie.`
    : ''
  return `${share}${also}.${tie}`
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

  const groups = useMemo(() => {
    const seen: string[] = []
    for (const c of meta.corpora) if (!seen.includes(c.group)) seen.push(c.group)
    return seen.map((g) => ({
      group: g,
      items: meta.corpora.filter((c) => c.group === g),
    }))
  }, [meta])

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
  const fitPoints = useMemo(() => {
    if (!cs.fit || frontier.length < 2) return []
    const lo = frontier[0]?.[0] ?? 0
    const hi = frontier[frontier.length - 1]?.[0] ?? 0
    if (!(lo > 0) || !(hi > lo)) return []
    const out: [number, number][] = []
    for (let i = 0; i <= 48; i += 1) {
      const c = lo * Math.pow(hi / lo, i / 48)
      out.push([c, predictFit(cs.fit, c)])
    }
    return out
  }, [cs, frontier])

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
        {kept.length === total
          ? `All ${total.toLocaleString()} scored ladder points for this corpus are drawn.`
          : `${kept.length.toLocaleString()} of ${total.toLocaleString()} scored ladder points drawn (every ${Math.ceil(total / kept.length)}${ordinal(Math.ceil(total / kept.length))} in compute order), capped at ${MAX_RAW_POINTS} circles.`}
        {' '}Hover a point for its compute, loss, parameter count, ensemble size
        and round.
      </p>
      <HoverReadout payload={payload} />
    </>
  )
}

function ordinal(n: number): string {
  if (n % 10 === 1 && n % 100 !== 11) return 'st'
  if (n % 10 === 2 && n % 100 !== 12) return 'nd'
  if (n % 10 === 3 && n % 100 !== 13) return 'rd'
  return 'th'
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
  const fit = cs.fit
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
            {(cs.bestLoss < fit.floor || fit.floor === 0) ? (
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
  const row = scaling.exponents.find((e) => e.key === key)
  if (!row) return null
  const entries = [
    ...row.literatureValues.map((v) => ({ v, derived: false })),
    ...row.literatureChinchilla.map((v) => ({ v, derived: true })),
  ].sort((a, b) => a.v - b.v)
  if (entries.length === 0) {
    return row.literatureDash ? (
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
            <th style={{ textAlign: 'left' }}>reference</th>
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
                <td style={{ textAlign: 'left', color: 'var(--text-faint)' }}>
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
  const totals = ARM_ORDER.reduce(
    (acc, arm) => {
      acc[arm] = verdicts.filter((v) => v.winner === arm).length
      return acc
    },
    {} as Record<ArmKey, number>,
  )
  const tieCount = verdicts.filter((v) => v.margin !== null && v.margin < TIE_MARGIN).length

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
                      : v.margin < TIE_MARGIN
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