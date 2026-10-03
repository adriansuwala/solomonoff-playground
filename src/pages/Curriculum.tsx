/**
 * Curriculum value (Figure 3): does a later generator actually produce better
 * training data, or just more of it?
 *
 * The paper answers this by fixing a corpus per generator checkpoint and training
 * a fresh 1M-parameter learner on each, then measuring epiplexity and
 * out-of-distribution transfer. Epiplexity is the excess training loss a
 * compute-bounded learner accumulates before converging — a proxy for how much
 * structure the data contains.
 */
import { useState } from 'react'
import { loadCurriculum } from '@/data/loaders'
import {
  Async, Card, Claim, Disclosure, Legend, PaperRef,
} from '@/components/UI'
import { ChartFrame, Series } from '@/components/Chart'
import type { CurriculumRow } from '@/data/types'

const MODALITIES = [
  { key: 'dclm', label: 'Text (DCLM)', color: 'var(--arm-selfplay)' },
  { key: 'audio', label: 'Audio (16-bit PCM)', color: 'var(--arm-pcfg)' },
  { key: 'cifar', label: 'Images (CIFAR-10 planar)', color: 'var(--arm-uniform)' },
] as const

type ModalityKey = (typeof MODALITIES)[number]['key']

/** Which corpus field backs each modality's transfer measure. */
const METRIC = {
  dclm: 'dclmSubsetMean',
  audio: 'audioSubsetMean',
  cifar: 'cifarSubsetMean',
} as const

const SD_METRIC = {
  dclm: 'dclmSubsetSd',
  audio: 'audioSubsetSd',
  cifar: 'cifarSubsetSd',
} as const

export function Curriculum() {
  const [modality, setModality] = useState<ModalityKey>('dclm')
  const [view, setView] = useState<'transfer' | 'epiplexity'>('transfer')
  const [showSd, setShowSd] = useState(true)

  return (
    <>
      <h1 className="page__title">Is the generator actually improving?</h1>
      <p className="page__lede">
        Scaling curves alone cannot tell you whether a curriculum is genuinely
        getting better. They are equally consistent with a generator that keeps
        producing more of the same material. This page isolates the question: fix
        a corpus from one generator checkpoint, train a fresh learner on it, and
        ask whether the corpus written <em>later</em> is worth more.
      </p>

      <Async load={loadCurriculum}>
        {(data) => (
          <>
            <div className="controls">
              <div className="field">
                <label className="field__label" htmlFor="curriculum-modality">
                  Held-out modality
                </label>
                <select
                  id="curriculum-modality"
                  value={modality}
                  onChange={(e) => setModality(e.target.value as ModalityKey)}
                >
                  {MODALITIES.map((m) => (
                    <option key={m.key} value={m.key}>{m.label}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label className="field__label" htmlFor="curriculum-view">Measure</label>
                <select
                  id="curriculum-view"
                  value={view}
                  onChange={(e) => setView(e.target.value as 'transfer' | 'epiplexity')}
                >
                  <option value="transfer">Transfer bits/byte (lower is better)</option>
                  <option value="epiplexity">Epiplexity (higher is better)</option>
                </select>
              </div>
              <label className="row row--tight" style={{ fontSize: 13 }}>
                <input
                  type="checkbox"
                  checked={showSd}
                  onChange={(e) => setShowSd(e.target.checked)}
                />
                show ±1 SD over seeds
              </label>
            </div>

            <CurriculumChart
              rows={data.rows}
              modality={modality}
              view={view}
              showSd={showSd}
            />

            <Legend
              items={[
                { label: 'individual seeds', color: 'var(--text-faint)' },
                { label: 'arm mean', color: 'var(--arm-selfplay)' },
              ]}
            />

            <h2 className="section">What the two panels say together</h2>
            <Claim reference="fig3">
              Epiplexity — the excess training loss a compute-bounded learner
              accumulates before converging — rises steadily with the generator
              endpoint. Later generators write corpora containing more extractable
              structure.
            </Claim>
            <Claim reference="fig3">
              Out-of-distribution validation loss on text, audio and images falls
              with the endpoint. The extra structure is not merely <em>more</em>{' '}
              structure: it is structure that transfers to data the learner has
              never seen.
            </Claim>
            <Claim reference="fig3">
              Taken together the panels rule out the deflationary reading. If later
              checkpoints were repeating earlier material, epiplexity would plateau
              and transfer would not improve.
            </Claim>

            <h2 className="section">Read the numbers</h2>
            <Card
              title={`${MODALITIES.find((m) => m.key === modality)!.label}: value by generator endpoint`}
              note="Endpoint T means the generator was trained for T rounds; the corpus for each arm mixes 16 snapshots spaced T/16 apart, with G₀ using the untrained generator."
            >
              <table className="data">
                <thead>
                  <tr>
                    <th>Arm</th>
                    <th>Endpoint T</th>
                    <th>Transfer bits/byte</th>
                    <th>±1 SD</th>
                    <th>Epiplexity</th>
                    <th>Seeds</th>
                  </tr>
                </thead>
                <tbody>
                  {data.rows.map((r) => (
                    <tr key={r.arm}>
                      <td>{r.arm}</td>
                      <td className="num">{r.T.toLocaleString()}</td>
                      <td className="num">{r[METRIC[modality]].toFixed(3)}</td>
                      <td className="num">{r[SD_METRIC[modality]].toFixed(3)}</td>
                      <td className="num">{r.epiplexityMean.toFixed(1)}</td>
                      <td className="num">{r.epiplexitySeeds.length}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <div className="grid-2">
              <Card title="Why 4-seed subsets, not a single seed?">
                <p className="body" style={{ margin: 0 }}>
                  A single 1M-parameter learner's loss is noisy enough to obscure
                  the trend. The paper reports the mean over 4-seed subsets with
                  the standard deviation across subsets, and separately the full
                  8-seed ensemble. Both are in the table above; the subsets are the
                  more honest of the two for judging a trend, because an 8-seed
                  ensemble also benefits from simply having more models.
                </p>
              </Card>
              <Card title="What epiplexity is and is not">
                <p className="body" style={{ margin: 0 }}>
                  Epiplexity (Finzi et al., 2026) is used instead of entropy or
                  Kolmogorov complexity because those unbounded notions cannot
                  capture structure only a compute-bounded observer could exploit —
                  which is the whole situation here. It measures excess loss{' '}
                  <em>before convergence</em>, so it reflects how much there was to
                  learn.
                </p>
              </Card>
            </div>

            <Disclosure summary="Methodology, if you want to check the construction">
              <p className="body">
                For each endpoint T the paper builds one fixed corpus by sampling
                4.19M programs uniformly across 16 generator checkpoints spaced
                T/16 apart and ending at T; g₀ uses the untrained generator. A fresh
                1M-parameter learner is then trained from scratch on each corpus for
                one epoch under a fixed token budget, four seeds per endpoint.{' '}
                <PaperRef reference="fig3" inline />
              </p>
              <p className="body">
                The construction deliberately removes generator <em>scale</em> from
                the comparison: what differs between arms is the training endpoint
                the corpus came from, not how much data there is.
              </p>
            </Disclosure>
          </>
        )}
      </Async>
    </>
  )
}

function CurriculumChart({
  rows, modality, view, showSd,
}: {
  rows: CurriculumRow[]
  modality: ModalityKey
  view: 'transfer' | 'epiplexity'
  showSd: boolean
}) {
  const isEpi = view === 'epiplexity'
  const values = rows.map((r) => (isEpi ? r.epiplexityMean : r[METRIC[modality]]))
  const sds = rows.map((r) => (isEpi ? r.epiplexitySd : r[SD_METRIC[modality]]))
  const color = MODALITIES.find((m) => m.key === modality)!.color

  const domain: [number, number] = (() => {
    const lo = Math.min(...values.map((v, i) => (showSd ? v - sds[i]! : v)))
    const hi = Math.max(...values.map((v, i) => (showSd ? v + sds[i]! : v)))
    const pad = Math.max((hi - lo) * 0.15, 0.1)
    return [Math.max(0, lo - pad), hi + pad]
  })()

  const first = rows[0]!
  const last = rows[rows.length - 1]!

  return (
    <ChartFrame
      height={360}
      xScale="linear"
      xDomain={[-(first.T || 200), last.T * 1.06]}
      yDomain={domain}
      xLabel="generator-training endpoint T (self-play rounds)"
      yLabel={isEpi ? 'epiplexity' : 'held-out bits/byte'}
    >
      {({ x, y }) => (
        <g>
          {showSd && (() => {
            // A band is the upper edge left-to-right, then the lower edge
            // right-to-left, closed. Building it with two passes avoids the
            // fragile string surgery an earlier version used.
            const upper = rows.map((r, i) =>
              `${i === 0 ? 'M' : 'L'}${x(r.T)},${y(values[i]! + sds[i]!)}`)
            const lower = rows.map((r, i) =>
              `L${x(r.T)},${y(values[i]! - sds[i]!)}`).reverse()
            return <path d={[...upper, ...lower, 'Z'].join(' ')} fill={color} opacity={0.13} />
          })()}
          {isEpi && rows.flatMap((r) =>
            r.epiplexitySeeds.map((v) => [r.T, v] as [number, number]),
          ).map(([t, v], i) => (
            <circle key={i} cx={x(t)} cy={y(v)} r={2.5}
                    fill="var(--text-faint)" opacity={0.55} />
          ))}
          <Series
            points={rows.map((r, i) => [x(r.T), y(values[i]!)] as [number, number])}
            color={color}
            width={2}
            showPoints
            pointRadius={4}
          />
          {rows.map((r, i) => (
            <text key={r.arm} x={x(r.T)} y={y(values[i]!) - 11} textAnchor="middle"
                  fontSize={10} fill="var(--text-faint)">
              {r.arm}
            </text>
          ))}
        </g>
      )}
    </ChartFrame>
  )
}