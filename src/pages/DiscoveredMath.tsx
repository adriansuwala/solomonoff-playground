/**
 * Discovered structure (Table 1, Appendix C): what the generator writes when it
 * writes mathematics.
 *
 * Every row in the browser below is a real program from the authors' released
 * scan output (hits.jsonl) — a training-time log, not a reconstruction. The
 * page's job is to let a reader check the claim rather than take it: the
 * recurrence a program emitted is re-verified in the browser against its own
 * first twelve terms, and shown passing or failing honestly.
 */
import { useEffect, useMemo, useState } from 'react'
import { loadMath, loadMeta } from '@/data/loaders'
import {
  Async, Card, Claim, Disclosure, Legend, PaperRef, Stat,
} from '@/components/UI'
import type { MathData, MathFamily, MathProgram, Rung } from '@/data/types'

const FAMILY_ORDER: MathFamily[] = [
  'arithmetic', 'fibonacci', 'geometric', 'quadratic', 'cubic',
]

const FAMILY_COLOR: Record<MathFamily, string> = {
  arithmetic: 'var(--arm-selfplay)',
  fibonacci: 'var(--arm-uniform)',
  geometric: 'var(--arm-pcfg)',
  quadratic: 'var(--warn)',
  cubic: 'var(--good)',
}

/** Rows per page in the browser. 20,045 rows never reach the DOM at once. */
const PAGE_SIZE = 60

export function DiscoveredMath() {
  return (
    <Async load={loadMeta}>
      {(meta) => (
        <Async load={loadMath}>
          {(math) => (
            <>
              <h1 className="page__title">Discovered mathematical structure</h1>
              <p className="page__lede">
                The generator learned to write programs whose output is a
                recognisable mathematical sequence — arithmetic, Fibonacci,
                geometric, quadratic, cubic — and it produced them far earlier
                than uniform sampling of the same program space allows. Those
                two halves of the sentence are worth separating, so this page
                shows both: the round-by-round comparison, and every program
                behind it.
              </p>

              <h2 className="section">The headline comparison</h2>
              <HeadlineComparison math={math} meta={meta} />

              <h2 className="section">What the statistics do and do not show</h2>
              <p className="body">
                The baseline is the sharpest part of the design. It drew{' '}
                {math.priorCounts.samples.toLocaleString()} programs by sampling
                i.i.d. uniformly over the instruction alphabet — including the
                augmented macro tokens — executed each with the same tape
                length, and passed the output through the{' '}
                <em>same detector</em> the self-play programs went through.
                It sampled{' '}
                {meta.mathPrior.programsPerRound.toLocaleString()} programs per
                round — two thirds of the {meta.compute.pool.toLocaleString()}{' '}
                the self-play generator emits, so the baseline is{' '}
                <em>not</em> rate-matched and its expected first discovery lands
                correspondingly later.{' '}
                <PaperRef reference="appC" inline />
              </p>
              <p className="body">
                Arithmetic was hit{' '}
                {math.priorCounts.hit_counts.arithmetic?.toLocaleString() ?? 0}{' '}
                times, giving p ≈ 9.3×10⁻⁶ and an expected first appearance near
                round 105. The other four families were hit{' '}
                <strong>zero</strong> times. With no hits in 1.64×10⁸ draws, the
                rule of three bounds p at {meta.mathPrior.ruleOfThree.toExponential(1)},
                which pushes the expected first round past 53,000 — while the
                self-play generator produced them by round 512 at the latest.{' '}
                <PaperRef reference="table1" also={['appC']} inline />
              </p>

              <Claim caveat reference="appC">
                Three asymmetries run in the generator's favour and are worth
                naming. First, all four rare families have <em>zero</em> baseline
                hits, so the experiment pins a common lower bound on how rare
                they are; it does not order them against each other, and the
                four rows are not comparable to the arithmetic row. Second, the
                self-play side was scanned only every 256 rounds, because that is
                how often program histories were checkpointed — so every
                "earliest round" reported here is a conservative{' '}
                <em>upper</em> bound on the true first appearance, and the real
                gap is likely wider, not narrower. Third, the arithmetic row at
                round 0 is not a discovery at all: it is what the generator's
                initialisation emits.
              </Claim>

              <Claim caveat reference="sec6">
                The causal claim stops here. That the generator <em>finds</em>
                these structures is established; that they are what drives
                transfer to natural data is not tested anywhere in the paper.
                The paper lists the experiments that would decide it — circuit
                analysis of the learner, and a curriculum ablation removing the
                mathematical families — as future work.
              </Claim>

              <h2 className="section">Browse the programs</h2>
              <p className="body">
                All {math.totalHits.toLocaleString()} hits from the released scan,
                filtered and paged below. Every recurrence shown is recomputed in
                your browser from the twelve terms the program emitted; nothing
                here is asserted on the paper's authority.
              </p>
              <ProgramBrowser
                math={math}
                rungs={meta.rungs}
                macros={meta.bf.macros.map((m) => m.token)}
                cellModulus={meta.bf.params.cell_modulus}
              />

              <h2 className="section">What counts as a hit</h2>
              <DetectionDisclosure
                examples={math.families.map((f) => ({
                  family: f.family,
                  program: f.program,
                  terms: f.terms,
                }))}
              />

              <h2 className="section">Provenance</h2>
              <Card
                title="Real training-time program logs, not a reconstruction"
                note={`${math.totalHits.toLocaleString()} hits across ${math.totalDistinctPrograms.toLocaleString()} distinct programs.`}
              >
                <p className="body" style={{ marginTop: 0 }}>
                  The rows come from the authors' released{' '}
                  <code>hits.jsonl</code>, the complete output of a scan over
                  every program the generator produced across all six ladder
                  rungs and 18 seeds — records of programs that were actually
                  executed during training, carrying the rung, seed and round
                  they came from. Nothing on this page is synthesised.{' '}
                  <PaperRef reference="appC" inline />
                </p>
                <p className="body">
                  {math.totalHits.toLocaleString()} hits contain{' '}
                  {math.totalDistinctPrograms.toLocaleString()} distinct program
                  strings: the same program is re-emitted in many rounds, which
                  is why the browser de-duplicates on rung, seed, round and text.
                </p>
              </Card>
            </>
          )}
        </Async>
      )}
    </Async>
  )
}

// ---------------------------------------------------------------------------
// Headline comparison
// ---------------------------------------------------------------------------

function HeadlineComparison({
  math, meta,
}: {
  math: MathData
  meta: { mathPrior: { programsPerRound: number; ruleOfThree: number } }
}) {
  const byFamily = useMemo(() => {
    const m = new Map<MathFamily, (typeof math.families)[number]>()
    for (const f of math.families) m.set(f.family, f)
    return m
  }, [math.families])

  // Log-scaled bars: the numbers span 0 → 53,000, so a linear bar would make
  // every self-play round indistinguishable from zero.
  const maxLog = Math.log10(53001)
  const barWidth = (round: number) => `${(Math.log10(round + 1) / maxLog) * 100}%`

  return (
    <Card
      title="Earliest round seen under self-play vs. expected first round under uniform sampling"
      note={`Baseline: ${math.priorCounts.samples.toLocaleString()} uniformly sampled programs, ${meta.mathPrior.programsPerRound.toLocaleString()} programs per round, same detector.`}
    >
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Family</th>
              <th>Self-play earliest round</th>
              <th>Expected first round, uniform prior</th>
              <th>Baseline hits</th>
              <th>Programs in the release</th>
            </tr>
          </thead>
          <tbody>
            {FAMILY_ORDER.map((fam) => {
              const f = byFamily.get(fam)
              if (!f) return null
              return (
                <tr key={fam}>
                  <td>
                    <span
                      className="legend__swatch"
                      style={{
                        background: FAMILY_COLOR[fam],
                        display: 'inline-block',
                        marginRight: 6,
                        verticalAlign: 'middle',
                      }}
                    />
                    {f.label}
                  </td>
                  <td className="num">
                    <div>{f.selfplay_round.toLocaleString()}</div>
                    <div
                      style={{
                        height: 3,
                        marginTop: 3,
                        background: FAMILY_COLOR[fam],
                        width: barWidth(f.selfplay_round),
                        minWidth: 2,
                      }}
                    />
                  </td>
                  <td className="num">
                    <div>
                      {f.prior_expected_round >= 53000
                        ? '> 53,000'
                        : `≈ ${f.prior_expected_round}`}
                    </div>
                    <div
                      style={{
                        height: 3,
                        marginTop: 3,
                        background: 'var(--text-faint)',
                        width: barWidth(f.prior_expected_round),
                      }}
                    />
                  </td>
                  <td className="num">
                    {f.prior_hits.toLocaleString()}
                    {f.prior_p !== null && (
                      <span style={{ color: 'var(--text-faint)' }}>
                        {' '}(p ≈ {f.prior_p.toExponential(1)})
                      </span>
                    )}
                  </td>
                  <td className="num">{f.hitCount.toLocaleString()}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <Legend
        items={[
          { label: 'self-play (learned sampler)', color: 'var(--arm-selfplay)' },
          { label: 'uniform Solomonoff-style prior', color: 'var(--text-faint)', dash: '4 3' },
        ]}
      />
      <p className="card__note">
        Bars are log-scaled in the round number; the numbers are the value. The
        four &gt;53,000 entries are the same bound: zero baseline hits means a
        shared limit, not four separate measurements.
      </p>
      <div className="claim claim--caveat" style={{ marginTop: 14 }}>
        <p className="claim__text">
          <strong>Arithmetic is at round 0, and that is not a discovery.</strong>{' '}
          The generator's initialisation — the Solomonoff-style prior g₀ —{' '}
          already emits an arithmetic ramp, so the earliest sampled program
          matches on the first scan. The interesting number in that row is the
          other column: even a programme the model starts with would take ≈105
          rounds to appear by chance.{' '}
          <PaperRef reference="table1" also={['appC']} inline />
        </p>
      </div>
      <div className="grid-3" style={{ marginTop: 16 }}>
        <Stat
          value={FAMILY_ORDER.map((fam) => byFamily.get(fam)?.selfplay_round)
            .filter((r): r is number => r !== undefined && r > 0)
            .reduce((a, b) => Math.max(a, b), 0).toLocaleString()}
          label="latest round any non-trivial family appeared"
        />
        <Stat
          value="> 53,000"
          label="expected first round under uniform sampling"
        />
        <Stat
          value={meta.mathPrior.programsPerRound.toLocaleString()}
          label="programs per round in the uniform baseline (self-play emits 1,536)"
        />
      </div>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// Program browser
// ---------------------------------------------------------------------------

function useDebounced<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setSettled(value), ms)
    return () => clearTimeout(t)
  }, [value, ms])
  return settled
}

/** Filter, then slice. The 20,045-row array is never mapped into the DOM. */
function ProgramBrowser({
  math, rungs, macros, cellModulus,
}: {
  math: MathData
  rungs: Rung[]
  macros: string[]
  cellModulus: number
}) {
  const [families, setFamilies] = useState<MathFamily[]>([])
  const [rungKeys, setRungKeys] = useState<string[]>([])
  const [minRound, setMinRound] = useState(0)
  const [rawQuery, setRawQuery] = useState('')
  const query = useDebounced(rawQuery.trim(), 200)
  const [page, setPage] = useState(0)
  const [selected, setSelected] = useState<MathProgram | null>(null)

  const maxRound = useMemo(
    () => math.programs.reduce((m, p) => Math.max(m, p.round), 0),
    [math.programs],
  )

  const rungLabel = useMemo(() => {
    const m = new Map<string, string>()
    for (const r of rungs) m.set(r.rung, r.label)
    return m
  }, [rungs])

  const filtered = useMemo(() => {
    const famSet = new Set(families)
    const rungSet = new Set(rungKeys)
    const q = query.toLowerCase()
    return math.programs.filter((p) => {
      if (famSet.size > 0 && !famSet.has(p.family)) return false
      if (rungSet.size > 0 && !rungSet.has(p.rung)) return false
      if (p.round < minRound) return false
      if (q && !p.program.toLowerCase().includes(q)) return false
      return true
    })
  }, [math.programs, families, rungKeys, minRound, query])

  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const safePage = Math.min(page, pageCount - 1)
  const rows = filtered.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE)

  // Counts per family under every filter *except* the family filter, so the
  // toggle labels stay honest while a family is deselected.
  const counts = useMemo(() => {
    const rungSet = new Set(rungKeys)
    const q = query.toLowerCase()
    const c: Record<string, number> = {}
    for (const p of math.programs) {
      if (rungSet.size > 0 && !rungSet.has(p.rung)) continue
      if (p.round < minRound) continue
      if (q && !p.program.toLowerCase().includes(q)) continue
      c[p.family] = (c[p.family] ?? 0) + 1
    }
    return c
  }, [math.programs, rungKeys, minRound, query])

  const reset = () => setPage(0)
  const toggle = <T,>(list: T[], v: T): T[] =>
    list.includes(v) ? list.filter((x) => x !== v) : [...list, v]

  return (
    <>
      <div className="controls">
        <div className="field">
          <span className="field__label">Family</span>
          <div className="row row--tight">
            {FAMILY_ORDER.map((fam) => (
              <button
                key={fam}
                type="button"
                aria-pressed={families.includes(fam)}
                onClick={() => {
                  setFamilies((f) => toggle(f, fam))
                  reset()
                }}
              >
                {fam} <span style={{ color: 'var(--text-faint)' }}>{counts[fam] ?? 0}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label className="field__label" htmlFor="dm-search">Program text</label>
          <input
            id="dm-search"
            type="text"
            placeholder="substring of the program text"
            value={rawQuery}
            onChange={(e) => { setRawQuery(e.target.value); reset() }}
            style={{ minWidth: 220 }}
          />
        </div>
      </div>

      <div className="controls">
        <div className="field">
          <span className="field__label">Rung</span>
          <div className="row row--tight">
            {rungs.map((r) => (
              <button
                key={r.rung}
                type="button"
                aria-pressed={rungKeys.includes(r.rung)}
                onClick={() => {
                  setRungKeys((k) => toggle(k, r.rung))
                  reset()
                }}
              >
                {r.label}
              </button>
            ))}
          </div>
        </div>
        <div className="field">
          <label className="field__label" htmlFor="dm-round">
            Minimum round — {minRound.toLocaleString()}
          </label>
          <input
            id="dm-round"
            type="range"
            min={0}
            max={maxRound}
            step={256}
            value={minRound}
            onChange={(e) => { setMinRound(Number(e.target.value)); reset() }}
          />
        </div>
        {(families.length > 0 || rungKeys.length > 0 || query || minRound > 0) && (
          <button
            type="button"
            onClick={() => {
              setFamilies([])
              setRungKeys([])
              setRawQuery('')
              setMinRound(0)
              reset()
            }}
          >
            clear filters
          </button>
        )}
      </div>

      <p className="card__note" style={{ marginTop: 0 }}>
        Showing{' '}
        {filtered.length === 0
          ? 'no programs'
          : `${(safePage * PAGE_SIZE + 1).toLocaleString()}–${Math.min(
              (safePage + 1) * PAGE_SIZE,
              filtered.length,
            ).toLocaleString()} of ${filtered.length.toLocaleString()}`}{' '}
        matching programs, out of {math.totalHits.toLocaleString()} hits in the
        release. Rounds advance in steps of 256 because that is how often program
        histories were saved.
      </p>

      {selected && (
        <ProgramDetail
          program={selected}
          macros={macros}
          cellModulus={cellModulus}
          rungLabel={rungLabel.get(selected.rung) ?? selected.rung}
          onClose={() => setSelected(null)}
        />
      )}

      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Family</th>
              <th>Round</th>
              <th>Rung</th>
              <th>Seed</th>
              <th>Program</th>
              <th>First terms</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((p, i) => (
              <tr
                key={`${p.rung}|${p.seed}|${p.round}|${i}`}
                onClick={() => setSelected(p)}
                style={{ cursor: 'pointer' }}
              >
                <td>
                  <span
                    className="legend__swatch"
                    style={{
                      background: FAMILY_COLOR[p.family],
                      display: 'inline-block',
                      marginRight: 6,
                      verticalAlign: 'middle',
                    }}
                  />
                  {p.family}
                </td>
                <td className="num">{p.round.toLocaleString()}</td>
                <td className="num">{rungLabel.get(p.rung) ?? p.rung}</td>
                <td className="num">{p.seed}</td>
                <td style={{ maxWidth: 320, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  <Program text={p.program} macros={macros} />
                </td>
                <td className="num">
                  {p.terms.slice(0, 6).join(', ')}…
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="row">
        <button type="button" disabled={safePage === 0}
                onClick={() => setPage(safePage - 1)}>
          ← previous
        </button>
        <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          page {safePage + 1} of {pageCount.toLocaleString()}
        </span>
        <button type="button" disabled={safePage >= pageCount - 1}
                onClick={() => setPage(safePage + 1)}>
          next →
        </button>
      </div>
    </>
  )
}

// ---------------------------------------------------------------------------
// Program rendering
// ---------------------------------------------------------------------------

/** The eight Brainfuck primitives plus the halt byte, per meta.bf.params. */
const PRIMITIVE_TOKENS = new Set(['>', '<', '+', '-', '[', ']', ',', '.', 'F'])

/** Colour-coded program text: prefix faint, primitives accented, macros amber. */
export function Program({ text, macros }: { text: string; macros: string[] }) {
  const macroSet = new Set(macros)
  return (
    <span className="program" style={{ display: 'inline-block', padding: '2px 6px' }}>
      {[...text].map((ch, i) => {
        const cls =
          ch === 'S'
            ? 'program__prefix'
            : macroSet.has(ch)
              ? 'program__macro'
              : PRIMITIVE_TOKENS.has(ch)
                ? 'program__op'
                : undefined
        return <span key={i} className={cls}>{ch}</span>
      })}
    </span>
  )
}

function ProgramDetail({
  program, macros, cellModulus, rungLabel, onClose,
}: {
  program: MathProgram
  macros: string[]
  cellModulus: number
  rungLabel: string
  onClose: () => void
}) {
  const check = verifyRecurrence(program.family, program.terms, cellModulus)
  return (
    <Card
      title={
        <span className="row row--tight">
          <span
            className="legend__swatch"
            style={{ background: FAMILY_COLOR[program.family] }}
          />
          {program.family} — round {program.round.toLocaleString()}
          <span className="pill">{rungLabel}</span>
          <span className="pill">{program.seed}</span>
        </span>
      }
      note={
        <span className="row row--tight">
          detected minimal period {program.period} bytes · tape{' '}
          {program.terms.length >= 0 ? '4095' : '—'} bytes
        </span>
      }
    >
      <div className="row" style={{ marginBottom: 12 }}>
        <Program text={program.program} macros={macros} />
        <button type="button" onClick={onClose}>close</button>
      </div>

      <h3 className="subhead">Emitted bytes</h3>
      <div className="tape">
        {program.terms.map((v, i) => (
          <div
            key={i}
            className={`tape__cell${i === 0 ? ' tape__cell--head' : ''}${v === 0 ? ' tape__cell--zero' : ''}`}
            title={`v[${i}] = ${v}`}
          >
            {v}
          </div>
        ))}
      </div>
      <p className="card__note">
        The first twelve bytes of the structured region the detector matched.
        The detector checked the recurrence over the whole matched region, not
        just these twelve, so a twelve-term check is a weaker statement.
      </p>

      <h3 className="subhead">Recurrence check, recomputed here</h3>
      <RecurrenceCheck family={program.family} check={check} />
    </Card>
  )
}

interface RecurrenceResult {
  ok: boolean
  /** The statement that was tested, e.g. "Δv[n] = 6 (mod 256)". */
  statement: string
  /** How many consecutive relations had to hold. */
  relations: number
  /** The first index at which the check failed, when it did. */
  firstFailure: number | null
}

/**
 * Re-verify a family's recurrence against a program's own emitted terms, mod
 * `m`. Local to this file: it is the only page that needs it, and the paper's
 * own detector is the authoritative version (see the disclosure below).
 */
function verifyRecurrence(
  family: MathFamily,
  terms: number[],
  m: number,
): RecurrenceResult {
  const n = terms.length
  const at = (i: number) => terms[i] ?? 0
  const order = family === 'arithmetic' ? 1
    : family === 'quadratic' ? 2
      : family === 'cubic' ? 3
        : 0

  if (order > 0) {
    // Difference table built forward: row 0 is the terms, row j the j-th
    // difference, every entry taken mod m. A hit is the row at this family's
    // order being constant — lower rows are *not* required to be constant
    // (a quadratic's first differences alternate).
    let row = terms
    let firstFailure: number | null = null
    let constant = 0
    for (let j = 1; j <= order; j += 1) {
      const next: number[] = []
      for (let i = 0; i + 1 < row.length; i += 1) {
        next.push(((row[i + 1]! - row[i]!) % m + m) % m)
      }
      row = next
      const head = row[0]
      if (head !== undefined) constant = head
      if (j === order) {
        const bad = row.findIndex((v) => v !== constant)
        if (bad >= 0) firstFailure = bad + 1
      }
    }
    const ord = ['Δ', 'Δ²', 'Δ³'][order - 1] ?? 'Δ'
    return {
      ok: firstFailure === null,
      statement: `${ord}v[n] = ${constant} (mod ${m})`,
      relations: n - order,
      firstFailure,
    }
  }

  if (family === 'fibonacci') {
    let firstFailure: number | null = null
    for (let i = 2; i < n; i += 1) {
      if ((((at(i) - at(i - 1) - at(i - 2)) % m) + m) % m !== 0) {
        firstFailure = i
        break
      }
    }
    return {
      ok: firstFailure === null,
      statement: `v[n] = v[n−1] + v[n−2] (mod ${m})`,
      relations: n - 2,
      firstFailure,
    }
  }

  // geometric: the ratio is unknown, so try all m of them (m = 256, trivial).
  // A leading zero makes the first step consistent with every ratio; the later
  // terms still discriminate, so the scan simply continues.
  for (let r = 0; r < m; r += 1) {
    let ok = true
    for (let i = 1; i < n; i += 1) {
      if ((((at(i) - r * at(i - 1)) % m) + m) % m !== 0) {
        ok = false
        break
      }
    }
    if (ok) {
      return {
        ok: true,
        statement: `v[n] = ${r} · v[n−1] (mod ${m})`,
        relations: n - 1,
        firstFailure: null,
      }
    }
  }
  return {
    ok: false,
    statement: `no ratio r satisfies v[n] = r · v[n−1] (mod ${m})`,
    relations: n - 1,
    firstFailure: null,
  }
}

function RecurrenceCheck({
  family, check,
}: {
  family: MathFamily
  check: RecurrenceResult
}) {
  const label = family === 'arithmetic' ? 'constant first difference'
    : family === 'quadratic' ? 'constant second difference'
      : family === 'cubic' ? 'constant third difference'
        : family === 'fibonacci' ? 'two-term sum'
          : 'constant multiplier'
  return (
    <>
      <div className="kv">
        <span className="kv__k">tested for</span>
        <span className="kv__v">{label}</span>
        <span className="kv__k">statement</span>
        <span className="kv__v">{check.statement}</span>
        <span className="kv__k">relations</span>
        <span className="kv__v">{check.relations} consecutive</span>
        <span className="kv__k">result</span>
        <span
          className="kv__v"
          style={{ color: check.ok ? 'var(--good)' : 'var(--warn)' }}
        >
          {check.ok
            ? 'holds on all twelve terms'
            : `fails at index ${check.firstFailure ?? '—'}`}
        </span>
      </div>
      <p className="card__note">
        {check.ok
          ? 'Verified in your browser from the bytes on the left, not taken from the paper. A program can pass this on twelve terms and still be a coincidence of the released slice; the detector additionally required the recurrence to hold to the end of the tape and the minimal period to exceed 30.'
          : 'This program does not satisfy the recurrence on the twelve released terms. The detector ran on the full tape, so this is a mismatch worth knowing about — it means the released slice is not always enough to see the pattern the detector saw.'}
      </p>
    </>
  )
}

// ---------------------------------------------------------------------------
// Detection criteria
// ---------------------------------------------------------------------------

function DetectionDisclosure({
  examples,
}: {
  examples: { family: MathFamily; program: string; terms: string }[]
}) {
  return (
    <Disclosure summary="How a tape counts as a hit (Appendix C)">
      <p className="body">
        A "hit" is a narrow, mechanical criterion, and a reader should know what
        it is before reading anything into what the generator discovered.{' '}
        <PaperRef reference="appC" inline />
      </p>
      <ul className="body" style={{ paddingLeft: 20, maxWidth: '78ch' }}>
        <li>
          The structured region may be preceded by up to 30 unrelated leading
          bytes. The matched region must then run <em>to the end of the tape</em>,
          so a program cannot emit structure and then pad.
        </li>
        <li>
          The minimal period of the matched region must be at least 30, checked
          both over the whole region and over the trailing 2048-byte window. The
          second check is what rejects "structured transient, then a constant
          tail" — the classic false positive.
        </li>
        <li>
          All arithmetic is modulo 256, the cell modulus. Any such mod-256
          sequence lifts to an integer-valued unmodded sequence of the same
          family, so the detector's "arbitrarily large cells" requirement is
          satisfied automatically rather than tested.
        </li>
        <li>
          Tapes shorter than 200 bytes are never considered.
        </li>
      </ul>
      <h3 className="subhead">The paper's illustrative programs</h3>
      <p className="body">
        Table 1 shows one short program per family. They are the hand-readable
        versions; the browser above holds the twenty thousand real ones, which
        are mostly longer and much less tidy.{' '}
        <PaperRef reference="table1" inline />
      </p>
      <div className="table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Family</th>
              <th>Example program</th>
              <th>Output</th>
            </tr>
          </thead>
          <tbody>
            {examples.map((e) => (
              <tr key={e.family}>
                <td>{e.family}</td>
                <td className="num">{e.program}</td>
                <td className="num">{e.terms}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="body">
        One thing the criterion does <em>not</em> require: any particular cell
        ever being written by the same instruction twice, or any arithmetic
        mnemonic appearing. These programs were not taught arithmetic. They were
        rewarded for being learnable, and arithmetic is what fell out.{' '}
        <PaperRef reference="appC" also={['sec2.1']} inline />
      </p>
    </Disclosure>
  )
}