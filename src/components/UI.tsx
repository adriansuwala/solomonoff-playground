/**
 * Paper section reference badge.
 *
 * D6: every claim in this app names where it comes from. This component is the
 * only sanctioned way to render a number, and `tools/verify_data.py` checks
 * that every reference in the data resolves against SECTIONS.
 */
import { useEffect, useState, type ReactNode } from 'react'
import { useSectionResolver } from '@/lib/sections'
import type { SectionKey } from '@/data/types'

export interface PaperRefProps {
  reference: SectionKey
  also?: SectionKey[]
  /** Render inline after a sentence instead of on its own line. */
  inline?: boolean
  prefix?: string
}

export function PaperRef({ reference, also, inline, prefix = 'from' }: PaperRefProps) {
  const resolve = useSectionResolver()
  const label = resolve(reference)
  const extras = (also ?? []).map((k) => resolve(k))
  return (
    <span
      className={`paperef${inline ? ' paperef--inline' : ''}`}
      title={`Source: ${label}${extras.length ? `; ${extras.join('; ')}` : ''}`}
    >
      <span style={{ color: 'var(--text-faint)' }}>{prefix}</span>{' '}
      <span className="paperef__label">{label}</span>
      {extras.length > 0 && (
        <span style={{ color: 'var(--text-faint)' }}> · {extras.join(' · ')}</span>
      )}
    </span>
  )
}

/**
 * A claim block: the statement, its evidence, and where it is from.
 * `caveat` renders in the warning style for claims that need a limit stated.
 */
export function Claim({
  children, reference, also, caveat,
}: {
  children: ReactNode
  reference: SectionKey
  also?: SectionKey[]
  caveat?: boolean
}) {
  return (
    <div className={`claim${caveat ? ' claim--caveat' : ''}`}>
      <p className="claim__text">
        {children}{' '}
        <PaperRef reference={reference} also={also} inline />
      </p>
    </div>
  )
}

/** Card with a title and an optional provenance note. */
export function Card({
  title, note, children, className,
}: {
  title?: ReactNode
  note?: ReactNode
  children: ReactNode
  className?: string
}) {
  return (
    <div className={`card${className ? ` ${className}` : ''}`}>
      {title && <div className="card__title">{title}</div>}
      {note && <div className="card__note">{note}</div>}
      {children}
    </div>
  )
}

/** A labelled statistic. */
export function Stat({ value, label }: { value: ReactNode; label: ReactNode }) {
  return (
    <div className="stat">
      <div className="stat__value">{value}</div>
      <div className="stat__label">{label}</div>
    </div>
  )
}

export function Legend({
  items,
}: {
  items: { label: string; color: string; dash?: string | null }[]
}) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span className="legend__item" key={it.label}>
          {it.dash ? (
            <span
              className="legend__swatch legend__swatch--dashed"
              style={{ color: it.color, width: 18 }}
            />
          ) : (
            <span className="legend__swatch" style={{ background: it.color }} />
          )}
          {it.label}
        </span>
      ))}
    </div>
  )
}

/** Collapsible detail, for methodology a reader can skip. */
export function Disclosure({
  summary, children, defaultOpen = false,
}: {
  summary: ReactNode
  children: ReactNode
  defaultOpen?: boolean
}) {
  return (
    <details className="disclosure" open={defaultOpen}>
      <summary>{summary}</summary>
      {children}
    </details>
  )
}

/** Loading / error placeholder. */
export function Async<T>({
  load, children,
}: {
  load: () => Promise<T>
  children: (data: T) => ReactNode
}) {
  const [state, setState] = useState<
    { kind: 'loading' } | { kind: 'error'; error: Error } | { kind: 'ready'; data: T }
  >({ kind: 'loading' })

  useEffect(() => {
    let live = true
    load().then(
      (data) => live && setState({ kind: 'ready', data }),
      (error: Error) => live && setState({ kind: 'error', error }),
    )
    return () => { live = false }
  }, [load])

  if (state.kind === 'loading') return <div className="loading">Loading…</div>
  if (state.kind === 'error') {
    return (
      <div className="error">
        Failed to load data: {state.error.message}
        <br />
        <small>Run <code>npm run data:build</code> to regenerate public/data/.</small>
      </div>
    )
  }
  return <>{children(state.data)}</>
}