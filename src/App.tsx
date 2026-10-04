/**
 * App shell: sidebar navigation plus the section resolver from meta.json.
 *
 * The resolver is created once from the loaded meta bundle and provided through
 * context, so a paper reference that does not resolve throws at render time
 * instead of silently rendering a bare label (D6).
 */
import { useLayoutEffect, useMemo, useRef, useState } from 'react'
import { Async } from '@/components/UI'
import { loadMeta } from '@/data/loaders'
import { makeResolver, SectionProvider } from '@/lib/sections'
import type { Meta, SectionKey } from '@/data/types'

import { Overview } from '@/pages/Overview'
import { Method } from '@/pages/Method'
import { Scaling } from '@/pages/Scaling'
import { Reward } from '@/pages/Reward'
import { RewardSim } from '@/pages/RewardSim'
import { Curriculum } from '@/pages/Curriculum'
import { DiscoveredMath } from '@/pages/DiscoveredMath'
import { InContextLearning } from '@/pages/InContextLearning'
import { Encodings } from '@/pages/Encodings'

export interface NavEntry {
  id: string
  label: string
  group: string
  ref: SectionKey
  render: () => JSX.Element
}

export const NAV: NavEntry[] = [
  { id: 'overview', label: 'Overview', group: 'The claim', ref: 'abstract',
    render: () => <Overview /> },
  { id: 'method', label: 'How it works', group: 'The claim', ref: 'sec2',
    render: () => <Method /> },
  { id: 'scaling', label: 'Scaling laws', group: 'The evidence', ref: 'sec3.1',
    render: () => <Scaling /> },
  { id: 'reward', label: 'Reward ablations', group: 'The evidence', ref: 'appF',
    render: () => <Reward /> },
  { id: 'reward-sim', label: 'Inside the reward', group: 'The evidence', ref: 'eq2',
    render: () => <RewardSim /> },
  { id: 'curriculum', label: 'Curriculum value', group: 'The evidence', ref: 'fig3',
    render: () => <Curriculum /> },
  { id: 'math', label: 'Discovered structure', group: 'The evidence', ref: 'appC',
    render: () => <DiscoveredMath /> },
  { id: 'icl', label: 'In-context learning', group: 'The evidence', ref: 'sec3.2',
    render: () => <InContextLearning /> },
  { id: 'encodings', label: 'What the bytes are', group: 'Reference', ref: 'appB',
    render: () => <Encodings /> },
]

export function App() {
  const [page, setPage] = useState(NAV[0]!.id)

  return (
    <Async load={loadMeta}>
      {(meta) => <Shell meta={meta} page={page} onNavigate={setPage} />}
    </Async>
  )
}

/**
 * Put the reader at the top of a page when the page changes.
 *
 * The window is the scroller here, not `.main` -- body scrolls and the sidebar
 * has its own overflow -- so this is window.scrollTo rather than a container
 * reset. Without it, arriving halfway down a long page such as Scaling lands you
 * halfway down the next one, which reads as a broken page rather than a
 * continuation.
 *
 * `useLayoutEffect` rather than `useEffect`: the reset has to land in the same
 * commit as the new content. With `useEffect` the browser paints the new page at
 * the old scroll offset first, and the jump is visible.
 *
 * The ref makes this fire only on a real page change, so a re-render triggered by
 * data arriving does not yank the reader back to the top mid-page.
 */
function resetScrollOnChange(page: string) {
  const first = useRef(true)
  useLayoutEffect(() => {
    if (first.current) {
      // Mounting: whatever offset the browser restored from history or a
      // deep-link anchor is deliberate. Leave it alone.
      first.current = false
      return
    }
    window.scrollTo(0, 0)
  }, [page])
}

/**
 * Split out of App so the resolver's useMemo is a hook of a component rather
 * than of the render callback Async invokes. A callback passed to Async is not
 * a component boundary: Async skips calling it on the loading pass, so any
 * hook inside it changes hook count between renders and React tears the tree
 * down with "rendered more hooks than during the previous render". That blanked
 * every page.
 */
function Shell({
  meta, page, onNavigate,
}: {
  meta: Meta
  page: string
  onNavigate: (id: string) => void
}) {
  const resolve = useMemo(() => makeResolver(meta.sections), [meta])
  const active = NAV.find((n) => n.id === page) ?? NAV[0]!
  const groups = [...new Set(NAV.map((n) => n.group))]
  resetScrollOnChange(page)

  return (
    <SectionProvider value={resolve}>
      <div className="app">
        <nav className="sidebar" aria-label="Sections">
          <div className="sidebar__title">{meta.paper.title}</div>
          <div className="sidebar__subtitle">
            arXiv:{meta.paper.arxiv} · an interactive reading
          </div>
          {groups.map((g) => (
            <div key={g}>
              <div className="sidebar__group">{g}</div>
              {NAV.filter((n) => n.group === g).map((n) => (
                <a
                  key={n.id}
                  className={`sidebar__link${n.id === active.id ? ' sidebar__link--active' : ''}`}
                  href={`#${n.id}`}
                  aria-current={n.id === active.id ? 'page' : undefined}
                  onClick={(e) => { e.preventDefault(); onNavigate(n.id) }}
                >
                  {n.label}
                </a>
              ))}
            </div>
          ))}
        </nav>
        <main className="main">
          {active.render()}
          <div className="footer">
            <p>
              Every figure on these pages is recomputed in the browser from
              data released by the paper's authors at{' '}
              <a href={meta.paper.codeRepo} target="_blank" rel="noreferrer">
                {meta.paper.codeRepo}
              </a>
              {' '}and cross-checked against the paper's own tables. Nothing
              here is transcribed by hand.
            </p>
            <p>{meta.provenance.license}</p>
          </div>
        </main>
      </div>
    </SectionProvider>
  )
}
