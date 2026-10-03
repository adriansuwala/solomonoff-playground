/**
 * Scopes for the paper's claim -> section reference requirement (D6).
 *
 * Loading the meta bundle resolves every key used here; an unresolvable
 * reference throws at render time in development rather than silently showing a
 * bare label. tools/verify_data.py independently checks that every reference in
 * the data resolves against SECTIONS.
 */
import { createContext, useContext } from 'react'
import type { SectionKey, Sections } from '@/data/types'

export type SectionResolver = (key: SectionKey) => string

const SectionContext = createContext<SectionResolver | null>(null)

export const SectionProvider = SectionContext.Provider

export function useSectionResolver(): SectionResolver {
  const fn = useContext(SectionContext)
  if (!fn) {
    throw new Error('useSectionResolver used outside <SectionProvider>')
  }
  return fn
}

/** Build a resolver from the meta bundle. Unknown keys throw. */
export function makeResolver(sections: Sections): SectionResolver {
  return (key: SectionKey) => {
    const label = sections[key]
    if (!label) {
      throw new Error(
        `unknown paper section reference ${JSON.stringify(key)}; ` +
        'add it to SECTIONS in tools/paper_refs.py',
      )
    }
    return label
  }
}