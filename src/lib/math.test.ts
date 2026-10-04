/**
 * Every LaTeX string on the site renders.
 *
 * This exists because `throwOnError` is on in renderMath(). A malformed equation
 * would otherwise crash the page that holds it -- and the crash is at runtime, in
 * a browser, on a page nobody is looking at until someone notices it is blank.
 * Here it is a test failure instead.
 *
 * The companion check is that SITE_MATH is not silently falling behind the pages.
 * Asserting the count is crude but it fails loudly when someone adds an equation
 * and forgets to register it, which is the failure mode a render-only test cannot
 * see. If this number needs bumping, that is the signal, not an obstacle.
 */
import { describe, expect, it } from 'vitest'
import { renderMath, SITE_MATH, siteMathSources } from '@/components/Math'

describe('site mathematics', () => {
  it('renders every registered expression without throwing', () => {
    for (const { tex, display } of SITE_MATH) {
      const html = renderMath(tex, display)
      expect(html).toContain('katex')
      expect(html.length).toBeGreaterThan(0)
    }
  })

  it('renders each expression in both display and inline mode', () => {
    // Display mode wraps in .katex-display and centres; inline does not. A
    // string that only works in one mode is a string that will break somewhere.
    for (const tex of siteMathSources()) {
      expect(renderMath(tex, true)).toContain('katex-display')
      expect(renderMath(tex, false)).not.toContain('katex-display')
    }
  })

  it('escapes markup rather than emitting it', () => {
    // KaTeX has no raw-HTML passthrough, which is what makes the
    // dangerouslySetInnerHTML in <Math> sound. Prove it rather than assume it.
    const html = renderMath(String.raw`\text{<img src=x onerror=alert(1)>}`)
    expect(html).not.toContain('<img')
  })

  it('actually throws on a malformed expression', () => {
    // If this ever stops throwing, throwOnError has been turned off and the
    // loop above is testing much less than it appears to.
    expect(() => renderMath(String.raw`\frac{1}{`)).toThrow()
  })

  it('has a registry count matching what the pages declare', () => {
    // 25 -> 32 when the reward simulator page landed: the factored reward, its
    // three underbraced terms, delta-theta_e restated, and g_0 / g_phi on the
    // limits table.
    expect(SITE_MATH.length).toBe(32)
  })
})