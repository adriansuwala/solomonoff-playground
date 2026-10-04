/**
 * KaTeX rendering.
 *
 * The equations used to live in `.program` divs -- monospace, hand-set Unicode,
 * no fractions. That read acceptably for one line and badly for Equation 2, which
 * has a nested floor in a subscript. Everything mathematical now goes through
 * <Math>, so there is one place that decides how notation is typeset.
 *
 * KaTeX is synchronous. The markup is injected via dangerouslySetInnerHTML,
 * which is sound here for one specific reason: KaTeX HTML-escapes every token it
 * emits, and there is no raw-HTML escape hatch, so no author-supplied string can
 * reach the DOM as markup. `throwOnError` is on, so a malformed expression is a
 * visible crash rather than a silent red box -- and the test suite renders every
 * equation on the site, so a typo fails `npm test` before it fails a reader.
 */
import katex from 'katex'

export type MathProps = {
  /** LaTeX source. Use `\\` in a JS string literal, or String.raw for clarity. */
  tex: string
  /** Block display (centred, own line) vs inline (sits in a sentence). */
  display?: boolean
}

/** Render LaTeX to KaTeX HTML. Throws on a malformed expression. */
export function renderMath(tex: string, display = false): string {
  return katex.renderToString(tex, {
    displayMode: display,
    throwOnError: true,
    // The site is already dark; KaTeX would otherwise emit black-on-dark.
    output: 'html',
    strict: false,
  })
}

/**
 * Every equation string on the site, so the test suite can render them all and
 * prove none of them throw. Keep this list in sync when adding an equation --
 * a missing entry is a silent gap in coverage, not a failure.
 */
export const SITE_MATH: { tex: string; display: boolean }[] = [
  // Method: Equation 1 -- the learner's loss.
  { tex: String.raw`L(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\log p_\theta\!\left(y_t \mid y_{<t}\right)`, display: true },
  // Method: Equation 2 -- the reward, with the preconditioner on its own line.
  { tex: String.raw`r_i = \left|\left\langle \nabla_\theta L(y_i;\theta_{\text{now}}),\, P_e \odot \delta\theta_e \right\rangle\right|`, display: true },
  { tex: String.raw`\delta\theta_e = \theta_{\lfloor e/2 \rfloor} - \theta_e, \qquad P_e = \operatorname{diag}\!\left(\sqrt{\hat{v}_e} + \varepsilon\right)`, display: true },
  // Method: Equation 3 -- the generator's RL objective.
  { tex: String.raw`J(\phi) = \mathbb{E}_x\!\left[\frac{r(x)}{\beta}\log\frac{g_\phi(x)}{g_0(x)}\right] - \beta\, \mathrm{KL}\!\left(g_\phi \| g_0\right)`, display: true },
  { tex: String.raw`g_0(x) = |A|^{-\mathrm{len}(x)}, \qquad |A| = \text{alphabet size}`, display: true },
  // Method: Equation 4 -- the GRPO advantage.
  { tex: String.raw`\hat{A}_i = \frac{r_i - \mathrm{mean}_j r_j}{\mathrm{std}_j r_j + \varepsilon}`, display: true },
  // Method: Equation 5 -- expert iteration.
  { tex: String.raw`\mathcal{L}_{\mathrm{SFT}}(\phi) = -\,\mathbb{E}_{x \sim g_\phi}\!\left[w(x)\log g_\phi(x)\right], \qquad w(x) \propto \max\left(0,\ r(x) - \tau\right)`, display: true },
  // Reward / Overview: the same Equation 2 restated inline in prose.
  { tex: String.raw`r_i = |\langle \nabla_\theta L(y_i;\theta_{\text{now}}), P_e \odot \delta\theta_e\rangle|`, display: false },
  { tex: String.raw`\delta\theta_e`, display: false },
  { tex: String.raw`\theta_{\lfloor e/2 \rfloor} - \theta_e`, display: false },
  { tex: String.raw`P_e = \operatorname{diag}(\sqrt{\hat{v}_e} + \varepsilon)`, display: false },
  { tex: String.raw`L(y_i;\theta)`, display: false },
  { tex: String.raw`\theta_{\text{now}}`, display: false },
  { tex: String.raw`\theta_e`, display: false },
  { tex: String.raw`\delta\theta`, display: false },
  { tex: String.raw`L(\theta_{\text{pre}}) - L(\theta_{\text{post}})`, display: false },
  { tex: String.raw`\tau`, display: false },
  { tex: String.raw`\varepsilon`, display: false },
  { tex: String.raw`\beta`, display: false },
  // Prose fragments. Small, but each one was hand-set Unicode before, and the
  // floor and the abs-value bars are the two that read worst in monospace.
  { tex: String.raw`\lfloor e/2 \rfloor`, display: false },
  { tex: String.raw`\theta_{\lfloor e/2 \rfloor}`, display: false },
  { tex: String.raw`\theta_{e-1}`, display: false },
  { tex: String.raw`|\cdot|`, display: false },
  { tex: String.raw`\mathcal{L}_{\mathrm{SFT}}(\phi)`, display: false },
  { tex: String.raw`g_\phi`, display: false },
]

/** Every LaTeX string on the site, flattened for tests. */
export function siteMathSources(): string[] {
  return SITE_MATH.map((m) => m.tex)
}

/**
 * Render LaTeX. Display mode gets its own centred line; inline mode sits in the
 * sentence around it and inherits the current font size.
 */
export function Math({ tex, display = false }: MathProps) {
  const html = renderMath(tex, display)
  if (display) {
    return (
      <div
        className="math math--display"
        // KaTeX output, escaped by KaTeX. See the note at the top of this file.
        dangerouslySetInnerHTML={{ __html: html }}
      />
    )
  }
  return (
    <span
      className="math math--inline"
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}