import { describe, expect, it } from 'vitest'
import {
  chinchillaToComputeExponent, decadeTicks, formatCompute, formatPowerOfTen,
  lossDomain, makeLinearY, makeLogX, predictFit,
} from './scales'

describe('lossDomain', () => {
  it('falls back to the uniform-over-256 reference when nothing is finite', () => {
    expect(lossDomain([])).toEqual([0, 8])
    expect(lossDomain([NaN, Infinity, -1, 0])).toEqual([0, 8])
  })

  it('brackets the data with a small positive pad and never goes below zero', () => {
    const [lo, hi] = lossDomain([6.1, 6.2, 6.15])
    expect(lo).toBeLessThan(6.1)
    expect(hi).toBeGreaterThan(6.2)
    expect(lo).toBeGreaterThanOrEqual(0)
  })

  it('clamps to zero rather than returning a negative axis bound', () => {
    const [lo] = lossDomain([0.01, 0.02])
    expect(lo).toBe(0)
  })
})

describe('makeLogX', () => {
  it('maps the domain endpoints onto the range', () => {
    const x = makeLogX([1e15, 1e18], [0, 300])
    expect(x(1e15)).toBeCloseTo(0, 6)
    expect(x(1e18)).toBeCloseTo(300, 6)
  })

  it('is monotonic increasing', () => {
    const x = makeLogX([1e14, 1e20], [0, 100])
    expect(x(1e16)).toBeLessThan(x(1e18))
  })

  it('drops non-positive and non-finite values, which log10 cannot represent', () => {
    // 0 is the round index of the untrained model: including it must not
    // produce NaN, which is the F5 bug in docs/findings.md.
    const x = makeLogX([0, 1e15, NaN, 1e18], [0, 300])
    expect(Number.isFinite(x(1e16))).toBe(true)
  })

  it('returns a usable scale when given nothing usable', () => {
    const x = makeLogX([0, -5, NaN], [0, 300])
    expect(Number.isFinite(x(5))).toBe(true)
  })
})

describe('makeLinearY', () => {
  it('is inverted: larger loss sits higher on screen (smaller y)', () => {
    const y = makeLinearY([5, 7], [300, 0])
    expect(y(5)).toBe(300)
    expect(y(7)).toBe(0)
    expect(y(5)).toBeGreaterThan(y(7))
  })

  it('clamps outside the domain instead of drawing off-canvas', () => {
    const y = makeLinearY([5, 7], [300, 0])
    expect(y(1)).toBe(300)
    expect(y(99)).toBe(0)
  })
})

describe('predictFit', () => {
  const fit = { amplitude: 3.0, floor: 4.75, alpha: 0.123 }

  it('evaluates L(C) = floor + amplitude * C^-alpha', () => {
    expect(predictFit(fit, 1e18)).toBeCloseTo(
      fit.floor + fit.amplitude * Math.pow(1e18, -fit.alpha), 12,
    )
  })

  it('decreases with compute, as a scaling law must', () => {
    expect(predictFit(fit, 1e20)).toBeLessThan(predictFit(fit, 1e16))
  })

  it('approaches the fitted floor asymptotically and never crosses it', () => {
    let prev = Infinity
    for (const c of [1e15, 1e18, 1e21, 1e24, 1e30]) {
      const v = predictFit(fit, c)
      expect(v).toBeGreaterThan(fit.floor)
      expect(v).toBeLessThan(prev)
      prev = v
    }
    expect(prev).toBeCloseTo(fit.floor, 2)
  })

  it('decreases in alpha at fixed compute, given positive amplitude', () => {
    const slow = predictFit({ ...fit, alpha: 0.05 }, 1e18)
    const fast = predictFit({ ...fit, alpha: 0.30 }, 1e18)
    expect(fast).toBeLessThan(slow)
  })

  it('reproduces the text-vs-DNA shape: a larger alpha decays faster', () => {
    // Guards the visual claim on the Scaling page that DNA (0.435) falls much
    // faster than text (0.123) over the same compute range.
    const dna = predictFit({ amplitude: 3, floor: 0.4, alpha: 0.435 }, 1e20)
      - predictFit({ amplitude: 3, floor: 0.4, alpha: 0.435 }, 1e16)
    const dclm = predictFit({ amplitude: 3, floor: 4.75, alpha: 0.123 }, 1e20)
      - predictFit({ amplitude: 3, floor: 4.75, alpha: 0.123 }, 1e16)
    expect(dna).toBeGreaterThan(dclm)
  })
})

describe('chinchillaToComputeExponent', () => {
  it('converts a published (alpha, beta) pair to a compute exponent', () => {
    expect(chinchillaToComputeExponent(0.1, 0.1)).toBeCloseTo(0.05, 12)
  })

  it('lands well below the smaller of the two data exponents', () => {
    // b = ab/(a+b) is half the smaller exponent when they are equal, so for
    // (0.2, 0.3) it is 0.12 -- below half of 0.2, not between them.
    const b = chinchillaToComputeExponent(0.2, 0.3)
    expect(b).toBeCloseTo(0.12, 12)
    expect(b).toBeLessThan(0.2)
    expect(b).toBeLessThan(0.3)
    // Less than half the smaller exponent, which is the equal case's value.
    expect(b).toBeLessThan(0.15)
  })

  it('agrees with the paper caption form b = alpha*beta/(alpha+beta)', () => {
    const alpha = 0.123, beta = 0.153
    expect(chinchillaToComputeExponent(alpha, beta))
      .toBeCloseTo((alpha * beta) / (alpha + beta), 12)
  })

  it('does not need special-casing when alpha and beta are equal', () => {
    expect(chinchillaToComputeExponent(0.08, 0.08)).toBeCloseTo(0.04, 12)
  })
})

describe('formatCompute', () => {
  it('renders SI-prefixed FLOP counts', () => {
    expect(formatCompute(2.5e18)).toBe('2.5E')
    expect(formatCompute(1e15)).toBe('1.0P')
    expect(formatCompute(3.2e9)).toBe('3.2G')
  })

  it('keeps one decimal below a mantissa of 100 and drops it above', () => {
    expect(formatCompute(5e18)).toBe('5.0E')
    expect(formatCompute(250e15)).toBe('250P')
    expect(formatCompute(1.2e21)).toBe('1200E')
  })

  it('degrades to an em dash for non-positive or non-finite input', () => {
    expect(formatCompute(0)).toBe('—')
    expect(formatCompute(-1)).toBe('—')
    expect(formatCompute(NaN)).toBe('—')
  })
})

describe('formatPowerOfTen', () => {
  it('uses superscript digits so ticks need no MathML', () => {
    expect(formatPowerOfTen(18)).toBe('10¹⁸')
    expect(formatPowerOfTen(0)).toBe('10⁰')
  })

  it('handles negative exponents', () => {
    expect(formatPowerOfTen(-3)).toBe('10⁻³')
  })

  it('rounds rather than emitting a decimal point', () => {
    expect(formatPowerOfTen(15.2)).toBe('10¹⁵')
  })
})

describe('decadeTicks', () => {
  it('emits one tick per decade inside the domain', () => {
    const ticks = decadeTicks([1e15, 1e18])
    expect(ticks).toEqual([1e15, 1e16, 1e17, 1e18])
  })

  it('excludes decades outside the domain', () => {
    expect(decadeTicks([2e15, 5e16])).toEqual([1e16])
  })

  it('returns nothing for a domain containing zero, which log10 cannot use', () => {
    expect(decadeTicks([0, 1e18])).toEqual([])
    expect(decadeTicks([-1, 10])).toEqual([])
  })

  it('refuses to emit thousands of ticks for a pathological domain', () => {
    expect(decadeTicks([1e-40, 1e40])).toEqual([])
  })

  it('falls back to the domain endpoints when no decade falls inside it', () => {
    // Without this the axis renders blank, which reads as a broken chart.
    expect(decadeTicks([5e16, 9e16])).toEqual([5e16, 9e16])
  })
})